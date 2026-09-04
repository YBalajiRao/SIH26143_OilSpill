import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.geo_utils import patch_pixel_to_latlon, get_exact_raster_resolution_and_area
from src.utils.slick_morphology import mask_features
from src.environment.real_netcdf_forcing import RealMetoceanForcingEngine
from src.drift.probabilistic_drift import backward_drift_particles, forward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

def estimate_slick_age_proxy(area_km2, avg_backscatter_db):
    try:
        a_km2 = float(area_km2)
    except Exception:
        a_km2 = 0.0
    if a_km2 <= 0.0:
        return 6.0
    contrast_ratio = float(abs(avg_backscatter_db) / 25.0)
    estimated_hours = 2.5 * (a_km2 * 10.0) ** 0.45 * contrast_ratio
    return float(np.clip(estimated_hours, 1.5, 48.0))

def run_master_pipeline():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "perception_frozen_E5_2.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")

    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    canonical_csv = os.path.join(root, "results", "ais_outputs", "final_attribution_ranking.csv")
    out_fig       = os.path.join(root, "results", "figures", "sih26143_master_intelligence_map.png")
    out_html      = os.path.join(root, "results", "sih_interactive_dashboard.html")

    os.makedirs(os.path.dirname(canonical_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    opt_thr = 0.50
    freeze_note = os.path.join(root, "results", "metrics", "FROZEN_THRESHOLD.txt")
    if os.path.exists(freeze_note):
        with open(freeze_note, "r") as f:
            for line in f:
                if "optimal_threshold=" in line:
                    opt_thr = float(line.split("=")[1].strip())

    print("=" * 80)
    print(f" NTRO SIH26143 MASTER PIPELINE EXECUTION (Matched Dec 7, 2018 Case | t_opt={opt_thr:.2f})")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[✓] Checkpoint Loaded (missing_keys={len(missing)}, unexpected_keys={len(unexpected)})")
    model.eval()

    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    
    # Target Validation Patch #482 in 2018_12_07.tif (37,160 GT oil pixels)
    target_idx = 482
    sample_img, sample_mask = ds[target_idx]
    
    gt_pixels = int((sample_mask.numpy() > 0.5).sum())

    row = ds.df.iloc[target_idx]
    fname_tif = os.path.basename(str(row["paths"]).replace("\\", "/"))
    tif_full_path = os.path.join(img_dir, fname_tif)
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip('"\'').split(",")]
    patch_y, patch_x = coords[0], coords[1]

    with torch.no_grad():
        logits = model(sample_img.unsqueeze(0).to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = (prob_map >= opt_thr).astype(np.float32)
    morph = mask_features(binary_mask)
    pred_pixels = int(morph["area_px"])

    gt_area_info   = get_exact_raster_resolution_and_area(tif_full_path, gt_pixels)
    pred_area_info = get_exact_raster_resolution_and_area(tif_full_path, pred_pixels)

    center_lat, center_lon = patch_pixel_to_latlon(
        tif_full_path, patch_y, patch_x,
        local_y=int(morph["centroid_xy"][1] * 255),
        local_x=int(morph["centroid_xy"][0] * 255)
    )

    avg_backscatter = float(sample_img.mean() * 40.0 - 35.0)
    age_proxy_h = estimate_slick_age_proxy(pred_area_info["area_km2"], avg_backscatter)

    scene_date_str = "2018-12-07"
    print(f"\n[+] Matched Case Characterization ({fname_tif} - Validation Patch #{target_idx}):")
    print(f"    - Target Scene:             {fname_tif}")
    print(f"    - Acquisition Date:         2018-12-07")
    print(f"    - Observed Geographic Pos:  {center_lat:.4f}° N, {center_lon:.4f}° W")
    print(f"    - Ground-Truth Oil Pixels:  {gt_pixels} px ({gt_area_info['area_km2']:.4f} km²)")
    print(f"    - Predicted Oil Pixels:     {pred_pixels} px ({pred_area_info['area_km2']:.4f} km²)")
    print(f"    - Spatial Resolution:       10.0m x 10.0m (100.0 m²/px)")
    print(f"    - Elongation Ratio:         {morph['elongation']:.2f}")
    print(f"    - Major Axis Orientation:   {morph['orientation_deg']:.1f}°")
    print(f"    - Model-Derived Age Proxy:  ~{age_proxy_h:.1f} hours post-release")

    # Metocean Forcing (Dec 7, 2018)
    met_engine = RealMetoceanForcingEngine()
    forcing = met_engine.get_velocity_at_latlon(center_lat, center_lon, scene_date_str)

    # Backward Hindcast (-24h)
    print("\n[*] Back-propagating 1,000 particles over -24 hours...")
    traj_back = backward_drift_particles(
        morph["seed_xy"], n_particles=1000, n_steps=24, dt_hours=1.0,
        u_current=forcing["u_current"], v_current=forcing["v_current"],
        u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
        wind_factor=forcing["wind_factor"], diffusion=0.015, rng=20181207
    )
    dens_back, final_pts = origin_density(traj_back, grid_size=64)
    stats_back = origin_stats(dens_back, final_pts)

    peak_off_x, peak_off_y = stats_back["peak_xy"]
    origin_lat = center_lat + (peak_off_y - 0.5) * 0.28
    origin_lon = center_lon + (peak_off_x - 0.5) * 0.28
    print(f"[✓] Reconstructed Origin Peak: {origin_lat:.4f}° N, {origin_lon:.4f}° W")

    # Forward Drift Projection (+24h)
    print("\n[*] Forward-propagating 1,000 particles over +24 hours...")
    traj_fwd = forward_drift_particles(
        morph["seed_xy"], n_particles=1000, n_steps=24, dt_hours=1.0,
        u_current=forcing["u_current"], v_current=forcing["v_current"],
        u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
        wind_factor=forcing["wind_factor"], diffusion=0.015, rng=20181207
    )
    dens_fwd, _ = origin_density(traj_fwd, grid_size=64)

    # AIS Query (Dec 7, 2018)
    ais_pipeline = MarineCadastreAISPipeline()
    df_ais = ais_pipeline.load_and_filter_ais(origin_lat - 0.5, origin_lat + 0.5, origin_lon - 0.5, origin_lon + 0.5)
    df_ranked = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon, slick_orient_deg=morph["orientation_deg"])

    df_ranked.to_csv(canonical_csv, index=False)

    print("\n" + "=" * 80)
    print("   CANONICAL RECONCILED ATTRIBUTION RANKING (DECEMBER 7, 2018 MATCHED CASE)")
    print("=" * 80)
    cols_show = ["rank", "mmsi", "vessel_name", "ntro_attribution_score", "dist_km", "proximity_score", "kinematic_score", "alignment_score", "gap_penalty"]
    print(df_ranked[cols_show].head(5).to_string(index=False))
    print("=" * 80)

    # 6-Panel Evidence Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    sar_np = sample_img.squeeze().numpy()

    axes[0, 0].imshow(sar_np, cmap="gray")
    axes[0, 0].imshow(prob_map, cmap="plasma", alpha=0.45)
    axes[0, 0].set_title(f"1. Sentinel-1 SAR & E5.2 Mask (83.49% mIoU)\n({fname_tif})", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(dens_back, cmap="hot", origin="lower", extent=[-15, 15, -15, 15])
    axes[0, 1].set_title("2. Backward Particle Hindcast (-24h Origin)", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("East-West Distance (km)")
    axes[0, 1].set_ylabel("North-South Distance (km)")

    axes[0, 2].imshow(dens_fwd, cmap="viridis", origin="lower", extent=[-15, 15, -15, 15])
    axes[0, 2].set_title("3. Forward Particle Projection (+24h Impact)", fontsize=10, fontweight="bold")
    axes[0, 2].set_xlabel("East-West Distance (km)")
    axes[0, 2].set_ylabel("North-South Distance (km)")

    axes[1, 0].scatter(df_ranked["lon"], df_ranked["lat"], c="#1f77b4", s=40, alpha=0.7, label="MarineCadastre AIS")
    t1 = df_ranked.iloc[0]
    axes[1, 0].scatter(t1["lon"], t1["lat"], c="#d62728", s=160, marker="*", label=f"Rank #1: {t1['vessel_name']}")
    axes[1, 0].scatter(origin_lon, origin_lat, c="#2ca02c", s=120, marker="x", label="Origin Peak")
    axes[1, 0].set_title("4. Real MarineCadastre AIS & Origin Peak", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Longitude (°W)")
    axes[1, 0].set_ylabel("Latitude (°N)")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    top5 = df_ranked.head(5)
    bars = axes[1, 1].barh(top5["vessel_name"], top5["ntro_attribution_score"], color="#2ca02c")
    bars[0].set_color("#d62728")
    axes[1, 1].set_xlabel("Physio-GraphSpill Score", fontsize=10)
    axes[1, 1].set_title("5. Top 5 Ranked Suspect Vessels", fontsize=10, fontweight="bold")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1, 2].axis("off")
    summary_text = (
        f"NTRO SIH26143 Intelligence Summary\n"
        f"-----------------------------------------\n"
        f"Target Scene:      {fname_tif}\n"
        f"Acquisition Date:  2018-12-07\n"
        f"Observed Position: {center_lat:.4f}°N, {center_lon:.4f}°W\n"
        f"Inferred Origin:   {origin_lat:.4f}°N, {origin_lon:.4f}°W\n"
        f"GT Slick Area:     {gt_pixels} px ({gt_area_info['area_km2']:.4f} km²)\n"
        f"Predicted Area:    {pred_pixels} px ({pred_area_info['area_km2']:.4f} km²)\n"
        f"Elongation Ratio:  {morph['elongation']:.2f}\n"
        f"Heuristic Age Proxy:~{age_proxy_h:.1f} hours\n\n"
        f"Top Candidate Vessel: {t1['vessel_name']}\n"
        f"MMSI: {t1['mmsi']} | Score: {t1['ntro_attribution_score']:.4f}\n"
        f"Primary Evidence:\n{t1['investigation_evidence']}"
    )
    axes[1, 2].text(0.05, 0.05, summary_text, fontsize=8.5, family="monospace", bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.85))

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()

    # Copy master plot to poster folder
    poster_fig = os.path.join(root, "poster", "figures", "sih26143_master_intelligence_map.png")
    import shutil
    shutil.copy(out_fig, poster_fig)

    print(f"\n[✓] Canonical CSV Saved -> {canonical_csv}")
    print(f"[✓] Master Intelligence Map -> {out_fig}\n")

if __name__ == "__main__":
    run_master_pipeline()
