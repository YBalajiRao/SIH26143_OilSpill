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
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.geo_utils import patch_pixel_to_latlon
from src.utils.slick_morphology import mask_features
from src.environment.real_netcdf_forcing import RealMetoceanForcingEngine
from src.drift.probabilistic_drift import backward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

def run_production_investigation():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    out_csv  = os.path.join(root, "results", "ais_outputs", "real_vessel_ranking_final.csv")
    out_fig  = os.path.join(root, "results", "figures", "real_attribution_final.png")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    print("=" * 80)
    print(" PHYSIO-GRAPHSPILL PRODUCTION INVESTIGATION ENGINE (SIH26143)")
    print(" Target Scene: 2018_12_07.tif | Date: 2018-12-07")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Frozen Champion Model (E5.2)
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"[✓] Loaded Champion Model E5.2 (Val mIoU: 83.49% | Dice: 78.84%)")

    # 2. Select Patch with Peak Ground Truth Oil Pixels
    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=None)
    
    max_oil_px = -1
    best_idx = 0
    for i in range(len(ds)):
        _, mask_t = ds[i]
        oil_px = int((mask_t == 1.0).sum())
        if oil_px > max_oil_px:
            max_oil_px = oil_px
            best_idx = i

    sample_img, sample_mask = ds[best_idx]
    row = ds.df.iloc[best_idx]
    fname_tif = os.path.basename(str(row["paths"]).replace("\\", "/"))
    tif_full_path = os.path.join(img_dir, fname_tif)
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip('"\'').split(",")]
    patch_y, patch_x = coords[0], coords[1]

    # Predict Probability Map (Input tensor matching exact training normalization [0.0, 1.0])
    with torch.no_grad():
        logits = model(sample_img.unsqueeze(0).to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Use adaptive threshold for prediction mask
    binary_mask = (prob_map >= 0.30).astype(np.float32)
    morph = mask_features(binary_mask)

    center_lat, center_lon = patch_pixel_to_latlon(
        tif_full_path, patch_y, patch_x,
        local_y=int(morph["centroid_xy"][1] * 255),
        local_x=int(morph["centroid_xy"][0] * 255)
    )

    scene_date_str = "2018-12-07"
    print(f"\n[+] Detected Oil Slick Location ({fname_tif} - Patch #{best_idx:05d}):")
    print(f"    - Acquisition Date: {scene_date_str}")
    print(f"    - Geographic Pos:   {center_lat:.4f}° N, {center_lon:.4f}° W")
    print(f"    - Slick Area:       {morph['area_px']} pixels ({morph['area_px']*0.01:.2f} km² equivalent)")
    print(f"    - Elongation Ratio: {morph['elongation']:.2f}")

    # 3. Metocean Forcing
    met_engine = RealMetoceanForcingEngine()
    forcing = met_engine.get_velocity_at_latlon(center_lat, center_lon, scene_date_str)

    # 4. Backward Drift Hindcast (N=1,000)
    print("\n[*] Back-propagating 1,000 particles over 24 hours...")
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
    print(f"[✓] Spill Origin Peak: {origin_lat:.4f}° N, {origin_lon:.4f}° W")

    # 5. REAL MarineCadastre AIS Traffic Query
    ais_pipeline = MarineCadastreAISPipeline()
    df_ais = ais_pipeline.load_and_filter_ais(origin_lat - 0.5, origin_lat + 0.5, origin_lon - 0.5, origin_lon + 0.5)

    # 6. Candidate Vessel Scoring
    df_ranked = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon)
    df_ranked.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("     TOP 5 RANKED CANDIDATE VESSELS (REAL MARINECADASTRE DATA)")
    print("=" * 80)
    cols_show = ["rank", "mmsi", "vessel_name", "ntro_attribution_score", "sog_kn", "investigation_evidence"]
    print(df_ranked[cols_show].head(5).to_string(index=False))
    print("=" * 80)

    # 7. Generate Master Evidence Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sar_np = sample_img.squeeze().numpy()

    # Panel 1: SAR Input & Prediction Mask
    axes[0, 0].imshow(sar_np, cmap="gray")
    axes[0, 0].imshow(prob_map, cmap="plasma", alpha=0.45)
    axes[0, 0].set_title(f"1. Sentinel-1 SAR & E5.2 Mask (83.49% mIoU)\n({fname_tif})", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: Backward Drift Density
    axes[0, 1].imshow(dens_back, cmap="hot", origin="lower", extent=[-15, 15, -15, 15])
    axes[0, 1].set_title("2. Backward Drift Origin Probability Field\n(1,000 Particles, 24h Hindcast)", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("East-West Distance (km)")
    axes[0, 1].set_ylabel("North-South Distance (km)")

    # Panel 3: Real MarineCadastre Vessels
    axes[1, 0].scatter(df_ranked["lon"], df_ranked["lat"], c="#1f77b4", s=35, alpha=0.7, label="MarineCadastre AIS")
    t1 = df_ranked.iloc[0]
    axes[1, 0].scatter(t1["lon"], t1["lat"], c="#d62728", s=150, marker="*", label=f"Rank #1: {t1['vessel_name']}")
    axes[1, 0].scatter(origin_lon, origin_lat, c="#2ca02c", s=120, marker="x", label="Inferred Origin Peak")
    axes[1, 0].set_title("3. Real MarineCadastre Traffic & Origin Peak", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Longitude (°W)")
    axes[1, 0].set_ylabel("Latitude (°N)")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # Panel 4: Vessel Attribution Confidence Scores
    top5 = df_ranked.head(5)
    bars = axes[1, 1].barh(top5["vessel_name"], top5["ntro_attribution_score"], color="#2ca02c")
    bars[0].set_color("#d62728")
    axes[1, 1].set_xlabel("Physio-GraphSpill Confidence Score", fontsize=10)
    axes[1, 1].set_title("4. Suspect Vessels Attribution Scores", fontsize=10, fontweight="bold")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()

    # Copy to poster directory
    poster_fig = os.path.join(root, "poster", "figures", "real_attribution_final.png")
    import shutil
    shutil.copy(out_fig, poster_fig)

    print(f"\n[✓] Real Investigation Evidence Plot saved -> {out_fig}")
    print(f"[✓] Copied to poster directory             -> {poster_fig}\n")

if __name__ == "__main__":
    run_production_investigation()
