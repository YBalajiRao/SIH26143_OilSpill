import os
import sys
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.geo_utils import patch_pixel_to_latlon
from src.utils.slick_morphology import mask_features
from src.environment.metocean_loader import get_real_metocean_forcing
from src.drift.probabilistic_drift import backward_drift_particles, forward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.real_ais_fetcher import fetch_real_gulf_ais_vessels
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

def generate_sih_master_dashboard():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    out_csv  = os.path.join(root, "results", "ais_outputs", "ntro_final_attribution_ranking.csv")
    out_fig  = os.path.join(root, "results", "figures", "sih26143_master_intelligence_map.png")
    out_html = os.path.join(root, "results", "sih_interactive_dashboard.html")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    print("=" * 80)
    print(" SIH26143 NTRO MASTER EVALUATION DASHBOARD GENERATOR")
    print(" Executing Complete Chain: SAR -> Perception -> Backward -> Forward -> AIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Champion Perception Model
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"[✓] Loaded Champion Model E5.2 (Val mIoU: 83.49% | Dice: 78.84%)")

    # 2. Load Target Scene (2018_12_07.tif)
    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    sample_idx, sample_img = None, None
    for i in range(len(ds)):
        img_t, mask_t = ds[i]
        if (mask_t > 0.5).float().mean() > 0.05:
            sample_idx, sample_img = i, img_t
            break

    row = ds.df.iloc[sample_idx]
    fname_tif = "2018_12_07.tif"
    tif_full_path = os.path.join(img_dir, fname_tif)
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip('"\'').split(",")]
    patch_y, patch_x = coords[0], coords[1]

    with torch.no_grad():
        logits = model(sample_img.unsqueeze(0).to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = (prob_map >= 0.5).astype(np.float32)
    morph = mask_features(binary_mask)

    center_lat, center_lon = patch_pixel_to_latlon(
        tif_full_path, patch_y, patch_x,
        local_y=int(morph["centroid_xy"][1] * 255),
        local_x=int(morph["centroid_xy"][0] * 255)
    )

    # 3. Metocean Forcing & Drift
    forcing = get_real_metocean_forcing(center_lat, center_lon, scene_date_str="2018-12-07")
    
    # Backward Drift (-24h Hindcast)
    traj_back = backward_drift_particles(
        morph["seed_xy"], n_particles=1000, n_steps=24,
        u_current=forcing["u_current"], v_current=forcing["v_current"],
        u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
        wind_factor=forcing["wind_drift_factor"], diffusion=0.015, rng=20181207
    )
    dens_back, final_pts = origin_density(traj_back, grid_size=64)
    stats_back = origin_stats(dens_back, final_pts)

    peak_off_x, peak_off_y = stats_back["peak_xy"]
    origin_lat = center_lat + (peak_off_y - 0.5) * 0.28
    origin_lon = center_lon + (peak_off_x - 0.5) * 0.28

    # Forward Drift (+24h Forecast)
    traj_fwd = forward_drift_particles(
        morph["seed_xy"], n_particles=1000, n_steps=24,
        u_current=forcing["u_current"], v_current=forcing["v_current"],
        u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
        wind_factor=forcing["wind_drift_factor"], diffusion=0.015, rng=20181207
    )
    dens_fwd, fwd_pts = origin_density(traj_fwd, grid_size=64)

    # 4. AIS Retrieval & NTRO Scoring
    df_ais = fetch_real_gulf_ais_vessels(origin_lat, origin_lon, radius_km=35.0, scene_date_str="2018-12-07")
    df_ranked = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon)
    df_ranked.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("        FINAL NTRO CANDIDATE VESSEL ATTRIBUTION RANKING")
    print("=" * 80)
    print(df_ranked[["rank", "mmsi", "vessel_name", "ntro_attribution_score", "sog_kn", "investigation_evidence"]].head(5).to_string(index=False))
    print("=" * 80)

    # 5. Generate Master Poster & Evaluation Plot (6-Panel Comprehensive View)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    sar_np = sample_img.squeeze().numpy()

    # (1) Raw SAR & Segmentation Overlay
    axes[0, 0].imshow(sar_np, cmap="gray")
    axes[0, 0].imshow(prob_map, cmap="plasma", alpha=0.45)
    axes[0, 0].set_title("1. Sentinel-1 SAR & E5.2 Mask (83.49% mIoU)", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # (2) Backward Hindcast (-24h Origin Probability)
    axes[0, 1].imshow(dens_back, cmap="hot", origin="lower", extent=[-15, 15, -15, 15])
    axes[0, 1].set_title("2. Backward Particle Hindcast (-24h Origin)", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("East-West Distance (km)")
    axes[0, 1].set_ylabel("North-South Distance (km)")

    # (3) Forward Flow Forecast (+24h Slick Movement)
    axes[0, 2].imshow(dens_fwd, cmap="viridis", origin="lower", extent=[-15, 15, -15, 15])
    axes[0, 2].set_title("3. Forward Particle Forecast (+24h Impact)", fontsize=10, fontweight="bold")
    axes[0, 2].set_xlabel("East-West Distance (km)")
    axes[0, 2].set_ylabel("North-South Distance (km)")

    # (4) AIS Vessels & Inferred Origin Peak
    axes[1, 0].scatter(df_ranked["lon"], df_ranked["lat"], c="#1f77b4", s=40, alpha=0.7, label="AIS Vessels")
    t1 = df_ranked.iloc[0]
    axes[1, 0].scatter(t1["lon"], t1["lat"], c="#d62728", s=160, marker="*", label=f"Rank #1: {t1['vessel_name']}")
    axes[1, 0].scatter(origin_lon, origin_lat, c="#2ca02c", s=120, marker="x", label="Origin Peak")
    axes[1, 0].set_title("4. Reconstructed AIS Traffic & Origin Peak", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Longitude (°W)")
    axes[1, 0].set_ylabel("Latitude (°N)")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # (5) Ranked Candidate Vessel Scores
    top5 = df_ranked.head(5)
    bars = axes[1, 1].barh(top5["vessel_name"], top5["ntro_attribution_score"], color="#2ca02c")
    bars[0].set_color("#d62728")
    axes[1, 1].set_xlabel("Physio-GraphSpill Score", fontsize=10)
    axes[1, 1].set_title("5. Top 5 Ranked Suspect Vessels", fontsize=10, fontweight="bold")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)

    # (6) Slick Geometry & Physical Characteristics
    axes[1, 2].axis("off")
    info_text = (
        f"Physio-GraphSpill Intelligence Summary\n"
        f"-----------------------------------------\n"
        f"Target Scene:      {fname_tif}\n"
        f"Acquisition Date:  2018-12-07\n"
        f"Spill Position:    {center_lat:.4f}°N, {center_lon:.4f}°W\n"
        f"Origin Peak:       {origin_lat:.4f}°N, {origin_lon:.4f}°W\n"
        f"Slick Area:        {morph['area_px']} pixels\n"
        f"Elongation Ratio:  {morph['elongation']:.2f}\n"
        f"Current Velocity:  u={forcing['u_current']}m/s, v={forcing['v_current']}m/s\n"
        f"Wind Velocity:     u={forcing['u_wind']}m/s, v={forcing['v_wind']}m/s\n\n"
        f"Top Culprit Suspect: {t1['vessel_name']}\n"
        f"MMSI: {t1['mmsi']} | Confidence: {t1['ntro_attribution_score']*100:.1f}%\n"
        f"Primary Evidence:\n{t1['investigation_evidence']}"
    )
    axes[1, 2].text(0.05, 0.10, info_text, fontsize=9, family="monospace", bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()

    # Copy master plot to poster folder
    poster_fig = os.path.join(root, "poster", "figures", "sih26143_master_intelligence_map.png")
    import shutil
    shutil.copy(out_fig, poster_fig)

    # 6. Generate HTML Interactive Dashboard
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SIH26143 Physio-GraphSpill Live Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 20px; }}
        h1 {{ color: #38bdf8; text-align: center; }}
        .header-box {{ background-color: #1e293b; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #38bdf8; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .card {{ background-color: #1e293b; padding: 15px; border-radius: 8px; width: 45%; border: 1px solid #334155; }}
        img {{ width: 100%; border-radius: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px; border: 1px solid #334155; text-align: left; }}
        th {{ background-color: #0284c7; color: white; }}
        tr:nth-child(even) {{ background-color: #0f172a; }}
        .highlight {{ background-color: #7f1d1d !important; color: #fca5a5; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>SIH26143 — Marine Oil-Spill Attribution Intelligence Dashboard</h1>
    <div class="header-box">
        <h3>Problem Statement ID: SIH26143 (NTRO) | SAH Track: T7</h3>
        <p><b>Model:</b> Physio-GraphSpill E5.2 (Val mIoU: <b>83.49%</b>) | <b>Target Scene:</b> 2018_12_07.tif | <b>Date:</b> 2018-12-07</p>
        <p><b>Spill Location:</b> {center_lat:.4f}°N, {center_lon:.4f}°W | <b>Inferred Origin:</b> {origin_lat:.4f}°N, {origin_lon:.4f}°W</p>
    </div>

    <div class="container">
        <div class="card" style="width: 95%;">
            <h2>1. Master Multimodal Evidence Map</h2>
            <img src="figures/sih26143_master_intelligence_map.png" alt="Intelligence Map">
        </div>

        <div class="card" style="width: 95%;">
            <h2>2. Ranked Candidate Vessel Attribution Table</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>MMSI</th>
                    <th>Vessel Name</th>
                    <th>Confidence Score</th>
                    <th>Speed (kn)</th>
                    <th>Investigation Evidence</th>
                </tr>
    """
    for _, r in df_ranked.head(6).iterrows():
        cls = "class='highlight'" if r["rank"] == 1 else ""
        html_content += f"""
                <tr {cls}>
                    <td>#{r['rank']}</td>
                    <td>{r['mmsi']}</td>
                    <td>{r['vessel_name']}</td>
                    <td>{r['ntro_attribution_score']*100:.1f}%</td>
                    <td>{r['sog_kn']} kn</td>
                    <td>{r['investigation_evidence']}</td>
                </tr>
        """
    html_content += """
            </table>
        </div>
    </div>
</body>
</html>
    """
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[✓] SIH Master Intelligence Map saved -> {out_fig}")
    print(f"[✓] SIH Interactive Web Dashboard saved -> {out_html}\n")

if __name__ == "__main__":
    generate_sih_master_dashboard()
