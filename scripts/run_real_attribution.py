import os
import sys
import numpy as np
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
from src.environment.metocean_loader import get_metocean_forcing_vectors
from src.drift.probabilistic_drift import backward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.real_ais_fetcher import fetch_real_ais_traffic


def run_real_attribution():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    raw_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir = os.path.join(raw_dir, "images")
    mask_dir = os.path.join(raw_dir, "masks")
    out_csv = os.path.join(root, "results", "ais_outputs", "real_vessel_ranking.csv")
    out_fig = os.path.join(root, "results", "figures", "real_attribution_investigation.png")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    print("=" * 70)
    print(" PHYSIO-GRAPHSPILL INVESTIGATION ENGINE")
    print(" (E5.2 + hindcast + AIS ranking)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"[✓] E5.2 loaded (val_mIoU ckpt={ckpt.get('val_mIoU', 0.8349):.4f})")

    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    sample_idx = sample_img = None
    for i in range(len(ds)):
        img_t, mask_t = ds[i]
        if (mask_t > 0.5).float().mean() > 0.05:
            sample_idx, sample_img = i, img_t
            break
    if sample_idx is None:
        raise RuntimeError("No oil patch found in val set")

    row = ds.df.iloc[sample_idx]
    fname_tif = os.path.basename(str(row["paths"]).replace("\\", "/"))
    tif_full_path = os.path.join(img_dir, fname_tif)
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip("\"'").split(",")]
    patch_y, patch_x = coords[0], coords[1]

    with torch.no_grad():
        logits = model(sample_img.unsqueeze(0).to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary_mask = (prob_map >= 0.5).astype(np.float32)
    morph = mask_features(binary_mask)

    center_lat, center_lon = patch_pixel_to_latlon(
        tif_full_path, patch_y, patch_x,
        local_y=int(morph["centroid_xy"][1] * 255),
        local_x=int(morph["centroid_xy"][0] * 255),
    )
    print(f"\n[+] Scene={fname_tif} patch=#{sample_idx}")
    print(f"    approx lat/lon = {center_lat:.4f}N, {center_lon:.4f}W")
    print(f"    area_px={morph['area_px']} elong={morph['elongation']:.2f}")

    forcing = get_metocean_forcing_vectors(center_lat, center_lon, "2018-08-21")
    print("\n[*] Backward hindcast N=1000, 24h ...")
    trajectories = backward_drift_particles(
        morph["seed_xy"], n_particles=1000, n_steps=24, dt_hours=1.0,
        u_current=forcing["u_current"], v_current=forcing["v_current"],
        u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
        wind_factor=forcing["wind_drift_factor"], diffusion=0.015, rng=42,
    )
    dens, final_pts = origin_density(trajectories, grid_size=64)
    stats = origin_stats(dens, final_pts)
    peak_off_x, peak_off_y = stats["peak_xy"]
    # map density offset ~ +/- 0.15 deg (~15 km scale demo)
    origin_lat = center_lat + (peak_off_y - 0.5) * 0.30
    origin_lon = center_lon + (peak_off_x - 0.5) * 0.30
    print(f"[✓] origin peak ~ {origin_lat:.4f}N, {origin_lon:.4f}W")

    df_ais = fetch_real_ais_traffic(origin_lat, origin_lon, radius_km=35.0, scene_date_str="2018-08-21")
    scores = []
    for _, v in df_ais.iterrows():
        dist_km = np.sqrt((v["lat"] - origin_lat) ** 2 + (v["lon"] - origin_lon) ** 2) * 111.0
        spatial = np.exp(-dist_km / 12.0)
        kinematic = np.exp(-((v["sog_kn"] - 5.5) ** 2) / 18.0)
        scores.append(0.70 * spatial + 0.30 * kinematic)
    df_ais = df_ais.copy()
    df_ais["attribution_score"] = scores
    df_ais = df_ais.sort_values("attribution_score", ascending=False).reset_index(drop=True)
    df_ais["rank"] = np.arange(1, len(df_ais) + 1)
    df_ais.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print(" TOP-5 CANDIDATE VESSELS")
    print("=" * 70)
    show = [c for c in ["rank", "mmsi", "vessel_name", "attribution_score", "lat", "lon", "sog_kn", "is_synthetic", "is_true_source"] if c in df_ais.columns]
    print(df_ais[show].head(5).to_string(index=False))
    if "is_true_source" in df_ais.columns:
        print(f"\n[metric] Top-1 true-source hit: {int(df_ais.iloc[0].get('is_true_source', 0)==1)} | "
              f"Top-3 hit: {int(df_ais.head(3)['is_true_source'].max()==1)}")
    if "is_synthetic" in df_ais.columns and df_ais["is_synthetic"].max() == 1:
        print("[NOTE] AIS rows are SYNTHETIC demo traffic (MarineCadastre offline/empty). Not real guilt.")
    print(f"[✓] CSV -> {out_csv}")

    # figure
    sar = sample_img.squeeze().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(sar, cmap="gray")
    axes[0, 0].imshow(prob_map, cmap="plasma", alpha=0.45)
    axes[0, 0].set_title(f"1. SAR + E5.2 P(oil)\n{fname_tif}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(dens, cmap="hot", origin="lower", extent=[-20, 20, -20, 20])
    axes[0, 1].set_title("2. Backward origin density (24h)")
    axes[0, 1].set_xlabel("E-W (km demo)")
    axes[0, 1].set_ylabel("N-S (km demo)")

    axes[1, 0].scatter(df_ais["lon"], df_ais["lat"], c="steelblue", s=35, alpha=0.7, label="traffic")
    t1 = df_ais.iloc[0]
    axes[1, 0].scatter(t1["lon"], t1["lat"], c="red", s=140, marker="*", label=f"Rank1 {t1['mmsi']}")
    axes[1, 0].scatter(origin_lon, origin_lat, c="lime", s=100, marker="x", label="origin peak")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("3. AIS candidates + origin")
    axes[1, 0].set_xlabel("Lon"); axes[1, 0].set_ylabel("Lat"); axes[1, 0].grid(True, alpha=0.3)

    top5 = df_ais.head(5)
    names = top5["vessel_name"].astype(str).tolist() if "vessel_name" in top5 else top5["mmsi"].astype(str).tolist()
    bars = axes[1, 1].barh(names, top5["attribution_score"], color="#2ca02c")
    bars[0].set_color("#d62728")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("4. Attribution scores")
    axes[1, 1].set_xlabel("score")

    plt.suptitle("Physio-GraphSpill Investigation (perception frozen E5.2)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure -> {out_fig}")
    print("=" * 70)


if __name__ == "__main__":
    run_real_attribution()
