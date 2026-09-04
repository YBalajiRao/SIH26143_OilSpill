import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.geo_utils import patch_pixel_to_latlon
from src.utils.slick_morphology import mask_features
from src.environment.real_netcdf_forcing import RealMetoceanForcingEngine
from src.drift.probabilistic_drift import backward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

def run_batch_attribution_suite():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    out_csv = os.path.join(root, "results", "ais_outputs", "batch_attribution_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print("=" * 80)
    print(" BATCH PHYSIO-GRAPHSPILL ATTRIBUTION EVALUATION SUITE")
    print(" Evaluating 15 Oil-Dense Satellite Patches")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model & Dataset
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    met_engine = RealMetoceanForcingEngine()
    ais_pipeline = MarineCadastreAISPipeline()

    # Collect top 15 patches with oil pixels
    target_patches = []
    for i in range(len(ds)):
        _, mask_t = ds[i]
        oil_px = int((mask_t > 0.5).sum())
        if oil_px > 3000:
            target_patches.append((i, oil_px))
            if len(target_patches) == 15:
                break

    print(f"[✓] Selected {len(target_patches)} patches with >3,000 oil pixels.")

    summary_records = []

    for rank_i, (patch_idx, oil_px) in enumerate(target_patches, 1):
        sample_img, _ = ds[patch_idx]
        row = ds.df.iloc[patch_idx]
        fname_tif = os.path.basename(str(row["paths"]).replace("\\", "/"))
        tif_full_path = os.path.join(img_dir, fname_tif)
        coords = [int(c.strip()) for c in str(row["coordinates"]).strip('"\'').split(",")]
        patch_y, patch_x = coords[0], coords[1]

        with torch.no_grad():
            logits = model(sample_img.unsqueeze(0).to(device))
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (prob_map >= 0.35).astype(np.float32)
        morph = mask_features(binary_mask)

        center_lat, center_lon = patch_pixel_to_latlon(
            tif_full_path, patch_y, patch_x,
            local_y=int(morph["centroid_xy"][1] * 255),
            local_x=int(morph["centroid_xy"][0] * 255)
        )

        forcing = met_engine.get_velocity_at_latlon(center_lat, center_lon, "2018-12-07")

        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=500, n_steps=12, dt_hours=1.0,
            u_current=forcing["u_current"], v_current=forcing["v_current"],
            u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
            wind_factor=forcing["wind_factor"], diffusion=0.015, rng=patch_idx
        )

        dens, final_pts = origin_density(traj, grid_size=32)
        stats = origin_stats(dens, final_pts)

        peak_off_x, peak_off_y = stats["peak_xy"]
        origin_lat = center_lat + (peak_off_y - 0.5) * 0.28
        origin_lon = center_lon + (peak_off_x - 0.5) * 0.28

        df_ais = ais_pipeline.load_and_filter_ais(origin_lat - 0.4, origin_lat + 0.4, origin_lon - 0.4, origin_lon + 0.4)
        df_ranked = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon)

        top1 = df_ranked.iloc[0]

        print(f"  Patch #{patch_idx:05d} [{fname_tif}] -> Area: {morph['area_px']:5d}px | Origin: ({origin_lat:.4f}°N, {origin_lon:.4f}°W) | Top-1 Vessel: {top1['vessel_name']} (Score: {top1['ntro_attribution_score']:.4f})")

        summary_records.append({
            "patch_index": patch_idx,
            "scene_file": fname_tif,
            "slick_area_px": morph["area_px"],
            "observed_lat": center_lat,
            "observed_lon": center_lon,
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "top1_vessel_name": top1["vessel_name"],
            "top1_mmsi": top1["mmsi"],
            "top1_score": top1["ntro_attribution_score"]
        })

    df_sum = pd.DataFrame(summary_records)
    df_sum.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 80)
    print(" BATCH ATTRIBUTION SUMMARY RESULTS (N = 15 Patches)")
    print("=" * 80)
    print(df_sum[["patch_index", "slick_area_px", "origin_lat", "origin_lon", "top1_vessel_name", "top1_score"]].to_string(index=False))
    print(f"\n[✓] Batch Attribution Summary Exported -> {out_csv}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_batch_attribution_suite()
