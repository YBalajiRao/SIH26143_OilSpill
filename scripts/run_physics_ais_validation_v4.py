import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# 1. Suppress OpenCV C++ warnings completely
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.slick_morphology import mask_features
from src.utils.geo_utils import get_exact_raster_resolution_and_area
from src.environment.real_netcdf_forcing import RealMetoceanForcingEngine
from src.drift.probabilistic_drift import backward_drift_particles, forward_drift_particles, origin_density
from src.drift.origin_estimation import haversine_km, origin_stats_geodesic
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

ROOT = r"D:\SIH26143_OilSpill"
OUT_DIR = os.path.join(ROOT, "results", "physics_ais_validation")
os.makedirs(OUT_DIR, exist_ok=True)

THR = 0.50  # FROZEN PERCEPTION PROTOCOL
N_PART = 1000
N_NULL = 1000
SEED = 20181207

def batch_patch_pixels_to_latlon(patch_y, patch_x, local_ys, local_xs, h=2259, w=4424):
    """Vectorized instant conversion for array of particles without re-opening disk TIFF."""
    default_lat_top, default_lat_bottom = 29.2, 27.8
    default_lon_left, default_lon_right = -89.8, -87.2

    full_ys = patch_y + local_ys
    full_xs = patch_x + local_xs

    lats = default_lat_top - (full_ys / float(h)) * (default_lat_top - default_lat_bottom)
    lons = default_lon_left + (full_xs / float(w)) * (default_lon_right - default_lon_left)

    return lats, lons

def load_e52(device):
    ckpt = os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
    m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    st = torch.load(ckpt, map_location=device)
    sd = st["model_state_dict"] if isinstance(st, dict) and "model_state_dict" in st else st
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, ckpt

def age_proxy_hours(area_km2, contrast_db_abs=18.0):
    if area_km2 <= 0: return 6.0
    c = abs(contrast_db_abs) / 25.0
    return float(np.clip(2.5 * (area_km2 * 10.0) ** 0.45 * c, 1.5, 48.0))

def run_v4_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" GEODESIC PHYSICS & AIS VALIDATION SUITE (v4 SILENT & FAST)")
    print(" Segmentation FROZEN: Raw [0,1] | t = 0.50 | Morphology = NONE")
    print("=" * 80)

    model, ckpt_path = load_e52(device)

    raw = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path = os.path.join(raw, "dataframe_val_dataset_256_90.csv")
    img_dir  = os.path.join(raw, "images")
    mask_dir = os.path.join(raw, "masks")

    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    
    # Target Validation Patch #482 (37,160 GT oil pixels)
    idx = 482
    sample_img, sample_mask = ds[idx]
    row = ds.df.iloc[idx]
    fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip('"\'').split(",")]
    py, px = coords[0], coords[1]
    tif_path = os.path.join(img_dir, fname)

    # Read image once for height/width
    img_full = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    scene_h, scene_w = (img_full.shape[0], img_full.shape[1]) if img_full is not None else (2259, 4424)

    with torch.no_grad():
        prob = torch.sigmoid(model(sample_img.unsqueeze(0).to(device))).squeeze().cpu().numpy()

    gt = sample_mask.numpy().squeeze()
    binary = (prob >= THR).astype(np.float32)
    morph = mask_features(binary)

    gt_px = int((gt >= 0.5).sum())
    pred_px = int(morph["area_px"])
    
    gt_area_info = get_exact_raster_resolution_and_area(tif_path, gt_px)
    pred_area_info = get_exact_raster_resolution_and_area(tif_path, pred_px)

    cy = int(morph["centroid_xy"][1] * 255)
    cx = int(morph["centroid_xy"][0] * 255)
    center_lat, center_lon = batch_pixel_to_latlon_single = (
        float(29.2 - ((py + cy) / float(scene_h)) * (29.2 - 27.8)),
        float(-89.8 + ((px + cx) / float(scene_w)) * (-89.8 - (-87.2)))
    )

    avg_bs = float(sample_img.mean() * 40.0 - 35.0)
    age_proxy_h = age_proxy_hours(pred_area_info["area_km2"], contrast_db_abs=avg_bs)
    orient = float(morph.get("orientation_deg", -170.4))

    print(f"\n[+] Matched Case Characterization ({fname} - Validation Patch #{idx}):")
    print(f"    - Observed Lat/Lon: {center_lat:.4f}° N, {center_lon:.4f}° W")
    print(f"    - GT Oil Pixels:    {gt_px} px ({gt_area_info['area_km2']:.4f} km²)")
    print(f"    - Predicted Pixels: {pred_px} px ({pred_area_info['area_km2']:.4f} km²)")
    print(f"    - Model Age Proxy:  ~{age_proxy_h:.1f} hours [Heuristic Estimate]")

    met = RealMetoceanForcingEngine()
    forcing = met.get_velocity_at_latlon(center_lat, center_lon, "2018-12-07")
    wf = forcing.get("wind_drift_factor", forcing.get("wind_factor", 0.035))

    # =========================================================================
    # 1. AGE-PROXY SENSITIVITY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [1] AGE-PROXY SENSITIVITY (Origin Peak Shift vs Release Time)")
    print("=" * 80)
    age_list = [6, 8, 10, 12, 18, 24]
    age_rows = []

    for age_h in age_list:
        n_steps = int(age_h)
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PART, n_steps=n_steps, dt_hours=1.0,
            u_current=forcing["u_current"], v_current=forcing["v_current"],
            u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
            wind_factor=wf, diffusion=0.015, rng=SEED + n_steps
        )
        dens, final_pts = origin_density(traj, grid_size=64)
        pkx, pky = np.unravel_index(np.argmax(dens), dens.shape)
        
        o_lat = center_lat + (pky / 63.0 - 0.5) * 0.28
        o_lon = center_lon + (pkx / 63.0 - 0.5) * 0.28
        disp_km = haversine_km(center_lat, center_lon, o_lat, o_lon)

        age_rows.append({
            "age_proxy_h": age_h,
            "origin_lat": o_lat,
            "origin_lon": o_lon,
            "obs_to_origin_km": disp_km
        })
        print(f"  Release Age = {age_h:2d}h -> Origin: ({o_lat:.4f}°N, {o_lon:.4f}°W) | Shift: {disp_km:.2f} km")

    pd.DataFrame(age_rows).to_csv(os.path.join(OUT_DIR, "age_proxy_sensitivity.csv"), index=False)

    # Base -24h Backward Hindcast
    traj_back = backward_drift_particles(
        morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
        u_current=forcing["u_current"], v_current=forcing["v_current"],
        u_wind=forcing["u_wind"], v_wind=forcing["v_wind"],
        wind_factor=wf, diffusion=0.015, rng=SEED
    )
    dens_back, final_pts_back = origin_density(traj_back, grid_size=64)
    pkx_b, pky_b = np.unravel_index(np.argmax(dens_back), dens_back.shape)
    
    origin_lat = center_lat + (pky_b / 63.0 - 0.5) * 0.28
    origin_lon = center_lon + (pkx_b / 63.0 - 0.5) * 0.28

    # Fast Vectorized Conversion for all 1,000 particles at once
    particle_ys_local = final_pts_back[:, 1] * 255.0
    particle_xs_local = final_pts_back[:, 0] * 255.0
    particle_lats, particle_lons = batch_patch_pixels_to_latlon(py, px, particle_ys_local, particle_xs_local, h=scene_h, w=scene_w)
    particle_latlon = np.stack([particle_lats, particle_lons], axis=1)

    # =========================================================================
    # 2. GEODESIC CONTAINMENT RADII (TRUE PERCENTILES)
    # =========================================================================
    geo_stats = origin_stats_geodesic(particle_latlon, origin_lat, origin_lon)
    disp_base_km = haversine_km(center_lat, center_lon, origin_lat, origin_lon)

    print("\n" + "=" * 80)
    print(" [2] GEODESIC PARTICLE UNCERTAINTY & CONTAINMENT RADII (-24h, N=1,000)")
    print("=" * 80)
    print(f"  Origin Probability Peak: ({origin_lat:.4f}° N, {origin_lon:.4f}° W)")
    print(f"  Observed-to-Origin Peak Displacement: {disp_base_km:.2f} km")
    print(f"  Particle Centroid: ({geo_stats['mean_lat']:.4f}° N, {geo_stats['mean_lon']:.4f}° W)")
    print(f"  Particle Distance Range: Min = {geo_stats['min_km']:.2f} km | Max = {geo_stats['max_km']:.2f} km")
    print(f"  True 50% Containment Radius (r50): {geo_stats['r50_km']:.2f} km")
    print(f"  True 90% Containment Radius (r90): {geo_stats['r90_km']:.2f} km")
    print(f"  True 95% Containment Radius (r95): {geo_stats['r95_km']:.2f} km")

    # =========================================================================
    # 3. REAL METOCEAN PARTICLE CLOUD DISPLACEMENT SENSITIVITY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [3] REAL METOCEAN PARTICLE CLOUD DISPLACEMENT SENSITIVITY (±10% Perturbations)")
    print("=" * 80)
    
    perturbs = [
        ("baseline", 1.0, 1.0),
        ("wind_+10%", 1.0, 1.1),
        ("wind_-10%", 1.0, 0.9),
        ("curr_+10%", 1.1, 1.0),
        ("curr_-10%", 0.9, 1.0),
        ("both_+10%", 1.1, 1.1),
        ("both_-10%", 0.9, 0.9),
    ]

    met_rows = []
    base_u, base_v = forcing["u_current"], forcing["v_current"]
    base_uw, base_vw = forcing["u_wind"], forcing["v_wind"]

    for name, cf, wfct in perturbs:
        traj_p = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
            u_current=base_u * cf, v_current=base_v * cf,
            u_wind=base_uw * wfct, v_wind=base_vw * wfct,
            wind_factor=wf, diffusion=0.015, rng=SEED + hash(name) % 999
        )
        _, final_p = origin_density(traj_p, grid_size=64)
        
        p_lats, p_lons = batch_patch_pixels_to_latlon(py, px, final_p[:, 1] * 255.0, final_p[:, 0] * 255.0, h=scene_h, w=scene_w)
        cloud_shifts = haversine_km(particle_lats, particle_lons, p_lats, p_lons)
        mean_cloud_shift = float(np.mean(cloud_shifts))
        median_cloud_shift = float(np.median(cloud_shifts))

        met_rows.append({
            "perturbation": name,
            "mean_cloud_shift_km": mean_cloud_shift,
            "median_cloud_shift_km": median_cloud_shift
        })
        print(f"  {name:12s} -> Particle Cloud Mean Shift: {mean_cloud_shift:.2f} km | Median Shift: {median_cloud_shift:.2f} km")

    pd.DataFrame(met_rows).to_csv(os.path.join(OUT_DIR, "metocean_particle_sensitivity.csv"), index=False)

    # =========================================================================
    # 4. FORWARD DRIFT PROJECTION INITIAL STATE VERIFICATION (+24h)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [4] FORWARD DRIFT PROJECTION TRAJECTORY VERIFICATION (+24h)")
    print("=" * 80)
    traj_fwd = forward_drift_particles(
        morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
        u_current=base_u, v_current=base_v, u_wind=base_uw, v_wind=base_vw,
        wind_factor=wf, diffusion=0.015, rng=SEED + 1
    )
    
    steps_check = [0, 6, 12, 18, 24]
    for step_i in steps_check:
        pts_step = traj_fwd[:, step_i, :]
        mean_pos = pts_step.mean(axis=0)
        s_lat, s_lon = batch_patch_pixels_to_latlon(py, px, mean_pos[1]*255.0, mean_pos[0]*255.0, h=scene_h, w=scene_w)
        fwd_shift = haversine_km(center_lat, center_lon, float(s_lat), float(s_lon))
        print(f"  Forward Step t = +{step_i:02d}h -> Centroid: ({s_lat:.4f}° N, {s_lon:.4f}° W) | Shift: {fwd_shift:.2f} km")

    # =========================================================================
    # 5. REAL AIS CANDIDATES & ATTRIBUTION WEIGHT GRID STABILITY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [5] REAL MARINECADASTRE AIS CANDIDATES & WEIGHT GRID STABILITY")
    print("=" * 80)
    ais_pipeline = MarineCadastreAISPipeline()
    df_ais = ais_pipeline.load_and_filter_ais(origin_lat - 0.5, origin_lat + 0.5, origin_lon - 0.5, origin_lon + 0.5)
    print(f"  Real MarineCadastre Vessels in Search Bounding Box: {len(df_ais)}")

    df_ranked = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon, slick_orient_deg=orient)
    top_base = df_ranked.iloc[0]
    print(f"  Baseline Top Candidate: {top_base['vessel_name']} (MMSI: {top_base['mmsi']}) | Score: {top_base['ntro_attribution_score']:.4f} | Dist: {top_base['dist_km']:.2f} km")

    wsets = [
        ("baseline_40_25_25_10", 0.40, 0.25, 0.25, 0.10),
        ("prox_heavy_50_20_20_10", 0.50, 0.20, 0.20, 0.10),
        ("prox_light_30_30_30_10", 0.30, 0.30, 0.30, 0.10),
        ("kin_heavy_30_40_20_10",  0.30, 0.40, 0.20, 0.10),
        ("align_heavy_30_20_40_10", 0.30, 0.20, 0.40, 0.10),
        ("equal_weights_25", 0.25, 0.25, 0.25, 0.25)
    ]

    sens_rows = []
    print("\n  Weight Sensitivity Grid Analysis:")
    for wname, wp, wk, wa, wt in wsets:
        sc = (wp * df_ranked["proximity_score"].values +
              wk * df_ranked["kinematic_score"].values +
              wa * df_ranked["alignment_score"].values +
              wt * df_ranked["temporal_score"].values)
        if "gap_penalty" in df_ranked.columns:
            sc = sc - df_ranked["gap_penalty"].values
            
        best_i = int(np.argmax(sc))
        v_best = df_ranked.iloc[best_i]
        
        sens_rows.append({
            "weight_set": wname, "top_vessel": str(v_best["vessel_name"]),
            "mmsi": str(v_best["mmsi"]), "score": float(sc[best_i]),
            "dist_km": float(v_best["dist_km"])
        })
        print(f"    {wname:22s} -> Top: {v_best['vessel_name']:22s} | Score: {sc[best_i]:.4f} | Dist: {v_best['dist_km']:.1f} km")

    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "ais_weight_sensitivity_grid.csv"), index=False)
    unique_mmsis = set(r["mmsi"] for r in sens_rows)
    top_vessel_counts = pd.Series([r["top_vessel"] for r in sens_rows]).value_counts()
    dominant_vessel = top_vessel_counts.index[0]
    dominant_retention_pct = (top_vessel_counts.iloc[0] / len(wsets)) * 100.0

    print(f"\n  [✓] Unique Top MMSIs Across Weight Grid: {len(unique_mmsis)} -> {unique_mmsis}")
    print(f"  [✓] Candidate Stability: '{dominant_vessel}' retained Rank #1 in {dominant_retention_pct:.1f}% of weight configurations ({top_vessel_counts.iloc[0]}/{len(wsets)}).")

    # =========================================================================
    # 6. SPATIAL NULL PERMUTATION TEST (N = 1,000)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [6] SPATIAL NULL PERMUTATION TEST (N = 1,000 RANDOMIZED ORIGIN PLACEMENTS)")
    print("=" * 80)
    
    s_obs = float(df_ranked.iloc[0]["ntro_attribution_score"])
    rng_null = np.random.default_rng(SEED)
    
    null_top_scores = []
    exceeded_count = 0

    print(f"[*] Running {N_NULL} Monte Carlo spatial origin shifts...")
    for _ in range(N_NULL):
        null_lat = center_lat + rng_null.uniform(-0.5, 0.5)
        null_lon = center_lon + rng_null.uniform(-0.5, 0.5)
        
        df_null = score_and_rank_vessels_ntro(df_ais, null_lat, null_lon, slick_orient_deg=orient)
        null_s = float(df_null.iloc[0]["ntro_attribution_score"])
        null_top_scores.append(null_s)
        if null_s >= s_obs:
            exceeded_count += 1

    null_top_scores = np.array(null_top_scores)
    p_empirical = (1.0 + exceeded_count) / (1.0 + N_NULL)

    print(f"\n  Observed Top Candidate Score (S_obs): {s_obs:.6f}")
    print(f"  Null Distribution Mean ± Std:          {null_top_scores.mean():.6f} ± {null_top_scores.std():.6f}")
    print(f"  Null Distribution Median:              {np.median(null_top_scores):.6f}")
    print(f"  Null Distribution 95th Percentile:     {np.percentile(null_top_scores, 95):.6f}")
    print(f"  Empirical p-value:                     p = {p_empirical:.4f}")
    
    if p_empirical < 0.05:
        p_conclusion = f"p = {p_empirical:.4f} indicates statistically unusual ranking under the tested spatial null."
    else:
        p_conclusion = f"p = {p_empirical:.4f} indicates the ranking provides investigative prioritization rather than statistical proof under spatial null."

    print(f"  Statistical Conclusion:                {p_conclusion}")

    # Export master JSON summary
    summary_v4 = {
        "case_metadata": {
            "scene": fname, "patch_idx": idx,
            "observed_lat": center_lat, "observed_lon": center_lon,
            "origin_lat": origin_lat, "origin_lon": origin_lon,
            "obs_to_origin_displacement_km": disp_base_km,
            "gt_area_km2": gt_area_info["area_km2"],
            "pred_area_km2": pred_area_info["area_km2"],
            "heuristic_age_proxy_hours": age_proxy_h
        },
        "containment_radii_km": {
            "r50": geo_stats["r50_km"],
            "r90": geo_stats["r90_km"],
            "r95": geo_stats["r95_km"],
            "min_km": geo_stats["min_km"],
            "max_km": geo_stats["max_km"]
        },
        "attribution_stability": {
            "dominant_vessel": dominant_vessel,
            "stability_pct": dominant_retention_pct,
            "unique_top_mmsis": list(unique_mmsis)
        },
        "spatial_null_test": {
            "s_obs": s_obs,
            "null_mean": float(null_top_scores.mean()),
            "null_p95": float(np.percentile(null_top_scores, 95)),
            "empirical_p_value": float(p_empirical),
            "conclusion": p_conclusion
        }
    }

    with open(os.path.join(OUT_DIR, "validation_summary_v4.json"), "w") as f:
        json.dump(summary_v4, f, indent=2)

    # Generate Publication Figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    ages = [r["age_proxy_h"] for r in age_rows]
    disps = [r["obs_to_origin_km"] for r in age_rows]
    axes[0, 0].plot(ages, disps, "o-", color="#1f77b4", lw=2)
    axes[0, 0].set_title("1. Model-Derived Age-Proxy vs Origin Shift", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Heuristic Age Proxy (hours)")
    axes[0, 0].set_ylabel("Displacement to Origin (km)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    m_names = [r["perturbation"] for r in met_rows[1:]]
    m_shifts = [r["mean_cloud_shift_km"] for r in met_rows[1:]]
    axes[0, 1].barh(m_names, m_shifts, color="#ff7f0e")
    axes[0, 1].set_title("2. Particle Cloud Shift under ±10% Metocean Forcing", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Mean Geodesic Cloud Shift (km)")
    axes[0, 1].grid(axis="x", linestyle="--", alpha=0.5)

    axes[1, 0].hist(null_top_scores, bins=30, color="steelblue", alpha=0.75, edgecolor="black")
    axes[1, 0].axvline(s_obs, color="red", lw=2, label=f"Observed S_obs = {s_obs:.3f}")
    axes[1, 0].axvline(np.percentile(null_top_scores, 95), color="orange", ls="--", label="Null p95")
    axes[1, 0].set_title(f"3. Spatial Null Distribution (N=1,000 | p = {p_empirical:.3f})", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Top Candidate Score")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    axes[1, 1].bar([r["weight_set"] for r in sens_rows], [r["score"] for r in sens_rows], color="#2ca02c")
    axes[1, 1].set_title(f"4. Top Candidate Score Stability ({dominant_vessel})", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Attribution Score")
    axes[1, 1].set_xticks(range(len(sens_rows)))
    axes[1, 1].set_xticklabels([r["weight_set"] for r in sens_rows], rotation=35, ha="right", fontsize=8)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "geodesic_physics_ais_validation_panel_v4.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\n[✓] Validation Panel Figure Saved -> {fig_path}")
    print(f"[✓] Master Summary Saved          -> {os.path.join(OUT_DIR, 'validation_summary_v4.json')}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_v4_validation()
