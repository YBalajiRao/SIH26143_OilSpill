import os, sys, json
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import numpy as np
import pandas as pd
import torch
import cv2
try:
    cv2.setLogLevel(0)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.slick_morphology import mask_features
from src.utils.geo_utils import (
    patch_pixel_to_latlon, batch_norm_xy_to_latlon, get_exact_raster_resolution_and_area,
    haversine_km, origin_stats_geodesic, get_scene_hw, LON_LEFT, LON_RIGHT
)
from src.environment.real_netcdf_forcing import RealMetoceanForcingEngine
from src.drift.probabilistic_drift import backward_drift_particles, forward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

ROOT = r"D:\SIH26143_OilSpill"
OUT = os.path.join(ROOT, "results", "physics_ais_validation")
os.makedirs(OUT, exist_ok=True)

THR = 0.50
N_PARTICLES = 1000
N_NULL_TRIALS = 1000
MASTER_SEED = 20181207
REF_LAT, REF_LON = 28.3987, -88.3660
SAR_ACQUISITION_TIME = "2018-12-07 12:00:00"

PERT_SEEDS = {
    "baseline": 20181207,
    "wind_+10%": 20181208,
    "wind_-10%": 20181209,
    "curr_+10%": 20181210,
    "curr_-10%": 20181211,
    "both_+10%": 20181212,
    "both_-10%": 20181213,
}

WEIGHT_CONFIGS = [
    ("baseline_40_25_25_10",   (0.40, 0.25, 0.25, 0.10)),
    ("prox_heavy_50_20_20_10", (0.50, 0.20, 0.20, 0.10)),
    ("prox_light_30_30_30_10", (0.30, 0.30, 0.30, 0.10)),
    ("kin_heavy_30_40_20_10",  (0.30, 0.40, 0.20, 0.10)),
    ("align_heavy_30_20_40_10",(0.30, 0.20, 0.40, 0.10)),
    ("equal_weights_25",       (0.25, 0.25, 0.25, 0.25)),
]

def load_frozen_e52(device):
    ckpt = os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
    m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    st = torch.load(ckpt, map_location=device)
    sd = st["model_state_dict"] if isinstance(st, dict) and "model_state_dict" in st else st
    m.load_state_dict(sd, strict=False)
    return m.eval(), ckpt

def compute_age_proxy_hours(area_km2, cabs=18.0):
    if area_km2 <= 0:
        return 6.0
    return float(np.clip(2.5 * (area_km2 * 10.0) ** 0.45 * (abs(cabs) / 25.0), 1.5, 48.0))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" PHYSIO-GRAPHSPILL — FINAL PHYSICS & AIS VALIDATION SUITE v4.2")
    print(" Protocol: Frozen E5.2 | Raw [0,1] | Thr = 0.50 | Morphology = NONE")
    print("=" * 80)

    # 1. Perception
    model, ckpt_path = load_frozen_e52(device)
    raw_dir = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train")
    ds = GulfSARPatchDataset(
        os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv"),
        os.path.join(raw_dir, "images"),
        os.path.join(raw_dir, "masks"),
        transform=get_val_transforms(),
    )
    patch_idx = 482
    img_t, msk_t = ds[patch_idx]
    row = ds.df.iloc[patch_idx]
    fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
    py, px = [int(c.strip()) for c in str(row["coordinates"]).strip("\"'").split(",")]
    tif_path = os.path.join(raw_dir, "images", fname)
    H, W = get_scene_hw(tif_path)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t.unsqueeze(0).to(device))).squeeze().cpu().numpy()
    
    gt_mask = msk_t.numpy().squeeze()
    pred_binary = (prob >= THR).astype(np.uint8)
    
    morph = mask_features(pred_binary)
    gt_px = int((gt_mask >= 0.5).sum())
    pred_px = int(morph["area_px"])
    gt_area_info = get_exact_raster_resolution_and_area(tif_path, gt_px)
    pred_area_info = get_exact_raster_resolution_and_area(tif_path, pred_px)
    
    local_cy = int(np.clip(morph["centroid_xy"][1] * 255.0, 0, 255))
    local_cx = int(np.clip(morph["centroid_xy"][0] * 255.0, 0, 255))
    slick_orient_deg = float(morph.get("orientation_deg", -170.4))
    age_proxy = compute_age_proxy_hours(pred_area_info["area_km2"])

    obs_lat, obs_lon = patch_pixel_to_latlon(tif_path, py, px, local_y=local_cy, local_x=local_cx)
    d_ref = float(haversine_km(REF_LAT, REF_LON, obs_lat, obs_lon))

    print("\n[+] CASE CHARACTERIZATION:")
    print(f"  - Scene / Patch:      {fname} (Patch #{patch_idx})")
    print(f"  - Observed Centroid:  {obs_lat:.4f}° N, {obs_lon:.4f}° W (Δref = {d_ref:.3f} km)")
    print(f"  - Ground Truth Area:  {gt_px} px ({gt_area_info['area_km2']:.4f} km²)")
    print(f"  - Predicted Area:     {pred_px} px ({pred_area_info['area_km2']:.4f} km²)")
    print(f"  - Model-Derived Age:  {age_proxy:.2f} hours (Proxy heuristic)")
    print(f"  - Slick Orientation:  {slick_orient_deg:.2f}° (Elongation = {morph['elongation']:.2f})")

    if d_ref > 5.0:
        print("  [!] FATAL: Georeferencing outside allowable bounds. Terminating.")
        sys.exit(1)

    # 2. Metocean Forcing
    forcing_engine = RealMetoceanForcingEngine()
    forcing = forcing_engine.get_velocity_at_latlon(obs_lat, obs_lon, "2018-12-07")
    wf = forcing.get("wind_drift_factor", forcing.get("wind_factor", 0.035))
    u, v = forcing["u_current"], forcing["v_current"]
    uw, vw = forcing["u_wind"], forcing["v_wind"]
    print(f"\n[+] METOCEAN FORCING (ERA5 + CMEMS Dec 7, 2018):")
    print(f"  - Current (u, v):     ({u:.4f}, {v:.4f}) m/s")
    print(f"  - Wind (uw, vw):      ({uw:.4f}, {vw:.4f}) m/s (Drift Factor = {wf:.3f})")

    # 3. Backward Lagrangian Drift (-24h, N=1000)
    traj_b = backward_drift_particles(
        morph["seed_xy"], n_particles=N_PARTICLES, n_steps=24, dt_hours=1.0,
        u_current=u, v_current=v, u_wind=uw, v_wind=vw, wind_factor=wf,
        diffusion=0.015, rng=MASTER_SEED
    )
    dens_b, final_b = origin_density(traj_b, grid_size=64)
    st = origin_stats(dens_b, final_b)
    peak_norm = np.array([[st["peak_xy"][0], st["peak_xy"][1]]], dtype=np.float64)
    plat, plon = batch_norm_xy_to_latlon(tif_path, py, px, peak_norm, h=H, w=W)
    origin_lat, origin_lon = float(plat[0]), float(plon[0])

    part_lat, part_lon = batch_norm_xy_to_latlon(tif_path, py, px, final_b, h=H, w=W)
    part_ll = np.stack([part_lat, part_lon], axis=1)

    geo_centroid = origin_stats_geodesic(part_ll, float(part_lat.mean()), float(part_lon.mean()))
    disp_peak = float(haversine_km(obs_lat, obs_lon, origin_lat, origin_lon))
    disp_centroid = float(haversine_km(obs_lat, obs_lon, geo_centroid['mean_lat'], geo_centroid['mean_lon']))

    print("\n[+] BACKWARD SOURCE RECONSTRUCTION (-24h):")
    print(f"  - Origin Peak:        {origin_lat:.4f}° N, {origin_lon:.4f}° W (Obs->Peak: {disp_peak:.2f} km)")
    print(f"  - Particle Centroid:  {geo_centroid['mean_lat']:.4f}° N, {geo_centroid['mean_lon']:.4f}° W (Obs->Centroid: {disp_centroid:.2f} km)")
    print(f"  - Centroid Radii:     r50 = {geo_centroid['r50_km']:.2f} km | r90 = {geo_centroid['r90_km']:.2f} km | r95 = {geo_centroid['r95_km']:.2f} km")

    # 4. Age Sensitivity Grid
    print("\n[+] AGE-PROXY SENSITIVITY GRID:")
    age_rows = []
    for age_h in [6, 8, 10, 12, 18, 24]:
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PARTICLES, n_steps=int(age_h), dt_hours=1.0,
            u_current=u, v_current=v, u_wind=uw, v_wind=vw, wind_factor=wf,
            diffusion=0.015, rng=MASTER_SEED + int(age_h)
        )
        dens, fin = origin_density(traj, grid_size=64)
        st_age = origin_stats(dens, fin)
        pn = np.array([[st_age["peak_xy"][0], st_age["peak_xy"][1]]], dtype=np.float64)
        olat, olon = batch_norm_xy_to_latlon(tif_path, py, px, pn, h=H, w=W)
        flat, flon = batch_norm_xy_to_latlon(tif_path, py, px, fin, h=H, w=W)
        d_pk = float(haversine_km(obs_lat, obs_lon, float(olat[0]), float(olon[0])))
        d_ct = float(haversine_km(obs_lat, obs_lon, float(flat.mean()), float(flon.mean())))
        age_rows.append({
            "age_hours": age_h,
            "peak_lat": float(olat[0]), "peak_lon": float(olon[0]), "disp_peak_km": d_pk,
            "centroid_lat": float(flat.mean()), "centroid_lon": float(flon.mean()), "disp_centroid_km": d_ct
        })
        print(f"  - Age = {age_h:2d}h | Peak: ({olat[0]:.4f}, {olon[0]:.4f}) -> {d_pk:5.2f} km | Centroid: ({flat.mean():.4f}, {flon.mean():.4f}) -> {d_ct:5.2f} km")
    
    df_age = pd.DataFrame(age_rows)
    df_age.to_csv(os.path.join(OUT, "age_sensitivity_v42.csv"), index=False)

    # 5. Deterministic Metocean Sensitivity
    print("\n[+] METOCEAN FORCING SENSITIVITY (±10% Deterministic Perturbations):")
    met_rows = []
    for name, cf, wfc in [
        ("baseline", 1.0, 1.0),
        ("wind_+10%", 1.0, 1.1), ("wind_-10%", 1.0, 0.9),
        ("curr_+10%", 1.1, 1.0), ("curr_-10%", 0.9, 1.0),
        ("both_+10%", 1.1, 1.1), ("both_-10%", 0.9, 0.9),
    ]:
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PARTICLES, n_steps=24, dt_hours=1.0,
            u_current=u * cf, v_current=v * cf, u_wind=uw * wfc, v_wind=vw * wfc,
            wind_factor=wf, diffusion=0.015, rng=PERT_SEEDS[name]
        )
        _, fin = origin_density(traj, grid_size=64)
        la, lo = batch_norm_xy_to_latlon(tif_path, py, px, fin, h=H, w=W)
        shifts = haversine_km(part_lat, part_lon, la, lo)
        met_rows.append({
            "perturbation": name,
            "mean_shift_km": float(np.mean(shifts)),
            "median_shift_km": float(np.median(shifts)),
            "p95_shift_km": float(np.percentile(shifts, 95))
        })
        print(f"  - {name:12s} | Mean Shift: {met_rows[-1]['mean_shift_km']:5.2f} km | Median: {met_rows[-1]['median_shift_km']:5.2f} km | p95: {met_rows[-1]['p95_shift_km']:5.2f} km")
    
    df_met = pd.DataFrame(met_rows)
    df_met.to_csv(os.path.join(OUT, "metocean_sensitivity_v42.csv"), index=False)

    # 6. Forward Drift Projection (+24h)
    print("\n[+] FORWARD DRIFT PROJECTION (+24h under available forcing):")
    traj_f = forward_drift_particles(
        morph["seed_xy"], n_particles=N_PARTICLES, n_steps=24, dt_hours=1.0,
        u_current=u, v_current=v, u_wind=uw, v_wind=vw, wind_factor=wf,
        diffusion=0.015, rng=MASTER_SEED + 1
    )
    fwd_rows = []
    for step in [0, 6, 12, 18, 24]:
        mxy = traj_f[:, step, :].mean(axis=0, keepdims=True)
        la, lo = batch_norm_xy_to_latlon(tif_path, py, px, mxy, h=H, w=W)
        d_obs = float(haversine_km(obs_lat, obs_lon, float(la[0]), float(lo[0])))
        fwd_rows.append({"step_hours": step, "lat": float(la[0]), "lon": float(lo[0]), "disp_from_obs_km": d_obs})
        print(f"  - Step t = +{step:02d}h | Centroid: ({la[0]:.4f}° N, {lo[0]:.4f}° W) | Displacement from Obs: {d_obs:5.2f} km")
    
    df_fwd = pd.DataFrame(fwd_rows)
    df_fwd.to_csv(os.path.join(OUT, "forward_trajectory_v42.csv"), index=False)

    # 7. Real AIS Trajectory Ranking (CPA Matching)
    print("\n[+] MARINECADASTRE AIS TRAJECTORY CANDIDATE EVALUATION:")
    lat_mid = 0.5 * (obs_lat + origin_lat)
    lon_mid = 0.5 * (obs_lon + origin_lon)
    df_ais = MarineCadastreAISPipeline().load_and_filter_ais(
        lat_min=lat_mid - 0.6, lat_max=lat_mid + 0.6,
        lon_min=lon_mid - 0.6, lon_max=lon_mid + 0.6
    )
    
    df_rank = score_and_rank_vessels_ntro(
        df_ais, origin_lat, origin_lon,
        slick_orient_deg=slick_orient_deg,
        obs_datetime=SAR_ACQUISITION_TIME,
        age_proxy_hours=age_proxy,
        weights=(0.40, 0.25, 0.25, 0.10)
    )
    df_rank.to_csv(os.path.join(OUT, "ais_ranking_v42.csv"), index=False)
    
    top_cand = df_rank.iloc[0]
    s_obs = float(top_cand["ntro_attribution_score"])
    print(f"\n  [✓] BASELINE TOP CANDIDATE VESSEL:")
    print(f"      - Vessel Name:      {top_cand.get('vessel_name')}")
    print(f"      - MMSI:             {top_cand.get('mmsi')}")
    print(f"      - CPA Distance:     {float(top_cand['dist_km']):.2f} km to origin peak")
    print(f"      - CPA SOG:          {float(top_cand['sog_kn']):.1f} knots")
    print(f"      - CPA Time:         {top_cand.get('cpa_time')}")
    print(f"      - Total AIS Pings:  {int(top_cand.get('pings_in_bbox', 1))}")
    print(f"      - Proximity Score:  {float(top_cand['proximity_score']):.4f}")
    print(f"      - Kinematic Score:  {float(top_cand['kinematic_score']):.4f}")
    print(f"      - Alignment Score:  {float(top_cand['alignment_score']):.4f}")
    print(f"      - Temporal Score:   {float(top_cand['temporal_score']):.4f}")
    print(f"      - Final Score:      {s_obs:.6f} (Rank #1 / {len(df_rank)})")

    # 8. Predefined Weight Sensitivity Grid
    print("\n[+] WEIGHT SENSITIVITY GRID EVALUATION:")
    sens_rows = []
    base_mmsi = str(top_cand["mmsi"])
    for wname, w_tuple in WEIGHT_CONFIGS:
        dfr_w = score_and_rank_vessels_ntro(
            df_ais, origin_lat, origin_lon,
            slick_orient_deg=slick_orient_deg,
            obs_datetime=SAR_ACQUISITION_TIME,
            age_proxy_hours=age_proxy,
            weights=w_tuple
        )
        winner = dfr_w.iloc[0]
        base_match = dfr_w[dfr_w["mmsi"] == base_mmsi]
        base_rank = int(base_match["rank"].values[0]) if len(base_match) > 0 else -1

        sens_rows.append({
            "config_name": wname,
            "weights": str(w_tuple),
            "top_vessel": str(winner["vessel_name"]),
            "top_mmsi": str(winner["mmsi"]),
            "top_score": float(winner["ntro_attribution_score"]),
            "top_dist_km": float(winner["dist_km"]),
            "baseline_mmsi_rank": base_rank
        })
        print(f"  - {wname:22s} -> Top: {str(winner['vessel_name'])[:20]:20s} (MMSI: {winner['mmsi']}) | Score: {float(winner['ntro_attribution_score']):.4f} | Dist: {float(winner['dist_km']):5.1f} km")
    
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(os.path.join(OUT, "ais_weight_sensitivity_v42.csv"), index=False)
    
    top_mmsis = set(df_sens["top_mmsi"])
    stability_pct = float(np.mean(df_sens["top_mmsi"] == base_mmsi) * 100.0)
    print(f"\n  [✓] WEIGHT STABILITY SUMMARY:")
    print(f"      - Baseline MMSI:        {base_mmsi} ({top_cand.get('vessel_name')})")
    print(f"      - Top-1 Stability Rate: {stability_pct:.1f}% ({int(np.sum(df_sens['top_mmsi'] == base_mmsi))}/{len(WEIGHT_CONFIGS)} configurations)")
    print(f"      - Unique Top MMSIs:     {len(top_mmsis)} across grid: {top_mmsis}")

    # 9. Spatial Null Permutation (N = 1000)
    print(f"\n[+] SPATIAL NULL HYPOTHESIS TEST (N = {N_NULL_TRIALS} Randomized Origin Placements):")
    rng = np.random.default_rng(MASTER_SEED)
    null_scores = []
    
    for _ in range(N_NULL_TRIALS):
        rand_lat = obs_lat + rng.uniform(-0.5, 0.5)
        rand_lon = obs_lon + rng.uniform(-0.5, 0.5)
        dfr_null = score_and_rank_vessels_ntro(
            df_ais, rand_lat, rand_lon,
            slick_orient_deg=slick_orient_deg,
            obs_datetime=SAR_ACQUISITION_TIME,
            age_proxy_hours=age_proxy,
            weights=(0.40, 0.25, 0.25, 0.10)
        )
        null_scores.append(float(dfr_null.iloc[0]["ntro_attribution_score"]))
    
    null_scores = np.asarray(null_scores, dtype=np.float64)
    p_emp = float((1.0 + np.sum(null_scores >= s_obs)) / (1.0 + N_NULL_TRIALS))
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))
    null_med = float(np.median(null_scores))
    null_p95 = float(np.percentile(null_scores, 95))
    null_p99 = float(np.percentile(null_scores, 99))

    df_null = pd.DataFrame({"trial": np.arange(1, N_NULL_TRIALS + 1), "null_top_score": null_scores})
    df_null.to_csv(os.path.join(OUT, "ais_null_distribution_v42.csv"), index=False)

    if p_emp < 0.05:
        concl_str = f"Empirical p = {p_emp:.4f} < 0.05: Candidate score is statistically unusual relative to spatial null distribution."
    else:
        concl_str = f"Empirical p = {p_emp:.4f} >= 0.05: Candidate score provides investigative prioritization without decisive spatial significance under this null."

    print(f"  - Observed Top Score (S_obs): {s_obs:.6f}")
    print(f"  - Null Distribution:          {null_mean:.6f} ± {null_std:.6f} (Median: {null_med:.6f})")
    print(f"  - Null 95th / 99th Pct:       {null_p95:.6f} / {null_p99:.6f}")
    print(f"  - Empirical p-value:          p = {p_emp:.4f}")
    print(f"  - Statistical Conclusion:     {concl_str}")

    # 10. Master Summary Export
    summary = {
        "pipeline_version": "v4.2 (Rigorous Trajectory-Aware Benchmark)",
        "protocol": {
            "model_checkpoint": ckpt_path,
            "input_normalization": "Raw [0,1]",
            "segmentation_threshold": THR,
            "morphological_operations": "NONE / 0 px",
            "sar_observation_time": SAR_ACQUISITION_TIME,
        },
        "case_metadata": {
            "scene": fname,
            "patch_index": patch_idx,
            "observed_centroid": {"lat": obs_lat, "lon": obs_lon},
            "gt_area_km2": gt_area_info["area_km2"],
            "pred_area_km2": pred_area_info["area_km2"],
            "gt_pixel_count": gt_px,
            "pred_pixel_count": pred_px,
            "model_age_proxy_hours": age_proxy,
            "slick_orientation_deg": slick_orient_deg,
        },
        "backward_source_reconstruction": {
            "origin_peak": {"lat": origin_lat, "lon": origin_lon, "disp_from_obs_km": disp_peak},
            "particle_centroid": {"lat": geo_centroid["mean_lat"], "lon": geo_centroid["mean_lon"], "disp_from_obs_km": disp_centroid},
            "containment_radii_centroid_km": {
                "r50": geo_centroid["r50_km"],
                "r90": geo_centroid["r90_km"],
                "r95": geo_centroid["r95_km"],
                "min": geo_centroid["min_km"],
                "max": geo_centroid["max_km"],
            },
        },
        "forward_drift_projection_km": {f"t_{r['step_hours']:02d}h": r["disp_from_obs_km"] for r in fwd_rows},
        "ais_attribution": {
            "unique_vessels_searched": len(df_rank),
            "total_trajectory_pings_evaluated": len(df_ais),
            "top_candidate": {
                "vessel_name": str(top_cand.get("vessel_name")),
                "mmsi": str(top_cand.get("mmsi")),
                "cpa_dist_km": float(top_cand["dist_km"]),
                "cpa_sog_kn": float(top_cand["sog_kn"]),
                "cpa_time": str(top_cand.get("cpa_time")),
                "pings_in_bbox": int(top_cand.get("pings_in_bbox", 1)),
                "proximity_score": float(top_cand["proximity_score"]),
                "kinematic_score": float(top_cand["kinematic_score"]),
                "alignment_score": float(top_cand["alignment_score"]),
                "temporal_score": float(top_cand["temporal_score"]),
                "gap_penalty": float(top_cand["gap_penalty"]),
                "ntro_attribution_score": s_obs,
            },
            "weight_sensitivity": {
                "top1_stability_pct": stability_pct,
                "unique_top_mmsis": list(top_mmsis),
            },
            "spatial_null_test": {
                "n_trials": N_NULL_TRIALS,
                "null_mean": null_mean,
                "null_std": null_std,
                "null_p95": null_p95,
                "null_p99": null_p99,
                "empirical_p_value": p_emp,
                "conclusion": concl_str,
            }
        },
        "scientific_disclaimers": {
            "attribution": "Prioritized candidate vessel for investigation; does not constitute causal or legal proof of responsibility.",
            "age_proxy": "Model-derived empirical proxy; not a verified release timestamp.",
            "forward_drift": "24-hour forward drift projection under available static/reanalyzed metocean forcing.",
        }
    }

    with open(os.path.join(OUT, "physics_ais_v42_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Master Table CSV
    summary_flat = [{
        "Scene": fname,
        "Obs_Lat": obs_lat,
        "Obs_Lon": obs_lon,
        "GT_km2": gt_area_info["area_km2"],
        "Pred_km2": pred_area_info["area_km2"],
        "Age_Proxy_h": age_proxy,
        "Origin_Peak_Lat": origin_lat,
        "Origin_Peak_Lon": origin_lon,
        "Origin_Disp_km": disp_peak,
        "Radius_r50_km": geo_centroid["r50_km"],
        "Radius_r90_km": geo_centroid["r90_km"],
        "Radius_r95_km": geo_centroid["r95_km"],
        "Top_Candidate": str(top_cand.get("vessel_name")),
        "Top_MMSI": str(top_cand.get("mmsi")),
        "Top_Dist_km": float(top_cand["dist_km"]),
        "Top_Score": s_obs,
        "Top1_Stability_Pct": stability_pct,
        "Spatial_Null_p": p_emp
    }]
    pd.DataFrame(summary_flat).to_csv(os.path.join(OUT, "physics_ais_v42_summary.csv"), index=False)

    # 11. Multi-Panel Scientific Validation Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Age vs Origin Displacement
    axes[0, 0].plot(df_age["age_hours"], df_age["disp_centroid_km"], "o-", color="#1f77b4", lw=2, label="Particle Centroid")
    axes[0, 0].plot(df_age["age_hours"], df_age["disp_peak_km"], "s--", color="#d62728", lw=1.5, label="Probability Peak")
    axes[0, 0].axvline(age_proxy, color="green", linestyle=":", label=f"Model Age Proxy ({age_proxy:.1f}h)")
    axes[0, 0].set_title("(A) Backward Drift vs Age Proxy", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Hypothetical Age (hours)")
    axes[0, 0].set_ylabel("Displacement from Observation (km)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # Subplot 2: Metocean Sensitivity
    pert_names = [r["perturbation"] for r in met_rows[1:]]
    pert_shifts = [r["mean_shift_km"] for r in met_rows[1:]]
    axes[0, 1].barh(pert_names, pert_shifts, color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("(B) Metocean Forcing Sensitivity (±10%)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Mean Particle Cloud Shift (km)")
    axes[0, 1].grid(axis="x", alpha=0.3)

    # Subplot 3: Spatial Null Distribution
    axes[1, 0].hist(null_scores, bins=30, color="#4c72b0", edgecolor="black", alpha=0.75, label="Spatial Null Trials")
    axes[1, 0].axvline(s_obs, color="red", lw=2.5, linestyle="-", label=f"Observed Top Score ({s_obs:.3f})")
    axes[1, 0].axvline(null_p95, color="orange", lw=1.5, linestyle="--", label=f"Null 95th Pct ({null_p95:.3f})")
    axes[1, 0].set_title(f"(C) Spatial Null Permutation (N=1000, p={p_emp:.4f})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Candidate Attribution Score")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    # Subplot 4: Candidate Investigation Dossier
    axes[1, 1].axis("off")
    dossier_text = (
        f"PHYSIO-GRAPHSPILL ATTRIBUTION DOSSIER\n"
        f"--------------------------------------------------\n"
        f"Scene:             {fname}\n"
        f"Observed Slick:    {obs_lat:.4f}° N, {obs_lon:.4f}° W\n"
        f"Estimated Origin:  {origin_lat:.4f}° N, {origin_lon:.4f}° W\n"
        f"Containment (r90): {geo_centroid['r90_km']:.2f} km (r95: {geo_centroid['r95_km']:.2f} km)\n\n"
        f"TOP CANDIDATE VESSEL:\n"
        f"Name:              {top_cand.get('vessel_name')}\n"
        f"MMSI:              {top_cand.get('mmsi')}\n"
        f"CPA Distance:      {float(top_cand['dist_km']):.2f} km to peak\n"
        f"Attribution Score: {s_obs:.4f}\n"
        f"  - Proximity:     {float(top_cand['proximity_score']):.4f}  (w=0.40)\n"
        f"  - Kinematic:     {float(top_cand['kinematic_score']):.4f}  (w=0.25)\n"
        f"  - Alignment:     {float(top_cand['alignment_score']):.4f}  (w=0.25)\n"
        f"  - Temporal:      {float(top_cand['temporal_score']):.4f}  (w=0.10)\n\n"
        f"STATISTICAL ROBUSTNESS:\n"
        f"Weight Stability:  {stability_pct:.1f}% top-rank retention\n"
        f"Spatial Null Test: p = {p_emp:.4f} (N = 1,000 trials)\n\n"
        f"STATUS: Candidate Prioritization (Non-Causal)"
    )
    axes[1, 1].text(0.04, 0.50, dossier_text, family="monospace", fontsize=9, va="center",
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#cccccc", lw=1.2))

    fig.suptitle("Physio-GraphSpill: Physics-Coupled AIS Vessel Attribution Validation (v4.2)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    panel_path = os.path.join(OUT, "physics_ais_validation_panel_v42.png")
    fig.savefig(panel_path, dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(" [✓] VALIDATION SUITE v4.2 COMPLETED SUCCESSFULLY")
    print(f" [✓] Master Summary JSON : {os.path.join(OUT, 'physics_ais_v42_summary.json')}")
    print(f" [✓] Master Summary CSV  : {os.path.join(OUT, 'physics_ais_v42_summary.csv')}")
    print(f" [✓] High-Res Panel Plot : {panel_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
