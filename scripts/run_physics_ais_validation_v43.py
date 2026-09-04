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

ROOT = r"D:\SIH26143_OilSpill"
OUT = os.path.join(ROOT, "results", "physics_ais_validation")
os.makedirs(OUT, exist_ok=True)

# Frozen experimental parameters
THR = 0.50
N_PARTICLES = 1000
N_NULL_TRIALS = 1000
MASTER_SEED = 20181207
REF_LAT, REF_LON = 28.3987, -88.3660
SAR_ACQUISITION_TIME = "2018-12-07 12:00:00"
TEMPORAL_WINDOW_HOURS = 6.0

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

def score_vessels_in_window(df_pings, origin_lat, origin_lon, slick_orient_deg, t_release, window_hours=6.0, weights=(0.40, 0.25, 0.25, 0.10)):
    """
    Release-window constrained scoring:
    Filters pings strictly within [t_release - window_hours, t_release + window_hours],
    identifies CPA within that window, and scores each vessel.
    """
    w_prox, w_kin, w_align, w_temp = weights
    df = df_pings.copy()
    df["dt_hours"] = (df["base_date_time"] - t_release).dt.total_seconds() / 3600.0
    
    # Filter to contemporaneous window
    df_win = df[df["dt_hours"].abs() <= window_hours].copy()
    if len(df_win) == 0:
        return pd.DataFrame()

    df_win["dist_km"] = haversine_km(origin_lat, origin_lon, df_win["lat"].values, df_win["lon"].values)
    
    records = []
    for mmsi, grp in df_win.groupby("mmsi"):
        v_name = str(grp["vessel_name"].iloc[0])
        min_idx = grp["dist_km"].idxmin()
        cpa = grp.loc[min_idx]
        
        d_min_km = float(cpa["dist_km"])
        cpa_lat = float(cpa["lat"])
        cpa_lon = float(cpa["lon"])
        cpa_sog = float(cpa["sog_kn"]) if not np.isnan(cpa["sog_kn"]) else 6.0
        cpa_cog = float(cpa["cog_deg"]) if not np.isnan(cpa["cog_deg"]) else np.nan
        cpa_time = str(cpa["base_date_time"])
        dt_h = float(cpa["dt_hours"])
        
        # Sub-scores
        s_prox = float(np.clip(np.exp(-d_min_km / 15.0), 0.0, 1.0))
        s_kin = float(np.clip(np.exp(-((cpa_sog - 5.8) ** 2) / 12.0), 0.0, 1.0))
        
        if np.isnan(cpa_cog) or cpa_cog < 0 or cpa_cog > 360:
            s_align = 0.50
        else:
            diff = abs((slick_orient_deg - cpa_cog) % 360.0)
            diff = min(diff, 360.0 - diff)
            delta_theta_axial = min(diff, abs(180.0 - diff))
            s_align = float(np.clip(1.0 - (delta_theta_axial / 90.0), 0.0, 1.0))
            
        s_temp = float(np.clip(np.exp(-(dt_h ** 2) / (2.0 * (window_hours ** 2))), 0.0, 1.0))
        
        ais_gap = int(grp.get("ais_gap_flag", pd.Series([0])).iloc[0]) if "ais_gap_flag" in grp else 0
        p_gap = 0.20 if ais_gap == 1 else 0.0
        
        score = (w_prox * s_prox) + (w_kin * s_kin) + (w_align * s_align) + (w_temp * s_temp) - p_gap
        score = float(np.clip(score, 0.0001, 0.9999))
        
        if d_min_km < 10.0 and 3.0 <= cpa_sog <= 8.0:
            evidence = "HIGH CONTEMPORANEOUS CORRELATION: Direct spatial overlap during release window & transit speed match."
        elif d_min_km < 25.0:
            evidence = "MODERATE CANDIDATE: Active in Lagrangian dispersion corridor during release window."
        else:
            evidence = "LOW SPATIAL OVERLAP: Active during release window but outside primary containment radius."

        records.append({
            "mmsi": str(mmsi),
            "vessel_name": v_name,
            "cpa_lat": cpa_lat,
            "cpa_lon": cpa_lon,
            "cpa_dist_km": d_min_km,
            "cpa_sog_kn": cpa_sog,
            "cpa_cog_deg": cpa_cog,
            "cpa_time": cpa_time,
            "dt_hours_to_release": dt_h,
            "pings_in_window": len(grp),
            "proximity_score": s_prox,
            "kinematic_score": s_kin,
            "alignment_score": s_align,
            "temporal_score": s_temp,
            "gap_penalty": p_gap,
            "attribution_score": score,
            "investigation_evidence": evidence
        })

    df_out = pd.DataFrame(records).sort_values("attribution_score", ascending=False).reset_index(drop=True)
    df_out["rank"] = np.arange(1, len(df_out) + 1)
    return df_out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(" PHYSIO-GRAPHSPILL — FINAL PHYSICS & AIS VALIDATION SUITE v4.3")
    print(" Protocol: Frozen E5.2 | Raw [0,1] | Thr = 0.50 | Morphology = NONE")
    print(" Feature: Release-Window Constrained Attribution (|t - T_release| <= 6.0h)")
    print("=" * 85)

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

    t_obs = pd.to_datetime(SAR_ACQUISITION_TIME)
    t_release = t_obs - pd.Timedelta(hours=float(age_proxy))

    print("\n[+] CASE CHARACTERIZATION:")
    print(f"  - Scene / Patch:               {fname} (Patch #{patch_idx})")
    print(f"  - Observed Centroid:           {obs_lat:.4f}° N, {obs_lon:.4f}° W (Δref = {d_ref:.3f} km)")
    print(f"  - Ground Truth Area:           {gt_px} px ({gt_area_info['area_km2']:.4f} km²)")
    print(f"  - Predicted Area:              {pred_px} px ({pred_area_info['area_km2']:.4f} km²)")
    print(f"  - Model-Derived Age Proxy:     {age_proxy:.2f} hours")
    print(f"  - SAR Observation Time:        {t_obs}")
    print(f"  - Inferred Release Window:     {t_release} (±{TEMPORAL_WINDOW_HOURS:.1f}h)")
    print(f"  - Slick Orientation:           {slick_orient_deg:.2f}° (Elongation = {morph['elongation']:.2f})")

    # 2. Metocean Forcing
    forcing_engine = RealMetoceanForcingEngine()
    forcing = forcing_engine.get_velocity_at_latlon(obs_lat, obs_lon, "2018-12-07")
    wf = forcing.get("wind_drift_factor", forcing.get("wind_factor", 0.035))
    u, v = forcing["u_current"], forcing["v_current"]
    uw, vw = forcing["u_wind"], forcing["v_wind"]
    print(f"\n[+] METOCEAN FORCING (ERA5 + CMEMS Dec 7, 2018):")
    print(f"  - Current (u, v):              ({u:.4f}, {v:.4f}) m/s")
    print(f"  - Wind (uw, vw):               ({uw:.4f}, {vw:.4f}) m/s (Drift Factor = {wf:.3f})")

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
    print(f"  - Origin Peak:                 {origin_lat:.4f}° N, {origin_lon:.4f}° W (Obs->Peak: {disp_peak:.2f} km)")
    print(f"  - Particle Centroid:           {geo_centroid['mean_lat']:.4f}° N, {geo_centroid['mean_lon']:.4f}° W (Obs->Centroid: {disp_centroid:.2f} km)")
    print(f"  - Centroid Radii:              r50 = {geo_centroid['r50_km']:.2f} km | r90 = {geo_centroid['r90_km']:.2f} km | r95 = {geo_centroid['r95_km']:.2f} km")

    # 4. Age Sensitivity
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
    df_age = pd.DataFrame(age_rows)
    df_age.to_csv(os.path.join(OUT, "age_sensitivity_v43.csv"), index=False)

    # 5. Deterministic Metocean Sensitivity
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
    df_met = pd.DataFrame(met_rows)
    df_met.to_csv(os.path.join(OUT, "metocean_sensitivity_v43.csv"), index=False)

    # 6. Forward Drift Projection (+24h)
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
    df_fwd = pd.DataFrame(fwd_rows)
    df_fwd.to_csv(os.path.join(OUT, "forward_trajectory_v43.csv"), index=False)

    # 7. Release-Window Constrained AIS Candidate Attribution
    print("\n[+] CONTEMPORANEOUS AIS ATTRIBUTION (|t - T_release| <= 6.0h):")
    lat_mid = 0.5 * (obs_lat + origin_lat)
    lon_mid = 0.5 * (obs_lon + origin_lon)
    df_pings = MarineCadastreAISPipeline().load_and_filter_ais(
        lat_min=lat_mid - 0.6, lat_max=lat_mid + 0.6,
        lon_min=lon_mid - 0.6, lon_max=lon_mid + 0.6
    )
    
    df_rank_win = score_vessels_in_window(
        df_pings, origin_lat, origin_lon,
        slick_orient_deg=slick_orient_deg,
        t_release=t_release,
        window_hours=TEMPORAL_WINDOW_HOURS,
        weights=(0.40, 0.25, 0.25, 0.10)
    )
    df_rank_win.to_csv(os.path.join(OUT, "ais_ranking_release_window_v43.csv"), index=False)

    print(f"  [✓] Evaluated {len(df_rank_win)} Contemporaneous Vessels active during Release Window.")
    print("\n  [✓] TOP 5 CONTEMPORANEOUS CANDIDATE VESSELS:")
    print(df_rank_win[["rank", "mmsi", "vessel_name", "cpa_dist_km", "cpa_sog_kn", "dt_hours_to_release", "proximity_score", "kinematic_score", "temporal_score", "attribution_score"]].head(5).to_string(index=False))

    top_cand = df_rank_win.iloc[0]
    s_obs = float(top_cand["attribution_score"])

    # 8. Weight Sensitivity Grid on Contemporaneous Fleet
    print("\n[+] WEIGHT SENSITIVITY GRID (Contemporaneous Fleet):")
    sens_rows = []
    base_mmsi = str(top_cand["mmsi"])
    for wname, w_tuple in WEIGHT_CONFIGS:
        dfr_w = score_vessels_in_window(
            df_pings, origin_lat, origin_lon,
            slick_orient_deg=slick_orient_deg,
            t_release=t_release,
            window_hours=TEMPORAL_WINDOW_HOURS,
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
            "top_score": float(winner["attribution_score"]),
            "top_dist_km": float(winner["cpa_dist_km"]),
            "baseline_mmsi_rank": base_rank
        })
        print(f"  - {wname:22s} -> Top: {str(winner['vessel_name'])[:20]:20s} (MMSI: {winner['mmsi']}) | Score: {float(winner['attribution_score']):.4f} | Dist: {float(winner['cpa_dist_km']):5.1f} km | Base MMSI Rank: #{base_rank}")
    
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(os.path.join(OUT, "ais_weight_sensitivity_v43.csv"), index=False)
    
    top_mmsis = set(df_sens["top_mmsi"])
    stability_pct = float(np.mean(df_sens["top_mmsi"] == base_mmsi) * 100.0)
    print(f"\n  [✓] Contemporaneous Top-1 Stability Rate: {stability_pct:.1f}% ({int(np.sum(df_sens['top_mmsi'] == base_mmsi))}/{len(WEIGHT_CONFIGS)} configs)")

    # 9. Spatial Null Permutation on Contemporaneous Fleet (N = 1000)
    print(f"\n[+] SPATIAL NULL HYPOTHESIS TEST (N = {N_NULL_TRIALS} Trials on Contemporaneous Fleet):")
    rng = np.random.default_rng(MASTER_SEED)
    null_scores = []
    
    for _ in range(N_NULL_TRIALS):
        rand_lat = obs_lat + rng.uniform(-0.5, 0.5)
        rand_lon = obs_lon + rng.uniform(-0.5, 0.5)
        dfr_null = score_vessels_in_window(
            df_pings, rand_lat, rand_lon,
            slick_orient_deg=slick_orient_deg,
            t_release=t_release,
            window_hours=TEMPORAL_WINDOW_HOURS,
            weights=(0.40, 0.25, 0.25, 0.10)
        )
        if len(dfr_null) > 0:
            null_scores.append(float(dfr_null.iloc[0]["attribution_score"]))
        else:
            null_scores.append(0.0)
    
    null_scores = np.asarray(null_scores, dtype=np.float64)
    p_emp = float((1.0 + np.sum(null_scores >= s_obs)) / (1.0 + N_NULL_TRIALS))
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))
    null_med = float(np.median(null_scores))
    null_p95 = float(np.percentile(null_scores, 95))
    null_p99 = float(np.percentile(null_scores, 99))

    df_null = pd.DataFrame({"trial": np.arange(1, N_NULL_TRIALS + 1), "null_top_score": null_scores})
    df_null.to_csv(os.path.join(OUT, "ais_null_distribution_v43.csv"), index=False)

    if p_emp < 0.05:
        concl_str = f"Empirical p = {p_emp:.4f} < 0.05: Observed candidate score is statistically significant relative to the spatial null distribution."
    else:
        concl_str = f"Empirical p = {p_emp:.4f} >= 0.05: Candidate score provides investigative prioritization without decisive spatial significance under this null."

    print(f"  - Observed Top Score (S_obs): {s_obs:.6f}")
    print(f"  - Null Distribution:          {null_mean:.6f} ± {null_std:.6f} (Median: {null_med:.6f})")
    print(f"  - Null 95th / 99th Pct:       {null_p95:.6f} / {null_p99:.6f}")
    print(f"  - Empirical p-value:          p = {p_emp:.4f}")
    print(f"  - Conclusion:                 {concl_str}")

    # 10. Summary Export
    summary = {
        "pipeline_version": "v4.3 (Release-Window Constrained Attribution)",
        "protocol": {
            "model_checkpoint": ckpt_path,
            "input_normalization": "Raw [0,1]",
            "segmentation_threshold": THR,
            "morphological_operations": "NONE / 0 px",
            "sar_observation_time": SAR_ACQUISITION_TIME,
            "temporal_window_hours": TEMPORAL_WINDOW_HOURS,
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
            "inferred_release_time": str(t_release),
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
        "contemporaneous_ais_attribution": {
            "contemporaneous_vessels_searched": len(df_rank_win),
            "top_candidate": {
                "vessel_name": str(top_cand.get("vessel_name")),
                "mmsi": str(top_cand.get("mmsi")),
                "cpa_dist_km": float(top_cand["cpa_dist_km"]),
                "cpa_sog_kn": float(top_cand["cpa_sog_kn"]),
                "cpa_cog_deg": float(top_cand["cpa_cog_deg"]),
                "cpa_time": str(top_cand.get("cpa_time")),
                "dt_hours_to_release": float(top_cand["dt_hours_to_release"]),
                "proximity_score": float(top_cand["proximity_score"]),
                "kinematic_score": float(top_cand["kinematic_score"]),
                "alignment_score": float(top_cand["alignment_score"]),
                "temporal_score": float(top_cand["temporal_score"]),
                "attribution_score": s_obs,
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

    with open(os.path.join(OUT, "physics_ais_v43_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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
        "Top_Contemporaneous_Candidate": str(top_cand.get("vessel_name")),
        "Top_MMSI": str(top_cand.get("mmsi")),
        "Top_Dist_km": float(top_cand["cpa_dist_km"]),
        "Top_Score": s_obs,
        "Top1_Stability_Pct": stability_pct,
        "Spatial_Null_p": p_emp
    }]
    pd.DataFrame(summary_flat).to_csv(os.path.join(OUT, "physics_ais_v43_summary.csv"), index=False)

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
    axes[1, 0].set_title(f"(C) Contemporaneous Spatial Null (N=1000, p={p_emp:.4f})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Attribution Score")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    # Subplot 4: Candidate Investigation Dossier
    axes[1, 1].axis("off")
    dossier_text = (
        f"PHYSIO-GRAPHSPILL CONTEMPORANEOUS ATTRIBUTION\n"
        f"--------------------------------------------------\n"
        f"Scene:             {fname}\n"
        f"Observed Slick:    {obs_lat:.4f}° N, {obs_lon:.4f}° W\n"
        f"Estimated Origin:  {origin_lat:.4f}° N, {origin_lon:.4f}° W\n"
        f"Release Window:    {str(t_release)[:19]} (±6h)\n\n"
        f"TOP CONTEMPORANEOUS CANDIDATE:\n"
        f"Name:              {top_cand.get('vessel_name')}\n"
        f"MMSI:              {top_cand.get('mmsi')}\n"
        f"CPA in Window:     {float(top_cand['cpa_dist_km']):.2f} km to peak\n"
        f"Time Delta (Δt):   {float(top_cand['dt_hours_to_release']):+.2f} hours\n"
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

    fig.suptitle("Physio-GraphSpill: Release-Window Constrained Vessel Attribution (v4.3)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    panel_path = os.path.join(OUT, "physics_ais_validation_panel_v43.png")
    fig.savefig(panel_path, dpi=300)
    plt.close()

    print("\n" + "=" * 85)
    print(" [✓] VALIDATION SUITE v4.3 COMPLETED SUCCESSFULLY")
    print(f" [✓] Master Summary JSON : {os.path.join(OUT, 'physics_ais_v43_summary.json')}")
    print(f" [✓] Master Summary CSV  : {os.path.join(OUT, 'physics_ais_v43_summary.csv')}")
    print(f" [✓] High-Res Panel Plot : {panel_path}")
    print("=" * 85)

if __name__ == "__main__":
    main()
