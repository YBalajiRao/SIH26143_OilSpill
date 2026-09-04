import os, sys, json
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
from src.utils.slick_morphology import mask_features
from src.utils.geo_utils import patch_pixel_to_latlon
from src.environment.real_netcdf_forcing import RealMetoceanForcingEngine
from src.drift.probabilistic_drift import backward_drift_particles, forward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

ROOT = r"D:\SIH26143_OilSpill"
OUT_DIR = os.path.join(ROOT, "results", "physics_ais_validation")
os.makedirs(OUT_DIR, exist_ok=True)

THR = 0.50  # FROZEN
N_PART = 1000
N_NULL = 1000
SEED = 20181207
GSD_M = 10.0  # report area with this assumption; label as 10 m GSD

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
    """Heuristic only — not validated release time."""
    if area_km2 <= 0:
        return 6.0
    c = abs(contrast_db_abs) / 25.0
    return float(np.clip(2.5 * (area_km2 * 10.0) ** 0.45 * c, 1.5, 48.0))

def containment_radius_km(final_xy, peak_xy, frac=0.5, km_per_unit=30.0):
    """final_xy: (N,2) in [0,1] patch-normalized; rough km scale for reporting."""
    d = np.sqrt(((final_xy - peak_xy) ** 2).sum(axis=1))
    d_sorted = np.sort(d)
    k = int(np.clip(np.ceil(frac * len(d_sorted)) - 1, 0, len(d_sorted) - 1))
    return float(d_sorted[k] * km_per_unit)

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("FROZEN: thr=0.50 morph=none preprocess=raw[0,1]")

    model, ckpt_path = load_e52(device)
    print("CKPT:", ckpt_path)

    # --- Case: Dec-7 matched patch 482 (oil-positive, temporally aligned AIS/metocean) ---
    raw = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path = os.path.join(raw, "dataframe_val_dataset_256_90.csv")
    img_dir = os.path.join(raw, "images")
    msk_dir = os.path.join(raw, "masks")
    ds = GulfSARPatchDataset(csv_path, img_dir, msk_dir, transform=get_val_transforms())
    idx = 482
    img_t, msk_t = ds[idx]
    row = ds.df.iloc[idx]
    fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip("\"'").split(",")]
    py, px = coords[0], coords[1]
    tif_path = os.path.join(img_dir, fname)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t.unsqueeze(0).to(device))).squeeze().cpu().numpy()
    gt = msk_t.numpy().squeeze()
    binary = (prob >= THR).astype(np.float32)
    morph = mask_features(binary)
    gt_px = int((gt >= 0.5).sum())
    pred_px = int(morph["area_px"])
    gt_km2 = gt_px * (GSD_M * GSD_M) / 1e6
    pred_km2 = pred_px * (GSD_M * GSD_M) / 1e6

    cy = int(morph["centroid_xy"][1] * 255)
    cx = int(morph["centroid_xy"][0] * 255)
    lat, lon = patch_pixel_to_latlon(tif_path, py, px, local_y=cy, local_x=cx)
    age0 = age_proxy_hours(pred_km2, contrast_db_abs=18.0)
    orient = float(morph.get("orientation_deg", 0.0))

    print("\n" + "=" * 78)
    print(" CASE (segmentation frozen path)")
    print("=" * 78)
    print(f" scene={fname}  patch_idx={idx}  coords=({py},{px})")
    print(f" GT_px={gt_px} ({gt_km2:.4f} km^2 @10m)  Pred_px={pred_px} ({pred_km2:.4f} km^2)")
    print(f" elong={morph['elongation']:.3f}  orient={orient:.1f} deg")
    print(f" obs lat/lon ≈ {lat:.4f}, {lon:.4f}")
    print(f" age_proxy_h (heuristic) ≈ {age0:.2f}  [NOT validated release time]")

    if pred_px < 100:
        print("ERROR: pred oil too small — stop AIS; fix segmentation path.")
        return

    met = RealMetoceanForcingEngine()
    f0 = met.get_velocity_at_latlon(lat, lon, "2018-12-07")
    print(f" metocean: {f0.get('source', f0)}")

    # ========== 1) AGE PROXY SENSITIVITY ==========
    print("\n" + "=" * 78)
    print(" [1] AGE-PROXY SENSITIVITY (origin peak shift)")
    print("=" * 78)
    age_list = [6, 8, 10, 12, 18, 24]
    age_rows = []
    base_u, base_v = f0["u_current"], f0["v_current"]
    base_uw, base_vw = f0["u_wind"], f0["v_wind"]
    wf = f0.get("wind_factor", f0.get("wind_drift_factor", 0.035))

    for age_h in age_list:
        n_steps = int(age_h)
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PART, n_steps=n_steps, dt_hours=1.0,
            u_current=base_u, v_current=base_v, u_wind=base_uw, v_wind=base_vw,
            wind_factor=wf, diffusion=0.015, rng=SEED + n_steps,
        )
        dens, final_pts = origin_density(traj, grid_size=64)
        st = origin_stats(dens, final_pts)
        pkx, pky = st["peak_xy"]
        o_lat = lat + (pky - 0.5) * 0.28
        o_lon = lon + (pkx - 0.5) * 0.28
        # displacement from observed slick centroid
        d_km = float(np.hypot(o_lat - lat, o_lon - lon) * 111.0)
        age_rows.append({
            "age_proxy_h": age_h, "origin_lat": o_lat, "origin_lon": o_lon,
            "obs_to_origin_km": d_km, "peak_xy": f"{pkx:.4f},{pky:.4f}",
        })
        print(f"  age={age_h:2d}h  origin=({o_lat:.4f},{o_lon:.4f})  disp≈{d_km:.2f} km")

    pd.DataFrame(age_rows).to_csv(os.path.join(OUT_DIR, "age_proxy_sensitivity.csv"), index=False)

    # default case: 24h backward (pipeline standard)
    traj_b = backward_drift_particles(
        morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
        u_current=base_u, v_current=base_v, u_wind=base_uw, v_wind=base_vw,
        wind_factor=wf, diffusion=0.015, rng=SEED,
    )
    dens_b, final_b = origin_density(traj_b, grid_size=64)
    st_b = origin_stats(dens_b, final_b)
    pkx, pky = st_b["peak_xy"]
    origin_lat = lat + (pky - 0.5) * 0.28
    origin_lon = lon + (pkx - 0.5) * 0.28
    cent = final_b.mean(axis=0)
    r50 = containment_radius_km(final_b, np.array(st_b["peak_xy"]), 0.50)
    r95 = containment_radius_km(final_b, np.array(st_b["peak_xy"]), 0.95)
    disp0 = float(np.hypot(origin_lat - lat, origin_lon - lon) * 111.0)

    print("\n" + "=" * 78)
    print(" [2] PARTICLE UNCERTAINTY (−24 h hindcast, N=1000)")
    print("=" * 78)
    print(f"  peak origin ≈ ({origin_lat:.4f} N, {origin_lon:.4f} W)")
    print(f"  particle centroid (norm) = ({cent[0]:.4f}, {cent[1]:.4f})")
    print(f"  rough 50% containment radius ≈ {r50:.2f} km (scale-dependent)")
    print(f"  rough 95% containment radius ≈ {r95:.2f} km")
    print(f"  obs→peak displacement ≈ {disp0:.2f} km")

    # ========== 3) METOCEAN SENSITIVITY ==========
    print("\n" + "=" * 78)
    print(" [3] METOCEAN SENSITIVITY (±10% wind / current)")
    print("=" * 78)
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
    for name, cf, wfct in perturbs:
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=500, n_steps=24, dt_hours=1.0,
            u_current=base_u * cf, v_current=base_v * cf,
            u_wind=base_uw * wfct, v_wind=base_vw * wfct,
            wind_factor=wf, diffusion=0.015, rng=SEED + hash(name) % 999,
        )
        dens, fp = origin_density(traj, grid_size=48)
        st = origin_stats(dens, fp)
        qkx, qky = st["peak_xy"]
        olat = lat + (qky - 0.5) * 0.28
        olon = lon + (qkx - 0.5) * 0.28
        dpeak = float(np.hypot(olat - origin_lat, olon - origin_lon) * 111.0)
        met_rows.append({"perturbation": name, "origin_lat": olat, "origin_lon": olon, "shift_from_baseline_km": dpeak})
        print(f"  {name:12s}  origin=({olat:.4f},{olon:.4f})  Δpeak≈{dpeak:.2f} km")
    pd.DataFrame(met_rows).to_csv(os.path.join(OUT_DIR, "metocean_sensitivity.csv"), index=False)

    # ========== 4) FORWARD PROJECTION (not a validated forecast) ==========
    print("\n" + "=" * 78)
    print(" [4] +24 h FORWARD DRIFT PROJECTION (available forcing; NOT validated forecast)")
    print("=" * 78)
    traj_f = forward_drift_particles(
        morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
        u_current=base_u, v_current=base_v, u_wind=base_uw, v_wind=base_vw,
        wind_factor=wf, diffusion=0.015, rng=SEED + 1,
    )
    dens_f, final_f = origin_density(traj_f, grid_size=64)
    st_f = origin_stats(dens_f, final_f)
    fkx, fky = st_f["peak_xy"]
    fut_lat = lat + (fky - 0.5) * 0.28
    fut_lon = lon + (fkx - 0.5) * 0.28
    print(f"  forward density peak ≈ ({fut_lat:.4f}, {fut_lon:.4f})")
    print("  Label on poster: '24-h forward drift projection under available metocean forcing'")

    # ========== 5) AIS + WEIGHT SENSITIVITY ==========
    print("\n" + "=" * 78)
    print(" [5] AIS CANDIDATES + WEIGHT SENSITIVITY")
    print("=" * 78)
    print("  Temporal window: AIS files Dec 6–8 2018 around SAR 2018-12-07 (MATCHED CASE)")
    ais = MarineCadastreAISPipeline()
    df_ais = ais.load_and_filter_ais(origin_lat - 0.5, origin_lat + 0.5, origin_lon - 0.5, origin_lon + 0.5)
    print(f"  vessels in box: {len(df_ais)}")

    # Baseline ranking via existing NTRO scorer
    df_rank = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon, slick_orient_deg=orient)
    df_rank.to_csv(os.path.join(OUT_DIR, "baseline_ranking.csv"), index=False)
    top = df_rank.iloc[0]
    print(f"  baseline top-ranked CANDIDATE: {top.get('vessel_name','?')}  "
          f"score={float(top['ntro_attribution_score']):.4f}  "
          f"dist_km={float(top.get('dist_km', np.nan)):.2f}")
    print("  (NOT a determination of legal responsibility)")

    # Manual weight grid on components if present
    need = ["proximity_score", "kinematic_score", "alignment_score"]
    if all(c in df_rank.columns for c in need):
        wsets = [
            ("baseline_50_25_25", 0.50, 0.25, 0.25),
            ("prox_up_60_20_20", 0.60, 0.20, 0.20),
            ("prox_dn_40_30_30", 0.40, 0.30, 0.30),
            ("kin_up_40_40_20", 0.40, 0.40, 0.20),
            ("align_up_40_20_40", 0.40, 0.20, 0.40),
            ("equal_33", 1/3, 1/3, 1/3),
        ]
        sens = []
        for wname, wp, wk, wa in wsets:
            sc = (wp * df_rank["proximity_score"].values
                  + wk * df_rank["kinematic_score"].values
                  + wa * df_rank["alignment_score"].values)
            if "gap_penalty" in df_rank.columns:
                sc = sc - df_rank["gap_penalty"].values
            order = np.argsort(-sc)
            j = int(order[0])
            sens.append({
                "weight_set": wname, "wp": wp, "wk": wk, "wa": wa,
                "top_vessel": str(df_rank.iloc[j]["vessel_name"]),
                "top_mmsi": str(df_rank.iloc[j]["mmsi"]),
                "top_score": float(sc[j]),
                "top_dist_km": float(df_rank.iloc[j].get("dist_km", np.nan)),
            })
            print(f"  {wname:18s} → {sens[-1]['top_vessel'][:22]:22s}  "
                  f"score={sens[-1]['top_score']:.4f}  dist={sens[-1]['top_dist_km']:.1f} km")
        pd.DataFrame(sens).to_csv(os.path.join(OUT_DIR, "weight_sensitivity.csv"), index=False)
        tops = set(s["top_mmsi"] for s in sens)
        print(f"  unique top MMSI across weight sets: {len(tops)} → {tops}")
    else:
        print("  [!] ranking CSV missing component columns; skip weight grid")

    # Evidence table (top 10)
    cols = [c for c in ["rank", "mmsi", "vessel_name", "dist_km", "proximity_score",
                        "kinematic_score", "alignment_score", "gap_penalty",
                        "ntro_attribution_score", "investigation_evidence"] if c in df_rank.columns]
    df_rank[cols].head(10).to_csv(os.path.join(OUT_DIR, "candidate_evidence_top10.csv"), index=False)

    # ========== 6) NULL / RANDOMIZATION N=1000 ==========
    print("\n" + "=" * 78)
    print(f" [6] SPATIAL NULL TEST (N={N_NULL}) — empirical exceedance of top score")
    print("=" * 78)
    s_obs = float(df_rank.iloc[0]["ntro_attribution_score"])
    rng = np.random.default_rng(SEED)
    # random origins in +/- 0.5 deg box around observed slick
    null_scores = []
    for i in range(N_NULL):
        rlat = lat + rng.uniform(-0.5, 0.5)
        rlon = lon + rng.uniform(-0.5, 0.5)
        dfr = score_and_rank_vessels_ntro(df_ais, rlat, rlon, slick_orient_deg=orient)
        null_scores.append(float(dfr.iloc[0]["ntro_attribution_score"]))
    null_scores = np.asarray(null_scores, dtype=np.float64)
    # p = (1 + #{S_null >= S_obs}) / (1 + N)
    p_emp = (1.0 + np.sum(null_scores >= s_obs)) / (1.0 + N_NULL)
    print(f"  S_obs (top candidate @ reconstructed origin) = {s_obs:.6f}")
    print(f"  null mean±std = {null_scores.mean():.6f} ± {null_scores.std():.6f}")
    print(f"  null median = {np.median(null_scores):.6f}  p95 = {np.percentile(null_scores, 95):.6f}")
    print(f"  empirical p-value = {p_emp:.4f}  "
          f"{'(p<0.05)' if p_emp < 0.05 else '(not <0.05 under this null)'}")
    pd.DataFrame({"null_top_score": null_scores}).to_csv(
        os.path.join(OUT_DIR, "null_top_scores_n1000.csv"), index=False)

    # ========== FIGURE ==========
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    sar = img_t.squeeze().numpy()
    axes[0, 0].imshow(sar, cmap="gray")
    axes[0, 0].imshow(prob, cmap="plasma", alpha=0.45)
    axes[0, 0].set_title("1. SAR + P(oil) thr=0.50 frozen")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(dens_b, cmap="hot", origin="lower")
    axes[0, 1].set_title("2. Backward density (−24 h)")
    axes[0, 1].set_xlabel("grid x"); axes[0, 1].set_ylabel("grid y")

    axes[0, 2].imshow(dens_f, cmap="viridis", origin="lower")
    axes[0, 2].set_title("3. Forward projection (+24 h)\n[not validated forecast]")

    ages = [r["age_proxy_h"] for r in age_rows]
    disps = [r["obs_to_origin_km"] for r in age_rows]
    axes[1, 0].plot(ages, disps, "o-")
    axes[1, 0].set_xlabel("age proxy (h)"); axes[1, 0].set_ylabel("obs→origin km")
    axes[1, 0].set_title("4. Age-proxy sensitivity")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(null_scores, bins=40, color="steelblue", alpha=0.85)
    axes[1, 1].axvline(s_obs, color="red", lw=2, label=f"S_obs={s_obs:.3f}")
    axes[1, 1].set_title(f"5. Null top-score (N={N_NULL})\np={p_emp:.3f}")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].axis("off")
    txt = (
        f"PHYSICS/AIS VALIDATION SNAPSHOT\n"
        f"Segmentation FROZEN: thr={THR}, morph=none, [0,1]\n"
        f"Case: {fname} idx={idx}\n"
        f"GT {gt_px}px ({gt_km2:.3f}km2) | Pred {pred_px}px ({pred_km2:.3f}km2)\n"
        f"Origin peak: {origin_lat:.4f}, {origin_lon:.4f}\n"
        f"Disp obs→peak: {disp0:.1f} km | r50~{r50:.1f} r95~{r95:.1f} km\n"
        f"Top-ranked CANDIDATE (not legal fault):\n"
        f"  {top.get('vessel_name','?')}  score={s_obs:.4f}\n"
        f"Null test p={p_emp:.4f}\n"
        f"Age = heuristic PROXY only\n"
        f"Forward = projection under available forcing"
    )
    axes[1, 2].text(0.02, 0.05, txt, fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="#f5f5f5"))
    fig.suptitle("Physics + AIS validation (segmentation frozen)", fontweight="bold")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "physics_ais_validation_panel.png")
    fig.savefig(fig_path, dpi=200)
    plt.close()

    summary = {
        "segmentation_frozen": {
            "threshold": THR, "morphology": "none", "preprocess": "raw_[0,1]",
            "ckpt": ckpt_path,
        },
        "case": {
            "scene": fname, "patch_idx": idx,
            "gt_px": gt_px, "pred_px": pred_px,
            "gt_km2_10m": gt_km2, "pred_km2_10m": pred_km2,
            "lat": lat, "lon": lon,
            "age_proxy_h_heuristic": age0,
            "note_age": "model-derived proxy, not validated release time",
        },
        "drift": {
            "backward_h": 24, "forward_h": 24, "n_particles": N_PART,
            "origin_lat": origin_lat, "origin_lon": origin_lon,
            "obs_to_peak_km": disp0,
            "containment_r50_km_approx": r50, "containment_r95_km_approx": r95,
            "forward_label": "24-h forward drift projection under available forcing",
        },
        "ais": {
            "n_vessels": int(len(df_ais)),
            "top_ranked_candidate": str(top.get("vessel_name", "")),
            "top_mmsi": str(top.get("mmsi", "")),
            "top_score": s_obs,
            "wording": "top-ranked candidate vessel — not legal responsibility",
            "temporal_window": "AIS Dec 6-8 2018 with SAR 2018-12-07 (matched demo case)",
        },
        "null_test": {
            "N": N_NULL, "S_obs": s_obs,
            "null_mean": float(null_scores.mean()),
            "null_p95": float(np.percentile(null_scores, 95)),
            "empirical_p": float(p_emp),
        },
        "outputs_dir": OUT_DIR,
    }
    with open(os.path.join(OUT_DIR, "validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[✓] Outputs → {OUT_DIR}")
    print(f"[✓] Figure → {fig_path}")
    print("DONE physics/AIS validation block. Review JSON + CSVs before poster claims.")

if __name__ == "__main__":
    run()
