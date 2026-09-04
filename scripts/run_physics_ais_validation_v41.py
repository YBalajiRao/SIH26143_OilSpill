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
THR, N_PART, N_NULL, SEED = 0.50, 1000, 1000, 20181207
REF_LAT, REF_LON = 28.3987, -88.3660

def load_e52(device):
    ckpt = os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
    m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    st = torch.load(ckpt, map_location=device)
    sd = st["model_state_dict"] if isinstance(st, dict) and "model_state_dict" in st else st
    m.load_state_dict(sd, strict=False)
    return m.eval(), ckpt

def age_proxy_hours(area_km2, cabs=18.0):
    if area_km2 <= 0:
        return 6.0
    return float(np.clip(2.5 * (area_km2 * 10.0) ** 0.45 * (abs(cabs) / 25.0), 1.5, 48.0))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" PHYSICS/AIS v4.1 — FIXED LON + SAME GEOREF EVERYWHERE")
    print("=" * 80)
    print(f"LON_LEFT={LON_LEFT} LON_RIGHT={LON_RIGHT} (eastward span={LON_RIGHT-LON_LEFT})")

    model, ckpt = load_e52(device)
    raw = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train")
    ds = GulfSARPatchDataset(
        os.path.join(raw, "dataframe_val_dataset_256_90.csv"),
        os.path.join(raw, "images"),
        os.path.join(raw, "masks"),
        transform=get_val_transforms(),
    )
    idx = 482
    img_t, msk_t = ds[idx]
    row = ds.df.iloc[idx]
    fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
    py, px = [int(c.strip()) for c in str(row["coordinates"]).strip("\"'").split(",")]
    tif_path = os.path.join(raw, "images", fname)
    H, W = get_scene_hw(tif_path)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t.unsqueeze(0).to(device))).squeeze().cpu().numpy()
    gt = msk_t.numpy().squeeze()
    binary = (prob >= THR).astype(np.float32)
    morph = mask_features(binary)
    gt_px = int((gt >= 0.5).sum())
    pred_px = int(morph["area_px"])
    gt_a = get_exact_raster_resolution_and_area(tif_path, gt_px)
    pr_a = get_exact_raster_resolution_and_area(tif_path, pred_px)
    cy = int(np.clip(morph["centroid_xy"][1] * 255.0, 0, 255))
    cx = int(np.clip(morph["centroid_xy"][0] * 255.0, 0, 255))
    orient = float(morph.get("orientation_deg", -170.4))

    print("\n[0] COORDINATE SANITY")
    obs_lat, obs_lon = patch_pixel_to_latlon(tif_path, py, px, local_y=cy, local_x=cx)
    dref = float(haversine_km(REF_LAT, REF_LON, obs_lat, obs_lon))
    print(f"  scene={fname} patch=({py},{px}) local=({cy},{cx}) HxW={H}x{W}")
    print(f"  OBSERVED = {obs_lat:.4f}, {obs_lon:.4f}")
    print(f"  vs REF {REF_LAT}, {REF_LON} -> {dref:.3f} km")
    if dref > 5.0:
        print("  FATAL: lon/lat still wrong. Abort.")
        raise SystemExit(2)
    print("  [OK] georef matches trusted reference")

    for name, ly, lx in [("TL", 0, 0), ("CTR", 128, 128), ("BR", 255, 255)]:
        la, lo = patch_pixel_to_latlon(tif_path, py, px, local_y=ly, local_x=lx)
        print(f"  {name}: {la:.4f}, {lo:.4f}")

    age0 = age_proxy_hours(pr_a["area_km2"])
    print(f"  GT {gt_px}px ({gt_a['area_km2']:.4f} km2) Pred {pred_px}px ({pr_a['area_km2']:.4f} km2) age_proxy={age0:.2f}h")

    forcing = RealMetoceanForcingEngine().get_velocity_at_latlon(obs_lat, obs_lon, "2018-12-07")
    wf = forcing.get("wind_drift_factor", forcing.get("wind_factor", 0.035))
    u, v, uw, vw = forcing["u_current"], forcing["v_current"], forcing["u_wind"], forcing["v_wind"]

    # backward 24h
    traj_b = backward_drift_particles(
        morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
        u_current=u, v_current=v, u_wind=uw, v_wind=vw, wind_factor=wf, diffusion=0.015, rng=SEED)
    dens_b, final_b = origin_density(traj_b, grid_size=64)
    st = origin_stats(dens_b, final_b)
    peak_norm = np.array([[st["peak_xy"][0], st["peak_xy"][1]]], dtype=np.float64)
    olat, olon = batch_norm_xy_to_latlon(tif_path, py, px, peak_norm, h=H, w=W)
    origin_lat, origin_lon = float(olat[0]), float(olon[0])
    part_lat, part_lon = batch_norm_xy_to_latlon(tif_path, py, px, final_b, h=H, w=W)
    part_ll = np.stack([part_lat, part_lon], 1)
    geo_c = origin_stats_geodesic(part_ll, float(part_lat.mean()), float(part_lon.mean()))
    geo_p = origin_stats_geodesic(part_ll, origin_lat, origin_lon)
    disp = float(haversine_km(obs_lat, obs_lon, origin_lat, origin_lon))

    print("\n[2] PARTICLE UNCERTAINTY")
    print(f"  origin peak {origin_lat:.4f},{origin_lon:.4f}  obs->peak {disp:.2f} km")
    print(f"  centroid {geo_c['mean_lat']:.4f},{geo_c['mean_lon']:.4f}")
    print(f"  FROM CENTROID r50/r90/r95 = {geo_c['r50_km']:.2f} / {geo_c['r90_km']:.2f} / {geo_c['r95_km']:.2f} km")
    print(f"  FROM PEAK     r50/r90/r95 = {geo_p['r50_km']:.2f} / {geo_p['r90_km']:.2f} / {geo_p['r95_km']:.2f} km")
    print(f"  min/max from centroid {geo_c['min_km']:.2f}/{geo_c['max_km']:.2f} unique~{geo_c['n_unique']}")

    print("\n[1] AGE SENSITIVITY")
    age_rows = []
    for age_h in [6, 8, 10, 12, 18, 24]:
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PART, n_steps=int(age_h), dt_hours=1.0,
            u_current=u, v_current=v, u_wind=uw, v_wind=vw, wind_factor=wf, diffusion=0.015, rng=SEED+int(age_h))
        dens, fin = origin_density(traj, grid_size=64)
        st2 = origin_stats(dens, fin)
        pn = np.array([[st2["peak_xy"][0], st2["peak_xy"][1]]])
        plat, plon = batch_norm_xy_to_latlon(tif_path, py, px, pn, h=H, w=W)
        flat, flon = batch_norm_xy_to_latlon(tif_path, py, px, fin, h=H, w=W)
        dpk = float(haversine_km(obs_lat, obs_lon, float(plat[0]), float(plon[0])))
        dct = float(haversine_km(obs_lat, obs_lon, float(flat.mean()), float(flon.mean())))
        age_rows.append({"age_h": age_h, "disp_peak_km": dpk, "disp_centroid_km": dct,
                         "peak_lat": float(plat[0]), "peak_lon": float(plon[0]),
                         "cent_lat": float(flat.mean()), "cent_lon": float(flon.mean())})
        print(f"  age={age_h:2d}h d_peak={dpk:.2f} d_cent={dct:.2f} km")
    pd.DataFrame(age_rows).to_csv(os.path.join(OUT, "age_v41.csv"), index=False)

    print("\n[3] METOCEAN +/-10%")
    met_rows = []
    for name, cf, wfc in [("baseline",1,1),("wind_+10%",1,1.1),("wind_-10%",1,0.9),
                          ("curr_+10%",1.1,1),("curr_-10%",0.9,1),("both_+10%",1.1,1.1),("both_-10%",0.9,0.9)]:
        traj = backward_drift_particles(
            morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
            u_current=u*cf, v_current=v*cf, u_wind=uw*wfc, v_wind=vw*wfc, wind_factor=wf,
            diffusion=0.015, rng=SEED+(abs(hash(name))%997))
        _, fin = origin_density(traj, grid_size=64)
        la, lo = batch_norm_xy_to_latlon(tif_path, py, px, fin, h=H, w=W)
        sh = haversine_km(part_lat, part_lon, la, lo)
        met_rows.append({"pert": name, "mean_km": float(np.mean(sh)), "median_km": float(np.median(sh)), "p95_km": float(np.percentile(sh, 95))})
        print(f"  {name:12s} mean={met_rows[-1]['mean_km']:.2f} med={met_rows[-1]['median_km']:.2f} p95={met_rows[-1]['p95_km']:.2f}")
    pd.DataFrame(met_rows).to_csv(os.path.join(OUT, "metocean_v41.csv"), index=False)

    print("\n[4] FORWARD t=0..24 (t=0 must be ~0 km from OBS)")
    traj_f = forward_drift_particles(
        morph["seed_xy"], n_particles=N_PART, n_steps=24, dt_hours=1.0,
        u_current=u, v_current=v, u_wind=uw, v_wind=vw, wind_factor=wf, diffusion=0.015, rng=SEED+1)
    fr = []
    for step in [0, 6, 12, 18, 24]:
        mxy = traj_f[:, step, :].mean(axis=0, keepdims=True)
        la, lo = batch_norm_xy_to_latlon(tif_path, py, px, mxy, h=H, w=W)
        d0 = float(haversine_km(obs_lat, obs_lon, float(la[0]), float(lo[0])))
        fr.append({"t_h": step, "lat": float(la[0]), "lon": float(lo[0]), "disp_from_obs_km": d0})
        print(f"  t=+{step:02d}h ({la[0]:.4f},{lo[0]:.4f}) |d_obs|={d0:.2f} km")
    print("  [OK] t=0 near OBS" if fr[0]["disp_from_obs_km"] < 5 else "  WARNING t=0 far from OBS")
    pd.DataFrame(fr).to_csv(os.path.join(OUT, "forward_v41.csv"), index=False)

    print("\n[5] AIS")
    lat0 = 0.5 * (obs_lat + origin_lat)
    lon0 = 0.5 * (obs_lon + origin_lon)
    df_ais = MarineCadastreAISPipeline().load_and_filter_ais(lat0-0.6, lat0+0.6, lon0-0.6, lon0+0.6)
    df_rank = score_and_rank_vessels_ntro(df_ais, origin_lat, origin_lon, slick_orient_deg=orient)
    df_rank.to_csv(os.path.join(OUT, "ranking_v41.csv"), index=False)
    top = df_rank.iloc[0]
    s_obs = float(top["ntro_attribution_score"])
    print(f"  n={len(df_ais)} top CANDIDATE={top.get('vessel_name')} score={s_obs:.4f} dist={float(top.get('dist_km',np.nan)):.2f} km")

    print("\n[6] NULL N=1000")
    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(N_NULL):
        dfr = score_and_rank_vessels_ntro(df_ais, obs_lat+rng.uniform(-0.5,0.5), obs_lon+rng.uniform(-0.5,0.5), slick_orient_deg=orient)
        nulls.append(float(dfr.iloc[0]["ntro_attribution_score"]))
    nulls = np.asarray(nulls)
    p_emp = (1.0 + np.sum(nulls >= s_obs)) / (1.0 + N_NULL)
    concl = (f"p={p_emp:.4f}: unusual vs null." if p_emp < 0.05
             else f"p={p_emp:.4f}: investigative prioritization, NOT statistical proof under this null.")
    print(f"  S_obs={s_obs:.6f} null_mean={nulls.mean():.6f} p95={np.percentile(nulls,95):.6f}")
    print(f"  {concl}")

    summary = {
        "coord_ok": True,
        "obs": {"lat": obs_lat, "lon": obs_lon},
        "origin_peak": {"lat": origin_lat, "lon": origin_lon, "disp_km": disp},
        "radii_centroid_km": {"r50": geo_c["r50_km"], "r90": geo_c["r90_km"], "r95": geo_c["r95_km"]},
        "forward_t0_km": fr[0]["disp_from_obs_km"],
        "forward_t24_km": fr[-1]["disp_from_obs_km"],
        "ais_top": {"name": str(top.get("vessel_name")), "mmsi": str(top.get("mmsi")),
                    "score": s_obs, "dist_km": float(top.get("dist_km", np.nan))},
        "null_p": float(p_emp),
        "conclusion": concl,
        "claims": {
            "vessel": "top-ranked candidate only",
            "age": "model-derived age proxy",
            "forward": "24-h forward drift projection under available forcing",
        },
    }
    with open(os.path.join(OUT, "validation_summary_v41.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].plot([r["age_h"] for r in age_rows], [r["disp_centroid_km"] for r in age_rows], "o-")
    ax[0].set_title("Age proxy vs centroid shift"); ax[0].set_xlabel("h"); ax[0].set_ylabel("km")
    ax[1].barh([m["pert"] for m in met_rows[1:]], [m["mean_km"] for m in met_rows[1:]])
    ax[1].set_title("Metocean mean shift km")
    ax[2].hist(nulls, bins=30, alpha=0.85); ax[2].axvline(s_obs, color="r"); ax[2].set_title(f"Null p={p_emp:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "panel_v41.png"), dpi=200); plt.close()
    print(f"\n[OK] {os.path.join(OUT, 'validation_summary_v41.json')}")
    print("[OK] Expect OBS lon ~ -88.37 not -91.23")

if __name__ == "__main__":
    main()
