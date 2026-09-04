import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.geo_utils import haversine_km
from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
from src.ais.vessel_ranking import score_and_rank_vessels_ntro

ROOT = r"D:\SIH26143_OilSpill"
OUT = os.path.join(ROOT, "results", "physics_ais_validation")
os.makedirs(OUT, exist_ok=True)

# Frozen experimental parameters
OBS_LAT, OBS_LON = 28.3987, -88.3660
ORIGIN_LAT, ORIGIN_LON = 28.4712, -88.2831
SLICK_ORIENT_DEG = 149.32
SAR_ACQUISITION_TIME = "2018-12-07 12:00:00"
AGE_PROXY_HOURS = 7.72

def main():
    print("=" * 85)
    print(" PHYSIO-GRAPHSPILL — AIS SPATIOTEMPORAL CANDIDATE AUDIT")
    print("=" * 85)

    t_obs = pd.to_datetime(SAR_ACQUISITION_TIME)
    t_release = t_obs - pd.Timedelta(hours=float(AGE_PROXY_HOURS))
    print(f"[+] SAR Observation Timestamp:      {t_obs}")
    print(f"[+] Model-Derived Age Proxy:       {AGE_PROXY_HOURS:.2f} hours")
    print(f"[+] Estimated Release Window Center: {t_release}")
    print(f"[+] Reconstructed Lagrangian Origin: {ORIGIN_LAT:.4f}° N, {ORIGIN_LON:.4f}° W")

    # 1. Load all trajectory pings
    lat_mid = 0.5 * (OBS_LAT + ORIGIN_LAT)
    lon_mid = 0.5 * (OBS_LON + ORIGIN_LON)
    pipeline = MarineCadastreAISPipeline()
    df_pings = pipeline.load_and_filter_ais(
        lat_min=lat_mid - 0.6, lat_max=lat_mid + 0.6,
        lon_min=lon_mid - 0.6, lon_max=lon_mid + 0.6
    )

    # 2. Run baseline ranking
    df_rank = score_and_rank_vessels_ntro(
        df_pings, ORIGIN_LAT, ORIGIN_LON,
        slick_orient_deg=SLICK_ORIENT_DEG,
        obs_datetime=SAR_ACQUISITION_TIME,
        age_proxy_hours=AGE_PROXY_HOURS,
        weights=(0.40, 0.25, 0.25, 0.10)
    )

    # 3. Detailed Spatiotemporal Audit of Top 10 Candidates
    top10_mmsis = df_rank["mmsi"].head(10).tolist()
    audit_rows = []

    print("\n" + "=" * 85)
    print(" [1] TOP 10 RANKED CANDIDATES — SPATIOTEMPORAL TRAJECTORY AUDIT")
    print("=" * 85)

    for rank_idx, mmsi in enumerate(top10_mmsis, 1):
        v_rank_row = df_rank[df_rank["mmsi"] == mmsi].iloc[0]
        v_pings = df_pings[df_pings["mmsi"] == mmsi].sort_values("base_date_time").copy()
        
        v_name = str(v_rank_row["vessel_name"])
        n_pings = len(v_pings)
        t_first = v_pings["base_date_time"].min()
        t_last = v_pings["base_date_time"].max()
        
        # Calculate distance for every ping of this vessel
        p_lats = v_pings["lat"].values
        p_lons = v_pings["lon"].values
        p_dists = haversine_km(ORIGIN_LAT, ORIGIN_LON, p_lats, p_lons)
        v_pings["dist_to_origin_km"] = p_dists
        v_pings["dt_hours_to_release"] = (v_pings["base_date_time"] - t_release).dt.total_seconds() / 3600.0

        # Global CPA
        cpa_idx = int(np.argmin(p_dists))
        cpa_dist = float(p_dists[cpa_idx])
        cpa_time = v_pings["base_date_time"].iloc[cpa_idx]
        cpa_dt_h = float(v_pings["dt_hours_to_release"].iloc[cpa_idx])
        cpa_sog = float(v_pings["sog_kn"].iloc[cpa_idx])
        cpa_cog = float(v_pings["cog_deg"].iloc[cpa_idx])

        # Release Window Subsets (±6h and ±12h)
        window_6h = v_pings[v_pings["dt_hours_to_release"].abs() <= 6.0]
        window_12h = v_pings[v_pings["dt_hours_to_release"].abs() <= 12.0]
        
        cpa_6h_dist = float(window_6h["dist_to_origin_km"].min()) if len(window_6h) > 0 else np.nan
        cpa_12h_dist = float(window_12h["dist_to_origin_km"].min()) if len(window_12h) > 0 else np.nan

        audit_rows.append({
            "Rank": rank_idx,
            "MMSI": mmsi,
            "Vessel_Name": v_name,
            "Total_Pings": n_pings,
            "First_Ping": str(t_first),
            "Last_Ping": str(t_last),
            "Global_CPA_Dist_km": cpa_dist,
            "Global_CPA_Time": str(cpa_time),
            "Global_CPA_dt_hours": cpa_dt_h,
            "Global_CPA_SOG_kn": cpa_sog,
            "Pings_in_6h_Window": len(window_6h),
            "Min_Dist_6h_Window_km": cpa_6h_dist,
            "Pings_in_12h_Window": len(window_12h),
            "Min_Dist_12h_Window_km": cpa_12h_dist,
            "S_prox": float(v_rank_row["proximity_score"]),
            "S_kin": float(v_rank_row["kinematic_score"]),
            "S_align": float(v_rank_row["alignment_score"]),
            "S_temp": float(v_rank_row["temporal_score"]),
            "Final_Score": float(v_rank_row["ntro_attribution_score"]),
        })

        print(f"Rank #{rank_idx:02d} | MMSI: {mmsi} | Vessel: {v_name[:22]:22s}")
        print(f"  - Total Pings:         {n_pings:4d} (Span: {str(t_first)[:19]} to {str(t_last)[:19]})")
        print(f"  - Global CPA:          {cpa_dist:5.2f} km at {str(cpa_time)[:19]} (Δt = {cpa_dt_h:+6.1f}h from release)")
        print(f"  - Release Window ±6h:  {len(window_6h):3d} pings | Closest Approach in 6h Window:  {cpa_6h_dist:5.2f} km" if len(window_6h) > 0 else "  - Release Window ±6h:    0 pings")
        print(f"  - Release Window ±12h: {len(window_12h):3d} pings | Closest Approach in 12h Window: {cpa_12h_dist:5.2f} km" if len(window_12h) > 0 else "  - Release Window ±12h:   0 pings")
        print(f"  - Scores: S_prox={v_rank_row['proximity_score']:.4f}, S_kin={v_rank_row['kinematic_score']:.4f}, S_align={v_rank_row['alignment_score']:.4f}, S_temp={v_rank_row['temporal_score']:.4f} -> Final={v_rank_row['ntro_attribution_score']:.6f}\n")

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(os.path.join(OUT, "ais_temporal_audit_v42.csv"), index=False)

    # 4. Contemporaneous Window Search (Vessels present within ±6h of release)
    print("=" * 85)
    print(" [2] CONTEMPORANEOUS RELEASE WINDOW AUDIT (|t - T_release| <= 6 hours)")
    print("=" * 85)
    
    # Filter all pings within ±6h of estimated release
    pings_6h = df_pings.copy()
    pings_6h["dt_hours"] = (pings_6h["base_date_time"] - t_release).dt.total_seconds() / 3600.0
    pings_6h = pings_6h[pings_6h["dt_hours"].abs() <= 6.0].copy()
    
    if len(pings_6h) > 0:
        pings_6h["dist_km"] = haversine_km(ORIGIN_LAT, ORIGIN_LON, pings_6h["lat"].values, pings_6h["lon"].values)
        contemp_summary = []
        for mmsi, grp in pings_6h.groupby("mmsi"):
            min_idx = grp["dist_km"].idxmin()
            row_min = grp.loc[min_idx]
            contemp_summary.append({
                "mmsi": mmsi,
                "vessel_name": row_min["vessel_name"],
                "min_dist_window_km": float(row_min["dist_km"]),
                "ping_time": str(row_min["base_date_time"]),
                "dt_hours": float(row_min["dt_hours"]),
                "sog_kn": float(row_min["sog_kn"]),
                "pings_in_6h": len(grp)
            })
        df_contemp = pd.DataFrame(contemp_summary).sort_values("min_dist_window_km").reset_index(drop=True)
        print(f"Found {len(df_contemp)} vessels active within ±6h of estimated release window.")
        print("\nTop 5 Closest Vessels DURING Release Window:")
        print(df_contemp.head(5).to_string())
        df_contemp.to_csv(os.path.join(OUT, "ais_contemporaneous_6h_candidates.csv"), index=False)
    else:
        print("No AIS pings found in MarineCadastre extract within ±6h of release window.")

    print("\n" + "=" * 85)
    print(" [✓] AIS TEMPORAL AUDIT COMPLETED SUCCESSFULLY")
    print(f" [✓] Audit Output CSV: {os.path.join(OUT, 'ais_temporal_audit_v42.csv')}")
    print("=" * 85)

if __name__ == "__main__":
    main()
