import numpy as np
import pandas as pd
from src.utils.geo_utils import haversine_km

def score_and_rank_vessels_ntro(
    df_ais, 
    origin_lat, 
    origin_lon, 
    slick_orient_deg=-170.4,
    obs_datetime="2018-12-07 12:00:00",
    age_proxy_hours=7.72,
    weights=(0.40, 0.25, 0.25, 0.10)
):
    """
    Multimodal NTRO Candidate Vessel Scoring Engine (v4.2 Trajectory-Aware CPA Matching):
    Evaluates all recorded pings per vessel and selects the Closest Point of Approach (CPA) 
    state relative to the reconstructed Lagrangian origin and estimated release time window.
    
    Score = (w_prox * S_prox) + (w_kin * S_kin) + (w_align * S_align) + (w_temp * S_temp) - P_gap
    All sub-scores are strictly bounded in [0.0, 1.0].
    """
    w_prox, w_kin, w_align, w_temp = weights
    
    if len(df_ais) == 0:
        return pd.DataFrame()

    t_obs = pd.to_datetime(obs_datetime)
    t_release = t_obs - pd.Timedelta(hours=float(age_proxy_hours))

    vessel_records = []

    # Group by MMSI to evaluate full vessel trajectories
    for mmsi, group in df_ais.groupby("mmsi"):
        v_name = str(group["vessel_name"].iloc[0])
        lats = group["lat"].values
        lons = group["lon"].values
        sogs = group["sog_kn"].values
        cogs = group["cog_deg"].values
        times = group["base_date_time"].values

        # 1. Geodesic distance vector (km)
        dists = haversine_km(origin_lat, origin_lon, lats, lons)
        
        # 2. Identify CPA ping (closest physical approach to reconstructed origin)
        cpa_idx = int(np.argmin(dists))
        d_min_km = float(dists[cpa_idx])
        cpa_lat = float(lats[cpa_idx])
        cpa_lon = float(lons[cpa_idx])
        cpa_sog = float(sogs[cpa_idx]) if not np.isnan(sogs[cpa_idx]) else 6.0
        cpa_cog = float(cogs[cpa_idx]) if not np.isnan(cogs[cpa_idx]) else np.nan
        cpa_time = times[cpa_idx]

        # 3. Sub-score computation on CPA state
        # A. Spatial Proximity (15 km exponential decay) -> [0, 1]
        s_prox = float(np.clip(np.exp(-d_min_km / 15.0), 0.0, 1.0))

        # B. Kinematic Speed Match (3-8 knots ideal discharge velocity) -> [0, 1]
        s_kin = float(np.clip(np.exp(-((cpa_sog - 5.8) ** 2) / 12.0), 0.0, 1.0))

        # C. Trajectory / Slick Axial Alignment -> [0, 1]
        if np.isnan(cpa_cog) or cpa_cog < 0 or cpa_cog > 360:
            s_align = 0.50  # Neutral prior if course unrecorded
        else:
            diff = abs((slick_orient_deg - cpa_cog) % 360.0)
            diff = min(diff, 360.0 - diff)
            delta_theta_axial = min(diff, abs(180.0 - diff))
            s_align = float(np.clip(1.0 - (delta_theta_axial / 90.0), 0.0, 1.0))

        # D. Temporal Consistency relative to release window -> [0, 1]
        if pd.isna(cpa_time):
            s_temp = 0.50
            dt_hours = np.nan
        else:
            dt_hours = abs((pd.to_datetime(cpa_time) - t_release).total_seconds()) / 3600.0
            s_temp = float(np.clip(np.exp(-(dt_hours ** 2) / (2.0 * (6.0 ** 2))), 0.0, 1.0))

        # E. AIS Gap Penalty
        ais_gap = int(group.get("ais_gap_flag", pd.Series([0])).iloc[0]) if "ais_gap_flag" in group else 0
        p_gap = 0.20 if ais_gap == 1 else 0.0

        # Convex fusion
        score = (w_prox * s_prox) + (w_kin * s_kin) + (w_align * s_align) + (w_temp * s_temp) - p_gap
        score = float(np.clip(score, 0.0001, 0.9999))

        if d_min_km < 10.0 and 3.0 <= cpa_sog <= 8.0:
            evidence = "HIGH PROXIMITY: Direct spatial overlap with reconstructed origin & discharge velocity match."
        elif d_min_km < 25.0:
            evidence = "MODERATE CANDIDATE: Trajectory intersects Lagrangian dispersion corridor."
        elif ais_gap == 1:
            evidence = "BEHAVIORAL ANOMALY: Transmission gap detected during estimated release window."
        else:
            evidence = "LOW CORRELATION: Trajectory outside primary reconstructed origin uncertainty radius."

        vessel_records.append({
            "mmsi": str(mmsi),
            "vessel_name": v_name,
            "cpa_lat": cpa_lat,
            "cpa_lon": cpa_lon,
            "dist_km": d_min_km,
            "sog_kn": cpa_sog,
            "cog_deg": cpa_cog,
            "cpa_time": str(cpa_time),
            "delta_t_hours": dt_hours,
            "pings_in_bbox": len(group),
            "proximity_score": s_prox,
            "kinematic_score": s_kin,
            "alignment_score": s_align,
            "temporal_score": s_temp,
            "gap_penalty": p_gap,
            "ntro_attribution_score": score,
            "investigation_evidence": evidence
        })

    df_out = pd.DataFrame(vessel_records)
    df_out = df_out.sort_values("ntro_attribution_score", ascending=False).reset_index(drop=True)
    df_out["rank"] = np.arange(1, len(df_out) + 1)

    return df_out
