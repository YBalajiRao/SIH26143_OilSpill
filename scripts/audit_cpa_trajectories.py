import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = r"D:\SIH26143_OilSpill"
OUT = os.path.join(ROOT, "results", "physics_ais_validation")

print("=" * 90)
print(" PHYSIO-GRAPHSPILL — CPA TRAJECTORY ALIGNMENT AUDIT (v4.3)")
print("=" * 90)

# Load raw filtered AIS pings and the computed ranking
ranking_path = os.path.join(OUT, "ais_ranking_release_window_v43.csv")
if not os.path.exists(ranking_path):
    print(f"[!] Ranking file not found: {ranking_path}")
    sys.exit(1)

df_rank = pd.read_csv(ranking_path)
print(f"[✓] Loaded ranking with {len(df_rank)} contemporaneous candidates.")

from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
pipeline = MarineCadastreAISPipeline()

# Query same spatiotemporal window to examine individual trajectory pings
# Center lat/lon of search
obs_lat, obs_lon = 28.3987, -88.3660
origin_lat, origin_lon = 28.4712, -88.2831
lat_mid = 0.5 * (obs_lat + origin_lat)
lon_mid = 0.5 * (obs_lon + origin_lon)

df_pings = pipeline.load_and_filter_ais(
    lat_min=lat_mid - 0.6, lat_max=lat_mid + 0.6,
    lon_min=lon_mid - 0.6, lon_max=lon_mid + 0.6
)

target_mmsis = ["352683000", "366939810", "636016002"] # West Capricorn, Sheila Moran, Pacific Sharav

from src.utils.geo_utils import haversine_km
t_release = pd.to_datetime("2018-12-07 04:16:57")

print("\n" + "=" * 90)
print(" INDIVIDUAL TRAJECTORY PING TRACE FOR KEY CANDIDATES")
print("=" * 90)

for mmsi in target_mmsis:
    v_rank = df_rank[df_rank["mmsi"] == int(mmsi)]
    if len(v_rank) == 0:
        v_rank = df_rank[df_rank["mmsi"] == str(mmsi)]
        
    if len(v_rank) == 0:
        print(f"[!] MMSI {mmsi} not found in ranking.")
        continue
        
    row = v_rank.iloc[0]
    print(f"\nCandidate: {row['vessel_name']} (MMSI: {mmsi}) | Rank #{row['rank']}")
    print(f"  - Summary CPA Dist: {row['cpa_dist_km']:.4f} km | SOG: {row['cpa_sog_kn']:.1f} kn | dt: {row['dt_hours_to_release']:.4f} h")
    print(f"  - Scores: Proximity={row['proximity_score']:.4f} | Kinematic={row['kinematic_score']:.4f} | Alignment={row['alignment_score']:.4f} | Temporal={row['temporal_score']:.4f}")
    
    # Trace pings
    pings = df_pings[df_pings["mmsi"] == str(mmsi)].copy()
    pings["base_date_time"] = pd.to_datetime(pings["base_date_time"])
    pings["dist_km"] = haversine_km(origin_lat, origin_lon, pings["lat"].values, pings["lon"].values)
    pings["dt_hours"] = (pings["base_date_time"] - t_release).dt.total_seconds() / 3600.0
    pings_sorted = pings.sort_values("dist_km")
    
    print("  - Closest 3 pings recorded in bounding box:")
    for idx, r in pings_sorted.head(3).iterrows():
        print(f"    Ping Time: {r['base_date_time']} | Dist: {r['dist_km']:6.3f} km | SOG: {r['sog_kn']:4.1f} kn | COG: {r['cog_deg']:5.1f}° | dt: {r['dt_hours']:+6.2f} hours")

print("\n" + "=" * 90)
print(" DIAGNOSTIC COMPLETE — VERIFY ALIGNMENT")
print("=" * 90)
