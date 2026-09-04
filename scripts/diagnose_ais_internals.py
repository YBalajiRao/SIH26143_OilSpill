import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = r"D:\SIH26143_OilSpill"

print("=" * 80)
print(" DIAGNOSTIC: AIS & MORPHOLOGY PIPELINE INTERNALS")
print("=" * 80)

# 1. vessel_ranking.py source
vr_path = os.path.join(ROOT, "src", "ais", "vessel_ranking.py")
print("\n" + "=" * 80)
print(" [1] INSPECTING: src/ais/vessel_ranking.py")
print("=" * 80)
if os.path.exists(vr_path):
    with open(vr_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print(f" [!] NOT FOUND: {vr_path}")

# 2. marinecadastre_pipeline.py source
mp_path = os.path.join(ROOT, "src", "ais", "marinecadastre_pipeline.py")
print("\n" + "=" * 80)
print(" [2] INSPECTING: src/ais/marinecadastre_pipeline.py")
print("=" * 80)
if os.path.exists(mp_path):
    with open(mp_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print(f" [!] NOT FOUND: {mp_path}")

# 3. slick_morphology.py source
sm_path = os.path.join(ROOT, "src", "utils", "slick_morphology.py")
print("\n" + "=" * 80)
print(" [3] INSPECTING: src/utils/slick_morphology.py")
print("=" * 80)
if os.path.exists(sm_path):
    with open(sm_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print(f" [!] NOT FOUND: {sm_path}")

# 4. AIS raw CSV schema inspection
ais_dir = os.path.join(ROOT, "data", "raw", "ais")
print("\n" + "=" * 80)
print(" [4] INSPECTING: data/raw/ais CSV schema")
print("=" * 80)
csvs = sorted(glob.glob(os.path.join(ais_dir, "*.csv")))
print(f" Found {len(csvs)} CSV files in {ais_dir}")
for c in csvs:
    print(f"\n --- File: {os.path.basename(c)} ---")
    try:
        df_head = pd.read_csv(c, nrows=3)
        print(f"  Columns ({len(df_head.columns)}): {list(df_head.columns)}")
        print(df_head.to_string())
    except Exception as e:
        print(f"  Error reading {c}: {e}")

# 5. Live Pipeline Execution & DataFrame Schema Trace
print("\n" + "=" * 80)
print(" [5] LIVE TRACE: MarineCadastreAISPipeline & score_and_rank_vessels_ntro")
print("=" * 80)
try:
    from src.ais.marinecadastre_pipeline import MarineCadastreAISPipeline
    from src.ais.vessel_ranking import score_and_rank_vessels_ntro
    
    pipe = MarineCadastreAISPipeline()
    # Search around observed/origin bounding box (Dec 7, 2018)
    lat_min, lat_max = 28.0, 29.0
    lon_min, lon_max = -89.0, -87.5
    print(f" Loading AIS bounding box: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]...")
    df_ais = pipe.load_and_filter_ais(lat_min, lat_max, lon_min, lon_max)
    print(f" [✓] Loaded {len(df_ais)} aggregated vessel records")
    print(f" AIS DataFrame Columns: {list(df_ais.columns)}")
    print("\n First 2 records of loaded AIS:")
    print(df_ais.head(2).to_string())

    if len(df_ais) > 0:
        print("\n Running ranking around observed origin (28.4712, -88.2831)...")
        df_rank = score_and_rank_vessels_ntro(df_ais, 28.4712, -88.2831, slick_orient_deg=-170.4)
        print(f" [✓] Ranking complete. Output DataFrame Columns ({len(df_rank.columns)}): {list(df_rank.columns)}")
        print("\n Top 5 Ranked Candidates:")
        cols_to_show = [c for c in ["mmsi", "vessel_name", "dist_km", "proximity_score", 
                                    "kinematic_score", "alignment_score", "temporal_score", 
                                    "gap_penalty", "ntro_attribution_score"] if c in df_rank.columns]
        print(df_rank[cols_to_show].head(5).to_string() if cols_to_show else df_rank.head(5).to_string())
        
        # Check specific score properties
        print("\n Diagnostics on Ranking Columns:")
        for score_col in ["proximity_score", "kinematic_score", "alignment_score", "temporal_score", "gap_penalty"]:
            if score_col in df_rank.columns:
                vals = df_rank[score_col].dropna().values
                print(f"  - {score_col:18s}: min={vals.min():.4f}, max={vals.max():.4f}, unique_len={len(np.unique(vals))}")
            else:
                print(f"  - {score_col:18s}: [!] COLUMN NOT IN OUTPUT DATAFRAME")

except Exception as e:
    import traceback
    print(f" [!] ERROR DURING LIVE TRACE: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print(" DIAGNOSTIC COMPLETE — Paste the entire terminal output below")
print("=" * 80)
