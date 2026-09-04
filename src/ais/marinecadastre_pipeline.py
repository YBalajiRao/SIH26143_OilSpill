import os
import glob
import pandas as pd
import numpy as np

class MarineCadastreAISPipeline:
    def __init__(self, ais_dir=r"D:\SIH26143_OilSpill\data\raw\ais"):
        self.ais_dir = ais_dir

    def get_available_csvs(self):
        return sorted([f for f in glob.glob(os.path.join(self.ais_dir, "*.csv")) if not f.endswith(".zst")])

    def load_and_filter_ais(self, lat_min, lat_max, lon_min, lon_max):
        csv_files = self.get_available_csvs()
        print(f"[*] Inspecting {len(csv_files)} MarineCadastre CSV files in {self.ais_dir}...")

        dfs = []
        for f in csv_files:
            fname = os.path.basename(f)
            if "backup" in fname or "MarineCadastre_AIS_Gulf_2018_12.csv" in fname:
                if len(csv_files) > 1:
                    continue

            try:
                header_preview = pd.read_csv(f, nrows=2, encoding="utf-8-sig")
                col_map = {str(c).strip().lower().replace('\ufeff', ''): c for c in header_preview.columns}

                lat_c  = next((col_map[k] for k in col_map if k in ['lat', 'latitude']), None)
                lon_c  = next((col_map[k] for k in col_map if k in ['lon', 'longitude', 'lng']), None)
                mmsi_c = next((col_map[k] for k in col_map if 'mmsi' in k), None)
                sog_c  = next((col_map[k] for k in col_map if k in ['sog', 'speed']), None)
                cog_c  = next((col_map[k] for k in col_map if k in ['cog', 'course', 'heading']), None)
                name_c = next((col_map[k] for k in col_map if 'vessel' in k or 'name' in k), None)
                time_c = next((col_map[k] for k in col_map if 'date' in k or 'time' in k), None)

                if not (lat_c and lon_c and mmsi_c):
                    continue

                use_cols = [lat_c, lon_c, mmsi_c]
                if sog_c: use_cols.append(sog_c)
                if cog_c: use_cols.append(cog_c)
                if name_c: use_cols.append(name_c)
                if time_c: use_cols.append(time_c)

                df_chunk = pd.read_csv(f, usecols=use_cols, encoding="utf-8-sig", low_memory=False)

                df_chunk[lat_c] = pd.to_numeric(df_chunk[lat_c], errors='coerce')
                df_chunk[lon_c] = pd.to_numeric(df_chunk[lon_c], errors='coerce')

                mask = (df_chunk[lat_c] >= lat_min - 0.5) & (df_chunk[lat_c] <= lat_max + 0.5) & \
                       (df_chunk[lon_c] >= lon_min - 0.5) & (df_chunk[lon_c] <= lon_max + 0.5)
                filtered = df_chunk[mask].copy()

                if len(filtered) > 0:
                    rename_dict = {mmsi_c: "mmsi", lat_c: "lat", lon_c: "lon"}
                    if sog_c: rename_dict[sog_c] = "sog_kn"
                    if cog_c: rename_dict[cog_c] = "cog_deg"
                    if name_c: rename_dict[name_c] = "vessel_name"
                    if time_c: rename_dict[time_c] = "base_date_time"

                    filtered.rename(columns=rename_dict, inplace=True)
                    dfs.append(filtered)
            except Exception as e:
                print(f"[!] Error reading {fname}: {e}")

        if not dfs:
            return pd.DataFrame(columns=["mmsi", "lat", "lon", "sog_kn", "cog_deg", "vessel_name", "base_date_time", "is_real_data"])

        combined = pd.concat(dfs, ignore_index=True)
        if "vessel_name" not in combined.columns: combined["vessel_name"] = "COMMERCIAL_VESSEL"
        if "sog_kn" not in combined.columns: combined["sog_kn"] = 6.0
        if "cog_deg" not in combined.columns: combined["cog_deg"] = np.nan
        if "base_date_time" not in combined.columns: combined["base_date_time"] = "2018-12-07 12:00:00"

        combined["vessel_name"] = combined["vessel_name"].fillna("COMMERCIAL_VESSEL").astype(str)
        combined["mmsi"] = combined["mmsi"].astype(str)
        combined["sog_kn"] = pd.to_numeric(combined["sog_kn"], errors="coerce").fillna(6.0)
        combined["cog_deg"] = pd.to_numeric(combined["cog_deg"], errors="coerce")
        combined["lat"] = pd.to_numeric(combined["lat"], errors="coerce")
        combined["lon"] = pd.to_numeric(combined["lon"], errors="coerce")
        combined["base_date_time"] = pd.to_datetime(combined["base_date_time"], errors="coerce")

        valid = combined.dropna(subset=["lat", "lon"]).copy()
        valid["is_real_data"] = 1
        n_unique_mmsi = valid["mmsi"].nunique()

        print(f"[✓] LOADED {len(valid)} AIS TRAJECTORY PINGS ACROSS {n_unique_mmsi} UNIQUE VESSELS!")
        return valid
