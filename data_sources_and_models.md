# Data Ingestion and Model Weights Download Guide

This repository contains the complete source code, evaluation suites, and interactive dashboard for the **Physio-GraphSpill** framework. Due to repository size limitations on GitHub, large-scale satellite datasets, meteorological reanalysis grids, and binary model checkpoints are stored externally. 

Follow this guide to download and reconstruct the local directory structure.

---

## 🛰️ 1. Satellite Imagery Data (Pillar 1)

The perception module is trained and evaluated on C-band Synthetic Aperture Radar (SAR) Sentinel-1 data.

* **Source**: Sentinel-1 SAR Oil Spill Dataset (Zenodo)
* **Dataset Download Link**: [https://zenodo.org/record/4642191](https://zenodo.org/record/4642191)
* **Local Path Setup**:
  Save the extracted GeoTIFF scenes (`.tif`) and corresponding binary masks into:
  `D:\SIH26143_OilSpill\data\raw\gulf_mexico\extracted\train\images\`
  `D:\SIH26143_OilSpill\data\raw\gulf_mexico\extracted\train\masks\`

---

## 🌊 2. Oceanographic & Meteorological Reanalysis (Pillar 2)

Hydrodynamic current and wind vectors are used to drive the backward and forward Lagrangian transport modeling.

* **ERA5 Wind Fields**: Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
  * **Link**: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
* **CMEMS Current Velocities**: Copernicus Marine Environment Monitoring Service (Global Ocean Physics Reanalysis).
  * **Link**: [https://marine.copernicus.eu/](https://marine.copernicus.eu/)
* **Local Path Setup**:
  The spatio-temporally interpolated NetCDF grid for the case study is located at:
  `D:\SIH26143_OilSpill\data\raw\metocean\era5_cmems_20181207.nc`

---

## ⚓ 3. Historical Automatic Identification System (AIS) Trajectories (Pillar 3)

Vessel traffic reconstruction around the release window is driven by real historical transits.

* **Source**: NOAA MarineCadastre Access AIS
* **Download Link**: [https://marinecadastre.gov/accessais/](https://marinecadastre.gov/accessais/)
* **Target Filter**: Gulf of Mexico (Zone 15 / 16), December 6–8, 2018.
* **Local Path Setup**:
  Place the filtered CSV trajectory logs into:
  `D:\SIH26143_OilSpill\data\raw\ais\`

---

## 🧠 4. Frozen Deep Learning Checkpoints

The perception module relies on the frozen E5.2 model parameters to execute live tiled inference.

* **Model Weights**: `perception_frozen_E5_2.pth` (approximately 114 MB)
* **Setup Instructions**:
  Download the weights from your release link and place the checkpoint file directly into:
  `D:\SIH26143_OilSpill\models\checkpoints\`
