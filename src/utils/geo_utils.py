import os
import numpy as np
import cv2

# Gulf fallback bbox (only if GeoTIFF has no geotransform)
LAT_TOP, LAT_BOT = 29.2, 27.8
LON_LEFT, LON_RIGHT = -89.8, -87.2  # west → east (LON_RIGHT > LON_LEFT algebraically)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def get_scene_hw(tif_path):
    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 2259, 4424
    return int(img.shape[0]), int(img.shape[1])

def get_exact_raster_resolution_and_area(tif_path, mask_pixel_count):
    pixel_width_m = 10.0
    pixel_height_m = 10.0
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            res = src.res
            if abs(res[0]) > 0:
                pixel_width_m = float(abs(res[0]))
            if abs(res[1]) > 0:
                pixel_height_m = float(abs(res[1]))
    except Exception:
        pass
    area_m2 = float(mask_pixel_count) * pixel_width_m * pixel_height_m
    return {
        "pixel_width_m": float(pixel_width_m),
        "pixel_height_m": float(pixel_height_m),
        "pixel_area_m2": float(pixel_width_m * pixel_height_m),
        "area_m2": float(area_m2),
        "area_km2": float(area_m2 / 1000000.0),
        "area_hectares": float(area_m2 / 10000.0),
    }

def patch_pixel_to_latlon(fname_tif, patch_y, patch_x, patch_size=256, local_y=128, local_x=128):
    """Single point. CORRECT lon: LON_LEFT + frac * (LON_RIGHT - LON_LEFT)."""
    h, w = get_scene_hw(fname_tif)
    full_y = float(patch_y) + float(local_y)
    full_x = float(patch_x) + float(local_x)
    lat = LAT_TOP - (full_y / float(h)) * (LAT_TOP - LAT_BOT)
    lon = LON_LEFT + (full_x / float(w)) * (LON_RIGHT - LON_LEFT)  # FIXED
    return float(lat), float(lon)

def batch_norm_xy_to_latlon(tif_path, patch_y, patch_x, norm_xy, h=None, w=None):
    """
    norm_xy: (N,2) with columns [x_norm, y_norm] in [0,1] within 256 patch
    returns lats (N,), lons (N,)
    """
    if h is None or w is None:
        h, w = get_scene_hw(tif_path)
    x_norm = np.asarray(norm_xy[:, 0], dtype=np.float64)
    y_norm = np.asarray(norm_xy[:, 1], dtype=np.float64)
    local_x = np.clip(x_norm * 255.0, 0.0, 255.0)
    local_y = np.clip(y_norm * 255.0, 0.0, 255.0)
    full_y = float(patch_y) + local_y
    full_x = float(patch_x) + local_x
    lats = LAT_TOP - (full_y / float(h)) * (LAT_TOP - LAT_BOT)
    lons = LON_LEFT + (full_x / float(w)) * (LON_RIGHT - LON_LEFT)  # FIXED
    return lats.astype(np.float64), lons.astype(np.float64)

def origin_stats_geodesic(particle_latlon, ref_lat, ref_lon):
    lats = particle_latlon[:, 0]
    lons = particle_latlon[:, 1]
    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))
    d = haversine_km(ref_lat, ref_lon, lats, lons)
    d = np.sort(np.asarray(d, dtype=np.float64))
    return {
        "mean_lat": mean_lat,
        "mean_lon": mean_lon,
        "r50_km": float(np.percentile(d, 50)),
        "r90_km": float(np.percentile(d, 90)),
        "r95_km": float(np.percentile(d, 95)),
        "min_km": float(d[0]),
        "max_km": float(d[-1]),
        "n_particles": int(len(d)),
        "n_unique": int(len(np.unique(np.round(d, 6)))),
    }
