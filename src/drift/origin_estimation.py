import numpy as np
from src.utils.geo_utils import haversine_km, origin_stats_geodesic

def origin_stats(dens, final_pts):
    """Legacy peak on density grid (normalized coords)."""
    pk = np.unravel_index(np.argmax(dens), dens.shape)
    # dens is (H,W) with y,x; return peak_xy as (x,y) in [0,1]
    H, W = dens.shape
    peak_xy = (float(pk[1]) / max(W - 1, 1), float(pk[0]) / max(H - 1, 1))
    return {"peak_xy": peak_xy, "peak_idx": pk}
