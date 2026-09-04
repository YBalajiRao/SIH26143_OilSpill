"""Morphology from binary mask."""
import numpy as np
from scipy import ndimage

def mask_features(mask_hw):
    m = (mask_hw > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return {
            "area_px": 0,
            "centroid_xy": (0.5, 0.5),
            "seed_xy": np.array([[0.5, 0.5]]),
            "orientation_deg": 0.0,
            "elongation": 1.0,
        }
    ys, xs = np.where(m > 0)
    cy, cx = ys.mean(), xs.mean()
    h, w = m.shape
    # principal axis via covariance
    pts = np.stack([(xs - cx) / w, (ys - cy) / h], axis=1)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    ang = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    elong = float(np.sqrt(max(eigvals[0], 1e-12) / max(eigvals[1], 1e-12)))
    seed_xy = np.stack([xs / max(w - 1, 1), ys / max(h - 1, 1)], axis=1)
    # subsample seeds for drift
    if len(seed_xy) > 2000:
        rng = np.random.default_rng(0)
        seed_xy = seed_xy[rng.choice(len(seed_xy), 2000, replace=False)]
    return {
        "area_px": int(m.sum()),
        "centroid_xy": (float(cx / max(w - 1, 1)), float(cy / max(h - 1, 1))),
        "seed_xy": seed_xy,
        "orientation_deg": float(ang),
        "elongation": elong,
    }
