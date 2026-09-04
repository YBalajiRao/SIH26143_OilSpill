import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.slick_morphology import mask_features
from src.drift.probabilistic_drift import backward_drift_particles, origin_density
from src.drift.origin_estimation import origin_stats
from src.ais.vessel_ranking import make_mock_ais, score_vessels

ROOT = r"D:\SIH26143_OilSpill"
CKPT = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
VAL_CSV = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_val_dataset_256_90.csv")
IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "images")
MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "masks")
OUT_DIR = os.path.join(ROOT, "results", "ais_outputs")
FIG = os.path.join(ROOT, "results", "figures", "attribution_mvp_demo.png")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FIG), exist_ok=True)


def load_e52(device):
    model = PhysioGraphSpillPerception(1, 1, dropout_rate=0.1).to(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" ATTRIBUTION MVP — Physio-GraphSpill (E5.2 → drift → mock AIS)")
    print("=" * 70)

    model = load_e52(device)
    ds = GulfSARPatchDataset(VAL_CSV, IMG, MSK, transform=get_val_transforms())

    # pick first patch with enough oil
    sample_idx, img_t, mask_gt = None, None, None
    for i in range(len(ds)):
        im, m = ds[i]
        if (m > 0.5).float().mean() > 0.05:
            sample_idx, img_t, mask_gt = i, im, m
            break
    if sample_idx is None:
        raise RuntimeError("No oil patch found")

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob >= 0.5).astype(np.float32)
    gt = mask_gt.squeeze().numpy()

    feats = mask_features(pred)
    print(f"[patch #{sample_idx}] area_px={feats['area_px']} centroid={feats['centroid_xy']} "
          f"orient={feats['orientation_deg']:.1f}° elong={feats['elongation']:.2f}")

    traj = backward_drift_particles(feats["seed_xy"], n_particles=500, n_steps=24, rng=42)
    dens, final = origin_density(traj, grid_size=64)
    ostats = origin_stats(dens, final)
    print(f"[origin] peak={ostats['peak_xy']} mean={ostats['mean_xy']}")

    ais = make_mock_ais(ostats["peak_xy"], rng=42)
    ranked = score_vessels(ais, dens)
    out_csv = os.path.join(OUT_DIR, "mvp_vessel_ranking.csv")
    ranked.to_csv(out_csv, index=False)

    top1 = ranked.iloc[0]
    print("\n=== TOP-5 CANDIDATE VESSELS (mock AIS) ===")
    print(ranked[["rank", "mmsi", "attr_score", "x", "y", "sog_kn", "is_true_source"]].head(5).to_string(index=False))
    hit = int(top1["is_true_source"] == 1)
    top3_hit = int(ranked.head(3)["is_true_source"].max() == 1)
    print(f"\n[metric] Top-1 correct: {hit} | Top-3 correct: {top3_hit}")
    print(f"[✓] Ranking CSV -> {out_csv}")

    # figure
    sar = img_t.squeeze().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(sar, cmap="gray"); axes[0, 0].set_title(f"SAR patch #{sample_idx}"); axes[0, 0].axis("off")
    axes[0, 1].imshow(gt, cmap="gray"); axes[0, 1].set_title("Ground truth"); axes[0, 1].axis("off")
    axes[0, 2].imshow(prob, cmap="plasma", vmin=0, vmax=1); axes[0, 2].set_title("E5.2 P(oil)"); axes[0, 2].axis("off")

    axes[1, 0].imshow(pred, cmap="gray"); axes[1, 0].set_title("Binary mask + seeds")
    if len(feats["seed_xy"]):
        s = feats["seed_xy"]
        axes[1, 0].scatter(s[:: max(1, len(s)//80), 0] * 255, s[:: max(1, len(s)//80), 1] * 255, s=3, c="cyan", alpha=0.4)
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(dens, cmap="hot", origin="lower", extent=[0, 1, 0, 1])
    axes[1, 1].set_title("Backward-origin density")
    # plot particle end points
    axes[1, 1].scatter(final[::5, 0], final[::5, 1], s=2, c="white", alpha=0.25)
    axes[1, 1].plot(*ostats["peak_xy"], "g*", markersize=14, label="origin peak")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    axes[1, 2].imshow(dens, cmap="gray", origin="lower", extent=[0, 1, 0, 1], alpha=0.85)
    for _, r in ranked.iterrows():
        c = "lime" if r["is_true_source"] == 1 else ("red" if r["rank"] <= 3 else "yellow")
        axes[1, 2].scatter(r["x"], r["y"], c=c, s=60 if r["rank"] <= 3 else 25, zorder=3)
        if r["rank"] <= 5:
            axes[1, 2].text(r["x"], r["y"], f" #{int(r['rank'])}", color="white", fontsize=8)
    axes[1, 2].set_title("AIS candidates (green=true)")
    axes[1, 2].set_xlim(0, 1); axes[1, 2].set_ylim(0, 1)

    plt.suptitle("Physio-GraphSpill MVP: Detect → Hindcast origin → Rank vessels (mock AIS)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure -> {FIG}")
    print("=" * 70)
    print("NOTE: Mock AIS proves pipeline wiring. Replace with real AIS/ERA5/CMEMS next.")
    print("=" * 70)


if __name__ == "__main__":
    main()
