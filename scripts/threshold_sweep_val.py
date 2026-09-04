import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.metrics import compute_segmentation_metrics

ROOT = r"D:\SIH26143_OilSpill"
CKPT = os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth")
if not os.path.exists(CKPT):
    CKPT = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")

CSV = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_val_dataset_256_90.csv")
IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "images")
MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "masks")
OUT = os.path.join(ROOT, "results", "metrics", "threshold_sweep_val.csv")
OUT_482 = os.path.join(ROOT, "results", "metrics", "threshold_sweep_patch482.csv")

def load_model(device):
    m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    with torch.no_grad():
        _ = m(torch.zeros(1, 1, 256, 256, device=device))
    state = torch.load(CKPT, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"CKPT: {CKPT}")
    print(f"logged val_mIoU: {state.get('val_mIoU', 'n/a') if isinstance(state, dict) else 'n/a'}")
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    m.eval()
    return m

def metrics_at_threshold(probs_list, masks_list, thr):
    """Same aggregation style as training: average compute_segmentation_metrics over patches."""
    acc = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
    n = 0
    # optional MCC via confusion (global) for secondary report
    tp = fp = fn = tn = 0
    for prob, mask in zip(probs_list, masks_list):
        # compute_segmentation_metrics expects probs or binary; pass binary at thr
        binary = (prob >= thr).astype(np.float32)
        m = compute_segmentation_metrics(binary, mask, threshold=0.5)
        # note: metrics fn re-thresholds at 0.5; so pass float binary 0/1 already
        # If metrics always does >=0.5 on preds, binary 0/1 is correct.
        for k in acc:
            acc[k] += m[k]
        n += 1
        p = (prob >= thr).astype(np.uint8).ravel()
        t = (mask >= 0.5).astype(np.uint8).ravel()
        tp += int(np.logical_and(p == 1, t == 1).sum())
        fp += int(np.logical_and(p == 1, t == 0).sum())
        fn += int(np.logical_and(p == 0, t == 1).sum())
        tn += int(np.logical_and(p == 0, t == 0).sum())
    out = {k: acc[k] / max(n, 1) for k in acc}
    # global MCC secondary
    num = float(tp * tn - fp * fn)
    den = float(np.sqrt(max(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn), 0.0)))
    out["MCC_global"] = (num / den) if den > 0 else 0.0
    out["n_patches"] = n
    out["threshold"] = thr
    return out

def patch482_table(model, device, thresholds):
    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    img, mask = ds[482]
    with torch.no_grad():
        prob = torch.sigmoid(model(img.unsqueeze(0).to(device))).squeeze().cpu().numpy()
    gt = mask.numpy().squeeze()
    rows = []
    for thr in thresholds:
        pred = (prob >= thr).astype(np.uint8)
        g = (gt >= 0.5).astype(np.uint8)
        inter = np.logical_and(pred, g).sum()
        union = np.logical_or(pred, g).sum()
        gt_px = g.sum()
        pr_px = pred.sum()
        iou = inter / (union + 1e-8)
        dice = 2 * inter / (gt_px + pr_px + 1e-8)
        prec = inter / (pr_px + 1e-8)
        rec = inter / (gt_px + 1e-8)
        rows.append({
            "threshold": thr, "gt_px": int(gt_px), "pred_px": int(pr_px),
            "IoU": float(iou), "Dice": float(dice),
            "Precision": float(prec), "Recall": float(rec),
            "pred_km2_10m": float(pr_px) * 0.0001,
            "gt_km2_10m": float(gt_px) * 0.0001,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_482, index=False)
    print("\nPATCH #482 @ each threshold:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[✓] {OUT_482}")
    return df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = load_model(device)

    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"Val patches: {len(ds)}")

    # sanity: first batch tensor range
    imgs0, _ = next(iter(loader))
    print(f"Batch0 min/max/mean: {imgs0.min():.4f} / {imgs0.max():.4f} / {imgs0.mean():.4f}")
    assert imgs0.min() >= -1e-3, "Input left [0,1] — stop"

    print("\n[*] Single forward pass over FULL validation (store probs)...")
    probs_list, masks_list = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="infer"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            # per-sample
            for b in range(probs.shape[0]):
                probs_list.append(probs[b, 0] if probs.ndim == 4 else probs[b])
                m = masks[b].numpy()
                masks_list.append(m.squeeze() if m.ndim == 3 else m)

    # reference @0.50 should ~match E5.2
    ref = metrics_at_threshold(probs_list, masks_list, 0.50)
    print(f"\nREFERENCE @0.50 (same metric API): "
          f"mIoU={ref['mIoU']*100:.2f}% Dice={ref['Dice_F1']*100:.2f}% "
          f"P={ref['Precision']*100:.2f}% R={ref['Recall']*100:.2f}% MCC_g={ref['MCC_global']:.4f}")
    print("Expected ~83.49 / 78.84 if definition matches training eval.")

    thresholds = [round(x, 2) for x in list(np.arange(0.20, 0.81, 0.05))]
    rows = []
    print("\n" + "=" * 95)
    print(f"{'thr':>6} {'mIoU%':>8} {'Dice%':>8} {'Prec%':>8} {'Rec%':>8} {'MCC_g':>8}")
    print("=" * 95)
    for thr in thresholds:
        r = metrics_at_threshold(probs_list, masks_list, thr)
        rows.append(r)
        print(f"{thr:6.2f} {r['mIoU']*100:8.2f} {r['Dice_F1']*100:8.2f} "
              f"{r['Precision']*100:8.2f} {r['Recall']*100:8.2f} {r['MCC_global']:8.4f}")

    df = pd.DataFrame(rows)
    # primary select: max Dice, tie-break mIoU
    best_i = df["Dice_F1"].idxmax()
    # also report max mIoU row
    best_miou_i = df["mIoU"].idxmax()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("=" * 95)
    print(f"[SELECT by max Dice] thr={df.loc[best_i,'threshold']:.2f} "
          f"Dice={df.loc[best_i,'Dice_F1']*100:.2f}% mIoU={df.loc[best_i,'mIoU']*100:.2f}%")
    print(f"[SELECT by max mIoU] thr={df.loc[best_miou_i,'threshold']:.2f} "
          f"mIoU={df.loc[best_miou_i,'mIoU']*100:.2f}% Dice={df.loc[best_miou_i,'Dice_F1']*100:.2f}%")
    print(f"[✓] Wrote {OUT}")

    # If improvement over 0.50 is tiny (<0.3 pp Dice), keep 0.50 for simplicity
    dice_50 = float(df.loc[df["threshold"] == 0.50, "Dice_F1"].values[0])
    dice_best = float(df.loc[best_i, "Dice_F1"])
    dpp = (dice_best - dice_50) * 100
    print(f"\nDelta Dice vs 0.50: {dpp:+.2f} percentage points")
    if dpp < 0.30:
        print("RECOMMEND: FREEZE threshold=0.50 (gain not meaningful).")
        freeze = 0.50
    else:
        freeze = float(df.loc[best_i, "threshold"])
        print(f"RECOMMEND: FREEZE threshold={freeze:.2f} (validation-selected).")

    # Patch 482 table
    patch482_table(model, device, thresholds)

    # write freeze note
    note = os.path.join(ROOT, "results", "metrics", "FROZEN_THRESHOLD.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write(f"frozen_threshold={freeze}\n")
        f.write(f"preprocess=raw_[0,1]+ToTensorV2\n")
        f.write(f"ckpt={CKPT}\n")
        f.write(f"ref_mIoU_at_0.50={ref['mIoU']}\n")
        f.write(f"ref_Dice_at_0.50={ref['Dice_F1']}\n")
        f.write(f"best_Dice_thr={df.loc[best_i,'threshold']}\n")
        f.write(f"best_Dice={df.loc[best_i,'Dice_F1']}\n")
        f.write(f"delta_Dice_pp_vs_0.50={dpp}\n")
    print(f"[✓] {note}")
    print("\nDO NOT run morphology/AIS master until you paste this table and we freeze thr.")

if __name__ == "__main__":
    main()
