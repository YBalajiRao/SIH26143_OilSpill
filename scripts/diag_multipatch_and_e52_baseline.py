import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms, get_val_transforms_norm01
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.metrics import compute_segmentation_metrics

ROOT = r"D:\SIH26143_OilSpill"
CKPT = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
# prefer frozen if present
FROZEN = os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth")
CKPT = FROZEN if os.path.exists(FROZEN) else CKPT

CSV = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_val_dataset_256_90.csv")
IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "images")
MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "masks")
OUT = os.path.join(ROOT, "results", "metrics", "E5_2_sanity_reval.csv")
OUT_PATCH = os.path.join(ROOT, "results", "metrics", "multipatch_scaling_diag.csv")

def load_model(device):
    m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    # build FiLM if lazy
    with torch.no_grad():
        _ = m(torch.zeros(1, 1, 256, 256, device=device))
    state = torch.load(CKPT, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[ckpt] {CKPT}")
    print(f"[ckpt] val_mIoU logged={state.get('val_mIoU', 'n/a') if isinstance(state, dict) else 'n/a'}")
    print(f"[ckpt] missing_keys={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", list(missing)[:8])
    m.eval()
    return m

def eval_loader(model, loader, device, tag):
    acc = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
    n = 0
    sum_prob = 0.0
    sum_pred_px = 0.0
    sum_gt_px = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=tag):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            m = compute_segmentation_metrics(probs, masks)
            bs = imgs.size(0)
            for k in acc:
                acc[k] += m[k] * bs
            n += bs
            sum_prob += probs.mean().item() * bs
            sum_pred_px += (probs >= 0.5).float().sum().item()
            sum_gt_px += (masks >= 0.5).float().sum().item()
    out = {k: v / max(n, 1) for k, v in acc.items()}
    out["mean_prob"] = sum_prob / max(n, 1)
    out["mean_pred_px_per_patch"] = sum_pred_px / max(n, 1) / (256 * 256)
    out["mean_gt_px_per_patch"] = sum_gt_px / max(n, 1) / (256 * 256)
    out["n"] = n
    return out

def multipatch(model, device):
    # raw [0,1] CHW from dataset transform=None
    ds0 = GulfSARPatchDataset(CSV, IMG, MSK, transform=None)
    idxs = [0, 50, 100, 200, 300, 400, 482, 600, 800, 1000, 1200, 2000]
    idxs = [i for i in idxs if i < len(ds0)]
    rows = []
    print("\n" + "=" * 90)
    print(" MULTI-PATCH DIAG (same checkpoint, two input modes)")
    print("=" * 90)
    for idx in idxs:
        img, mask = ds0[idx]
        gt = int((mask.numpy() > 0.5).sum())
        x1 = img.unsqueeze(0).to(device)                    # [0,1]
        x2 = ((img - 0.5) / 0.5).unsqueeze(0).to(device)   # ~[-1,1]
        with torch.no_grad():
            p1 = torch.sigmoid(model(x1)).squeeze().cpu().numpy()
            p2 = torch.sigmoid(model(x2)).squeeze().cpu().numpy()
        r = {
            "idx": idx,
            "gt_px": gt,
            "m1_min": float(img.min()), "m1_max": float(img.max()), "m1_mean": float(img.mean()),
            "p1_min": float(p1.min()), "p1_max": float(p1.max()), "p1_mean": float(p1.mean()),
            "pred1_t035": int((p1 >= 0.35).sum()), "pred1_t050": int((p1 >= 0.5).sum()),
            "p2_min": float(p2.min()), "p2_max": float(p2.max()), "p2_mean": float(p2.mean()),
            "pred2_t035": int((p2 >= 0.35).sum()), "pred2_t050": int((p2 >= 0.5).sum()),
        }
        rows.append(r)
        print(f"Patch {idx:5d} GT={gt:5d} | "
              f"[0,1] pred@0.5={r['pred1_t050']:5d} pmean={r['p1_mean']:.4f} | "
              f"[-1,1] pred@0.5={r['pred2_t050']:5d} pmean={r['p2_mean']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATCH, index=False)
    print(f"[✓] {OUT_PATCH}")
    return df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = load_model(device)

    multipatch(model, device)

    print("\n" + "=" * 90)
    print(" FULL VAL BASELINE (NO morphology, threshold=0.5 inside compute_segmentation_metrics)")
    print("=" * 90)

    # Path A: ToTensor only [0,1]  — intended E5.2 training-style
    ds_a = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    ld_a = DataLoader(ds_a, batch_size=16, shuffle=False, num_workers=0)
    res_a = eval_loader(model, ld_a, device, "val [0,1]+ToTensor")
    print(f"A [0,1]+ToTensor     n={res_a['n']} mIoU={res_a['mIoU']*100:.2f}% Dice={res_a['Dice_F1']*100:.2f}% "
          f"P={res_a['Precision']*100:.2f}% R={res_a['Recall']*100:.2f}% mean_prob={res_a['mean_prob']:.4f} "
          f"pred_frac={res_a['mean_pred_px_per_patch']*100:.2f}% gt_frac={res_a['mean_gt_px_per_patch']*100:.2f}%")

    # Path B: Normalize max_pixel_value=1 -> ~[-1,1]
    ds_b = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms_norm01())
    ld_b = DataLoader(ds_b, batch_size=16, shuffle=False, num_workers=0)
    res_b = eval_loader(model, ld_b, device, "val Normalize[-1,1]")
    print(f"B Normalize[-1,1]    n={res_b['n']} mIoU={res_b['mIoU']*100:.2f}% Dice={res_b['Dice_F1']*100:.2f}% "
          f"P={res_b['Precision']*100:.2f}% R={res_b['Recall']*100:.2f}% mean_prob={res_b['mean_prob']:.4f} "
          f"pred_frac={res_b['mean_pred_px_per_patch']*100:.2f}% gt_frac={res_b['mean_gt_px_per_patch']*100:.2f}%")

    # Path C: transform=None raw [0,1] CHW
    ds_c = GulfSARPatchDataset(CSV, IMG, MSK, transform=None)
    ld_c = DataLoader(ds_c, batch_size=16, shuffle=False, num_workers=0)
    res_c = eval_loader(model, ld_c, device, "val raw[0,1] no tf")
    print(f"C raw[0,1] no tf     n={res_c['n']} mIoU={res_c['mIoU']*100:.2f}% Dice={res_c['Dice_F1']*100:.2f}% "
          f"P={res_c['Precision']*100:.2f}% R={res_c['Recall']*100:.2f}% mean_prob={res_c['mean_prob']:.4f} "
          f"pred_frac={res_c['mean_pred_px_per_patch']*100:.2f}% gt_frac={res_c['mean_gt_px_per_patch']*100:.2f}%")

    df = pd.DataFrame([
        {"mode": "A_ToTensor_0_1", **res_a},
        {"mode": "B_Normalize_m0.5_s0.5", **res_b},
        {"mode": "C_raw_0_1", **res_c},
    ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[✓] Wrote {OUT}")

    print("\nINTERPRETATION GUIDE:")
    print("  - If one mode ~83% mIoU / ~79% Dice → THAT is the correct inference preprocess. Freeze it.")
    print("  - If all modes << 83% or pred_frac ~100% → checkpoint/architecture load mismatch; do not tune threshold yet.")
    print("  - Do NOT use attribution/morphology until a mode reproduces E5.2 ballpark metrics.")

if __name__ == "__main__":
    main()
