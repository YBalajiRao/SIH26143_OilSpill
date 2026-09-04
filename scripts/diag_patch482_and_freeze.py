import os, sys
import numpy as np
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CKPT:", CKPT)

    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))  # build FiLM if lazy

    state = torch.load(CKPT, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"logged val_mIoU: {state.get('val_mIoU', 'n/a') if isinstance(state, dict) else 'n/a'}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print("  missing sample:", list(missing)[:12])
    if unexpected:
        print("  unexpected sample:", list(unexpected)[:12])
    model.eval()

    # --- Dataset with FROZEN val transforms ---
    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    assert len(ds) > 482, f"val len={len(ds)}"

    # Prove NO manual [-1,1] anywhere in this path
    sample_img, sample_mask = ds[482]
    row = ds.df.iloc[482]
    fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
    coords = str(row.get("coordinates", ""))

    print("\n" + "=" * 80)
    print(" PREPROCESS CHECK (must stay ~[0,1] after ToTensor)")
    print("=" * 80)
    print(f"CSV row: 482 | scene: {fname} | coords: {coords}")
    print(f"tensor shape: {tuple(sample_img.shape)} dtype={sample_img.dtype}")
    print(f"tensor min/max/mean: {sample_img.min().item():.4f} / {sample_img.max().item():.4f} / {sample_img.mean().item():.4f}")
    if sample_img.min().item() < -0.05:
        print("WARNING: tensor looks standardized to negative range — transforms not frozen correctly!")

    with torch.no_grad():
        # FORBIDDEN path (for contrast only):
        bad = ((sample_img - 0.5) / 0.5).unsqueeze(0).to(device)
        good = sample_img.unsqueeze(0).to(device)  # CORRECT

        logits_good = model(good)
        logits_bad = model(bad)
        prob = torch.sigmoid(logits_good).squeeze().cpu().numpy()
        prob_bad = torch.sigmoid(logits_bad).squeeze().cpu().numpy()

    gt = sample_mask.numpy()
    if gt.ndim == 3:
        gt = gt.squeeze()
    gt_bin = (gt >= 0.5).astype(np.uint8)
    pred05 = (prob >= 0.50).astype(np.uint8)
    pred035 = (prob >= 0.35).astype(np.uint8)

    gt_px = int(gt_bin.sum())
    pred_px = int(pred05.sum())
    inter = int(np.logical_and(pred05, gt_bin).sum())
    union = int(np.logical_or(pred05, gt_bin).sum())
    iou = inter / (union + 1e-8)
    dice = (2 * inter) / (gt_px + pred_px + 1e-8)
    prec = inter / (pred_px + 1e-8)
    rec = inter / (gt_px + 1e-8)

    print("\n" + "=" * 80)
    print(" PATCH #482 SEGMENTATION (RAW [0,1] PATH — AUTHORITATIVE)")
    print("=" * 80)
    print(f"GT pixels:        {gt_px}")
    print(f"Pred@0.50 pixels: {pred_px}")
    print(f"Pred@0.35 pixels: {int(pred035.sum())}")
    print(f"IoU@0.50:         {iou:.4f}")
    print(f"Dice@0.50:        {dice:.4f}")
    print(f"Precision@0.50:   {prec:.4f}")
    print(f"Recall@0.50:      {rec:.4f}")
    print(f"Prob min/max/mean:{prob.min():.4f} / {prob.max():.4f} / {prob.mean():.4f}")
    print(f"BAD [-1,1] mean_prob={prob_bad.mean():.4f} pred@0.5={(prob_bad>=0.5).sum()}  (do NOT use this path)")
    print("=" * 80)

    # Area at 10m GSD only as info
    print(f"GT area if 10m GSD:   {gt_px * 100 / 1e6:.4f} km^2")
    print(f"Pred area if 10m GSD: {pred_px * 100 / 1e6:.4f} km^2")
    print("Label as predicted vs GT separately; do not call pred area 'slick area' alone.")

    # Quick full-val reconfirm (optional but strong)
    print("\n[*] Quick full-val reconfirm (raw [0,1])...")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    acc = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
    n = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="val"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs))
            m = compute_segmentation_metrics(probs, masks)
            bs = imgs.size(0)
            for k in acc:
                acc[k] += m[k] * bs
            n += bs
    print(f"FULL VAL n={n} mIoU={acc['mIoU']/n*100:.2f}% Dice={acc['Dice_F1']/n*100:.2f}% "
          f"P={acc['Precision']/n*100:.2f}% R={acc['Recall']/n*100:.2f}%")
    print("Expected ballpark: ~83.49 / ~78.84 if path matches training.")
    print("\nNEXT: only if FULL VAL ~83.5 and Patch482 Dice is plausible (not ~1.0 full-patch),")
    print("      then threshold sweep on VAL only. Do NOT run AIS master until then.")

if __name__ == "__main__":
    main()
