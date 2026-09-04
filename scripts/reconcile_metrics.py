import os
import sys
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"[*] Pre-computing validation set inferences (N = {len(ds)})...")

    probs_list, masks_list = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="[Inference]"):
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            m_np = masks.numpy()
            for b in range(probs.shape[0]):
                probs_list.append(probs[b, 0])
                masks_list.append(m_np[b, 0])

    print("\n" + "=" * 80)
    print(" METRIC FORMULATION RECONCILIATION SUMMARY (Threshold t = 0.50)")
    print("=" * 80)

    # Method A: Standard Sample-Wise Mean (All Patches)
    acc_a = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
    # Method B: Sample-Wise Mean (Oil-Positive Ground Truth Patches Only)
    acc_b = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
    count_b = 0

    # Method C: Global Confusion Matrix Across Entire Dataset
    gt_total_px = 0
    pred_total_px = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    for prob, mask in zip(probs_list, masks_list):
        p_bin = (prob >= 0.50).astype(np.float32)
        m = compute_segmentation_metrics(p_bin, mask, threshold=0.5)
        
        for k in acc_a:
            acc_a[k] += m[k]

        gt_px = (mask >= 0.5).sum()
        if gt_px > 0:
            count_b += 1
            for k in acc_b:
                acc_b[k] += m[k]

        p_flat = (prob >= 0.50).astype(np.uint8).ravel()
        t_flat = (mask >= 0.5).astype(np.uint8).ravel()

        tp_total += int(np.logical_and(p_flat == 1, t_flat == 1).sum())
        fp_total += int(np.logical_and(p_flat == 1, t_flat == 0).sum())
        fn_total += int(np.logical_and(p_flat == 0, t_flat == 1).sum())
        tn_total += int(np.logical_and(p_flat == 0, t_flat == 0).sum())

    res_a = {k: v / len(probs_list) for k, v in acc_a.items()}
    res_b = {k: v / max(count_b, 1) for k, v in acc_b.items()}

    # Global Dataset-level IoU
    global_fg_iou = tp_total / float(tp_total + fp_total + fn_total + 1e-8)
    global_bg_iou = tn_total / float(tn_total + fp_total + fn_total + 1e-8)
    global_miou   = (global_fg_iou + global_bg_iou) / 2.0
    global_dice   = (2.0 * tp_total) / float(2.0 * tp_total + fp_total + fn_total + 1e-8)
    global_prec   = tp_total / float(tp_total + fp_total + 1e-8)
    global_rec    = tp_total / float(tp_total + fn_total + 1e-8)

    print(f" Method A (Sample-Wise Mean, All {len(probs_list)} Patches):")
    print(f"   -> mIoU: {res_a['mIoU']*100:.2f}% | Dice: {res_a['Dice_F1']*100:.2f}% | Prec: {res_a['Precision']*100:.2f}% | Rec: {res_a['Recall']*100:.2f}%")

    print(f"\n Method B (Sample-Wise Mean, {count_b} Oil-Positive Patches Only):")
    print(f"   -> mIoU: {res_b['mIoU']*100:.2f}% | Dice: {res_b['Dice_F1']*100:.2f}% | Prec: {res_b['Precision']*100:.2f}% | Rec: {res_b['Recall']*100:.2f}%")

    print(f"\n Method C (Global Dataset Confusion Matrix):")
    print(f"   -> mIoU: {global_miou*100:.2f}% (FG IoU: {global_fg_iou*100:.2f}%) | Dice: {global_dice*100:.2f}% | Prec: {global_prec*100:.2f}% | Rec: {global_rec*100:.2f}%")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
