import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.advanced_metrics import compute_advanced_segmentation_metrics

def clean_connected_components(binary_mask, min_size_px=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    clean_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size_px:
            clean_mask[labels == i] = 1.0
    return clean_mask

def run_full_validation_sweep():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "perception_frozen_E5_2.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"], strict=False)
    model.eval()

    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    print(f"[*] Evaluating full validation dataset (N = {len(ds)} patches)...")

    # Pre-compute probability maps for efficiency
    probs_list = []
    masks_list = []

    for idx in tqdm(range(len(ds)), desc="[*] Pre-computing Validation Inferences"):
        img_t, mask_t = ds[idx]
        with torch.no_grad():
            prob = torch.sigmoid(model(img_t.unsqueeze(0).to(device))).squeeze().cpu().numpy()
        probs_list.append(prob)
        masks_list.append(mask_t.numpy().squeeze())

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    print("\n" + "=" * 85)
    print(" EXPERIMENT A: PURE THRESHOLD SWEEP (NO MORPHOLOGICAL FILTERING)")
    print("=" * 85)
    
    for t in thresholds:
        miou_l, dice_l, mcc_l, bf1_l = [], [], [], []
        for prob, mask in zip(probs_list, masks_list):
            m = compute_advanced_segmentation_metrics(prob >= t, mask, threshold=0.5)
            miou_l.append(m["mIoU"])
            dice_l.append(m["Dice_F1"])
            mcc_l.append(m["MCC"])
            bf1_l.append(m["Boundary_F1"])
            
        print(f"  t={t:.2f} -> mIoU: {np.mean(miou_l)*100:.2f}% | Dice: {np.mean(dice_l)*100:.2f}% | MCC: {np.mean(mcc_l):.4f} | Boundary F1: {np.mean(bf1_l)*100:.2f}%")

    print("\n" + "=" * 85)
    print(" EXPERIMENT B: THRESHOLD + CONNECTED-COMPONENT FILTERING (min_size = 20px)")
    print("=" * 85)

    for t in thresholds:
        miou_l, dice_l, mcc_l, bf1_l = [], [], [], []
        for prob, mask in zip(probs_list, masks_list):
            clean_pred = clean_connected_components(prob >= t, min_size_px=20)
            m = compute_advanced_segmentation_metrics(clean_pred, mask, threshold=0.5)
            miou_l.append(m["mIoU"])
            dice_l.append(m["Dice_F1"])
            mcc_l.append(m["MCC"])
            bf1_l.append(m["Boundary_F1"])

        print(f"  t={t:.2f} + Clean20 -> mIoU: {np.mean(miou_l)*100:.2f}% | Dice: {np.mean(dice_l)*100:.2f}% | MCC: {np.mean(mcc_l):.4f} | Boundary F1: {np.mean(bf1_l)*100:.2f}%")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    run_full_validation_sweep()
