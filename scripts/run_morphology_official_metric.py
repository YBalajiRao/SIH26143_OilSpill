import os
import sys
import glob
import cv2
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
OUT_CSV = os.path.join(ROOT, "results", "metrics", "morphology_official_protocol_val.csv")

def clean_connected_components(binary_mask, min_size_px=20):
    if min_size_px <= 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    clean_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size_px:
            clean_mask[labels == i] = 1.0
    return clean_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))

    ckpt = torch.load(CKPT, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"[*] Pre-computing probabilities across N={len(ds)} validation patches...")
    probs_list = []
    masks_list = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="[Inference]"):
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            m_np = masks.numpy()
            for b in range(probs.shape[0]):
                probs_list.append(probs[b, 0])
                masks_list.append(m_np[b, 0])

    sizes = [0, 5, 10, 20, 50]
    records = []

    print("\n" + "=" * 85)
    print(" MORPHOLOGICAL COMPONENT FILTER ABLATION (OFFICIAL E5.2 PROTOCOL, t = 0.50)")
    print("=" * 85)
    print(f" {'Min Size (px)':>12} | {'Oil-Pos mIoU%':>14} | {'Oil-Pos Dice%':>14} | {'Full Set mIoU%':>14}")
    print("-" * 85)

    for sz in sizes:
        oil_pos_miou = []
        oil_pos_dice = []
        full_set_miou = []

        for prob, mask in zip(probs_list, masks_list):
            raw_bin = (prob >= 0.50).astype(np.float32)
            clean_bin = clean_connected_components(raw_bin, min_size_px=sz)
            
            # compute using original compute_segmentation_metrics API
            m = compute_segmentation_metrics(clean_bin, mask, threshold=0.5)
            full_set_miou.append(m["mIoU"])

            if (mask >= 0.5).sum() > 0:
                oil_pos_miou.append(m["mIoU"])
                oil_pos_dice.append(m["Dice_F1"])

        mean_pos_miou = float(np.mean(oil_pos_miou) * 100.0)
        mean_pos_dice = float(np.mean(oil_pos_dice) * 100.0)
        mean_full_miou = float(np.mean(full_set_miou) * 100.0)

        records.append({
            "min_size_px": sz,
            "oil_pos_mIoU_pct": mean_pos_miou,
            "oil_pos_Dice_pct": mean_pos_dice,
            "full_set_mIoU_pct": mean_full_miou
        })

        print(f" {sz:12d} | {mean_pos_miou:14.2f} | {mean_pos_dice:14.2f} | {mean_full_miou:14.2f}")

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUT_CSV, index=False)
    print("=" * 85)
    print(f"[✓] Official Morphology Ablation Logged -> {OUT_CSV}\n")

if __name__ == "__main__":
    main()
