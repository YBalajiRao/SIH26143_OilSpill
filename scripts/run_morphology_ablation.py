import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
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
OUT_CSV = os.path.join(ROOT, "results", "metrics", "morphology_ablation_val.csv")

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
    
    # Read optimal threshold from step 1
    opt_thr = 0.50
    freeze_note = os.path.join(ROOT, "results", "metrics", "FROZEN_THRESHOLD.txt")
    if os.path.exists(freeze_note):
        with open(freeze_note, "r") as f:
            for line in f:
                if "optimal_threshold=" in line:
                    opt_thr = float(line.split("=")[1].strip())

    print(f"[*] Evaluating Morphological Filter Sizes at Optimal Threshold t = {opt_thr:.2f}...")

    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    state = torch.load(CKPT, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    probs_list = []
    masks_list = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="[Inference]"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            m_np = masks.numpy()
            for b in range(probs.shape[0]):
                probs_list.append(probs[b, 0])
                masks_list.append(m_np[b, 0])

    sizes = [0, 10, 20, 50]
    records = []

    print("\n" + "=" * 80)
    print(f" MORPHOLOGICAL FILTER ABLATION (Threshold t = {opt_thr:.2f})")
    print("=" * 80)
    print(f" {'Min Size (px)':>12} | {'mIoU%':>8} | {'Dice%':>8} | {'Prec%':>8} | {'Rec%':>8}")
    print("-" * 80)

    for sz in sizes:
        acc = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
        n_samples = len(probs_list)

        for prob, mask in zip(probs_list, masks_list):
            raw_bin = (prob >= opt_thr).astype(np.float32)
            clean_bin = clean_connected_components(raw_bin, min_size_px=sz)
            m = compute_segmentation_metrics(clean_bin, mask, threshold=0.5)
            for k in acc:
                acc[k] += m[k]

        res = {k: v / max(n_samples, 1) for k, v in acc.items()}
        res["min_size_px"] = sz
        records.append(res)

        print(f" {sz:12d} | {res['mIoU']*100:8.2f} | {res['Dice_F1']*100:8.2f} | {res['Precision']*100:8.2f} | {res['Recall']*100:8.2f}")

    df_morph = pd.DataFrame(records)
    df_morph.to_csv(OUT_CSV, index=False)
    print("=" * 80)
    print(f"[✓] Morphological Ablation Logged -> {OUT_CSV}\n")

if __name__ == "__main__":
    main()
