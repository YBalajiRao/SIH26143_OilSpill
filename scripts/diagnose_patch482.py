import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception

def diagnose_row_482():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "perception_frozen_E5_2.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load Dataset with corrected transforms
    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())
    
    # Target row index 482 explicitly
    target_row_idx = 482
    sample_img, sample_mask = ds[target_row_idx]
    row = ds.df.iloc[target_row_idx]
    fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
    coords = [int(c.strip()) for c in str(row["coordinates"]).strip('"\'').split(",")]

    # Run forward pass
    with torch.no_grad():
        logits = model(sample_img.unsqueeze(0).to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    gt_pixels = int((sample_mask.numpy() > 0.5).sum())

    print("=" * 75)
    print(f" DIAGNOSTIC AUDIT: ROW #{target_row_idx} ({fname} at y={coords[0]}, x={coords[1]})")
    print("=" * 75)
    print(f"  Input Tensor Min:          {sample_img.min().item():.4f}")
    print(f"  Input Tensor Max:          {sample_img.max().item():.4f}")
    print(f"  Input Tensor Mean:         {sample_img.mean().item():.4f}")
    print(f"  GT Oil Pixels:             {gt_pixels} / 65,536 ({gt_pixels*100/65536:.2f}%)")
    print(f"\n  Prediction Probability Min:  {prob_map.min():.4f}")
    print(f"  Prediction Probability Max:  {prob_map.max():.4f}")
    print(f"  Prediction Probability Mean: {prob_map.mean():.4f}")
    print(f"\n  Predicted Oil Pixels across Decision Thresholds:")
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        pred_px = int((prob_map >= t).sum())
        print(f"    - Threshold t={t:.2f} -> {pred_px} predicted oil pixels ({pred_px*100/65536:.2f}%)")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    diagnose_row_482()
