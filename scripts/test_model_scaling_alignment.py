import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.segmentation.proposed_model import PhysioGraphSpillPerception

def test_scaling():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "perception_frozen_E5_2.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load dataset raw
    ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=None)
    
    # Target patch #482 (37,160 ground truth oil pixels)
    img_t, mask_t = ds[482]
    gt_px = int((mask_t.numpy() > 0.5).sum())

    print("=" * 75)
    print(" MODEL INPUT SCALING ALIGNMENT TEST (PATCH #00482)")
    print("=" * 75)
    print(f" Ground Truth Oil Pixels in Patch #00482: {gt_px} / 65,536 ({gt_px*100/65536:.2f}%)")

    # Scaling Mode 1: Direct [0.0, 1.0] Range
    x1 = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        p1 = torch.sigmoid(model(x1)).squeeze().cpu().numpy()

    # Scaling Mode 2: Standardized [-1.0, +1.0] Range ((x - 0.5)/0.5)
    x2 = ((img_t - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.no_grad():
        p2 = torch.sigmoid(model(x2)).squeeze().cpu().numpy()

    print(f"\n Mode 1: Raw [0.0, 1.0] Tensor Input:")
    print(f"   - Input Range: [{x1.min():.4f}, {x1.max():.4f}]")
    print(f"   - Prob Range:  [{p1.min():.4f}, {p1.max():.4f}] | Mean = {p1.mean():.4f}")
    print(f"   - Predicted Oil Pixels (>= 0.35): {(p1 >= 0.35).sum()}")

    print(f"\n Mode 2: Standardized [-1.0, +1.0] Tensor Input:")
    print(f"   - Input Range: [{x2.min():.4f}, {x2.max():.4f}]")
    print(f"   - Prob Range:  [{p2.min():.4f}, {p2.max():.4f}] | Mean = {p2.mean():.4f}")
    print(f"   - Predicted Oil Pixels (>= 0.35): {(p2 >= 0.35).sum()}")
    print("=" * 75)

if __name__ == "__main__":
    test_scaling()
