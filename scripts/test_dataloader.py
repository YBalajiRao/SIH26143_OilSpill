import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_train_transforms

ROOT = r"D:\SIH26143_OilSpill\data\raw\gulf_mexico\extracted"
train_csv = os.path.join(ROOT, "train", "dataframe_train_dataset_256_90.csv")
train_img = os.path.join(ROOT, "train", "images")
train_msk = os.path.join(ROOT, "train", "masks")

print("=" * 60)
print(" TESTING PYTORCH DATA LOADER ON GULF OF MEXICO DATASET")
print("=" * 60)

ds = GulfSARPatchDataset(train_csv, train_img, train_msk, transform=get_train_transforms())
loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

imgs, masks = next(iter(loader))

print("\n[✓] PyTorch Batch Successfully Loaded:")
print(f"    - Image Batch Shape: {tuple(imgs.shape)}  | Dtype: {imgs.dtype}")
print(f"    - Range: [{imgs.min():.4f}, {imgs.max():.4f}]")
print(f"    - Mask Batch Shape:  {tuple(masks.shape)} | Dtype: {masks.dtype}")
print(f"    - Unique Mask Values: {masks.unique().tolist()}")
print(f"    - Oil Pixel Fraction in Batch: {(masks == 1.0).float().mean().item() * 100:.2f}%")
print("=" * 60)
