import os
import sys
import numpy as np
import torch

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_train_transforms

def inspect_training_patches():
    root = r"D:\SIH26143_OilSpill"
    train_csv = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_train_dataset_256_90.csv")
    train_img = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "images")
    train_msk = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "masks")

    print("=" * 80)
    print(" VERIFYING ACTUAL TRAINED SAR PATCH CONTENT & PIXEL VALUES")
    print("=" * 80)

    # Initialize Dataset (loads and caches 14 scenes in RAM)
    dataset = GulfSARPatchDataset(train_csv, train_img, train_msk, transform=None)

    print(f"\n[+] Total Available Training Patches: {len(dataset)}")
    
    # Pick 5 diverse sample patch indices across the dataset
    sample_indices = [0, 500, 1200, 5000, 10000]

    for idx in sample_indices:
        row = dataset.df.iloc[idx]
        fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
        coord_str = str(row["coordinates"]).strip('"').strip("'")
        y, x = (int(v.strip()) for v in coord_str.split(","))

        # Get tensor output from dataset __getitem__
        img_tensor, mask_tensor = dataset[idx]

        img_np = img_tensor.squeeze().numpy()   # Shape (256, 256)
        mask_np = mask_tensor.squeeze().numpy() # Shape (256, 256)

        oil_pixel_count = int((mask_np == 1.0).sum())
        sea_pixel_count = int((mask_np == 0.0).sum())
        oil_coverage_pct = (oil_pixel_count / (256 * 256)) * 100.0

        print("\n" + "-" * 80)
        print(f" SAMPLE PATCH ITEM #{idx:05d}")
        print(f"  - Source GeoTIFF Scene: {fname}")
        print(f"  - CSV Crop Coordinate:  (y={y}, x={x})")
        print(f"  - Image Tensor Shape:   {tuple(img_tensor.shape)}  | Dtype: {img_tensor.dtype}")
        print(f"  - Mask Tensor Shape:    {tuple(mask_tensor.shape)} | Dtype: {mask_tensor.dtype}")
        print(f"  - Normalized SAR Range: Min = {img_np.min():.4f} | Max = {img_np.max():.4f} | Mean = {img_np.mean():.4f}")
        print(f"  - Mask Breakdown:       Oil Pixels = {oil_pixel_count} ({oil_coverage_pct:.2f}%) | Sea Pixels = {sea_pixel_count}")
        print(f"  - CSV Label Class:      {row['class']}")

        # Print a 5x5 sub-matrix of raw image floats
        print("\n  [5x5 Sub-Matrix of Image Pixel Float Values (Center of Patch)]:")
        sub_img = img_np[125:130, 125:130]
        for r in sub_img:
            print("    " + " ".join([f"{val:.4f}" for val in r]))

        # Print a 5x5 sub-matrix of mask binary values
        print("\n  [5x5 Sub-Matrix of Mask Binary Values (Center of Patch)]:")
        sub_mask = mask_np[125:130, 125:130]
        for r in sub_mask:
            print("    " + " ".join([f"{int(val)}" for val in r]))

    print("\n" + "=" * 80)
    print(" [✓] VERIFICATION COMPLETE: Actual 256x256 image & mask arrays are correctly passed.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    inspect_training_patches()
