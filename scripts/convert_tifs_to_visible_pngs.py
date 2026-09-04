import os
import sys
import glob
import cv2
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset

def export_visible_pngs():
    root = r"D:\SIH26143_OilSpill"
    train_csv = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_train_dataset_256_90.csv")
    train_img = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "images")
    train_msk = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "masks")
    
    out_dir = os.path.join(root, "results", "visible_pngs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(" CONVERTING 32-BIT FLOAT GeoTIFFs TO VISIBLE 8-BIT PNG IMAGES")
    print("=" * 80)

    # Instantiate dataset
    ds = GulfSARPatchDataset(train_csv, train_img, train_msk, transform=None)

    # Find 5 patches with high oil coverage
    oil_patches = []
    for idx in range(len(ds)):
        img_t, mask_t = ds[idx]
        oil_px = int((mask_t == 1.0).sum())
        if oil_px > 3000: # At least 3,000 oil pixels in the 256x256 crop
            oil_patches.append((idx, img_t, mask_t, oil_px))
            if len(oil_patches) == 5:
                break

    print(f"[✓] Selected {len(oil_patches)} patches containing large oil slicks.\n")

    for rank, (idx, img_t, mask_t, oil_px) in enumerate(oil_patches, 1):
        img_np = img_t.squeeze().numpy()   # Range [0.0, 1.0]
        mask_np = mask_t.squeeze().numpy() # Values 0.0 and 1.0

        # Scale SAR Image [0.0, 1.0] -> [0, 255] uint8
        sar_png = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Scale Mask [0.0, 1.0] -> [0, 255] uint8 (0 = Black Sea, 255 = Bright White Oil)
        mask_png = (mask_np * 255.0).astype(np.uint8)

        # Create Side-by-Side Comparison Image (SAR Image | Bright White Mask)
        combined_png = np.hstack([sar_png, mask_png])

        sar_out_path  = os.path.join(out_dir, f"patch_{idx:05d}_sar_image.png")
        mask_out_path = os.path.join(out_dir, f"patch_{idx:05d}_oil_mask.png")
        side_out_path = os.path.join(out_dir, f"patch_{idx:05d}_comparison_side_by_side.png")

        cv2.imwrite(sar_out_path, sar_png)
        cv2.imwrite(mask_out_path, mask_png)
        cv2.imwrite(side_out_path, combined_png)

        print(f"  Patch #{idx:05d} ({oil_px} Oil Pixels):")
        print(f"    - Saved Visible SAR Image -> {sar_out_path}")
        print(f"    - Saved Bright White Mask -> {mask_out_path}")
        print(f"    - Saved Side-by-Side View -> {side_out_path}")
        print("-" * 80)

    print(f"\n[✓] All PNG conversions complete. Files saved in:\n    {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    export_visible_pngs()
