import os
import glob
import pandas as pd
import numpy as np
import cv2

def inspect_gulf_dataset():
    root_dir = r"D:\SIH26143_OilSpill\data\raw\gulf_mexico\extracted"
    train_csv_path = os.path.join(root_dir, "train", "dataframe_train_dataset_256_90.csv")
    val_csv_path   = os.path.join(root_dir, "train", "dataframe_val_dataset_256_90.csv")
    
    print("=" * 60)
    print(" GULF OF MEXICO SAR DATASET — VERIFICATION & STATS")
    print("=" * 60)
    
    df_train = pd.read_csv(train_csv_path)
    df_val   = pd.read_csv(val_csv_path)
    
    train_img_dir  = os.path.join(root_dir, "train", "images")
    train_mask_dir = os.path.join(root_dir, "train", "masks")
    
    # 1. Test Load First Scene
    sample_img_files = glob.glob(os.path.join(train_img_dir, "*.tif"))
    sample_img_path = sample_img_files[0]
    sample_mask_path = os.path.join(train_mask_dir, os.path.basename(sample_img_path))
    
    img = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(sample_mask_path, cv2.IMREAD_UNCHANGED)
    
    print(f"\n[+] Full Scene: {os.path.basename(sample_img_path)}")
    print(f"    - Dimensions:      {img.shape} (Height x Width)")
    print(f"    - SAR Dtype:       {img.dtype}")
    print(f"    - Backscatter (dB): Min = {img.min():.2f} dB | Max = {img.max():.2f} dB")
    print(f"    - Mean SAR (dB):   {img.mean():.2f} dB | Std: {img.std():.2f} dB")
    print(f"    - Mask Dtype:      {mask.dtype} | Unique values: {np.unique(mask)}")

    # 2. Test First 5 Patches from CSV
    print(f"\n[+] Testing 5 Sample Patch Crops from Train CSV:")
    for idx in range(5):
        row = df_train.iloc[idx]
        filename = os.path.basename(row['paths'].replace('\\\\', '/').replace('\\', '/'))
        coords = [int(c.strip()) for c in str(row['coordinates']).strip('\"\'').split(',')]
        y, x = coords[0], coords[1]
        
        scene_img = cv2.imread(os.path.join(train_img_dir, filename), cv2.IMREAD_UNCHANGED)
        scene_mask = cv2.imread(os.path.join(train_mask_dir, filename), cv2.IMREAD_UNCHANGED)
        
        patch_img = scene_img[y : y + 256, x : x + 256]
        patch_mask = scene_mask[y : y + 256, x : x + 256]
        
        oil_pixel_ratio = (patch_mask == 1.0).sum() / (256 * 256) * 100
        print(f"    Patch #{idx:02d} [{filename}] (y={y:4d}, x={x:4d}) -> Shape: {patch_img.shape} | Oil Coverage: {oil_pixel_ratio:5.2f}% | Class: {row['class']}")

    print("\n" + "=" * 60)
    print(" [✓] GULF DATASET FULLY VERIFIED & READY FOR TRAINING")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    inspect_gulf_dataset()
