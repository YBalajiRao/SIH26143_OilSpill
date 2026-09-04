import os
import glob
import cv2
import numpy as np

def inspect_folder(folder_path, folder_label, max_samples=5):
    print("=" * 75)
    print(f" INSPECTING: {folder_label}")
    print(f" Path: {folder_path}")
    print("=" * 75)

    if not os.path.exists(folder_path):
        print(f"[!] Path does not exist: {folder_path}\n")
        return

    # Gather images matching standard extensions
    exts = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    print(f"[+] Total files found: {len(all_files)}")
    if len(all_files) == 0:
        print("[!] Folder contains no image files.\n")
        return

    # Select 5 sample files
    samples = all_files[:max_samples]

    for idx, filepath in enumerate(samples, 1):
        rel_path = os.path.relpath(filepath, folder_path)
        
        # Must use cv2.IMREAD_UNCHANGED for 32-bit float GeoTIFFs
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"  Sample #{idx:02d} | {rel_path} -> [✗] ERROR: cv2.imread returned None!")
            continue

        shape = img.shape
        dtype = str(img.dtype)

        # NaN and Inf checking
        if np.issubdtype(img.dtype, np.floating):
            nan_count = int(np.isnan(img).sum())
            inf_count = int(np.isinf(img).sum())
            clean_img = np.nan_to_num(img, nan=-35.0, posinf=5.0, neginf=-35.0)
        else:
            nan_count = 0
            inf_count = 0
            clean_img = img

        min_val  = float(np.min(clean_img))
        max_val  = float(np.max(clean_img))
        mean_val = float(np.mean(clean_img))
        std_val  = float(np.std(clean_img))

        # Check unique labels for masks/annotations
        uniques = np.unique(clean_img)
        unique_str = str(uniques.tolist()) if len(uniques) <= 10 else f"{len(uniques)} unique values (Continuous)"

        print(f"  Sample #{idx:02d} | {rel_path}")
        print(f"    - Shape: {shape} | Dtype: {dtype}")
        print(f"    - Range: Min = {min_val:.2f} | Max = {max_val:.2f} | Mean = {mean_val:.2f} | Std = {std_val:.2f}")
        print(f"    - NaNs:  {nan_count} | Infs: {inf_count}")
        print(f"    - Value Distribution: {unique_str}")
        print("-" * 75)
    print()

def main():
    root = r"D:\SIH26143_OilSpill"
    downloads = r"C:\Users\Amma\Downloads"

    # Folders to inspect
    inspect_folder(os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "images"), "Gulf of Mexico Train Images")
    inspect_folder(os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "masks"),  "Gulf of Mexico Train Masks")
    inspect_folder(os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "test", "images"),  "Gulf of Mexico Test Images")
    inspect_folder(os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "test", "masks"),   "Gulf of Mexico Test Masks")
    
    inspect_folder(os.path.join(downloads, "Sentinel-1 SAR Oil Spill Detection Dataset"), "Yesterday's Downloaded Sentinel-1 Dataset")
    inspect_folder(os.path.join(downloads, "Deep-SAR Oil Spill Segmentation (Refined)"), "Yesterday's Downloaded Deep-SAR Dataset")

if __name__ == "__main__":
    main()
