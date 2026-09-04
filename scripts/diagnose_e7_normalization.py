import os
import sys
import glob
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception

def diagnose():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    test_img_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "test", "images")
    test_msk_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "test", "masks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load test scene
    img_files = sorted(glob.glob(os.path.join(test_img_dir, "*.tif")))
    test_img_path = img_files[0]
    test_msk_path = os.path.join(test_msk_dir, os.path.basename(test_img_path))

    full_img = cv2.imread(test_img_path, cv2.IMREAD_UNCHANGED)
    full_mask = cv2.imread(test_msk_path, cv2.IMREAD_UNCHANGED)

    # Clean NaNs
    full_img = np.nan_to_num(full_img, nan=-35.0, posinf=5.0, neginf=-35.0).astype(np.float32)
    full_mask = (full_mask > 0.5).astype(np.float32)

    # 1. Normalize dB [-35, 5] -> [0, 1]
    img_db = np.clip(full_img, -35.0, 5.0)
    img_01 = (img_db - (-35.0)) / (5.0 - (-35.0))

    # Crop a 256x256 patch containing oil
    ys, xs = np.where(full_mask > 0.5)
    if len(ys) > 0:
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = max(0, cy - 128), max(0, cx - 128)
    else:
        y0, x0 = 100, 100

    patch_img_01 = img_01[y0 : y0 + 256, x0 : x0 + 256]
    patch_mask = full_mask[y0 : y0 + 256, x0 : x0 + 256]

    # Test WITHOUT Albumentations Normalization
    tensor_unnorm = torch.from_numpy(patch_img_01).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob_unnorm = torch.sigmoid(model(tensor_unnorm)).squeeze().cpu().numpy()

    # Test WITH Albumentations Normalization (matching training)
    tf = get_val_transforms()
    augmented = tf(image=np.expand_dims(patch_img_01, axis=-1), mask=patch_mask)
    tensor_norm = augmented["image"].unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob_norm = torch.sigmoid(model(tensor_norm)).squeeze().cpu().numpy()

    print("=" * 75)
    print(" DIAGNOSTIC: INPUT NORMALIZATION IMPACT ON MODEL PREDICTION")
    print("=" * 75)
    print(f" Target Patch Oil Coverage: {(patch_mask == 1.0).sum() / (256*256) * 100:.2f}%")
    print(f"\n [Without Mean/Std Normalization]:")
    print(f"   - Input Tensor Range: [{tensor_unnorm.min():.2f}, {tensor_unnorm.max():.2f}]")
    print(f"   - Predicted Prob Range: [{prob_unnorm.min():.4f}, {prob_unnorm.max():.4f}]")
    print(f"   - Predicted Oil Pixels (>= 0.5): {(prob_unnorm >= 0.5).sum()}")

    print(f"\n [WITH Albumentations Mean/Std Normalization (Correct)]:")
    print(f"   - Input Tensor Range: [{tensor_norm.min():.2f}, {tensor_norm.max():.2f}]")
    print(f"   - Predicted Prob Range: [{prob_norm.min():.4f}, {prob_norm.max():.4f}]")
    print(f"   - Predicted Oil Pixels (>= 0.5): {(prob_norm >= 0.5).sum()}")
    print("=" * 75)

if __name__ == "__main__":
    diagnose()
