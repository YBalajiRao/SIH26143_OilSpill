import os
import sys
import glob
import pandas as pd
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.datasets.transforms import get_val_transforms
from src.utils.metrics import compute_segmentation_metrics

def run_adaptive_cross_region():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    test_img_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "test", "images")
    test_msk_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "test", "masks")
    out_csv = os.path.join(root, "results", "metrics", "E7_1_adaptive_norm_cross_region.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    img_files = sorted(glob.glob(os.path.join(test_img_dir, "*.tif")))
    val_tf = get_val_transforms()
    results = []

    print("    Evaluating Adaptive 2nd-98th Percentile Normalization on Held-Out Scenes...")

    for img_path in img_files:
        fname = os.path.basename(img_path)
        full_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        full_mask = cv2.imread(os.path.join(test_msk_dir, fname), cv2.IMREAD_UNCHANGED)
        
        if full_img is None: continue
        
        full_img = np.nan_to_num(full_img, nan=-35.0, posinf=5.0, neginf=-35.0).astype(np.float32)
        full_mask = (full_mask > 0.5).astype(np.float32)

        p2, p98 = np.percentile(full_img, 2), np.percentile(full_img, 98)
        img_norm = np.clip(full_img, p2, p98)
        img_norm = (img_norm - p2) / (p98 - p2 + 1e-8)

        h, w = full_img.shape[:2]
        patch_size = 256
        stride = 128
        pred_full = np.zeros((h, w), dtype=np.float32)
        count_full = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    crop_img = img_norm[y : y + patch_size, x : x + patch_size]
                    aug = val_tf(image=np.expand_dims(crop_img, -1))
                    tensor_in = aug["image"].unsqueeze(0).float().to(device)
                    probs = torch.sigmoid(model(tensor_in)).squeeze().cpu().numpy()
                    pred_full[y : y + patch_size, x : x + patch_size] += probs
                    count_full[y : y + patch_size, x : x + patch_size] += 1.0

        count_full[count_full == 0] = 1.0
        pred_full /= count_full
        
        metrics = compute_segmentation_metrics(pred_full, full_mask)
        print(f"      {fname} -> mIoU: {metrics['mIoU']*100:.2f}% | Dice: {metrics['Dice_F1']*100:.2f}%")
        results.append({"scene": fname, "mIoU": metrics["mIoU"], "Dice": metrics["Dice_F1"]})

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"    [✓] Mean Adaptive Test mIoU: {df['mIoU'].mean()*100:.2f}%")
    print(f"    [✓] Saved -> {out_csv}\n")

if __name__ == "__main__":
    run_adaptive_cross_region()
