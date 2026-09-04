import os
import sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.uncertainty.uncertainty_map import compute_mc_uncertainty

def run_uncertainty_analysis():
    root = r"D:\SIH26143_OilSpill"
    ckpt_path = os.path.join(root, "models", "checkpoints", "E5_proposed_best.pth")
    raw_dir   = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path  = os.path.join(raw_dir, "dataframe_val_dataset_256_90.csv")
    img_dir   = os.path.join(raw_dir, "images")
    mask_dir  = os.path.join(raw_dir, "masks")

    output_fig = os.path.join(root, "results", "figures", "pixel_uncertainty_analysis.png")
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)

    print("=" * 65)
    print(" GENERATING EPISTEMIC UNCERTAINTY MAPS (MC DROPOUT T=10)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[✓] Loaded checkpoint: {ckpt_path} (Best mIoU: {checkpoint.get('val_mIoU', 0.0):.4f})")

    # Load Dataset
    val_ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=get_val_transforms())

    # Find two distinct patches containing oil
    oil_samples = []
    for idx in range(len(val_ds)):
        img_t, mask_t = val_ds[idx]
        if (mask_t == 1.0).float().mean().item() > 0.05:
            oil_samples.append((idx, img_t, mask_t))
            if len(oil_samples) == 2:
                break

    fig, axes = plt.subplots(len(oil_samples), 4, figsize=(16, 4 * len(oil_samples)))

    for row_idx, (sample_idx, img_t, mask_t) in enumerate(oil_samples):
        input_tensor = img_t.unsqueeze(0).to(device)
        
        # Run MC Sampling
        mean_prob, std_unc = compute_mc_uncertainty(model, input_tensor, num_samples=10)

        sar_np  = img_t.squeeze().numpy()
        mask_np = mask_t.squeeze().numpy()

        # Col 1: Raw SAR
        axes[row_idx, 0].imshow(sar_np, cmap="gray")
        axes[row_idx, 0].set_title(f"Patch #{sample_idx} | SAR Input", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Col 2: Ground Truth
        axes[row_idx, 1].imshow(mask_np, cmap="gray")
        axes[row_idx, 1].set_title("Ground Truth Mask", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Col 3: Predicted Mean Probability
        im3 = axes[row_idx, 2].imshow(mean_prob, cmap="plasma", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title("MC Mean Probability μ(x,y)", fontsize=10)
        axes[row_idx, 2].axis("off")
        plt.colorbar(im3, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # Col 4: Epistemic Uncertainty
        im4 = axes[row_idx, 3].imshow(std_unc, cmap="inferno")
        axes[row_idx, 3].set_title("Epistemic Uncertainty σ(x,y)", fontsize=10)
        axes[row_idx, 3].axis("off")
        plt.colorbar(im4, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[✓] Uncertainty analysis map saved to:\n    {output_fig}")
    print("=" * 65)

if __name__ == "__main__":
    run_uncertainty_analysis()
