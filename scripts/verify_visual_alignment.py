import os
import sys
import numpy as np

# Use Agg backend for non-interactive / headless saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src package is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset

def generate_verification_plot():
    root = r"D:\SIH26143_OilSpill"
    raw_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train")
    csv_path = os.path.join(raw_dir, "dataframe_train_dataset_256_90.csv")
    img_dir = os.path.join(raw_dir, "images")
    mask_dir = os.path.join(raw_dir, "masks")
    output_fig_path = os.path.join(root, "results", "figures", "dataset_verification_samples.png")

    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)

    print("=" * 65)
    print(" GENERATING SAR <-> MASK VISUAL ALIGNMENT DIAGNOSTIC")
    print("=" * 65)

    # Instantiate dataset without stochastic augmentations for pure raw check
    dataset = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=None)

    oil_samples = []
    bg_samples = []

    print("[*] Scanning dataset for representative positive & negative patches...")
    # Scan through dataset to grab 3 distinct oil patches and 3 clean background patches
    for idx in range(len(dataset)):
        img_t, mask_t = dataset[idx]
        oil_fraction = (mask_t == 1.0).float().mean().item()

        if oil_fraction > 0.08 and len(oil_samples) < 3:
            oil_samples.append((idx, img_t, mask_t, oil_fraction))
        elif oil_fraction == 0.0 and len(bg_samples) < 3:
            bg_samples.append((idx, img_t, mask_t, oil_fraction))

        if len(oil_samples) == 3 and len(bg_samples) == 3:
            break

    selected_samples = oil_samples + bg_samples
    total_samples = len(selected_samples)

    fig, axes = plt.subplots(total_samples, 3, figsize=(12, 3.5 * total_samples))

    for row_idx, (sample_idx, img_t, mask_t, frac) in enumerate(selected_samples):
        # Format tensors: img_t is (1, 256, 256), mask_t is (1, 256, 256)
        sar_np = img_t.squeeze().numpy()
        mask_np = mask_t.squeeze().numpy()

        # Build RGB overlay: SAR grayscale base + Red highlighted mask
        sar_rgb = np.stack([sar_np, sar_np, sar_np], axis=-1)
        overlay = sar_rgb.copy()
        # Highlight mask in strong red channel
        overlay[mask_np > 0.5] = [0.95, 0.15, 0.15]
        blended = 0.65 * sar_rgb + 0.35 * overlay

        sample_type = "OIL SPILL" if frac > 0 else "SEA / LOOK-ALIKE"

        # Col 1: SAR Image
        im0 = axes[row_idx, 0].imshow(sar_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(f"Patch #{sample_idx:05d} | {sample_type}\nNormalized SAR Backscatter", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Col 2: Ground Truth Mask
        axes[row_idx, 1].imshow(mask_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title(f"Ground Truth Binary Mask\nOil Coverage: {frac * 100:.2f}%", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Col 3: Blended Overlay
        axes[row_idx, 2].imshow(blended)
        axes[row_idx, 2].set_title(f"SAR + Mask Spatial Registration\n(Red = Oil Annotation)", fontsize=10)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[✓] Diagnostic plot saved successfully to:")
    print(f"    {output_fig_path}")
    print("=" * 65)

if __name__ == "__main__":
    generate_verification_plot()
