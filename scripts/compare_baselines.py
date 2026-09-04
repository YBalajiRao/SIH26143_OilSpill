import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison():
    root = r"D:\SIH26143_OilSpill"
    metrics_dir = os.path.join(root, "results", "metrics")
    fig_dir = os.path.join(root, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    e1_path = os.path.join(metrics_dir, "E1_unet_metrics.csv")
    e2_path = os.path.join(metrics_dir, "E2_deeplabv3plus_metrics.csv")
    e5_path = os.path.join(metrics_dir, "E5_proposed_metrics.csv")

    if not (os.path.exists(e1_path) and os.path.exists(e2_path) and os.path.exists(e5_path)):
        print("[!] Error: One or more metrics CSV files are missing.")
        return

    df_e1 = pd.read_csv(e1_path)
    df_e2 = pd.read_csv(e2_path)
    df_e5 = pd.read_csv(e5_path)

    # Take the best epoch (highest val_mIoU) for each model
    best_e1 = df_e1.loc[df_e1["val_mIoU"].idxmax()]
    best_e2 = df_e2.loc[df_e2["val_mIoU"].idxmax()]
    best_e5 = df_e5.loc[df_e5["val_mIoU"].idxmax()]

    summary_df = pd.DataFrame([
        {"Model": "E1: U-Net (ResNet-34)", "Val mIoU": best_e1["val_mIoU"], "Val Dice/F1": best_e1["val_dice"], "Precision": best_e1["val_precision"], "Recall": best_e1["val_recall"]},
        {"Model": "E2: DeepLabV3+ (ResNet-34)", "Val mIoU": best_e2["val_mIoU"], "Val Dice/F1": best_e2["val_dice"], "Precision": best_e2["val_precision"], "Recall": best_e2["val_recall"]},
        {"Model": "E5: Physio-GraphSpill (Proposed)", "Val mIoU": best_e5["val_mIoU"], "Val Dice/F1": best_e5["val_dice"], "Precision": best_e5["val_precision"], "Recall": best_e5["val_recall"]}
    ])

    print("=" * 75)
    print("           EXPERIMENT BENCHMARK COMPARISON (5 EPOCHS)")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)

    # Plot Bar Chart Comparison
    models = ["U-Net", "DeepLabV3+", "Physio-GraphSpill\n(Proposed)"]
    metrics = ["Val mIoU", "Val Dice/F1", "Precision", "Recall"]
    
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, metric in enumerate(metrics):
        vals = summary_df[metric] * 100
        rects = ax.bar(x + (idx - 1.5) * width, vals, width, label=metric, color=colors[idx])
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Gulf of Mexico SAR Oil-Spill Segmentation Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(40, 100)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig_out = os.path.join(fig_dir, "baseline_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

    print(f"\n[✓] Benchmark comparison figure saved to:\n    {fig_out}")

if __name__ == "__main__":
    generate_comparison()
