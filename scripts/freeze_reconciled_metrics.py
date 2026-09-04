import os
import sys
import pandas as pd

def freeze_metrics_summary():
    root = r"D:\SIH26143_OilSpill"
    log_dir = os.path.join(root, "results", "metrics")
    doc_dir = os.path.join(root, "docs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)

    out_file = os.path.join(log_dir, "RECONCILED_FINAL_METRICS_LEADERBOARD.csv")
    doc_file = os.path.join(doc_dir, "OFFICIAL_METRIC_REPORTING_STANDARD.txt")

    leaderboard_data = [
        {
            "Experiment_ID": "E1",
            "Model_Architecture": "U-Net Baseline (ResNet-34)",
            "Epochs": 5,
            "mIoU_pct": 81.48,
            "Foreground_Dice_pct": 76.00,
            "Precision_pct": 74.66,
            "Recall_pct": 82.39,
            "Role": "Baseline Reference"
        },
        {
            "Experiment_ID": "E2",
            "Model_Architecture": "DeepLabV3+ Baseline (ResNet-34)",
            "Epochs": 5,
            "mIoU_pct": 80.16,
            "Foreground_Dice_pct": 73.69,
            "Precision_pct": 69.86,
            "Recall_pct": 84.23,
            "Role": "Baseline Reference"
        },
        {
            "Experiment_ID": "E5",
            "Model_Architecture": "Physio-GraphSpill Prototype",
            "Epochs": 5,
            "mIoU_pct": 80.49,
            "Foreground_Dice_pct": 74.37,
            "Precision_pct": 73.84,
            "Recall_pct": 80.59,
            "Role": "Architecture Iteration"
        },
        {
            "Experiment_ID": "E5.1",
            "Model_Architecture": "Physio-GraphSpill v1.5 (Multi-Scale FiLM)",
            "Epochs": 8,
            "mIoU_pct": 82.10,
            "Foreground_Dice_pct": 76.73,
            "Precision_pct": 75.41,
            "Recall_pct": 83.23,
            "Role": "Architecture Iteration"
        },
        {
            "Experiment_ID": "E5.2",
            "Model_Architecture": "Physio-GraphSpill v2.0 (Atrous DAFM + ASPP)",
            "Epochs": 12,
            "mIoU_pct": 83.49,
            "Foreground_Dice_pct": 78.84,
            "Precision_pct": 78.47,
            "Recall_pct": 83.16,
            "Role": "E5.2 Benchmark Model"
        }
    ]

    df = pd.DataFrame(leaderboard_data)
    df.to_csv(out_file, index=False)

    reporting_doc = f"""================================================================================
SIH26143 — OFFICIAL METRIC REPORTING STANDARD & RECONCILIATION
================================================================================
Model:       Physio-GraphSpill v2.0 (E5.2)
Checkpoint:  models/checkpoints/perception_frozen_E5_2.pth

OFFICIAL BENCHMARK VALUES (Gulf of Mexico Validation Set - 7,249 Patches):
- Validation Mean IoU (mIoU)  : 83.49%
- Foreground Dice / F1 Score   : 78.84%
- Precision                   : 78.47%
- Recall                      : 83.16%

RECONCILIATION STATEMENT:
83.49% mIoU and 78.84% Dice are retained as the official E5.2 benchmark values 
for consistency with the original training evaluation protocol. Additional 
foreground-focused (Oil-Positive Patch mIoU: 78.29%) and global pixel-level 
metrics (Global Confusion mIoU: 85.39%) are reported separately to characterize 
class-imbalance effects.
================================================================================
"""

    with open(doc_file, "w", encoding="utf-8") as f:
        f.write(reporting_doc)

    print(f"[✓] Saved Reconciled Leaderboard CSV -> {out_file}")
    print(f"[✓] Saved Official Metric Reporting Standard -> {doc_file}")

if __name__ == "__main__":
    freeze_metrics_summary()
