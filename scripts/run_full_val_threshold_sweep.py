import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.metrics import compute_segmentation_metrics

ROOT = r"D:\SIH26143_OilSpill"
CKPT = os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth")
if not os.path.exists(CKPT):
    CKPT = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")

CSV = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_val_dataset_256_90.csv")
IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "images")
MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "masks")
OUT_CSV = os.path.join(ROOT, "results", "metrics", "threshold_sweep_full_val.csv")
TXT_FREEZE = os.path.join(ROOT, "results", "metrics", "FROZEN_THRESHOLD.txt")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {os.path.basename(CKPT)}")

    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, 256, 256, device=device))

    state = torch.load(CKPT, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    m_missing, m_unexp = model.load_state_dict(sd, strict=False)
    print(f"[✓] Checkpoint loaded (missing={len(m_missing)}, unexpected={len(m_unexp)})")
    model.eval()

    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"[*] Running single GPU forward pass over N={len(ds)} validation patches...")

    # Store all probability arrays and masks in CPU memory
    probs_list = []
    masks_list = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="[Inference]"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            m_np = masks.numpy()
            for b in range(probs.shape[0]):
                probs_list.append(probs[b, 0])
                masks_list.append(m_np[b, 0])

    thresholds = [round(t, 2) for t in np.arange(0.20, 0.81, 0.05)]
    records = []

    print("\n" + "=" * 80)
    print(" FULL VALIDATION THRESHOLD SWEEP RESULTS (N = 7,249 PATCHES)")
    print("=" * 80)
    print(f" {'thr':>6} | {'mIoU%':>8} | {'Dice%':>8} | {'Prec%':>8} | {'Rec%':>8}")
    print("-" * 80)

    for thr in thresholds:
        acc = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0}
        n_samples = len(probs_list)

        for prob, mask in zip(probs_list, masks_list):
            binary_pred = (prob >= thr).astype(np.float32)
            m = compute_segmentation_metrics(binary_pred, mask, threshold=0.5)
            for k in acc:
                acc[k] += m[k]

        res = {k: v / max(n_samples, 1) for k, v in acc.items()}
        res["threshold"] = thr
        records.append(res)

        print(f" {thr:6.2f} | {res['mIoU']*100:8.2f} | {res['Dice_F1']*100:8.2f} | {res['Precision']*100:8.2f} | {res['Recall']*100:8.2f}")

    df_sweep = pd.DataFrame(records)
    df_sweep.to_csv(OUT_CSV, index=False)

    best_row_dice = df_sweep.loc[df_sweep["Dice_F1"].idxmax()]
    best_row_miou = df_sweep.loc[df_sweep["mIoU"].idxmax()]

    # Selection logic: highest Dice/F1 score
    opt_thr = float(best_row_dice["threshold"])
    
    # If improvement over standard 0.50 is minimal (<0.3%), keep 0.50 for simplicity
    val_050 = df_sweep.loc[df_sweep["threshold"] == 0.50].iloc[0]
    if (best_row_dice["Dice_F1"] - val_050["Dice_F1"]) < 0.003:
        opt_thr = 0.50

    print("=" * 80)
    print(f" [★] Optimal Validation Threshold Selected: t_opt = {opt_thr:.2f}")
    print(f"     - Val mIoU @ t_opt: {df_sweep.loc[df_sweep['threshold']==opt_thr, 'mIoU'].values[0]*100:.2f}%")
    print(f"     - Val Dice @ t_opt: {df_sweep.loc[df_sweep['threshold']==opt_thr, 'Dice_F1'].values[0]*100:.2f}%")
    print("=" * 80 + "\n")

    with open(TXT_FREEZE, "w") as f:
        f.write(f"optimal_threshold={opt_thr:.2f}\n")
        f.write(f"val_mIoU={df_sweep.loc[df_sweep['threshold']==opt_thr, 'mIoU'].values[0]:.6f}\n")
        f.write(f"val_Dice={df_sweep.loc[df_sweep['threshold']==opt_thr, 'Dice_F1'].values[0]:.6f}\n")

    print(f"[✓] Saved full sweep log -> {OUT_CSV}")
    print(f"[✓] Saved frozen threshold note -> {TXT_FREEZE}")

if __name__ == "__main__":
    main()
