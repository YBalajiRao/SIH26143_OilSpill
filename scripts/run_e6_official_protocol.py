import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_val_transforms
from src.segmentation.unet import UNetBaseline
from src.segmentation.deeplabv3plus import DeepLabV3PlusBaseline
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.metrics import compute_segmentation_metrics

ROOT = r"D:\SIH26143_OilSpill"
CSV = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_val_dataset_256_90.csv")
IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "images")
MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "train", "masks")

OUT_CSV = os.path.join(ROOT, "results", "metrics", "E6_official_protocol_stress.csv")
OUT_JSON = os.path.join(ROOT, "results", "metrics", "E6_official_protocol_summary.json")
OUT_FIG = os.path.join(ROOT, "results", "figures", "E6_official_protocol_retention.png")

# Frozen protocol settings
THR = 0.50
N_PATCHES = 1280
SEED = 20181207
LEVELS = [
    ("clean", 0.0),
    ("mild", 0.04),
    ("moderate", 0.10),
    ("severe", 0.20),
]

CKPTS = [
    ("U-Net", os.path.join(ROOT, "models", "checkpoints", "E1_unet_best.pth"), "unet"),
    ("DeepLabV3+", os.path.join(ROOT, "models", "checkpoints", "E2_deeplabv3plus_best.pth"), "deeplab"),
    ("E5.2 Physio-GraphSpill", os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth"), "e52"),
]

def load_exact_model(kind, ckpt_path, device):
    """Loads exact model class matching saved state_dict keys to guarantee 0 key drops."""
    if kind == "unet":
        m = UNetBaseline(in_channels=1, out_classes=1).to(device)
    elif kind == "deeplab":
        m = DeepLabV3PlusBaseline(in_channels=1, out_classes=1).to(device)
    else:
        m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)

    state = torch.load(ckpt_path, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  [{kind}] Loaded {os.path.basename(ckpt_path)} -> missing={len(missing)}, unexpected={len(unexpected)}")
    return m.eval()

def apply_speckle_batch(imgs, var, rng):
    if var <= 0:
        return imgs
    noise = torch.from_numpy(
        rng.normal(1.0, np.sqrt(var), size=tuple(imgs.shape)).astype(np.float32)
    ).to(imgs.device)
    return torch.clamp(imgs * noise, 0.0, 1.0)

def eval_model(model, loader, device, var, seed):
    rng = np.random.default_rng(seed)
    sum_m = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0, "Foreground_IoU": 0.0}
    n = 0
    
    sum_oil = {"mIoU": 0.0, "Dice_F1": 0.0, "Precision": 0.0, "Recall": 0.0, "Foreground_IoU": 0.0}
    n_oil = 0
    
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            imgs = apply_speckle_batch(imgs, var, rng)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()
            B = probs_np.shape[0]
            
            for b in range(B):
                p = probs_np[b, 0]
                msk = masks_np[b, 0] if masks_np[b].ndim == 3 else masks_np[b].squeeze()
                binary = (p >= THR).astype(np.float32)
                met = compute_segmentation_metrics(binary, msk, threshold=0.5)
                
                for k in sum_m:
                    if k in met:
                        sum_m[k] += met[k]
                n += 1
                
                if (msk >= 0.5).sum() > 0:
                    for k in sum_oil:
                        if k in met:
                            sum_oil[k] += met[k]
                    n_oil += 1
                    
                pb = (p >= THR).astype(np.uint8).ravel()
                tb = (msk >= 0.5).astype(np.uint8).ravel()
                tp += int(np.logical_and(pb == 1, tb == 1).sum())
                fp += int(np.logical_and(pb == 1, tb == 0).sum())
                fn += int(np.logical_and(pb == 0, tb == 1).sum())
                tn += int(np.logical_and(pb == 0, tb == 0).sum())

    out = {f"sw_{k}": sum_m[k] / max(n, 1) for k in sum_m}
    out.update({f"oil_{k}": sum_oil[k] / max(n_oil, 1) for k in sum_oil})
    out["n_patches"] = n
    out["n_oil_patches"] = n_oil
    
    fg_iou = tp / float(tp + fp + fn + 1e-8)
    bg_iou = tn / float(tn + fp + fn + 1e-8)
    out["global_mIoU"] = 0.5 * (fg_iou + bg_iou)
    out["global_fg_IoU"] = fg_iou
    out["global_Dice"] = (2.0 * tp) / float(2 * tp + fp + fn + 1e-8)
    out["global_Precision"] = tp / float(tp + fp + 1e-8)
    out["global_Recall"] = tp / float(tp + fn + 1e-8)
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Protocol: raw[0,1] thr={THR} N={N_PATCHES} seed={SEED}")

    ds = GulfSARPatchDataset(CSV, IMG, MSK, transform=get_val_transforms())
    rng = np.random.default_rng(SEED)
    n_take = min(N_PATCHES, len(ds))
    indices = np.sort(rng.choice(len(ds), size=n_take, replace=False))
    subset = Subset(ds, indices.tolist())
    loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=0)

    rows = []
    for name, path, kind in CKPTS:
        if not os.path.exists(path):
            if kind == "e52":
                alt = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
                if os.path.exists(alt):
                    path = alt
                else:
                    print(f"[SKIP] missing {path}")
                    continue
            else:
                print(f"[SKIP] missing {path}")
                continue

        print(f"\n[*] Evaluating {name}...")
        model = load_exact_model(kind, path, device)
        clean_stats = None
        
        for level_name, var in LEVELS:
            st = eval_model(model, loader, device, var, SEED + hash(level_name) % 1000)
            if level_name == "clean":
                clean_stats = st
                
            row = {
                "model": name,
                "level": level_name,
                "speckle_var": var,
                "threshold": THR,
                "n_patches": st["n_patches"],
                "n_oil_patches": st["n_oil_patches"],
                "sw_mIoU": st["sw_mIoU"],
                "sw_Dice": st["sw_Dice_F1"],
                "sw_Precision": st["sw_Precision"],
                "sw_Recall": st["sw_Recall"],
                "oil_mIoU": st["oil_mIoU"],
                "oil_Dice": st["oil_Dice_F1"],
                "oil_Precision": st["oil_Precision"],
                "oil_Recall": st["oil_Recall"],
                "global_mIoU": st["global_mIoU"],
                "global_fg_IoU": st["global_fg_IoU"],
                "global_Dice": st["global_Dice"],
            }
            rows.append(row)
            print(f"  {level_name:10s} | Oil-Pos mIoU: {st['oil_mIoU']*100:5.2f}% | Oil-Pos Dice: {st['oil_Dice_F1']*100:5.2f}% | Full mIoU: {st['sw_mIoU']*100:5.2f}%")

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[✓] Saved metric log -> {OUT_CSV}")

    # Plot retention curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in df["model"].unique():
            sub = df[df["model"] == name].set_index("level").loc[[l[0] for l in LEVELS]]
            axes[0].plot([l[0] for l in LEVELS], sub["oil_mIoU"].values * 100, marker="o", label=name)
            axes[1].plot([l[0] for l in LEVELS], sub["sw_mIoU"].values * 100, marker="o", label=name)
            
        axes[0].set_title("E6 Oil-Positive Patch mIoU % (Primary)", fontsize=11, fontweight="bold")
        axes[1].set_title("E6 Full Set Sample-Wise mIoU %", fontsize=11, fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Degradation Level")
            ax.set_ylabel("mIoU %")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=200)
        plt.close()
        print(f"[✓] Saved retention plot -> {OUT_FIG}")
    except Exception as e:
        print("[!] Plotting skip:", e)

    print("\n" + "=" * 80)
    print(" [✓] E6 OFFICIAL PROTOCOL STRESS TEST COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
