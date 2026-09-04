import os, sys, glob, json
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.utils.metrics import compute_segmentation_metrics

try:
    from src.segmentation.unet import UNetBaseline
    from src.segmentation.deeplabv3plus import DeepLabV3PlusBaseline
    HAS_BASE = True
except Exception:
    HAS_BASE = False

ROOT = r"D:\SIH26143_OilSpill"
TEST_IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "test", "images")
TEST_MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "test", "masks")
OUT_CSV = os.path.join(ROOT, "results", "metrics", "E7_scene_level_official.csv")
OUT_SUM = os.path.join(ROOT, "results", "metrics", "E7_scene_level_summary.json")
OUT_FIG = os.path.join(ROOT, "results", "figures", "E7_scene_level_bars.png")

THR = 0.50
DB_MIN, DB_MAX = -35.0, 5.0
PATCH, STRIDE = 256, 128

CKPTS = [
    ("E5.2 Physio-GraphSpill",
     os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth"),
     "e52"),
]
if HAS_BASE:
    CKPTS = [
        ("U-Net", os.path.join(ROOT, "models", "checkpoints", "E1_unet_best.pth"), "unet"),
        ("DeepLabV3+", os.path.join(ROOT, "models", "checkpoints", "E2_deeplabv3plus_best.pth"), "deeplab"),
    ] + CKPTS

def load_model(kind, path, device):
    if kind == "unet":
        m = UNetBaseline(in_channels=1, out_classes=1).to(device)
    elif kind == "deeplab":
        m = DeepLabV3PlusBaseline(in_channels=1, out_classes=1).to(device)
    else:
        m = PhysioGraphSpillPerception(in_channels=1, out_classes=1, dropout_rate=0.1).to(device)
    state = torch.load(path, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  [{kind}] {os.path.basename(path)} missing={len(miss)} unexpected={len(unexp)}")
    return m.eval()

def load_scene_pair(img_path, msk_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
    if img is None or msk is None:
        return None, None
    img = np.nan_to_num(img, nan=DB_MIN, posinf=DB_MAX, neginf=DB_MIN).astype(np.float32)
    img = np.clip(img, DB_MIN, DB_MAX)
    img = (img - DB_MIN) / (DB_MAX - DB_MIN)  # [0,1]
    msk = (msk > 0.5).astype(np.float32)
    return img, msk

def predict_full(model, img01, device, tf):
    h, w = img01.shape[:2]
    pred = np.zeros((h, w), dtype=np.float32)
    wgt = np.zeros((h, w), dtype=np.float32)
    ys = list(range(0, max(h - PATCH + 1, 1), STRIDE))
    xs = list(range(0, max(w - PATCH + 1, 1), STRIDE))
    if ys[-1] != h - PATCH and h >= PATCH:
        ys.append(h - PATCH)
    if xs[-1] != w - PATCH and w >= PATCH:
        xs.append(w - PATCH)
    if h < PATCH or w < PATCH:
        # pad
        pad = np.zeros((max(h, PATCH), max(w, PATCH)), dtype=np.float32)
        pad[:h, :w] = img01
        img01 = pad
        h2, w2 = img01.shape
        ys, xs = [0], [0]
    with torch.no_grad():
        for y in ys:
            for x in xs:
                crop = img01[y:y+PATCH, x:x+PATCH]
                if crop.shape[0] != PATCH or crop.shape[1] != PATCH:
                    continue
                aug = tf(image=np.ascontiguousarray(crop[..., None]))
                ten = aug["image"].unsqueeze(0).float().to(device)
                if ten.min() < -1e-3:
                    raise RuntimeError("Input left [0,1] — check transforms")
                pr = torch.sigmoid(model(ten)).squeeze().cpu().numpy()
                pred[y:y+PATCH, x:x+PATCH] += pr
                wgt[y:y+PATCH, x:x+PATCH] += 1.0
    wgt[wgt == 0] = 1.0
    return pred / wgt

def scene_metrics(prob, gt, thr=THR):
    binary = (prob >= thr).astype(np.float32)
    m = compute_segmentation_metrics(binary, gt, threshold=0.5)
    # also global fg on this scene
    p = (prob >= thr).astype(np.uint8).ravel()
    t = (gt >= 0.5).astype(np.uint8).ravel()
    tp = int(np.logical_and(p == 1, t == 1).sum())
    fp = int(np.logical_and(p == 1, t == 0).sum())
    fn = int(np.logical_and(p == 0, t == 1).sum())
    tn = int(np.logical_and(p == 0, t == 0).sum())
    fg_iou = tp / float(tp + fp + fn + 1e-8)
    dice = 2 * tp / float(2 * tp + fp + fn + 1e-8)
    m["scene_fg_IoU"] = fg_iou
    m["scene_Dice_global"] = dice
    m["gt_oil_px"] = int(t.sum())
    m["pred_oil_px"] = int(p.sum())
    m["gt_frac"] = float(t.mean())
    m["pred_frac"] = float(p.mean())
    return m

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Protocol: raw[0,1] thr={THR} morph=NONE stride={STRIDE}")

    img_files = sorted(glob.glob(os.path.join(TEST_IMG, "*.tif")))
    if not img_files:
        raise FileNotFoundError(TEST_IMG)
    print(f"Test scenes found: {len(img_files)}")
    for p in img_files:
        print(" ", os.path.basename(p))

    tf = get_val_transforms()
    rows = []

    for name, path, kind in CKPTS:
        if not os.path.exists(path):
            if kind == "e52":
                alt = os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth")
                path = alt if os.path.exists(alt) else path
            if not os.path.exists(path):
                print(f"[SKIP] {name}")
                continue
        print(f"\n[*] {name}")
        model = load_model(kind, path, device)

        for img_path in img_files:
            fname = os.path.basename(img_path)
            msk_path = os.path.join(TEST_MSK, fname)
            img01, gt = load_scene_pair(img_path, msk_path)
            if img01 is None:
                print(f"  [!] fail read {fname}")
                continue
            h, w = gt.shape[:2]
            prob = predict_full(model, img01, device, tf)
            prob = prob[:h, :w]
            met = scene_metrics(prob, gt, THR)
            row = {
                "model": name,
                "scene": fname,
                "height": h,
                "width": w,
                "mIoU": met["mIoU"],
                "Dice_F1": met["Dice_F1"],
                "Precision": met["Precision"],
                "Recall": met["Recall"],
                "Foreground_IoU": met.get("Foreground_IoU", met["mIoU"]),
                "scene_fg_IoU": met["scene_fg_IoU"],
                "scene_Dice_global": met["scene_Dice_global"],
                "gt_oil_px": met["gt_oil_px"],
                "pred_oil_px": met["pred_oil_px"],
                "gt_frac": met["gt_frac"],
                "pred_frac": met["pred_frac"],
            }
            rows.append(row)
            print(f"  {fname:20s} mIoU={met['mIoU']*100:5.2f}% Dice={met['Dice_F1']*100:5.2f}% "
                  f"P={met['Precision']*100:5.2f}% R={met['Recall']*100:5.2f}% "
                  f"GTpx={met['gt_oil_px']} Predpx={met['pred_oil_px']}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[✓] {OUT_CSV}")

    # Summary for E5.2 (and others)
    summary = {"protocol": {"preprocess": "raw_[0,1]", "threshold": THR, "morphology": "none"}, "models": {}}
    print("\n" + "=" * 80)
    print(" E7 SCENE-LEVEL SUMMARY (keep ALL scenes including worst)")
    print("=" * 80)
    for name in df["model"].unique():
        sub = df[df["model"] == name]
        miou = sub["mIoU"].values
        dice = sub["Dice_F1"].values
        stats = {
            "n_scenes": int(len(sub)),
            "mIoU_mean": float(miou.mean()),
            "mIoU_median": float(np.median(miou)),
            "mIoU_std": float(miou.std()),
            "mIoU_best": float(miou.max()),
            "mIoU_worst": float(miou.min()),
            "Dice_mean": float(dice.mean()),
            "Dice_median": float(np.median(dice)),
            "Dice_worst": float(dice.min()),
            "best_scene": str(sub.loc[sub["mIoU"].idxmax(), "scene"]),
            "worst_scene": str(sub.loc[sub["mIoU"].idxmin(), "scene"]),
        }
        summary["models"][name] = stats
        print(f"\n{name}:")
        print(f"  N={stats['n_scenes']}  mIoU mean={stats['mIoU_mean']*100:.2f}%  "
              f"median={stats['mIoU_median']*100:.2f}%  std={stats['mIoU_std']*100:.2f}%")
        print(f"  best={stats['mIoU_best']*100:.2f}% ({stats['best_scene']})")
        print(f"  worst={stats['mIoU_worst']*100:.2f}% ({stats['worst_scene']})")
        print(f"  Dice mean={stats['Dice_mean']*100:.2f}%  worst={stats['Dice_worst']*100:.2f}%")
        print(sub[["scene", "mIoU", "Dice_F1", "Precision", "Recall", "gt_oil_px"]].to_string(index=False))

    with open(OUT_SUM, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] {OUT_SUM}")

    # bar chart E5.2 if present
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e52 = df[df["model"].str.contains("E5.2", na=False)]
        if len(e52):
            fig, ax = plt.subplots(figsize=(10, 4.5))
            x = np.arange(len(e52))
            ax.bar(x - 0.2, e52["mIoU"].values * 100, 0.4, label="mIoU %")
            ax.bar(x + 0.2, e52["Dice_F1"].values * 100, 0.4, label="Dice %")
            ax.set_xticks(x)
            ax.set_xticklabels(e52["scene"].tolist(), rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("%")
            ax.set_title("E7 Scene-level (E5.2 | thr=0.50 | raw[0,1] | no morph)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_FIG, dpi=200)
            plt.close()
            print(f"[✓] {OUT_FIG}")
    except Exception as e:
        print("[!] fig skip", e)

    print("\nDO NOT drop worst scene. Use as failure/limitation case.")
    print("NEXT after paste: freeze E7 table; then held-out wording; AIS only after segmentation frozen.")

if __name__ == "__main__":
    main()
