import os, sys, json
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets.transforms import get_val_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.segmentation.unet import UNetBaseline
from src.segmentation.deeplabv3plus import DeepLabV3PlusBaseline
from src.utils.metrics import compute_segmentation_metrics

ROOT = r"D:\SIH26143_OilSpill"
SCENE = "20200224_b.tif"
IMG = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "test", "images", SCENE)
MSK = os.path.join(ROOT, "data", "raw", "gulf_mexico", "extracted", "test", "masks", SCENE)
OUT_DIR = os.path.join(ROOT, "results", "figures", "E7_failure_20200224_b")
OUT_JSON = os.path.join(ROOT, "results", "metrics", "E7_failure_20200224_b.json")
OUT_FREEZE = os.path.join(ROOT, "results", "metrics", "SEGMENTATION_PROTOCOL_FROZEN.txt")

THR, DB_MIN, DB_MAX, PATCH, STRIDE = 0.5, -35.0, 5.0, 256, 128

def load_ck(kind, path, device):
    if kind == "unet":
        m = UNetBaseline(1, 1).to(device)
    elif kind == "deeplab":
        m = DeepLabV3PlusBaseline(1, 1).to(device)
    else:
        m = PhysioGraphSpillPerception(1, 1, dropout_rate=0.1).to(device)
    st = torch.load(path, map_location=device)
    sd = st["model_state_dict"] if isinstance(st, dict) and "model_state_dict" in st else st
    m.load_state_dict(sd, strict=False)
    return m.eval()

def predict_full(model, img01, device, tf):
    h, w = img01.shape
    pred = np.zeros((h, w), np.float32)
    wgt = np.zeros((h, w), np.float32)
    ys = list(range(0, max(h - PATCH + 1, 1), STRIDE))
    xs = list(range(0, max(w - PATCH + 1, 1), STRIDE))
    if h >= PATCH and ys[-1] != h - PATCH: ys.append(h - PATCH)
    if w >= PATCH and xs[-1] != w - PATCH: xs.append(w - PATCH)
    with torch.no_grad():
        for y in ys:
            for x in xs:
                crop = img01[y:y+PATCH, x:x+PATCH]
                if crop.shape != (PATCH, PATCH):
                    continue
                ten = tf(image=np.ascontiguousarray(crop[..., None]))["image"].unsqueeze(0).float().to(device)
                pr = torch.sigmoid(model(ten)).squeeze().cpu().numpy()
                pred[y:y+PATCH, x:x+PATCH] += pr
                wgt[y:y+PATCH, x:x+PATCH] += 1.0
    wgt[wgt == 0] = 1.0
    return pred / wgt

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = get_val_transforms()

    img = cv2.imread(IMG, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(MSK, cv2.IMREAD_UNCHANGED)
    img = np.nan_to_num(img, nan=DB_MIN, posinf=DB_MAX, neginf=DB_MIN).astype(np.float32)
    img = np.clip(img, DB_MIN, DB_MAX)
    img01 = (img - DB_MIN) / (DB_MAX - DB_MIN)
    gt = (gt > 0.5).astype(np.float32)

    # SAR stats on GT oil vs sea
    oil = img[gt > 0.5]
    sea = img[gt <= 0.5]
    print("=" * 70)
    print(f" SCENE DIAGNOSTIC: {SCENE}")
    print("=" * 70)
    print(f" shape={img.shape}  GT oil px={int(gt.sum())}  oil frac={gt.mean()*100:.3f}%")
    if oil.size and sea.size:
        print(f" SAR dB oil: mean={oil.mean():.2f} std={oil.std():.2f} min={oil.min():.2f} max={oil.max():.2f}")
        print(f" SAR dB sea: mean={sea.mean():.2f} std={sea.std():.2f}")
        print(f" contrast (sea_mean - oil_mean) dB = {sea.mean()-oil.mean():.2f}")

    ckpts = [
        ("U-Net", os.path.join(ROOT, "models", "checkpoints", "E1_unet_best.pth"), "unet"),
        ("DeepLabV3+", os.path.join(ROOT, "models", "checkpoints", "E2_deeplabv3plus_best.pth"), "deeplab"),
        ("E5.2", os.path.join(ROOT, "models", "checkpoints", "perception_frozen_E5_2.pth"), "e52"),
    ]
    if not os.path.exists(ckpts[2][1]):
        ckpts[2] = ("E5.2", os.path.join(ROOT, "models", "checkpoints", "E5_2_proposed_best.pth"), "e52")

    panel = [("SAR", img01), ("GT", gt)]
    report = {"scene": SCENE, "protocol": {"thr": THR, "preprocess": "raw_[0,1]", "morph": "none"}, "models": {}}

    for name, path, kind in ckpts:
        if not os.path.exists(path):
            print(f"[SKIP] {name}")
            continue
        m = load_ck(kind, path, device)
        prob = predict_full(m, img01, device, tf)
        binary = (prob >= THR).astype(np.float32)
        met = compute_segmentation_metrics(binary, gt, threshold=0.5)
        pred_px = int(binary.sum())
        gt_px = int(gt.sum())
        # component count
        nlab, _, stats, _ = cv2.connectedComponentsWithStats((gt > 0.5).astype(np.uint8), 8)
        nlab_p, _, stats_p, _ = cv2.connectedComponentsWithStats((binary > 0.5).astype(np.uint8), 8)
        print(f"\n{name}:")
        print(f"  mIoU={met['mIoU']*100:.2f}% Dice={met['Dice_F1']*100:.2f}% P={met['Precision']*100:.2f}% R={met['Recall']*100:.2f}%")
        print(f"  Pred px={pred_px} GT px={gt_px}  GT components={nlab-1} Pred components={nlab_p-1}")
        print(f"  prob mean/max={prob.mean():.4f}/{prob.max():.4f}")
        report["models"][name] = {
            "mIoU": float(met["mIoU"]), "Dice": float(met["Dice_F1"]),
            "Precision": float(met["Precision"]), "Recall": float(met["Recall"]),
            "pred_px": pred_px, "gt_px": gt_px,
            "gt_components": int(nlab - 1), "pred_components": int(nlab_p - 1),
            "prob_mean": float(prob.mean()), "prob_max": float(prob.max()),
        }
        panel.append((f"{name} P(oil)", prob))
        panel.append((f"{name} bin", binary))
        # save overlays
        vis = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        over = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        over[binary > 0.5] = (0, 0, 255)  # pred red
        gt_edge = cv2.Canny((gt * 255).astype(np.uint8), 50, 150)
        over[gt_edge > 0] = (0, 255, 0)
        cv2.imwrite(os.path.join(OUT_DIR, f"{kind}_overlay.png"), over)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # multi-panel figure
    n = len(panel)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for i, (title, arr) in enumerate(panel):
        axes[i].imshow(arr, cmap="gray" if i < 2 else ("hot" if "P(oil)" in title else "gray"))
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"E7 Failure case: {SCENE} (high precision, low recall under-segmentation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "panel_comparison.png"), dpi=200)
    plt.close()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    freeze = f"""SEGMENTATION PROTOCOL FROZEN
preprocess=raw_[0,1]+ToTensorV2
threshold=0.50
morphology=none
champion_ckpt=perception_frozen_E5_2.pth
E5.2_val_benchmark=mIoU_83.49_Dice_78.84  (training eval protocol; oil-pos diagnostic ~78.3)
E6_oil_pos_severe_mIoU: UNet=69.97 Deeplab=54.19 E5.2=70.27
E7_scene_mean_mIoU: UNet=71.86 Deeplab=73.67 E5.2=72.32
E7_scene_median_mIoU: UNet=71.46 Deeplab=74.01 E5.2=75.13
E7_worst_scene=20200224_b.tif (shared failure; E5.2 P~99.9 R~6.7)
DO_NOT_HEADLINE_samplewise_94pct
DO_NOT_CLAIM_E52_wins_all_scene_means
NEXT=physics_AIS_validation_only_after_this_freeze
"""
    with open(OUT_FREEZE, "w", encoding="utf-8") as f:
        f.write(freeze)

    print(f"\n[✓] {OUT_DIR}")
    print(f"[✓] {OUT_JSON}")
    print(f"[✓] {OUT_FREEZE}")
    print("SEGMENTATION SIDE: freeze after this. Then physics/AIS only.")

if __name__ == "__main__":
    main()
