import numpy as np
import torch
import cv2

def compute_advanced_segmentation_metrics(preds, targets, threshold=0.35, smooth=1e-6):
    """
    Pure NumPy evaluation metrics suite for imbalanced SAR oil spill segmentation.
    Computes mIoU, Dice/F1, Precision, Recall, Specificity, Balanced Accuracy,
    MCC (Matthews Correlation Coefficient), and Tolerance-Buffered Boundary F1
    with ZERO Scikit-Learn warnings.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    p_bin = (preds >= threshold).astype(np.uint8).ravel()
    t_bin = (targets >= 0.5).astype(np.uint8).ravel()

    tp = int(np.logical_and(p_bin == 1, t_bin == 1).sum())
    fp = int(np.logical_and(p_bin == 1, t_bin == 0).sum())
    fn = int(np.logical_and(p_bin == 0, t_bin == 1).sum())
    tn = int(np.logical_and(p_bin == 0, t_bin == 0).sum())

    fg_iou = (tp + smooth) / (tp + fp + fn + smooth)
    bg_iou = (tn + smooth) / (tn + fp + fn + smooth)
    miou = (fg_iou + bg_iou) / 2.0

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    # Native NumPy Matthews Correlation Coefficient (MCC)
    num = float(tp * tn - fp * fn)
    den = float(np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)))
    mcc = num / den if den > 0.0 else 0.0

    # Balanced Accuracy
    bal_acc = (recall + specificity) / 2.0

    # Boundary F1 Score with 2-pixel dilation tolerance
    p_img = p_bin.reshape(256, 256) if len(p_bin) == 65536 else None
    t_img = t_bin.reshape(256, 256) if len(t_bin) == 65536 else None

    if p_img is not None and t_img is not None and p_img.sum() > 0 and t_img.sum() > 0:
        p_edge = cv2.Canny((p_img * 255).astype(np.uint8), 100, 200) > 0
        t_edge = cv2.Canny((t_img * 255).astype(np.uint8), 100, 200) > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        t_dil = cv2.dilate(t_edge.astype(np.uint8), kernel) > 0
        p_dil = cv2.dilate(p_edge.astype(np.uint8), kernel) > 0

        prec_b = (np.logical_and(p_edge, t_dil)).sum() / (p_edge.sum() + smooth)
        rec_b  = (np.logical_and(t_edge, p_dil)).sum() / (t_edge.sum() + smooth)
        boundary_f1 = (2.0 * prec_b * rec_b + smooth) / (prec_b + rec_b + smooth)
    else:
        boundary_f1 = dice

    return {
        "mIoU": float(miou),
        "Foreground_IoU": float(fg_iou),
        "Dice_F1": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "Balanced_Accuracy": float(bal_acc),
        "MCC": float(mcc),
        "Boundary_F1": float(boundary_f1)
    }
