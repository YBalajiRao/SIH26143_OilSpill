import torch
import numpy as np

def compute_segmentation_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Computes IoU, Dice/F1, Precision, and Recall for binary segmentation.
    Expects probabilities (0.0 to 1.0) and binary targets (0 or 1).
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Apply binarization threshold
    binary_preds = (preds >= threshold).astype(np.uint8)
    binary_targets = (targets >= 0.5).astype(np.uint8)

    intersection = np.logical_and(binary_preds, binary_targets).sum()
    union = np.logical_or(binary_preds, binary_targets).sum()
    
    pred_sum = binary_preds.sum()
    target_sum = binary_targets.sum()

    # Foreground IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Background IoU
    bg_intersection = np.logical_and(1 - binary_preds, 1 - binary_targets).sum()
    bg_union = np.logical_or(1 - binary_preds, 1 - binary_targets).sum()
    bg_iou = (bg_intersection + smooth) / (bg_union + smooth)
    
    # Mean IoU across foreground and background
    miou = (iou + bg_iou) / 2.0

    # Dice / F1
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Precision and Recall
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    return {
        "mIoU": float(miou),
        "Foreground_IoU": float(iou),
        "Dice_F1": float(dice),
        "Precision": float(precision),
        "Recall": float(recall)
    }
