import torch
import torch.nn as nn
import numpy as np

def compute_mc_uncertainty(model, image_tensor, num_samples=10):
    """
    Monte Carlo Dropout Uncertainty Quantification.
    Runs T stochastic forward passes with active dropout during inference.
    
    Returns:
        mean_prob: Array [H, W] - Calibrated probability prediction
        std_uncertainty: Array [H, W] - Epistemic uncertainty map (variance/std dev)
    """
    model.eval()
    
    # Force dropout layers to remain active during evaluation pass
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().squeeze().numpy())

    # Stack predictions into shape [T, H, W]
    pred_stack = np.stack(predictions, axis=0)
    
    mean_prob = np.mean(pred_stack, axis=0)
    std_uncertainty = np.std(pred_stack, axis=0)
    
    return mean_prob, std_uncertainty
