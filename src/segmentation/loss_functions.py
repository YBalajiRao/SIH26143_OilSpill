import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1.0 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)

class BoundaryLoss(nn.Module):
    """Soft boundary loss via morphological-style gradient."""
    def __init__(self):
        super().__init__()
        # Initialize kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # register_buffer ensures these move with the model to GPU
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edge(self, x):
        # Explicitly ensure weights are on the same device as input x
        ex = F.conv2d(x, self.kx, padding=1)
        ey = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(ex * ex + ey * ey + 1e-6)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return F.l1_loss(self._edge(probs), self._edge(targets))

class ComboBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        loss = self.bce_weight * self.bce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets) + \
               self.boundary_weight * self.boundary(logits, targets)
        return loss
