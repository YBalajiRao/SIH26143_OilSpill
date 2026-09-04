import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetBaseline(nn.Module):
    """
    U-Net Baseline architecture using ResNet-34 backbone for SAR oil-spill segmentation.
    Configured for 1-channel input (VV polarization backscatter).
    """
    def __init__(self, in_channels=1, out_classes=1, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation=None  # Returns raw logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.model(x)
