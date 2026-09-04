import torch
import torch.nn as nn
import torch.nn.functional as F

class SARDegradationEstimator(nn.Module):
    """
    Extracts high-level degradation and noise Cues (speckle level, local variance, gradients)
    from single-channel SAR backscatter imagery.
    Produces a 128-dimensional global degradation embedding vector.
    """
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        
        # Spatial gradient kernel (Sobel filter for edge/speckle roughness)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Degradation feature extractor CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Input: [Raw SAR, GradX, GradY]
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # Compute spatial intensity gradients
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        # Stack raw backscatter with horizontal and vertical gradients -> 3 channels
        stacked = torch.cat([x, grad_x, grad_y], dim=1)
        
        feats = self.conv_layers(stacked).squeeze(-1).squeeze(-1)  # [B, 128]
        degradation_embed = self.fc(feats)                         # [B, embed_dim]
        return degradation_embed
