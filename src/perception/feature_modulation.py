import torch
import torch.nn as nn

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Applies affine transformation (gamma * F + beta) to feature maps
    conditioned on the SAR degradation vector.
    """
    def __init__(self, feature_dim, embed_dim=128):
        super().__init__()
        self.fc_gamma = nn.Linear(embed_dim, feature_dim)
        self.fc_beta = nn.Linear(embed_dim, feature_dim)

    def forward(self, x, degradation_embed):
        # Compute dynamic scale (gamma) and shift (beta)
        gamma = self.fc_gamma(degradation_embed).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta  = self.fc_beta(degradation_embed).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        
        # Modulate feature representation
        return (1.0 + gamma) * x + beta
