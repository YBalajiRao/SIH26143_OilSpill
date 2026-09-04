import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from src.perception.feature_modulation import FiLMBlock

class AtrousDegradationEstimator(nn.Module):
    """
    Multi-scale Dilated Degradation Estimator for SAR imagery (E5.2).
    Extracts speckle noise profiles using dilated convolutions (dilation = 1, 2, 4).
    """
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        
        self.d1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=4, dilation=4)
        
        self.bn_init = nn.BatchNorm2d(96)
        self.silu = nn.SiLU()

        self.conv_body = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
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
        feat1 = self.d1(x)
        feat2 = self.d2(x)
        feat4 = self.d4(x)
        stacked = torch.cat([feat1, feat2, feat4], dim=1)
        x_init = self.silu(self.bn_init(stacked))
        pooled = self.conv_body(x_init).squeeze(-1).squeeze(-1)
        return self.fc(pooled)


class ASPPBottleneck(nn.Module):
    """Atrous Spatial Pyramid Pooling for Bottleneck Feature Context (E5.2)."""
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18), nn.BatchNorm2d(out_channels), nn.ReLU())
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        concat = torch.cat([feat0, feat1, feat2, feat3], dim=1)
        return self.out_conv(concat)


class PhysioGraphSpillPerception(nn.Module):
    """
    Physio-GraphSpill Perception v2.0 Champion Architecture (E5.2).
    - ResNet-34 Multi-scale Encoder
    - Atrous Degradation Estimator
    - ASPP + FiLM Bottleneck Modulation
    - Multi-scale Skip FiLM Modulation
    """
    def __init__(self, in_channels=1, out_classes=1, encoder_name="resnet34", encoder_weights="imagenet", dropout_rate=0.1):
        super().__init__()
        
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation=None
        )
        
        self.degradation_estimator = AtrousDegradationEstimator(in_channels=in_channels, embed_dim=128)
        self.aspp_bottleneck = ASPPBottleneck(in_channels=512, out_channels=512)
        self.film_bottleneck = FiLMBlock(feature_dim=512, embed_dim=128)
        
        self.film_modules = nn.ModuleDict({
            '3': FiLMBlock(feature_dim=128, embed_dim=128),
            '4': FiLMBlock(feature_dim=256, embed_dim=128),
            '5': FiLMBlock(feature_dim=512, embed_dim=128)
        })
        self.embed_dim = 128
        self.mc_dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        deg = self.degradation_estimator(x)
        features = list(self.unet.encoder(x))

        for i in range(len(features) - 1):
            key = str(i)
            if key in self.film_modules:
                features[i] = self.film_modules[key](features[i], deg)

        bottleneck = features[-1]
        aspp_feat = self.aspp_bottleneck(bottleneck)
        features[-1] = self.film_bottleneck(aspp_feat, deg)
        features[-1] = self.mc_dropout(features[-1])

        try:
            decoder_output = self.unet.decoder(features)
        except TypeError:
            decoder_output = self.unet.decoder(*features)

        logits = self.unet.segmentation_head(decoder_output)
        return logits

DAFM_UNet = PhysioGraphSpillPerception
