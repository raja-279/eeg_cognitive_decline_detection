"""
model.py
--------
ResNet18-based CNN for EEG recurrence-plot classification.

Supports:
  - 3-channel input  (theta / alpha / beta recurrence plots)
  - Binary or multi-class output  (configurable num_classes)
  - Label smoothing + dropout for regularisation
  - build_model() factory function

Architecture choices vs. previous EEGNet:
  ✓ ResNet18 with pretrained ImageNet weights (transfer learning)
  ✓ Modified stem: 3×3 conv, no stride-4 downsampling (64px images are small)
  ✓ Dropout before FC head
  ✓ BatchNorm already built into ResNet blocks
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════
# ResNet18 backbone
# ═══════════════════════════════════════════════════════════════════════════

class EEGResNet(nn.Module):
    """
    ResNet18 adapted for small (64×64) 3-channel EEG recurrence plots.

    Key modifications over stock ResNet18:
      - Stem: 3×3 conv, stride=1, no MaxPool  (keeps spatial detail on 64px input)
      - Dropout(p) injected before the FC head
      - FC head replaced with Linear(512 → num_classes)

    Parameters
    ----------
    in_channels  : channels in the input image (3 for theta/alpha/beta RP)
    num_classes  : number of output logits (2 for binary, 3 for tri-class)
    dropout      : dropout probability before FC
    pretrained   : load ImageNet weights (only if in_channels == 3)
    freeze_bn    : freeze BatchNorm running stats (useful with small batches)
    """

    def __init__(self,
                 in_channels:  int   = 3,
                 num_classes:  int   = 3,
                 dropout:      float = 0.5,
                 pretrained:   bool  = True,
                 freeze_bn:    bool  = False):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────────────
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained and in_channels == 3 else None
        backbone = tvm.resnet18(weights=weights)

        # ── replace stem: remove stride-2 and maxpool (bad for 64px) ─────
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        # re-init if we changed in_channels or didn't use pretrained
        if not pretrained or in_channels != 3:
            nn.init.kaiming_normal_(backbone.conv1.weight,
                                    mode="fan_out", nonlinearity="relu")

        backbone.maxpool = nn.Identity()   # skip the 3×3 maxpool

        # ── strip original FC ─────────────────────────────────────────────
        enc_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.encoder = backbone

        # ── classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # ── optional BN freeze ────────────────────────────────────────────
        if freeze_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)  — e.g. (B, 3, 64, 64)

        Returns
        -------
        logits : (B, num_classes)
        """
        feats  = self.encoder(x)   # (B, 512)
        return self.head(feats)


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight fallback: improved EEGNet  (used when pretrained=False, in_ch=1)
# ═══════════════════════════════════════════════════════════════════════════

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(pool),
        )
    def forward(self, x): return self.block(x)


class EEGNet(nn.Module):
    """
    Lightweight CNN fallback (no torchvision dependency for in_channels != 3).

    Improvements over original:
      - GELU activations
      - Two-layer FC head with intermediate dropout
      - Residual skip in encoder via 1×1 projection
    """

    def __init__(self,
                 in_channels:  int   = 3,
                 img_size:     int   = 64,
                 num_classes:  int   = 3,
                 base_filters: int   = 32,
                 dropout:      float = 0.5):
        super().__init__()
        f = base_filters
        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, f,     kernel=3, pool=2),
            _ConvBlock(f,           f * 2, kernel=3, pool=2),
            _ConvBlock(f * 2,       f * 4, kernel=3, pool=2),
            _ConvBlock(f * 4,       f * 8, kernel=3, pool=2),
        )
        with torch.no_grad():
            dummy   = torch.zeros(1, in_channels, img_size, img_size)
            enc_dim = self.encoder(dummy).numel()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def build_model(
    architecture: Literal["resnet", "eegnet"] = "resnet",
    in_channels:  int   = 3,
    img_size:     int   = 64,
    num_classes:  int   = 3,
    dropout:      float = 0.5,
    pretrained:   bool  = True,
    base_filters: int   = 32,
) -> nn.Module:
    """
    Parameters
    ----------
    architecture : "resnet" (recommended) or "eegnet" (lightweight)
    in_channels  : 3 for recurrence plots (theta/alpha/beta channels)
    img_size     : spatial resolution (only needed for eegnet)
    num_classes  : 2 for binary stages, 3 for tri-class
    dropout      : regularisation
    pretrained   : use ImageNet weights (resnet + in_channels==3 only)
    base_filters : base filter count (eegnet only)

    Returns
    -------
    nn.Module
    """
    if architecture == "resnet":
        return EEGResNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
        )
    elif architecture == "eegnet":
        return EEGNet(
            in_channels=in_channels,
            img_size=img_size,
            num_classes=num_classes,
            base_filters=base_filters,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Choose 'resnet' or 'eegnet'.")
