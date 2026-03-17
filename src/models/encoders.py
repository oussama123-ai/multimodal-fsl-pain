"""
encoders.py
-----------
Modality-specific feature encoders.

  - VideoEncoder  : ViT-Base/16 (768-d output)
  - AudioEncoder  : ResNet-18 on mel-spectrograms (512-d output)
  - PhysioEncoder : 1-D CNN for HR / SpO2 / RR signals (256-d output)

Each encoder exposes a `forward(x, available=True)` API; when `available=False`
the encoder returns a zero vector concatenated with the learned availability
token, enabling masked contrastive pretraining on UNBC-McMaster (no audio).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR-style)."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Video Encoder
# ---------------------------------------------------------------------------

class VideoEncoder(nn.Module):
    """
    ViT-Base/16 backbone for facial video frames.

    Input  : (B, T, 3, 224, 224)  — T temporal frames
    Output : (B, 768)             — mean-pooled CLS tokens

    Patch size   : 16×16
    Hidden dim   : 768
    Attn heads   : 12
    Blocks       : 12
    Dropout      : 0.1
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.1,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            drop_rate=dropout,
        )
        self.feat_dim = self.vit.embed_dim  # 768
        self.projection = ProjectionHead(self.feat_dim, proj_dim=proj_dim)

        # Availability token (used when video is missing at inference)
        self.availability_token = nn.Parameter(torch.zeros(self.feat_dim))

    def forward(self, x: torch.Tensor, available: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, T, 3, 224, 224)
        available : bool — if False return availability-token vector
        """
        if not available:
            B = x.shape[0]
            return self.availability_token.unsqueeze(0).expand(B, -1)

        B, T, C, H, W = x.shape
        # Merge batch and time dimensions → run ViT frame-by-frame
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)          # (B*T, 768)
        feats = feats.view(B, T, -1).mean(dim=1)  # (B, 768) — temporal mean
        return feats

    def get_projection(self, x: torch.Tensor, available: bool = True) -> torch.Tensor:
        return self.projection(self.forward(x, available))


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Basic ResNet residual block (no downsampling)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class _ResStage(nn.Module):
    """Downsampling stage: stride-2 conv + N residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class AudioEncoder(nn.Module):
    """
    ResNet-18–style CNN operating on mel-spectrograms.

    Input  : (B, 1, 128, 128)  — single-channel mel-spectrogram
    Output : (B, 512)

    Architecture
    ------------
      stem  : conv 7×7, stride 2  → 64 ch
      stage1: 64  → 64  ch, 2 res-blocks
      stage2: 64  → 128 ch, 2 res-blocks
      stage3: 128 → 256 ch, 2 res-blocks
      stage4: 256 → 512 ch, 2 res-blocks
      global average pool → 512-d vector
    """

    def __init__(self, dropout: float = 0.3, proj_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = _ResStage(64, 64, n_blocks=2)
        self.stage2 = _ResStage(64, 128, n_blocks=2)
        self.stage3 = _ResStage(128, 256, n_blocks=2)
        self.stage4 = _ResStage(256, 512, n_blocks=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.drop   = nn.Dropout(dropout)
        self.feat_dim = 512
        self.projection = ProjectionHead(self.feat_dim, proj_dim=proj_dim)

        self.availability_token = nn.Parameter(torch.zeros(self.feat_dim))

    def forward(self, x: torch.Tensor, available: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, 1, 128, 128)
        available : bool
        """
        if not available:
            B = x.shape[0]
            return self.availability_token.unsqueeze(0).expand(B, -1)

        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out).flatten(1)  # (B, 512)
        out = self.drop(out)
        return F.normalize(out, dim=-1)

    def get_projection(self, x: torch.Tensor, available: bool = True) -> torch.Tensor:
        return self.projection(self.forward(x, available))


# ---------------------------------------------------------------------------
# Physiological Signal Encoder
# ---------------------------------------------------------------------------

class _Conv1dResBlock(nn.Module):
    """1-D residual block."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class PhysioEncoder(nn.Module):
    """
    1-D CNN for multi-variate physiological signals (HR, SpO2, RR).

    Input  : (B, 3, L)  — 3 channels, L = 30 s × 1 Hz = 30 samples
    Output : (B, 256)

    Architecture
    ------------
      conv1 : kernel 7, 3  → 32  ch + BN + ReLU + res-block
      conv2 : kernel 5, 32 → 64  ch + BN + ReLU + res-block
      conv3 : kernel 3, 64 → 128 ch + BN + ReLU + res-block
      global average pool  → 128 ch
      linear 128 → 256
    """

    def __init__(self, n_signals: int = 3, proj_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_signals, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            _Conv1dResBlock(32, kernel_size=7),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            _Conv1dResBlock(64, kernel_size=5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            _Conv1dResBlock(128, kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, 256)
        self.feat_dim = 256
        self.projection = ProjectionHead(self.feat_dim, proj_dim=proj_dim)

        self.availability_token = nn.Parameter(torch.zeros(self.feat_dim))

    def forward(self, x: torch.Tensor, available: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, 3, L)
        available : bool
        """
        if not available:
            B = x.shape[0]
            return self.availability_token.unsqueeze(0).expand(B, -1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).squeeze(-1)   # (B, 128)
        out = F.relu(self.fc(out), inplace=True)
        return F.normalize(out, dim=-1)

    def get_projection(self, x: torch.Tensor, available: bool = True) -> torch.Tensor:
        return self.projection(self.forward(x, available))
