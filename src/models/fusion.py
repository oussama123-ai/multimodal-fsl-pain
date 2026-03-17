"""
fusion.py
---------
Transformer-based multimodal fusion.

Takes per-modality embeddings (video 768-d, audio 512-d, physio 256-d),
projects them to a shared 768-d space, appends positional tokens, and
passes them through L transformer encoder layers.  Missing modalities
are handled by substituting the encoder's learned availability token.

Output: fused representation of shape (B, 768).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Two-layer MLP with GELU used as FFN inside transformer blocks."""

    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TransformerEncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn  = _MLP(d_model, dropout=dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x          : (B, S, D)
        attn_weights: (B, S, S)
        """
        normed = self.ln1(x)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Alignment projection (heterogeneous dim → shared 768)
# ---------------------------------------------------------------------------

class ModalityAlignmentMLP(nn.Module):
    """Project one modality embedding to the shared d_model space."""

    def __init__(self, in_dim: int, d_model: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main fusion module
# ---------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """
    Transformer-based multimodal fusion (Section 3.5).

    Parameters
    ----------
    video_dim  : dimensionality of video encoder output (default 768)
    audio_dim  : dimensionality of audio encoder output (default 512)
    physio_dim : dimensionality of physio encoder output (default 256)
    d_model    : shared hidden dimension (768)
    n_heads    : number of attention heads (12)
    n_layers   : number of transformer layers (default 4)
    dropout    : dropout rate (0.1)

    Input
    -----
    z_video  : (B, video_dim)   — may be an availability token
    z_audio  : (B, audio_dim)
    z_physio : (B, physio_dim)

    Output
    ------
    z_fused  : (B, d_model)     — mean-pooled over the 3 modality tokens
    attn_weights : list of (B, 3, 3) per layer
    """

    MODALITIES = ["video", "audio", "physio"]

    def __init__(
        self,
        video_dim: int = 768,
        audio_dim: int = 512,
        physio_dim: int = 256,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-modality alignment projections
        self.align = nn.ModuleDict({
            "video":  ModalityAlignmentMLP(video_dim,  d_model, dropout),
            "audio":  ModalityAlignmentMLP(audio_dim,  d_model, dropout),
            "physio": ModalityAlignmentMLP(physio_dim, d_model, dropout),
        })

        # Learnable positional (modality) embeddings — 3 tokens
        self.pos_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_video:  torch.Tensor,
        z_audio:  torch.Tensor,
        z_physio: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        # 1. Project to shared space
        v = self.align["video"](z_video).unsqueeze(1)    # (B, 1, D)
        a = self.align["audio"](z_audio).unsqueeze(1)
        p = self.align["physio"](z_physio).unsqueeze(1)

        # 2. Concatenate modality tokens → (B, 3, D)
        Z = torch.cat([v, a, p], dim=1)

        # 3. Add positional / modality embeddings
        Z = Z + self.pos_emb

        # 4. Pass through transformer layers
        all_attn: list[torch.Tensor] = []
        for layer in self.layers:
            Z, attn = layer(Z)
            all_attn.append(attn)

        Z = self.norm(Z)

        # 5. Mean-pool over the 3 modality tokens
        z_fused = Z.mean(dim=1)   # (B, D)

        return z_fused, all_attn
