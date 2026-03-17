"""
interpretability.py
--------------------
Interpretability and uncertainty tools (Section 3.9).

  - AttentionVisualizer     : extract and plot cross-modal attention weights
  - GradCAM                 : gradient-weighted class activation maps for video
  - MCDropoutUncertainty    : Monte Carlo uncertainty estimation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention Visualizer
# ---------------------------------------------------------------------------

class AttentionVisualizer:
    """
    Extract per-layer, per-modality attention weights from the fusion module.

    Usage
    -----
        vis = AttentionVisualizer(model)
        attn = vis.get_attention(video, audio, physio)
        # attn : (n_layers, 3, 3) — rows = queries, cols = key modalities
    """

    MODALITY_NAMES = ["video", "audio", "physio"]

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def get_attention(
        self,
        video:  torch.Tensor,
        audio:  torch.Tensor,
        physio: torch.Tensor,
    ) -> np.ndarray:
        """
        Returns attention weights for a single sample (B=1).

        Returns
        -------
        attn : np.ndarray of shape (n_layers, 3, 3)
        """
        self.model.eval()
        _, attn_list = self.model.encode(
            video.unsqueeze(0) if video.ndim < 5 else video,
            audio.unsqueeze(0) if audio.ndim < 4 else audio,
            physio.unsqueeze(0) if physio.ndim < 3 else physio,
        )
        # Each element in attn_list is (1, 3, 3)
        stacked = torch.stack(
            [a.squeeze(0) for a in attn_list], dim=0
        ).cpu().numpy()   # (n_layers, 3, 3)
        return stacked

    def modality_importance(self, attn: np.ndarray) -> dict[str, float]:
        """
        Aggregate attention into per-modality importance scores.
        Average over layers and query tokens; use column sums as key-modality weight.
        """
        # attn : (L, 3, 3) → mean over layers → (3, 3) → col mean → (3,)
        mean_attn  = attn.mean(axis=0)          # (3, 3)
        col_weight = mean_attn.mean(axis=0)     # (3,) importance per modality
        return {
            name: float(w)
            for name, w in zip(self.MODALITY_NAMES, col_weight)
        }


# ---------------------------------------------------------------------------
# GradCAM for video frames
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for the ViT video encoder.

    Registers a forward hook on the last ViT attention layer to capture
    attention-weighted spatial activations.

    Usage
    -----
        gcam = GradCAM(model.video_enc)
        cam  = gcam(video_tensor, target_class=1)   # (H, W) heatmap
    """

    def __init__(self, video_encoder):
        self.encoder  = video_encoder
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook the last transformer block in ViT
        last_block = self.encoder.vit.blocks[-1]

        def _fwd_hook(module, input, output):
            self.activations = output.detach()

        def _bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(last_block.register_forward_hook(_fwd_hook))
        self._hooks.append(last_block.register_full_backward_hook(_bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        video: torch.Tensor,    # (T, 3, H, W)
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Returns
        -------
        cam : np.ndarray of shape (H, W), values in [0, 1]
        """
        self.encoder.zero_grad()
        self.encoder.train()  # enable grads

        feat = self.encoder(video.unsqueeze(0))  # (1, 768)
        # Create a one-hot score for the target class
        score = feat[0, target_class]
        score.backward()

        self.encoder.eval()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))

        # Global average pool gradients over sequence length
        weights = self.gradients.mean(dim=1, keepdim=True)  # (1, 1, D)
        cam = (weights * self.activations).sum(dim=-1)      # (1, seq_len)
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Reshape to spatial grid (ViT with patch 16: 14×14 for 224×224)
        patch_grid = int(cam.shape[0] ** 0.5)
        cam = cam[:patch_grid ** 2].reshape(patch_grid, patch_grid)
        # Upsample to input resolution
        cam = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# MC Dropout Uncertainty
# ---------------------------------------------------------------------------

class MCDropoutUncertainty:
    """
    Estimate predictive uncertainty via Monte Carlo Dropout (Gal & Ghahramani, 2016).

    T stochastic forward passes → mean probability + entropy.

    Usage
    -----
        mcd = MCDropoutUncertainty(model, n_passes=50)
        result = mcd(support_data, query_data)
        # result["uncertainty"] : (Q,) — predictive entropy per query sample
    """

    def __init__(self, model, n_passes: int = 50):
        self.model    = model
        self.n_passes = n_passes

    def __call__(
        self,
        support_video, support_audio, support_physio, support_labels,
        query_video,   query_audio,   query_physio,
        device: str = "cuda",
    ) -> dict[str, np.ndarray]:
        """
        Returns
        -------
        dict with:
          mean_probs  : (Q, N) — averaged probabilities
          uncertainty : (Q,)  — predictive entropy
          high_uncert_mask : (Q,) bool — uncertainty > 0.3 threshold
        """
        result = self.model.mc_dropout_uncertainty(
            support_video=support_video.to(device),
            support_audio=support_audio.to(device),
            support_physio=support_physio.to(device),
            support_labels=support_labels.to(device),
            query_video=query_video.to(device),
            query_audio=query_audio.to(device),
            query_physio=query_physio.to(device),
            n_passes=self.n_passes,
        )

        mean_probs  = result["mean_probs"].cpu().numpy()
        uncertainty = result["uncertainty"].cpu().numpy()

        return {
            "mean_probs":      mean_probs,
            "uncertainty":     uncertainty,
            "high_uncert_mask": uncertainty > 0.3,
        }
