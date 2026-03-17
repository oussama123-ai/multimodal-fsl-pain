"""
contrastive.py
--------------
Self-supervised cross-modal contrastive pretraining (Section 3.5).

Implements:
  - InfoNCE loss (oord et al., 2018)
  - Dataset-conditional pair generation:
      BioVid samples   → all intra + cross-modal pairs (v-a, v-p, a-p)
      UNBC-McMaster    → only video-physio pairs (v-p); audio encoder skips
  - MoCo-style momentum encoder update

Algorithm matches Algorithm 1 in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def info_nce_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Symmetric InfoNCE / NT-Xent loss.

    Parameters
    ----------
    z_i, z_j : (B, D) — L2-normalized embeddings of two views
    temperature : float

    Returns
    -------
    scalar loss
    """
    B = z_i.shape[0]
    # Concatenate both views
    z = torch.cat([z_i, z_j], dim=0)              # (2B, D)
    sim = torch.mm(z, z.t()) / temperature         # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Contrastive trainer
# ---------------------------------------------------------------------------

class ContrastiveTrainer(nn.Module):
    """
    Manages the full contrastive pretraining objective.

    Parameters
    ----------
    video_enc, audio_enc, physio_enc : encoder modules (with .get_projection())
    temperature : τ (default 0.1, optimized by grid search)
    lambda_intra, lambda_cross, lambda_momentum : loss weights
    momentum : EMA decay for momentum encoders (default 0.99)
    """

    def __init__(
        self,
        video_enc,
        audio_enc,
        physio_enc,
        temperature:       float = 0.1,
        lambda_intra:      float = 0.3,
        lambda_cross:      float = 0.5,
        lambda_momentum:   float = 0.2,
        momentum:          float = 0.99,
    ):
        super().__init__()
        self.video_enc  = video_enc
        self.audio_enc  = audio_enc
        self.physio_enc = physio_enc

        self.tau             = temperature
        self.lambda_intra    = lambda_intra
        self.lambda_cross    = lambda_cross
        self.lambda_momentum = lambda_momentum
        self.momentum        = momentum

        # Momentum (target) encoders — no gradient
        self.mom_video  = self._clone_no_grad(video_enc)
        self.mom_audio  = self._clone_no_grad(audio_enc)
        self.mom_physio = self._clone_no_grad(physio_enc)

    @staticmethod
    def _clone_no_grad(encoder: nn.Module) -> nn.Module:
        import copy
        clone = copy.deepcopy(encoder)
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    @torch.no_grad()
    def _update_momentum(self):
        """EMA update: θ_mom ← m·θ_mom + (1-m)·θ"""
        m = self.momentum
        pairs = [
            (self.video_enc,  self.mom_video),
            (self.audio_enc,  self.mom_audio),
            (self.physio_enc, self.mom_physio),
        ]
        for enc, mom in pairs:
            for p, pm in zip(enc.parameters(), mom.parameters()):
                pm.data.mul_(m).add_((1.0 - m) * p.data)

    def forward(
        self,
        view1: dict,
        view2: dict,
        audio_available: bool = True,
    ) -> torch.Tensor:
        """
        Compute total contrastive loss for one batch.

        Parameters
        ----------
        view1, view2 : dicts with keys 'video', 'audio', 'physio'
                       (augmented views of the same samples)
        audio_available : False for UNBC-McMaster batches

        Returns
        -------
        total_loss : scalar tensor
        """
        # ---- Online encoder projections (view 1) ----
        z_v1 = self.video_enc.get_projection( view1["video"],  available=True)
        z_a1 = self.audio_enc.get_projection( view1["audio"],  available=audio_available)
        z_p1 = self.physio_enc.get_projection(view1["physio"], available=True)

        # ---- Online encoder projections (view 2) ----
        z_v2 = self.video_enc.get_projection( view2["video"],  available=True)
        z_a2 = self.audio_enc.get_projection( view2["audio"],  available=audio_available)
        z_p2 = self.physio_enc.get_projection(view2["physio"], available=True)

        # ---- Momentum encoder projections (for consistency loss) ----
        with torch.no_grad():
            z_v_mom = self.mom_video.get_projection( view1["video"],  available=True)
            z_a_mom = self.mom_audio.get_projection( view1["audio"],  available=audio_available)
            z_p_mom = self.mom_physio.get_projection(view1["physio"], available=True)

        # ---- Intra-modal losses (always computed) ----
        L_intra = (
            info_nce_loss(z_v1, z_v2, self.tau)
            + info_nce_loss(z_p1, z_p2, self.tau)
        )
        if audio_available:
            L_intra = L_intra + info_nce_loss(z_a1, z_a2, self.tau)

        # ---- Cross-modal losses ----
        # Video–Physio always included
        L_cross = info_nce_loss(z_v1, z_p1, self.tau)

        if audio_available:
            # BioVid: all three cross-modal pairs
            L_cross = L_cross + info_nce_loss(z_v1, z_a1, self.tau) \
                               + info_nce_loss(z_a1, z_p1, self.tau)

        # ---- Momentum consistency loss ----
        L_mom = (
            F.mse_loss(z_v1, z_v_mom)
            + F.mse_loss(z_p1, z_p_mom)
        )
        if audio_available:
            L_mom = L_mom + F.mse_loss(z_a1, z_a_mom)

        total_loss = (
            self.lambda_intra    * L_intra
            + self.lambda_cross  * L_cross
            + self.lambda_momentum * L_mom
        )

        # Update momentum encoders
        self._update_momentum()

        return total_loss
