"""
prototypical.py
---------------
Prototypical Networks (Snell et al., NeurIPS 2017) adapted for multimodal
pain recognition.

Key design choices
------------------
- Subject-independent episode construction: support and query sets are drawn
  from disjoint subject partitions to prevent data leakage.
- Euclidean distance metric (standard for prototypical nets).
- Supports N-way K-shot evaluation with variable N ∈ {2, 3}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalHead(nn.Module):
    """
    Few-shot classification head using class prototypes.

    Given:
      support embeddings z_s : (N * K, D)
      support labels    y_s : (N * K,)
      query embeddings  z_q : (Q_total, D)

    Returns:
      log-probabilities : (Q_total, N)
    """

    def __init__(self, distance: str = "euclidean"):
        super().__init__()
        assert distance in ("euclidean", "cosine"), f"Unknown distance: {distance}"
        self.distance = distance

    def compute_prototypes(
        self,
        z_support: torch.Tensor,
        y_support: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-class prototype as the mean of support embeddings.

        Parameters
        ----------
        z_support : (N*K, D)
        y_support : (N*K,)  — integer class labels 0..N-1

        Returns
        -------
        prototypes : (N, D)
        """
        classes = y_support.unique(sorted=True)
        prototypes = torch.stack([
            z_support[y_support == c].mean(dim=0) for c in classes
        ])
        return prototypes

    def forward(
        self,
        z_support: torch.Tensor,
        y_support: torch.Tensor,
        z_query:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_support : (N*K, D)
        y_support : (N*K,)
        z_query   : (Q_total, D)

        Returns
        -------
        log_probs : (Q_total, N)
        """
        prototypes = self.compute_prototypes(z_support, y_support)  # (N, D)

        if self.distance == "euclidean":
            # Squared Euclidean via broadcasting
            dists = torch.cdist(z_query, prototypes, p=2)  # (Q, N)
            log_probs = F.log_softmax(-dists, dim=-1)
        else:  # cosine
            q = F.normalize(z_query, dim=-1)
            p = F.normalize(prototypes, dim=-1)
            sim = q @ p.T                                  # (Q, N)
            log_probs = F.log_softmax(sim, dim=-1)

        return log_probs

    def predict(
        self,
        z_support: torch.Tensor,
        y_support: torch.Tensor,
        z_query:   torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted class indices (argmax)."""
        with torch.no_grad():
            log_probs = self.forward(z_support, y_support, z_query)
        return log_probs.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Episode loss helper
# ---------------------------------------------------------------------------

def prototypical_loss(
    log_probs: torch.Tensor,
    y_query:   torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Cross-entropy loss for episodic training.

    Parameters
    ----------
    log_probs : (Q_total, N)  — from PrototypicalHead.forward
    y_query   : (Q_total,)    — ground-truth class indices

    Returns
    -------
    loss     : scalar tensor
    accuracy : float in [0, 1]
    """
    loss = F.nll_loss(log_probs, y_query)
    preds = log_probs.argmax(dim=-1)
    acc = (preds == y_query).float().mean().item()
    return loss, acc
