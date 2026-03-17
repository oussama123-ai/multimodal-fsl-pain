"""
full_model.py
-------------
End-to-end assembly of the SSTL pain recognition framework:

  VideoEncoder + AudioEncoder + PhysioEncoder
        ↓              ↓               ↓
           MultimodalFusion (transformer)
                       ↓
            PrototypicalHead  (few-shot)
                       ↓
              Classification / AUC

Also exposes:
  - get_embeddings()  for standalone feature extraction
  - mc_dropout_uncertainty()  for Monte Carlo uncertainty estimates
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders    import VideoEncoder, AudioEncoder, PhysioEncoder
from .fusion      import MultimodalFusion
from .prototypical import PrototypicalHead, prototypical_loss


class MultimodalPainModel(nn.Module):
    """
    Full multimodal few-shot pain recognition model.

    Parameters
    ----------
    video_pretrained : load ImageNet weights for ViT backbone
    n_fusion_layers  : number of transformer layers in fusion module (default 4)
    dropout          : dropout rate throughout (default 0.1)
    proj_dim         : contrastive projection head output dim (default 128)
    distance         : prototypical head distance metric ('euclidean'|'cosine')
    """

    def __init__(
        self,
        video_pretrained: bool = True,
        n_fusion_layers: int = 4,
        dropout: float = 0.1,
        proj_dim: int = 128,
        distance: str = "euclidean",
    ):
        super().__init__()

        # ----- Encoders -----
        self.video_enc  = VideoEncoder(pretrained=video_pretrained,
                                       dropout=dropout, proj_dim=proj_dim)
        self.audio_enc  = AudioEncoder(dropout=dropout, proj_dim=proj_dim)
        self.physio_enc = PhysioEncoder(proj_dim=proj_dim)

        # ----- Fusion -----
        self.fusion = MultimodalFusion(
            video_dim=self.video_enc.feat_dim,    # 768
            audio_dim=self.audio_enc.feat_dim,    # 512
            physio_dim=self.physio_enc.feat_dim,  # 256
            d_model=768,
            n_heads=12,
            n_layers=n_fusion_layers,
            dropout=dropout,
        )

        # ----- Few-shot head -----
        self.proto_head = PrototypicalHead(distance=distance)

        # ----- MC Dropout (for uncertainty at inference) -----
        self.mc_dropout = nn.Dropout(p=dropout)

    # ------------------------------------------------------------------
    # Core forward (embedding extraction)
    # ------------------------------------------------------------------

    def encode(
        self,
        video:   torch.Tensor,
        audio:   torch.Tensor,
        physio:  torch.Tensor,
        video_avail:  bool = True,
        audio_avail:  bool = True,
        physio_avail: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Extract fused embedding for a batch.

        Returns
        -------
        z_fused     : (B, 768)
        attn_weights: list of per-layer attention tensors
        """
        z_v = self.video_enc(video,   available=video_avail)
        z_a = self.audio_enc(audio,   available=audio_avail)
        z_p = self.physio_enc(physio, available=physio_avail)

        z_fused, attn = self.fusion(z_v, z_a, z_p)
        return z_fused, attn

    # ------------------------------------------------------------------
    # Few-shot forward (used during episodic training / inference)
    # ------------------------------------------------------------------

    def forward(
        self,
        support_video:   torch.Tensor,
        support_audio:   torch.Tensor,
        support_physio:  torch.Tensor,
        support_labels:  torch.Tensor,
        query_video:     torch.Tensor,
        query_audio:     torch.Tensor,
        query_physio:    torch.Tensor,
        query_labels:    torch.Tensor | None = None,
        modality_mask:   dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Episodic forward pass.

        Parameters
        ----------
        support_* / query_* : tensors for each modality
        support_labels       : (N*K,)
        query_labels         : (Q,) — if None, skip loss computation
        modality_mask        : {'video': bool, 'audio': bool, 'physio': bool}
                               defaults to all True

        Returns
        -------
        dict with keys: loss, log_probs, predictions, accuracy (if labels given)
        """
        mask = modality_mask or {"video": True, "audio": True, "physio": True}

        # Encode support set
        z_support, _ = self.encode(
            support_video, support_audio, support_physio,
            video_avail=mask["video"],
            audio_avail=mask["audio"],
            physio_avail=mask["physio"],
        )

        # Encode query set
        z_query, attn = self.encode(
            query_video, query_audio, query_physio,
            video_avail=mask["video"],
            audio_avail=mask["audio"],
            physio_avail=mask["physio"],
        )

        # Few-shot classification
        log_probs = self.proto_head(z_support, support_labels, z_query)
        predictions = log_probs.argmax(dim=-1)

        result = {
            "log_probs":   log_probs,
            "predictions": predictions,
            "attn_weights": attn,
        }

        if query_labels is not None:
            loss, acc = prototypical_loss(log_probs, query_labels)
            result["loss"]     = loss
            result["accuracy"] = acc

        return result

    # ------------------------------------------------------------------
    # Monte Carlo Dropout uncertainty
    # ------------------------------------------------------------------

    @torch.no_grad()
    def mc_dropout_uncertainty(
        self,
        support_video:  torch.Tensor,
        support_audio:  torch.Tensor,
        support_physio: torch.Tensor,
        support_labels: torch.Tensor,
        query_video:    torch.Tensor,
        query_audio:    torch.Tensor,
        query_physio:   torch.Tensor,
        n_passes: int = 50,
    ) -> dict[str, torch.Tensor]:
        """
        Estimate predictive uncertainty via T stochastic forward passes.

        Returns
        -------
        mean_probs   : (Q, N)   — average predicted probabilities
        uncertainty  : (Q,)    — predictive entropy H[p(y|x)]
        """
        self.train()   # enable dropout

        probs_list = []
        for _ in range(n_passes):
            out = self.forward(
                support_video, support_audio, support_physio, support_labels,
                query_video,   query_audio,   query_physio,
            )
            probs_list.append(out["log_probs"].exp())

        self.eval()

        probs  = torch.stack(probs_list)         # (T, Q, N)
        mean_p = probs.mean(dim=0)               # (Q, N)

        # Predictive entropy
        eps = 1e-8
        entropy = -(mean_p * (mean_p + eps).log()).sum(dim=-1)  # (Q,)

        return {"mean_probs": mean_p, "uncertainty": entropy}

    # ------------------------------------------------------------------
    # Convenience: extract embeddings without prototypical head
    # ------------------------------------------------------------------

    def get_embeddings(
        self,
        video:  torch.Tensor,
        audio:  torch.Tensor,
        physio: torch.Tensor,
        modality_mask: dict[str, bool] | None = None,
    ) -> torch.Tensor:
        mask = modality_mask or {"video": True, "audio": True, "physio": True}
        z, _ = self.encode(
            video, audio, physio,
            video_avail=mask["video"],
            audio_avail=mask["audio"],
            physio_avail=mask["physio"],
        )
        return z
