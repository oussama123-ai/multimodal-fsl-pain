"""
frozen_encoder_baseline.py
---------------------------
Implements the reviewer-requested "lightweight / frozen encoder" baseline:

  off-the-shelf pretrained encoders (ImageNet ViT / ImageNet-transferred
  ResNet-18 / randomly-initialized 1D-CNN), ALL FROZEN, feeding a
  concatenate-and-pool fusion into a linear or 1-layer-MLP probe — trained
  under the exact same k-shot / LOSO protocol as the proposed model.

This isolates the contribution of (a) contrastive pretraining on
UNBC-McMaster + BioVid and (b) end-to-end fine-tuning, by holding everything
else (architecture family, fusion protocol, few-shot evaluation protocol)
fixed relative to the proposed method.

Config: configs/baseline_frozen_encoder.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..models.encoders import AudioEncoder, PhysioEncoder, VideoEncoder
from .manifest import RunManifest


class FrozenEncoderProbe(nn.Module):
    """Frozen video/audio/physio encoders -> concat+pool -> linear/MLP probe."""

    def __init__(self, probe_head: str = "linear", mlp_hidden_dim: int = 128):
        super().__init__()

        self.video_enc = VideoEncoder(pretrained=True)
        self.audio_enc = AudioEncoder()
        self.physio_enc = PhysioEncoder()

        # Freeze all encoder weights — this is what makes it "lightweight":
        # only the probe head below is ever trained.
        for enc in (self.video_enc, self.audio_enc, self.physio_enc):
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()

        in_dim = self.video_enc.feat_dim + self.audio_enc.feat_dim + self.physio_enc.feat_dim

        if probe_head == "linear":
            self.probe = nn.Linear(in_dim, 2)
        elif probe_head == "mlp_1layer":
            self.probe = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dim, 2),
            )
        else:
            raise ValueError(f"Unknown probe_head: {probe_head}")

    @torch.no_grad()
    def _frozen_features(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        physio: torch.Tensor,
        audio_available: bool = True,
    ) -> torch.Tensor:
        self.video_enc.eval()
        self.audio_enc.eval()
        self.physio_enc.eval()
        z_v = self.video_enc(video, available=True)
        z_a = self.audio_enc(audio, available=audio_available)
        z_p = self.physio_enc(physio, available=True)
        return torch.cat([z_v, z_a, z_p], dim=-1)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        physio: torch.Tensor,
        audio_available: bool = True,
    ) -> torch.Tensor:
        feats = self._frozen_features(video, audio, physio, audio_available)
        return self.probe(feats)


@dataclass
class FrozenEncoderResult:
    per_seed_per_kshot_auc: dict[int, dict[int, float]]
    mean_auc_by_kshot: dict[int, float]
    std_auc_by_kshot: dict[int, float]
    manifest_path: str


def train_probe_one_seed(
    cfg: dict[str, Any],
    seed: int,
    k_shot: int,
    train_episode_fn,
    eval_loso_fn,
    device: str = "cuda",
) -> tuple[float, dict[str, Any]]:
    """
    Trains the probe head for a single (seed, k_shot) setting and returns
    (LOSO AUC, hyperparams_used).

    `train_episode_fn` and `eval_loso_fn` are injected so this module does
    not depend directly on the neonatal data loaders / episode sampler —
    wire them up in scripts/train_baselines.py using
    src/data/episode_sampler.py and src/evaluation/loso.py, mirroring how
    the proposed model is trained/evaluated for a fair comparison.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe_cfg = cfg["probe_training"]
    model = FrozenEncoderProbe(
        probe_head=cfg["fusion"]["probe_head"],
        mlp_hidden_dim=cfg["fusion"].get("mlp_hidden_dim", 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.probe.parameters(),  # only the probe head has requires_grad=True
        lr=probe_cfg["optimizer"]["lr"],
        weight_decay=probe_cfg["optimizer"]["weight_decay"],
    )

    train_episode_fn(model, optimizer, k_shot=k_shot, seed=seed, epochs=probe_cfg["optimizer"]["epochs"])
    auc = eval_loso_fn(model, k_shot=k_shot, seed=seed)

    hyperparams_used = {
        "lr": probe_cfg["optimizer"]["lr"],
        "weight_decay": probe_cfg["optimizer"]["weight_decay"],
        "epochs": probe_cfg["optimizer"]["epochs"],
        "probe_head": cfg["fusion"]["probe_head"],
        "k_shot": k_shot,
    }
    return float(auc), hyperparams_used


def run_frozen_encoder_baseline(
    cfg: dict[str, Any],
    train_episode_fn,
    eval_loso_fn,
    output_dir: str,
    config_path: str = "configs/baseline_frozen_encoder.yaml",
    device: str = "cuda",
) -> FrozenEncoderResult:
    seeds: list[int] = cfg["seeds"]
    k_shots: list[int] = cfg["few_shot"]["k_shot"]

    manifest = RunManifest(
        baseline_name="frozen_encoder_linear_probe",
        config_path=config_path,
        resolved_config=cfg,
        seeds_requested=seeds,
    )

    per_seed_per_kshot: dict[int, dict[int, float]] = {s: {} for s in seeds}

    for seed in seeds:
        for k in k_shots:
            auc, hp = train_probe_one_seed(
                cfg, seed=seed, k_shot=k,
                train_episode_fn=train_episode_fn,
                eval_loso_fn=eval_loso_fn,
                device=device,
            )
            per_seed_per_kshot[seed][k] = auc
            manifest.record_trial(trial_config=hp, score=auc, seed=seed)
        manifest.record_final(seed=seed, hyperparams={"k_shot_results": per_seed_per_kshot[seed]})

    mean_by_k, std_by_k = {}, {}
    for k in k_shots:
        vals = np.array([per_seed_per_kshot[s][k] for s in seeds])
        mean_by_k[k] = float(vals.mean())
        std_by_k[k] = float(vals.std())

    manifest_path = manifest.finalize_and_write(output_dir)

    return FrozenEncoderResult(
        per_seed_per_kshot_auc=per_seed_per_kshot,
        mean_auc_by_kshot=mean_by_k,
        std_auc_by_kshot=std_by_k,
        manifest_path=str(manifest_path),
    )
