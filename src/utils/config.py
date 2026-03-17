"""
config.py
---------
Typed configuration dataclasses for all training phases.
Used with Hydra or standalone via OmegaConf.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    unbc_root:   str = "data/unbc_mcmaster"
    biovid_root: str = "data/biovid"
    neo_root:    str = "data/neonatal"
    n_frames:    int = 16
    binary:      bool = True      # collapse BioVid 4-class → binary


@dataclass
class ModelConfig:
    video_pretrained:  bool = True
    n_fusion_layers:   int  = 4
    dropout:           float = 0.1
    proj_dim:          int  = 128
    distance:          str  = "euclidean"   # 'euclidean' | 'cosine'


@dataclass
class PretrainConfig:
    epochs:          int   = 100
    batch_size:      int   = 16
    lr:              float = 1e-4
    weight_decay:    float = 1e-4
    temperature:     float = 0.1
    lambda_intra:    float = 0.3
    lambda_cross:    float = 0.5
    lambda_momentum: float = 0.2
    momentum:        float = 0.99
    warmup_epochs:   int   = 10
    grad_clip:       float = 1.0
    output_dir:      str   = "checkpoints/pretrain"
    use_amp:         bool  = True
    seed:            int   = 42


@dataclass
class FinetuneConfig:
    epochs:             int   = 50
    episodes_per_epoch: int   = 1000
    val_episodes:       int   = 500
    k_shot:             int   = 10
    n_way:              int   = 2
    n_query:            int   = 15
    lr_pretrained:      float = 1e-5
    lr_new:             float = 1e-4
    weight_decay:       float = 1e-4
    patience:           int   = 15
    grad_clip:          float = 1.0
    output_dir:         str   = "checkpoints/finetuned"
    pretrained_ckpt:    str   = "checkpoints/pretrain/best.pt"
    use_amp:            bool  = True
    seed:               int   = 42


@dataclass
class EvalConfig:
    checkpoint:      str  = "checkpoints/finetuned/best.pt"
    domain:          str  = "neonatal"     # 'adult' | 'neonatal'
    eval_mode:       str  = "loso"         # 'loso' | 'kfold'
    k_shot:          int  = 10
    n_way:           int  = 2
    n_query:         int  = 15
    episodes:        int  = 1000
    mc_passes:       int  = 50
    output_dir:      str  = "results"
    device:          str  = "cuda"
    seed:            int  = 42


@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval:     EvalConfig     = field(default_factory=EvalConfig)
    wandb:    bool = False
    wandb_project: str = "multimodal-fsl-pain"
