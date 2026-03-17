#!/usr/bin/env python3
"""
finetune.py
-----------
Phase 2: Few-Shot Fine-Tuning via Episodic Prototypical Network Training.

Usage
-----
    python scripts/finetune.py \
        --pretrained_ckpt checkpoints/pretrain/best.pt \
        --data_root ./data \
        --k_shot 10 \
        --output_dir checkpoints/finetuned
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.full_model import MultimodalPainModel
from data.datasets import BioVidDataset, NeonatalDataset
from data.episode_sampler import EpisodeSampler
from training.fewshot_trainer import FewShotTrainer
from utils.logger import get_logger

logger = get_logger("finetune")


def parse_args():
    p = argparse.ArgumentParser(description="Few-Shot Fine-Tuning")
    p.add_argument("--pretrained_ckpt", type=str,
                   default="checkpoints/pretrain/best.pt")
    p.add_argument("--data_root",  type=str, default="data")
    p.add_argument("--domain",     type=str, default="neonatal",
                   choices=["adult", "neonatal"])
    p.add_argument("--k_shot",     type=int, default=10)
    p.add_argument("--n_way",      type=int, default=2)
    p.add_argument("--n_query",    type=int, default=15)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--ep_per_epoch", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="checkpoints/finetuned")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_splits(data_root: Path, domain: str, seed: int = 42):
    """
    Build subject-wise 60/20/20 train/val/test splits.
    """
    if domain == "neonatal":
        full_ds = NeonatalDataset(str(data_root / "neonatal"))
        subjects = full_ds.get_subject_ids()
    else:
        full_ds = BioVidDataset(str(data_root / "biovid"), binary=True)
        subjects = sorted({r["subject"] for r in full_ds.records})

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects := list(subjects))

    n = len(subjects)
    n_train = int(0.6 * n)
    n_val   = int(0.2 * n)

    train_subj = subjects[:n_train]
    val_subj   = subjects[n_train:n_train + n_val]
    test_subj  = subjects[n_train + n_val:]

    logger.info(
        f"Splits — train: {len(train_subj)}, val: {len(val_subj)}, "
        f"test: {len(test_subj)}"
    )

    if domain == "neonatal":
        train_ds = NeonatalDataset(str(data_root / "neonatal"), subjects=train_subj)
        val_ds   = NeonatalDataset(str(data_root / "neonatal"), subjects=val_subj)
        test_ds  = NeonatalDataset(str(data_root / "neonatal"), subjects=test_subj)
    else:
        # For adult domain we filter BioVid by subject attribute
        def _subset(subjs):
            class FilteredBioVid(BioVidDataset):
                def __init__(self_, *a, **kw):
                    super().__init__(*a, **kw)
                    self_.records = [r for r in self_.records
                                     if r["subject"] in subjs]
            return FilteredBioVid(str(data_root / "biovid"), binary=True)
        train_ds = _subset(set(train_subj))
        val_ds   = _subset(set(val_subj))
        test_ds  = _subset(set(test_subj))

    return train_ds, val_ds, test_ds


def main():
    args = parse_args()
    set_seed(args.seed)
    data_root = Path(args.data_root)

    # ------------------------------------------------------------------ #
    # 1. Build model and load pretrained encoders
    # ------------------------------------------------------------------ #
    model = MultimodalPainModel(video_pretrained=False)

    if Path(args.pretrained_ckpt).exists():
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
        model.video_enc.load_state_dict(ckpt["video_enc_state"],  strict=False)
        model.audio_enc.load_state_dict(ckpt["audio_enc_state"],  strict=False)
        model.physio_enc.load_state_dict(ckpt["physio_enc_state"], strict=False)
        logger.info(f"Loaded pretrained encoders from {args.pretrained_ckpt}")
    else:
        logger.warning(
            f"Pretrained checkpoint not found at {args.pretrained_ckpt}. "
            "Training from scratch."
        )

    # ------------------------------------------------------------------ #
    # 2. Data
    # ------------------------------------------------------------------ #
    train_ds, val_ds, test_ds = build_splits(data_root, args.domain, args.seed)

    train_sampler = EpisodeSampler(
        train_ds, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query
    )
    val_sampler = EpisodeSampler(
        val_ds, n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query
    )

    # ------------------------------------------------------------------ #
    # 3. Train
    # ------------------------------------------------------------------ #
    trainer = FewShotTrainer(
        model=model,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        episodes_per_epoch=args.ep_per_epoch,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=args.device,
    )
    trained_model = trainer.fit()

    logger.info(f"Fine-tuning complete. Model saved to {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
