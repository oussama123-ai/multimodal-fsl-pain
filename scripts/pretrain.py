#!/usr/bin/env python3
"""
pretrain.py
-----------
Phase 1: Self-Supervised Contrastive Pretraining (Algorithm 1).

Usage
-----
    python scripts/pretrain.py \
        --config configs/pretrain.yaml \
        --data_root ./data \
        --output_dir ./checkpoints/pretrain
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.encoders import VideoEncoder, AudioEncoder, PhysioEncoder
from data.datasets import UNBCMcMasterDataset, BioVidDataset, CombinedAdultDataset
from data.augmentations import MultimodalAugmentation
from training.contrastive import ContrastiveTrainer
from utils.logger import get_logger, WandbLogger
from utils.config import PretrainConfig, DataConfig

logger = get_logger("pretrain")


# ---------------------------------------------------------------------------
# Collate function: produce two augmented views per sample
# ---------------------------------------------------------------------------

class TwoViewCollate:
    """
    Wraps a MultimodalAugmentation and produces (view1_batch, view2_batch)
    from a list of dataset items.
    """

    def __init__(self):
        self.aug = MultimodalAugmentation()

    def __call__(self, batch: list[dict]) -> tuple[dict, dict, bool]:
        view1_v, view1_a, view1_p = [], [], []
        view2_v, view2_a, view2_p = [], [], []
        audio_available = all(item["audio_available"] for item in batch)

        for item in batch:
            av = item["audio_available"]
            (v1, a1, p1), (v2, a2, p2) = self.aug.generate_two_views(
                item["video"], item["audio"], item["physio"],
                audio_available=av,
            )
            view1_v.append(v1); view1_a.append(a1); view1_p.append(p1)
            view2_v.append(v2); view2_a.append(a2); view2_p.append(p2)

        def mk(lst):
            return torch.stack(lst)

        view1 = {"video": mk(view1_v), "audio": mk(view1_a), "physio": mk(view1_p)}
        view2 = {"video": mk(view2_v), "audio": mk(view2_a), "physio": mk(view2_p)}
        return view1, view2, audio_available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SSL Contrastive Pretraining")
    p.add_argument("--data_root",  type=str, default="data")
    p.add_argument("--output_dir", type=str, default="checkpoints/pretrain")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--temperature",type=float, default=0.1)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--wandb",      action="store_true")
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # 1. Datasets
    # ------------------------------------------------------------------ #
    data_root = Path(args.data_root)
    unbc   = UNBCMcMasterDataset(str(data_root / "unbc_mcmaster"))
    biovid = BioVidDataset(str(data_root / "biovid"), binary=True)
    combined = CombinedAdultDataset(unbc, biovid)

    logger.info(f"Dataset sizes — UNBC: {len(unbc)}, BioVid: {len(biovid)}, "
                f"Combined: {len(combined)}")

    collate_fn = TwoViewCollate()
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------ #
    # 2. Models
    # ------------------------------------------------------------------ #
    video_enc  = VideoEncoder(pretrained=True).to(device)
    audio_enc  = AudioEncoder().to(device)
    physio_enc = PhysioEncoder().to(device)

    trainer = ContrastiveTrainer(
        video_enc, audio_enc, physio_enc,
        temperature=args.temperature,
    ).to(device)

    # ------------------------------------------------------------------ #
    # 3. Optimizer + LR schedule
    # ------------------------------------------------------------------ #
    optimizer = optim.AdamW(
        trainer.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------ #
    # 4. W&B (optional)
    # ------------------------------------------------------------------ #
    wlog = WandbLogger(
        project="multimodal-fsl-pain",
        name=f"pretrain_ep{args.epochs}_tau{args.temperature}",
        config=vars(args),
        enabled=args.wandb,
    )

    # ------------------------------------------------------------------ #
    # 5. Training loop
    # ------------------------------------------------------------------ #
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        trainer.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (view1, view2, audio_available) in enumerate(loader):
            # Move to device
            view1 = {k: v.to(device) for k, v in view1.items()}
            view2 = {k: v.to(device) for k, v in view2.items()}

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                loss = trainer(view1, view2, audio_available=audio_available)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )
        wlog.log({"pretrain/loss": avg_loss, "pretrain/lr": scheduler.get_last_lr()[0]},
                 step=epoch)

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "video_enc_state":  video_enc.state_dict(),
                "audio_enc_state":  audio_enc.state_dict(),
                "physio_enc_state": physio_enc.state_dict(),
                "loss":             best_loss,
                "args":             vars(args),
            }, output_dir / "best.pt")
            logger.info(f"  ✓ Saved checkpoint (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "video_enc_state":  video_enc.state_dict(),
                "audio_enc_state":  audio_enc.state_dict(),
                "physio_enc_state": physio_enc.state_dict(),
            }, output_dir / f"epoch_{epoch:03d}.pt")

    wlog.finish()
    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
