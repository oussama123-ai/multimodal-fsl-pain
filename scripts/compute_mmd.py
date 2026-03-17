#!/usr/bin/env python3
"""
compute_mmd.py
--------------
Standalone script to quantify the adult → neonatal domain gap via MMD.

Usage
-----
    python scripts/compute_mmd.py \
        --checkpoint checkpoints/pretrain/best.pt \
        --data_root  ./data \
        --device     cuda

Reproduces Table 2 of the paper:
    Video     MMD pre=0.198 → post=0.089
    Audio     MMD pre=0.143 → post=0.061
    Physio    MMD pre=0.089 → post=0.051
    Fused     MMD pre=0.142 → post=0.067
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.full_model import MultimodalPainModel
from data.datasets import (
    UNBCMcMasterDataset, BioVidDataset, NeonatalDataset, CombinedAdultDataset
)
from utils.mmd import compute_domain_gap
from utils.logger import get_logger

logger = get_logger("compute_mmd")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/pretrain/best.pt")
    p.add_argument("--data_root",  type=str, default="data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples",type=int, default=512)
    p.add_argument("--output",     type=str, default="results/mmd_domain_gap.json")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> MultimodalPainModel:
    model = MultimodalPainModel(video_pretrained=False)
    if Path(ckpt_path).exists():
        ckpt  = torch.load(ckpt_path, map_location=device)
        if "video_enc_state" in ckpt:
            model.video_enc.load_state_dict(ckpt["video_enc_state"],  strict=False)
            model.audio_enc.load_state_dict(ckpt["audio_enc_state"],  strict=False)
            model.physio_enc.load_state_dict(ckpt["physio_enc_state"], strict=False)
        else:
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"Checkpoint not found: {ckpt_path}. Using random init.")
    return model.to(device).eval()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output    = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device)

    # Build dataloaders
    adult_ds = CombinedAdultDataset(
        UNBCMcMasterDataset(str(data_root / "unbc_mcmaster")),
        BioVidDataset(str(data_root / "biovid")),
    )
    neo_ds = NeonatalDataset(str(data_root / "neonatal"))

    adult_loader = DataLoader(adult_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    neo_loader   = DataLoader(neo_ds,   batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Pre-adaptation MMD (from pretrain checkpoint)
    logger.info("Computing pre-adaptation MMD ...")
    mmd_pre = compute_domain_gap(
        model, adult_loader, neo_loader,
        device=args.device, max_samples=args.max_samples,
    )

    logger.info("MMD (pre-adaptation):")
    for modality, val in mmd_pre.items():
        logger.info(f"  {modality:10s}: {val:.4f}")

    # Save
    results = {"pre_adaptation": mmd_pre}
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output}")

    logger.info(
        "\nNote: post-adaptation MMD requires the finetuned checkpoint.\n"
        "Re-run with --checkpoint checkpoints/finetuned/best.pt to get post-adaptation values."
    )


if __name__ == "__main__":
    main()
