#!/usr/bin/env python3
"""
evaluate.py
-----------
Full evaluation pipeline:
  - LOSO cross-validation (neonatal pilot cohort)
  - K-fold cross-validation (adult domain)
  - Modality ablation
  - MMD domain gap computation
  - Clinical calibration analysis

Usage
-----
    python scripts/evaluate.py \
        --checkpoint checkpoints/finetuned/best.pt \
        --data_root ./data \
        --eval_mode loso \
        --domain neonatal \
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.full_model import MultimodalPainModel
from data.datasets import NeonatalDataset, BioVidDataset, UNBCMcMasterDataset, CombinedAdultDataset
from data.episode_sampler import EpisodeSampler
from evaluation.loso import run_loso
from evaluation.metrics import compute_all_metrics, bootstrap_ci
from utils.logger import get_logger
from utils.mmd import compute_domain_gap
from torch.utils.data import DataLoader

logger = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluation pipeline")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_root",   type=str, default="data")
    p.add_argument("--eval_mode",   type=str, default="loso",
                   choices=["loso", "kfold", "ablation", "mmd"])
    p.add_argument("--domain",      type=str, default="neonatal",
                   choices=["adult", "neonatal"])
    p.add_argument("--k_shot",      type=int, default=10)
    p.add_argument("--n_way",       type=int, default=2)
    p.add_argument("--n_query",     type=int, default=15)
    p.add_argument("--episodes",    type=int, default=1000)
    p.add_argument("--output_dir",  type=str, default="results")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> MultimodalPainModel:
    model = MultimodalPainModel(video_pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    logger.info(f"Loaded model from {ckpt_path}")
    return model


def run_ablation(
    model: MultimodalPainModel,
    dataset,
    k_shot: int,
    n_way: int,
    n_query: int,
    episodes: int,
    device: str,
) -> dict[str, dict]:
    """
    Systematic modality ablation matching Table 5 / Table 7.
    """
    configs = {
        "video_only":      {"video": True,  "audio": False, "physio": False},
        "audio_only":      {"video": False, "audio": True,  "physio": False},
        "physio_only":     {"video": False, "audio": False, "physio": True},
        "video_audio":     {"video": True,  "audio": True,  "physio": False},
        "video_physio":    {"video": True,  "audio": False, "physio": True},
        "all_modalities":  {"video": True,  "audio": True,  "physio": True},
    }

    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot, n_query=n_query)
    results = {}

    for name, mask in configs.items():
        y_true_all, y_pred_all, y_prob_all = [], [], []

        with torch.no_grad():
            ep_iter = iter(sampler)
            for _ in range(episodes):
                ep = next(ep_iter)
                out = model(
                    ep.support_video.to(device),
                    ep.support_audio.to(device),
                    ep.support_physio.to(device),
                    ep.support_labels.to(device),
                    ep.query_video.to(device),
                    ep.query_audio.to(device),
                    ep.query_physio.to(device),
                    ep.query_labels.to(device),
                    modality_mask=mask,
                )
                probs = out["log_probs"].exp().cpu().numpy()
                preds = out["predictions"].cpu().numpy()
                y_q   = ep.query_labels.numpy()
                y_true_all.extend(y_q.tolist())
                y_pred_all.extend(preds.tolist())
                y_prob_all.extend(probs.tolist())

        metrics = compute_all_metrics(
            np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)
        )
        results[name] = metrics
        logger.info(
            f"  {name:20s} | acc={metrics['accuracy']*100:.2f}% | "
            f"AUC={metrics['auc_roc']:.3f}"
        )

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device)
    data_root = Path(args.data_root)

    # ------------------------------------------------------------------ #
    # LOSO evaluation (neonatal pilot)
    # ------------------------------------------------------------------ #
    if args.eval_mode == "loso":
        dataset = NeonatalDataset(str(data_root / "neonatal"))
        logger.info(f"Running LOSO on {len(dataset.get_subject_ids())} subjects")

        results = run_loso(
            model=model,
            dataset=dataset,
            k_shot=args.k_shot,
            n_way=args.n_way,
            n_query=args.n_query,
            episodes_per_fold=args.episodes,
            device=args.device,
            seed=args.seed,
        )

        summary = {
            "eval_mode":    "loso",
            "domain":       "neonatal",
            "k_shot":       args.k_shot,
            "mean_auc":     results["mean_auc"],
            "std_auc":      results["std_auc"],
            "ci":           [results["ci_lower"], results["ci_upper"]],
            "min_auc":      results["min_auc"],
            "max_auc":      results["max_auc"],
            "pooled_metrics": results["all_metrics"],
            "per_subject":  results["per_subject"],
        }

        out_path = output_dir / f"loso_{args.k_shot}shot.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"\nLOSO Results ({args.k_shot}-shot):\n"
            f"  Mean AUC : {results['mean_auc']:.3f} ± {results['std_auc']:.3f}\n"
            f"  95% CI   : [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]\n"
            f"  Accuracy : {results['all_metrics']['accuracy']*100:.1f}%\n"
            f"  F1 macro : {results['all_metrics']['f1_macro']:.3f}\n"
            f"Results saved to {out_path}"
        )

    # ------------------------------------------------------------------ #
    # Modality ablation
    # ------------------------------------------------------------------ #
    elif args.eval_mode == "ablation":
        if args.domain == "neonatal":
            dataset = NeonatalDataset(str(data_root / "neonatal"))
        else:
            dataset = BioVidDataset(str(data_root / "biovid"), binary=True)

        logger.info(f"Running modality ablation ({args.domain} domain)")
        results = run_ablation(
            model, dataset,
            k_shot=args.k_shot, n_way=args.n_way, n_query=args.n_query,
            episodes=args.episodes, device=args.device,
        )
        out_path = output_dir / f"ablation_{args.domain}_{args.k_shot}shot.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Ablation results saved to {out_path}")

    # ------------------------------------------------------------------ #
    # MMD domain gap
    # ------------------------------------------------------------------ #
    elif args.eval_mode == "mmd":
        logger.info("Computing MMD domain gap (adult → neonatal)")
        adult_ds  = CombinedAdultDataset(
            UNBCMcMasterDataset(str(data_root / "unbc_mcmaster")),
            BioVidDataset(str(data_root / "biovid")),
        )
        neo_ds = NeonatalDataset(str(data_root / "neonatal"))

        adult_loader = DataLoader(adult_ds, batch_size=32, shuffle=True)
        neo_loader   = DataLoader(neo_ds,   batch_size=32, shuffle=True)

        mmd_values = compute_domain_gap(model, adult_loader, neo_loader,
                                        device=args.device)
        logger.info("MMD domain gap:")
        for k, v in mmd_values.items():
            logger.info(f"  {k:10s}: {v:.4f}")

        out_path = output_dir / "mmd_domain_gap.json"
        with open(out_path, "w") as f:
            json.dump(mmd_values, f, indent=2)
        logger.info(f"MMD saved to {out_path}")


if __name__ == "__main__":
    main()
