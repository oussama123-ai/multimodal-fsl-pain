#!/usr/bin/env python
"""
scripts/train_baselines.py
---------------------------
Runs every baseline declared in configs/baselines.yaml (classical baselines)
and configs/baseline_frozen_encoder.yaml (lightweight frozen-encoder probe),
across every configured seed, and writes:

  results/baselines/<baseline_name>_manifest.json   — exact hyperparameters,
      search trials evaluated, and seeds completed (see src/baselines/manifest.py)
  results/baselines/summary.csv                     — mean +/- std per baseline,
      ready to paste into the paper's comparison table

Usage
-----
    python scripts/train_baselines.py \
        --baselines-config configs/baselines.yaml \
        --frozen-config configs/baseline_frozen_encoder.yaml \
        --data-root ./data \
        --output-dir results/baselines

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from src.baselines import run_classical_baseline, run_frozen_encoder_baseline


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_pooled_features(data_root: str):
    raise NotImplementedError(
        "Wire this up to src/data/datasets.py:NeonatalDataset for your "
        "actual data root before running classical baselines."
    )


def _loso_scorer(model, X_by_subject, y_by_subject) -> float:
    """
    TODO: implement nested nested-LOSO scoring (fit on N-1 subjects,
    score on the held-out subject, average AUC across folds) matching
    src/evaluation/loso.py's protocol, for use during both hyperparameter
    search (inner_cv) and final scoring.
    """
    raise NotImplementedError(
        "Implement LOSO scoring for classical (scikit-learn) models here."
    )


def _make_episode_and_eval_fns(data_root: str):
    """
    TODO: return (train_episode_fn, eval_loso_fn) callables matching the
    signatures expected by src/baselines/frozen_encoder_baseline.py,
    built on top of src/data/episode_sampler.py:EpisodeSampler and
    src/evaluation/loso.py:run_loso so the frozen-encoder baseline is
    evaluated identically to the proposed model.
    """
    raise NotImplementedError(
        "Wire episode sampling + LOSO evaluation for the frozen-encoder "
        "probe before running this baseline."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-config", default="configs/baselines.yaml")
    parser.add_argument("--frozen-config", default="configs/baseline_frozen_encoder.yaml")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of baseline names to run (default: all).",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_rows = []

    baselines_cfg = _load_yaml(args.baselines_config)
    common_cfg = baselines_cfg["common"]

    classical_names = [
        k for k in baselines_cfg
        if k not in ("common", "frozen_encoder_linear_probe")
    ]
    if args.only:
        classical_names = [n for n in classical_names if n in args.only]

    if classical_names:
        X_by_subject, y_by_subject = _extract_pooled_features(args.data_root)
        for name in classical_names:
            result = run_classical_baseline(
                baseline_name=name,
                baseline_cfg=baselines_cfg[name],
                common_cfg=common_cfg,
                X_by_subject=X_by_subject,
                y_by_subject=y_by_subject,
                loso_scorer=_loso_scorer,
                output_dir=args.output_dir,
                config_path=args.baselines_config,
            )
            summary_rows.append({
                "baseline": name,
                "mean_auc": f"{result.mean_auc:.3f}",
                "std_auc": f"{result.std_auc:.3f}",
                "n_seeds": len(result.per_seed_auc),
                "manifest": result.manifest_path,
            })
            print(f"[{name}] AUC = {result.mean_auc:.3f} ± {result.std_auc:.3f}  "
                  f"(n_seeds={len(result.per_seed_auc)})  -> {result.manifest_path}")

    if not args.only or "frozen_encoder_linear_probe" in args.only:
        frozen_cfg = _load_yaml(args.frozen_config)
        train_fn, eval_fn = _make_episode_and_eval_fns(args.data_root)
        result = run_frozen_encoder_baseline(
            cfg=frozen_cfg,
            train_episode_fn=train_fn,
            eval_loso_fn=eval_fn,
            output_dir=args.output_dir,
            config_path=args.frozen_config,
        )
        for k, mean_auc in result.mean_auc_by_kshot.items():
            summary_rows.append({
                "baseline": f"frozen_encoder_linear_probe (k={k})",
                "mean_auc": f"{mean_auc:.3f}",
                "std_auc": f"{result.std_auc_by_kshot[k]:.3f}",
                "n_seeds": len(frozen_cfg["seeds"]),
                "manifest": result.manifest_path,
            })
            print(f"[frozen_encoder k={k}] AUC = {mean_auc:.3f} ± {result.std_auc_by_kshot[k]:.3f}")

    summary_path = Path(args.output_dir) / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["baseline", "mean_auc", "std_auc", "n_seeds", "manifest"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
