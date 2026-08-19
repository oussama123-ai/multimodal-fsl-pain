"""
manifest.py
-----------
Writes a timestamped, machine-generated record of exactly what was run for
a baseline experiment: the resolved config, the optimizer hyperparameters
actually used, the hyperparameter-search trials evaluated (not just the
budget requested), and the random seeds completed.

This directly addresses the reviewer request for exact optimizer
hyperparameters / search budget / seed counts: rather than relying on a
hand-maintained table that can drift from what was actually executed, every
run of scripts/train_baselines.py emits one of these next to its results,
and docs/REPRODUCIBILITY.md points readers here as the source of truth.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return None


@dataclass
class RunManifest:
    baseline_name: str
    config_path: str
    resolved_config: dict[str, Any]

    seeds_requested: list[int] = field(default_factory=list)
    seeds_completed: list[int] = field(default_factory=list)

    # One entry per hyperparameter configuration actually evaluated during
    # the search (not just the search space definition) — this is the
    # ground truth for "search budget".
    search_trials: list[dict[str, Any]] = field(default_factory=list)

    # Hyperparameters of the FINAL model selected after search, per seed.
    final_hyperparams_per_seed: dict[int, dict[str, Any]] = field(default_factory=dict)

    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None

    def record_trial(self, trial_config: dict[str, Any], score: float, seed: int) -> None:
        self.search_trials.append({"config": trial_config, "score": score, "seed": seed})

    def record_final(self, seed: int, hyperparams: dict[str, Any]) -> None:
        self.final_hyperparams_per_seed[seed] = hyperparams
        if seed not in self.seeds_completed:
            self.seeds_completed.append(seed)

    def finalize_and_write(self, output_dir: str | Path) -> Path:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "baseline_name": self.baseline_name,
            "config_path": self.config_path,
            "resolved_config": self.resolved_config,
            "seeds_requested": self.seeds_requested,
            "seeds_completed": self.seeds_completed,
            "n_seeds_requested": len(self.seeds_requested),
            "n_seeds_completed": len(self.seeds_completed),
            "search_budget_actual_trials": len(self.search_trials),
            "search_trials": self.search_trials,
            "final_hyperparams_per_seed": self.final_hyperparams_per_seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
        }

        if len(self.seeds_completed) < len(self.seeds_requested):
            payload["WARNING"] = (
                f"Only {len(self.seeds_completed)}/{len(self.seeds_requested)} "
                "requested seeds completed. Do not report an aggregate mean/std "
                "from this manifest until all seeds have run."
            )

        out_path = out_dir / f"{self.baseline_name}_manifest.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        return out_path
