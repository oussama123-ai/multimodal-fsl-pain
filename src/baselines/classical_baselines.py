"""
classical_baselines.py
-----------------------
Non-deep / classical baselines: SVM (RBF), Random Forest, Logistic
Regression, and an MLP trained from scratch on pooled multimodal features.

Every baseline is trained under the SAME leave-one-subject-out (LOSO)
protocol as the proposed model (src/evaluation/loso.py), over the SAME
pooled feature representation, so comparisons in Table 2 are apples-to-
apples.

Hyperparameters, HP-search method/budget, and seeds are all read from
configs/baselines.yaml — nothing here is hard-coded, so the manifest
written per run is guaranteed to match what this file actually executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .manifest import RunManifest

_MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "svm_rbf": lambda **kw: SVC(kernel="rbf", probability=True, **kw),
    "random_forest": lambda **kw: RandomForestClassifier(**kw),
    "logistic_regression": lambda **kw: LogisticRegression(**kw),
    "mlp_scratch": lambda **kw: MLPClassifier(**kw),
}


@dataclass
class BaselineResult:
    baseline_name: str
    per_seed_auc: dict[int, float]
    mean_auc: float
    std_auc: float
    manifest_path: str


def _sample_configs(search_cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Materialize the exact list of hyperparameter configs to evaluate,
    honoring the declared search budget so it is identical to what gets
    logged in the manifest."""
    space = search_cfg["search_space"]
    method = search_cfg.get("method", "grid_search")

    if method == "grid_search":
        configs = list(ParameterGrid(space))
    elif method == "random_search":
        n_trials = search_cfg.get("n_trials") or search_cfg.get("budget")
        if n_trials is None:
            raise ValueError(
                "random_search requires 'n_trials' (or 'budget') in "
                "configs/baselines.yaml for this baseline."
            )
        configs = list(
            ParameterSampler(space, n_iter=int(n_trials), random_state=seed)
        )
    else:
        raise ValueError(f"Unknown hyperparameter_search.method: {method}")
    return configs


def run_classical_baseline(
    baseline_name: str,
    baseline_cfg: dict[str, Any],
    common_cfg: dict[str, Any],
    X_by_subject: dict[str, np.ndarray],
    y_by_subject: dict[str, np.ndarray],
    loso_scorer: Callable[[Any, dict[str, np.ndarray], dict[str, np.ndarray]], float],
    output_dir: str,
    config_path: str = "configs/baselines.yaml",
) -> BaselineResult:
    """
    Parameters
    ----------
    baseline_name : one of the keys under configs/baselines.yaml (e.g. 'svm_rbf')
    baseline_cfg  : the resolved sub-dict for this baseline
    common_cfg    : configs/baselines.yaml:common (seeds, eval_protocol, ...)
    X_by_subject, y_by_subject : pooled features / labels keyed by subject id,
        matching the subject-level LOSO split used for the proposed model.
    loso_scorer   : callable(model, X_by_subject, y_by_subject) -> mean AUC,
        implementing the nested-LOSO scoring described in
        configs/baselines.yaml (hyperparameter_search.inner_cv: loso).
        Left injectable so this module has no hard dependency on the
        neonatal data loaders.
    """
    if baseline_name not in _MODEL_FACTORIES:
        raise ValueError(f"No classical-baseline factory registered for '{baseline_name}'")

    factory = _MODEL_FACTORIES[baseline_name]
    seeds: list[int] = common_cfg["seeds"]
    assert len(seeds) == common_cfg["n_seeds"], (
        "configs/baselines.yaml: common.n_seeds does not match len(common.seeds)"
    )

    manifest = RunManifest(
        baseline_name=baseline_name,
        config_path=config_path,
        resolved_config=baseline_cfg,
        seeds_requested=seeds,
    )

    per_seed_auc: dict[int, float] = {}

    for seed in seeds:
        np.random.seed(seed)
        candidate_configs = _sample_configs(baseline_cfg["hyperparameter_search"], seed)

        best_score = -np.inf
        best_cfg: dict[str, Any] | None = None

        for cand in candidate_configs:
            model = factory(**cand, random_state=seed) if "random_state" not in cand else factory(**cand)
            score = loso_scorer(model, X_by_subject, y_by_subject)
            manifest.record_trial(trial_config=cand, score=float(score), seed=seed)
            if score > best_score:
                best_score, best_cfg = score, cand

        assert best_cfg is not None, f"No candidate configs evaluated for {baseline_name}"
        final_model = factory(**best_cfg, random_state=seed) if "random_state" not in best_cfg else factory(**best_cfg)
        final_auc = loso_scorer(final_model, X_by_subject, y_by_subject)

        manifest.record_final(seed=seed, hyperparams=best_cfg)
        per_seed_auc[seed] = float(final_auc)

    aucs = np.array(list(per_seed_auc.values()))
    manifest_path = manifest.finalize_and_write(output_dir)

    return BaselineResult(
        baseline_name=baseline_name,
        per_seed_auc=per_seed_auc,
        mean_auc=float(aucs.mean()),
        std_auc=float(aucs.std()),
        manifest_path=str(manifest_path),
    )
