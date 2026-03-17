"""
metrics.py
----------
All evaluation metrics used in the paper.

  accuracy, balanced_accuracy, f1_macro, f1_weighted,
  auc_roc, auc_pr,
  cohen_kappa,
  expected_calibration_error (ECE),
  brier_score,
  sensitivity_at_specificity,
  compute_all_metrics
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    brier_score_loss,
    confusion_matrix,
)
from scipy.stats import wilcoxon
from scipy import stats


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Parameters
    ----------
    y_prob : (N,) for binary or (N, C) for multi-class (one-vs-rest)
    """
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                   average="macro"))
    except Exception:
        return float("nan")


def auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return float("nan")


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted average of |accuracy - confidence| across confidence bins.

    Parameters
    ----------
    y_prob : (N,) — predicted probability for the positive class
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = accuracy_score(y_true[mask], (y_prob[mask] >= 0.5).astype(int))
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)

    return float(ece)


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    """
    Sensitivity (recall) when specificity is fixed at `target_specificity`.
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    thresholds = np.unique(y_prob)
    best_sens = 0.0

    for thr in sorted(thresholds, reverse=True):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-8)
        sens = tp / (tp + fn + 1e-8)
        if spec >= target_specificity:
            best_sens = sens
            break

    return float(best_sens)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """
    Compute the full metric suite reported in Table 4 and Table 6.
    """
    return {
        "accuracy":         accuracy(y_true, y_pred),
        "balanced_acc":     balanced_accuracy(y_true, y_pred),
        "f1_macro":         f1_macro(y_true, y_pred),
        "f1_weighted":      f1_weighted(y_true, y_pred),
        "auc_roc":          auc_roc(y_true, y_prob),
        "auc_pr":           auc_pr(y_true, y_prob),
        "cohen_kappa":      cohen_kappa(y_true, y_pred),
        "brier_score":      brier_score(y_true, y_prob),
        "ece":              expected_calibration_error(y_true, y_prob),
        "sens_at_95spec":   sensitivity_at_specificity(y_true, y_prob, 0.95),
    }


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def paired_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test.

    Returns
    -------
    statistic, p_value
    """
    stat, p = wilcoxon(scores_a, scores_b, alternative="two-sided")
    return float(stat), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    return float((a.mean() - b.mean()) / (pooled_std + 1e-8))


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval (percentile method).

    Returns
    -------
    (lower_bound, upper_bound)
    """
    rng  = np.random.default_rng(seed=42)
    boot = [rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(n_bootstrap)]
    boot = np.array(boot)
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )
