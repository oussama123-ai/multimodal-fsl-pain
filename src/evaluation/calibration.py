"""
calibration.py
--------------
Calibration analysis: reliability diagrams and ECE computation
(Figure 5 in the paper, Table 8).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title:  str = "Reliability Diagram",
    save_path: str | None = None,
) -> np.ndarray:
    """
    Plot reliability (calibration) diagram.

    Parameters
    ----------
    y_prob : (N,) — predicted probability for positive class

    Returns
    -------
    bin_accs : (n_bins,) — per-bin accuracy
    """
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_confs.append(y_prob[mask].mean())
        bin_accs.append((y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean())
        bin_counts.append(mask.sum())

    bin_confs = np.array(bin_confs)
    bin_accs  = np.array(bin_accs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7,
           color="#2ecc71", label="Our model")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title); ax.legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    return bin_accs


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | None = None,
):
    """Plot confidence histogram for correct vs incorrect predictions."""
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    y_pred   = (y_prob >= 0.5).astype(int)
    correct  = y_prob[y_pred == y_true]
    incorrect = y_prob[y_pred != y_true]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(correct,   bins=20, alpha=0.7, color="#2ecc71", label="Correct")
    ax.hist(incorrect, bins=20, alpha=0.7, color="#e74c3c", label="Incorrect")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
