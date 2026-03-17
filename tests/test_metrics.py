"""
test_metrics.py
---------------
Unit tests for evaluation metrics.
Run with: pytest tests/test_metrics.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest


class TestMetrics:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.y_true = rng.integers(0, 2, size=200)
        probs_raw   = rng.random((200, 2))
        self.y_prob = probs_raw / probs_raw.sum(axis=1, keepdims=True)
        self.y_pred = self.y_prob.argmax(axis=1)

    def test_accuracy_range(self):
        from evaluation.metrics import accuracy
        acc = accuracy(self.y_true, self.y_pred)
        assert 0.0 <= acc <= 1.0

    def test_ece_range(self):
        from evaluation.metrics import expected_calibration_error
        ece = expected_calibration_error(self.y_true, self.y_prob)
        assert 0.0 <= ece <= 1.0

    def test_auc_range(self):
        from evaluation.metrics import auc_roc
        auc = auc_roc(self.y_true, self.y_prob)
        assert 0.0 <= auc <= 1.0

    def test_compute_all_returns_all_keys(self):
        from evaluation.metrics import compute_all_metrics
        result = compute_all_metrics(self.y_true, self.y_pred, self.y_prob)
        expected_keys = [
            "accuracy", "balanced_acc", "f1_macro", "f1_weighted",
            "auc_roc", "auc_pr", "cohen_kappa", "brier_score",
            "ece", "sens_at_95spec",
        ]
        for k in expected_keys:
            assert k in result, f"Missing metric: {k}"

    def test_bootstrap_ci_bounds(self):
        from evaluation.metrics import bootstrap_ci
        values = np.random.default_rng(0).random(50)
        lo, hi = bootstrap_ci(values, n_bootstrap=1000)
        assert lo < hi
        assert lo >= 0.0 and hi <= 1.0


class TestMMD:
    def test_same_distribution_near_zero(self):
        import torch
        from utils.mmd import compute_mmd
        X = torch.randn(100, 64)
        Y = X + torch.randn_like(X) * 0.01   # nearly identical
        mmd = compute_mmd(X, Y)
        assert mmd < 0.1, f"MMD too large for near-identical distributions: {mmd}"

    def test_different_distribution_positive(self):
        import torch
        from utils.mmd import compute_mmd
        X = torch.randn(100, 64)
        Y = torch.randn(100, 64) + 5.0     # very different mean
        mmd = compute_mmd(X, Y)
        assert mmd > 0.5, f"MMD too small for very different distributions: {mmd}"

    def test_mmd_non_negative(self):
        import torch
        from utils.mmd import compute_mmd
        X = torch.randn(50, 32)
        Y = torch.randn(50, 32)
        mmd = compute_mmd(X, Y)
        assert mmd >= 0.0
