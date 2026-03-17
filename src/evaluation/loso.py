"""
loso.py
-------
Leave-One-Subject-Out (LOSO) cross-validation for the neonatal pilot cohort.

Implements the evaluation protocol described in Section 4.2 and
Table 3 (LOSO AUC for 34 neonates).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.full_model import MultimodalPainModel
from ..data.datasets import NeonatalDataset
from ..data.episode_sampler import EpisodeSampler
from .metrics import compute_all_metrics, bootstrap_ci, paired_wilcoxon

logger = logging.getLogger(__name__)


def run_loso(
    model:      MultimodalPainModel,
    dataset:    NeonatalDataset,
    k_shot:     int = 10,
    n_query:    int = 15,
    n_way:      int = 2,
    episodes_per_fold: int = 1_000,
    device:     str = "cuda",
    seed:       int = 42,
) -> dict[str, object]:
    """
    Run 34-fold LOSO cross-validation on the neonatal pilot cohort.

    For each fold:
      1. Hold out all recordings from subject S as test (query) set.
      2. Use remaining 33 subjects to build a support set.
      3. Classify test recordings via prototypical inference.
      4. Aggregate episode-level metrics to subject-level AUC.

    Returns
    -------
    dict with:
      subject_aucs : np.ndarray of shape (34,) — one AUC per subject
      mean_auc     : float
      std_auc      : float
      ci_lower/upper : bootstrap 95% CI
      all_metrics  : dict from compute_all_metrics (pooled across subjects)
      per_subject  : list of per-subject result dicts
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device).eval()

    subject_ids = dataset.get_subject_ids()
    logger.info(f"LOSO: {len(subject_ids)} subjects, k_shot={k_shot}")

    per_subject_results = []
    subject_aucs = []

    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold_idx, test_subject in enumerate(subject_ids):
        logger.info(f"  Fold {fold_idx+1}/{len(subject_ids)}: test_subject={test_subject}")

        # Build train/test split by subject
        train_subjects = [s for s in subject_ids if s != test_subject]
        train_dataset  = NeonatalDataset(
            dataset.root, subjects=train_subjects,
            transform=None, n_frames=16,
        )
        test_dataset = NeonatalDataset(
            dataset.root, subjects=[test_subject],
            transform=None, n_frames=16,
        )

        if len(test_dataset) == 0:
            logger.warning(f"  No samples for {test_subject}, skipping.")
            continue

        # Episodic evaluation
        train_sampler = EpisodeSampler(train_dataset, n_way=n_way,
                                       k_shot=k_shot, n_query=n_query)
        test_sampler  = EpisodeSampler(test_dataset,  n_way=n_way,
                                       k_shot=k_shot, n_query=n_query)

        ep_y_true, ep_y_pred, ep_y_prob = [], [], []

        with torch.no_grad():
            for ep_idx in range(episodes_per_fold):
                # Sample support from TRAIN subjects (leave test subject out)
                train_ep = train_sampler.sample_episode()
                test_ep  = test_sampler.sample_episode()

                mask = {"video": True, "audio": True, "physio": True}

                # Use train episode support + test episode query
                out = model(
                    train_ep.support_video.to(device),
                    train_ep.support_audio.to(device),
                    train_ep.support_physio.to(device),
                    train_ep.support_labels.to(device),
                    test_ep.query_video.to(device),
                    test_ep.query_audio.to(device),
                    test_ep.query_physio.to(device),
                    test_ep.query_labels.to(device),
                    modality_mask=mask,
                )

                probs = out["log_probs"].exp().cpu().numpy()  # (Q, N)
                preds = out["predictions"].cpu().numpy()
                y_q   = test_ep.query_labels.numpy()

                ep_y_true.extend(y_q.tolist())
                ep_y_pred.extend(preds.tolist())
                ep_y_prob.extend(probs.tolist())

        ep_y_true = np.array(ep_y_true)
        ep_y_pred = np.array(ep_y_pred)
        ep_y_prob = np.array(ep_y_prob)

        subj_metrics = compute_all_metrics(ep_y_true, ep_y_pred, ep_y_prob)
        subj_auc     = subj_metrics["auc_roc"]

        subject_aucs.append(subj_auc)
        per_subject_results.append({
            "subject":     test_subject,
            "auc":         subj_auc,
            "accuracy":    subj_metrics["accuracy"],
            "f1":          subj_metrics["f1_macro"],
            "n_samples":   len(ep_y_true),
        })

        all_y_true.extend(ep_y_true.tolist())
        all_y_pred.extend(ep_y_pred.tolist())
        all_y_prob.extend(ep_y_prob.tolist())

        logger.info(
            f"    AUC={subj_auc:.3f} | acc={subj_metrics['accuracy']*100:.1f}%"
        )

    # Aggregate
    subject_aucs = np.array(subject_aucs)
    mean_auc = float(subject_aucs.mean())
    std_auc  = float(subject_aucs.std())
    ci_lo, ci_hi = bootstrap_ci(subject_aucs, n_bootstrap=10_000)

    pooled_metrics = compute_all_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob),
    )

    results = {
        "subject_aucs":    subject_aucs,
        "mean_auc":        mean_auc,
        "std_auc":         std_auc,
        "ci_lower":        ci_lo,
        "ci_upper":        ci_hi,
        "min_auc":         float(subject_aucs.min()),
        "max_auc":         float(subject_aucs.max()),
        "all_metrics":     pooled_metrics,
        "per_subject":     per_subject_results,
    }

    logger.info(
        f"LOSO complete: mean AUC = {mean_auc:.3f} ± {std_auc:.3f} "
        f"[{ci_lo:.3f}, {ci_hi:.3f}]"
    )
    return results
