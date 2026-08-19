"""
src.baselines
--------------
Baseline methods compared against the proposed multimodal few-shot model:

  - classical_baselines.py   : SVM, Random Forest, Logistic Regression, MLP-from-scratch
  - frozen_encoder_baseline.py : lightweight frozen-encoder + linear/MLP probe baseline

Both modules read their hyperparameters, HP-search budget, and seed list
directly from configs/baselines.yaml and configs/baseline_frozen_encoder.yaml
and write a run manifest recording exactly what was used, so the numbers
reported in the paper are always traceable to a specific config + code
version. See scripts/train_baselines.py for the CLI entry point and
docs/REPRODUCIBILITY.md for the full protocol description.
"""

from .classical_baselines import run_classical_baseline
from .frozen_encoder_baseline import run_frozen_encoder_baseline
from .manifest import RunManifest

__all__ = [
    "run_classical_baseline",
    "run_frozen_encoder_baseline",
    "RunManifest",
]
