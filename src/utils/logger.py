"""
logger.py
---------
Unified logging: Python logging + optional Weights & Biases integration.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class WandbLogger:
    """
    Thin wrapper around W&B for optional experiment tracking.
    Falls back to a no-op if wandb is not installed or WANDB_MODE=disabled.
    """

    def __init__(
        self,
        project: str = "multimodal-fsl-pain",
        name:    str | None = None,
        config:  dict | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        if not enabled:
            return
        try:
            import wandb
            wandb.init(project=project, name=name, config=config or {})
            self._wandb = wandb
        except Exception:
            self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None):
        if not self.enabled:
            return
        try:
            self._wandb.log(metrics, step=step)
        except Exception:
            pass

    def finish(self):
        if not self.enabled:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass
