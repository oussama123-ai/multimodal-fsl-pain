"""
fewshot_trainer.py
------------------
Episodic training and validation for the prototypical network (Phase 2).

Implements Algorithm 3 (Few-Shot Adaptation via Prototypical Networks).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from ..models.full_model import MultimodalPainModel
from ..data.episode_sampler import EpisodeSampler, Episode
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FewShotTrainer:
    """
    Episodic trainer for few-shot fine-tuning.

    Parameters
    ----------
    model            : MultimodalPainModel
    train_sampler    : EpisodeSampler on the training split
    val_sampler      : EpisodeSampler on the validation split
    episodes_per_epoch : number of episodes per training epoch (default 1000)
    val_episodes     : number of validation episodes (default 500)
    lr_pretrained    : learning rate for pretrained layers (1e-5)
    lr_new           : learning rate for new components (1e-4)
    epochs           : total training epochs (default 50)
    patience         : early stopping patience in epochs (default 15)
    output_dir       : checkpoint directory
    device           : 'cuda' or 'cpu'
    use_amp          : mixed-precision training
    """

    def __init__(
        self,
        model:             MultimodalPainModel,
        train_sampler:     EpisodeSampler,
        val_sampler:       EpisodeSampler,
        episodes_per_epoch: int   = 1000,
        val_episodes:       int   = 500,
        lr_pretrained:      float = 1e-5,
        lr_new:             float = 1e-4,
        epochs:             int   = 50,
        patience:           int   = 15,
        output_dir:         str   = "checkpoints/finetuned",
        device:             str   = "cuda",
        use_amp:            bool  = True,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.train_s = train_sampler
        self.val_s   = val_sampler
        self.ep_per_epoch = episodes_per_epoch
        self.val_eps = val_episodes
        self.epochs  = epochs
        self.patience = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and device == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Separate parameter groups: lower LR for pretrained backbone
        pretrained_params = list(model.video_enc.vit.parameters())
        new_params = [
            p for n, p in model.named_parameters()
            if not any(pp is p for pp in pretrained_params)
        ]
        self.optimizer = optim.AdamW([
            {"params": pretrained_params, "lr": lr_pretrained},
            {"params": new_params,        "lr": lr_new},
        ], weight_decay=1e-4)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        self.best_val_acc = 0.0
        self.no_improve   = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _episode_to_device(self, ep: Episode) -> Episode:
        return Episode(
            support_video=  ep.support_video.to(self.device),
            support_audio=  ep.support_audio.to(self.device),
            support_physio= ep.support_physio.to(self.device),
            support_labels= ep.support_labels.to(self.device),
            query_video=    ep.query_video.to(self.device),
            query_audio=    ep.query_audio.to(self.device),
            query_physio=   ep.query_physio.to(self.device),
            query_labels=   ep.query_labels.to(self.device),
            audio_available=ep.audio_available,
        )

    # ------------------------------------------------------------------
    # One epoch of training
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        ep_iter = iter(self.train_s)

        for step in range(self.ep_per_epoch):
            ep = self._episode_to_device(next(ep_iter))

            self.optimizer.zero_grad()

            mask = {
                "video": True,
                "audio": ep.audio_available,
                "physio": True,
            }

            with autocast(enabled=self.use_amp):
                out = self.model(
                    ep.support_video, ep.support_audio, ep.support_physio,
                    ep.support_labels,
                    ep.query_video,   ep.query_audio,   ep.query_physio,
                    ep.query_labels,
                    modality_mask=mask,
                )
                loss = out["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_acc  += out["accuracy"]

            if (step + 1) % 200 == 0:
                logger.info(
                    f"  step {step+1}/{self.ep_per_epoch} | "
                    f"loss={total_loss/(step+1):.4f} | "
                    f"acc={total_acc/(step+1)*100:.2f}%"
                )

        return total_loss / self.ep_per_epoch, total_acc / self.ep_per_epoch

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0
        ep_iter = iter(self.val_s)

        for _ in range(self.val_eps):
            ep  = self._episode_to_device(next(ep_iter))
            mask = {"video": True, "audio": ep.audio_available, "physio": True}

            out = self.model(
                ep.support_video, ep.support_audio, ep.support_physio,
                ep.support_labels,
                ep.query_video,   ep.query_audio,   ep.query_physio,
                ep.query_labels,
                modality_mask=mask,
            )
            total_loss += out["loss"].item()
            total_acc  += out["accuracy"]

        return total_loss / self.val_eps, total_acc / self.val_eps

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self) -> MultimodalPainModel:
        """
        Run full episodic training with early stopping.

        Returns
        -------
        model : best checkpoint model
        """
        logger.info(f"Starting few-shot fine-tuning for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._validate()
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
                f"val loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.no_improve   = 0
                ckpt_path = self.output_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                }, ckpt_path)
                logger.info(f"  ✓ Saved best checkpoint (val_acc={val_acc*100:.2f}%)")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        # Load best weights
        best = torch.load(self.output_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(best["model_state_dict"])
        logger.info(
            f"Training complete. Best val accuracy: {self.best_val_acc*100:.2f}%"
        )
        return self.model
