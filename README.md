# Multimodal Few-Shot Neonatal Pain Recognition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Self-Supervised Contrastive Transfer Learning for Few-Shot Multimodal Infant Pain Recognition: A Pilot Cross-Domain Study**
>
> Oussama El Othmani, Riadh Ouersighni, Sami Naouali
> *Frontiers in Pediatrics* — Manuscript ID: 1813543

---

## Overview

This repository provides the full implementation of our proof-of-concept framework for neonatal pain recognition using:

- **Self-Supervised Contrastive Pretraining** on adult pain datasets (UNBC-McMaster + BioVid) with masked contrastive learning for missing modalities
- **Transformer-Based Multimodal Fusion** of facial video (ViT), cry audio (ResNet-18 + mel-spectrograms), and physiological signals (1D-CNN)
- **Few-Shot Prototypical Networks** requiring only 1–10 labeled neonatal examples per class
- **Subject-Level LOSO Evaluation** on a 34-neonate pilot cohort across 2 NICUs

> ⚠️ **Pilot Study Disclaimer**: Neonatal results are proof-of-concept (n=34, 2 NICUs). Multi-center validation is required before any clinical deployment.

---

## Repository Structure

```
multimodal-fsl-pain/
├── src/
│   ├── models/
│   │   ├── encoders.py          # ViT, Audio CNN, Physio 1D-CNN encoders
│   │   ├── fusion.py            # Transformer-based multimodal fusion
│   │   ├── prototypical.py      # Prototypical network few-shot head
│   │   └── full_model.py        # End-to-end model assembly
│   ├── data/
│   │   ├── datasets.py          # UNBC-McMaster, BioVid, Neonatal dataset classes
│   │   ├── augmentations.py     # Video, audio, and physio augmentations
│   │   ├── episode_sampler.py   # Few-shot episode construction
│   │   └── preprocessing.py     # Temporal alignment, quality filtering
│   ├── training/
│   │   ├── contrastive.py       # InfoNCE + masked cross-modal pretraining
│   │   ├── fewshot_trainer.py   # Prototypical network episodic training
│   │   └── domain_adaptation.py # Adversarial domain adaptation (CDAN)
│   ├── evaluation/
│   │   ├── metrics.py           # AUC, ECE, Cohen's kappa, MMD
│   │   ├── loso.py              # Leave-one-subject-out cross-validation
│   │   └── calibration.py      # Reliability diagrams, ECE computation
│   └── utils/
│       ├── config.py            # Hydra/dataclass configs
│       ├── logger.py            # W&B + file logging
│       ├── interpretability.py  # GradCAM, attention viz, MC dropout
│       └── mmd.py               # Maximum Mean Discrepancy
├── configs/
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   └── eval.yaml
├── scripts/
│   ├── pretrain.py              # Phase 1: SSL pretraining
│   ├── finetune.py              # Phase 2: Few-shot fine-tuning
│   ├── evaluate.py              # Full evaluation pipeline
│   └── compute_mmd.py           # Domain gap quantification
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pretraining_analysis.ipynb
│   └── 03_clinical_results.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
├── requirements.txt
├── setup.py
└── docs/
    └── DATASET_SETUP.md
```

---

## Installation

```bash
git clone https://github.com/oussama123-ai/multimodal-fsl-pain.git
cd multimodal-fsl-pain
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **Training**: NVIDIA GPU with ≥16GB VRAM (A100 40GB recommended)
- **Inference**: ≥4GB GPU VRAM or CPU-only mode (≤2s per assessment)
- **RAM**: ≥32GB recommended for preprocessing

---

## Quick Start

### 1. Prepare Datasets

See [`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md) for download instructions.

```bash
# Expected structure
data/
├── unbc_mcmaster/      # UNBC-McMaster Pain Database
├── biovid/             # BioVid Heat Pain Database
└── neonatal/           # Your neonatal cohort (IRB required)
```

### 2. Self-Supervised Pretraining (Phase 1)

```bash
python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --data_root ./data \
    --output_dir ./checkpoints/pretrain
```

### 3. Few-Shot Fine-Tuning (Phase 2)

```bash
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --pretrained_ckpt ./checkpoints/pretrain/best.pt \
    --k_shot 10 \
    --output_dir ./checkpoints/finetuned
```

### 4. Evaluate (LOSO on Neonatal Cohort)

```bash
python scripts/evaluate.py \
    --config configs/eval.yaml \
    --checkpoint ./checkpoints/finetuned/best.pt \
    --eval_mode loso \
    --domain neonatal
```

---

## Key Results

| Setting | Accuracy (%) | Subject AUC | LOSO AUC |
|---|---|---|---|
| Adult domain, 10-shot | 83.6 ± 2.9 | — | — |
| Neonatal pilot, 10-shot | 80.4 ± 3.2 | 0.851 ± 0.061 | 0.812 ± 0.071 |
| Neonatal pilot, 1-shot | 68.8 ± 4.5 | 0.776 ± 0.089 | 0.776 ± 0.089 |

SSL pretraining contributes **+14.1%** over no pretraining (adult domain, 10-shot).

Domain gap: MMD = 0.142 (pre-adaptation) → 0.067 (post-adaptation).

---

## Citation

```bibtex
@article{elothmani2025ssl,
  title   = {Self-Supervised Contrastive Transfer Learning for Few-Shot Multimodal
             Infant Pain Recognition: A Pilot Cross-Domain Study},
  author  = {El Othmani, Oussama and Ouersighni, Riadh and Naouali, Sami},
  journal = {Frontiers in Pediatrics},
  year    = {2025},
  note    = {Manuscript ID: 1813543}
}
```

---

## Ethics and Clinical Use

This work involves human subjects data collected under IRB Protocol #2023-IRB-045. The framework is intended **solely for research purposes** and must not be used clinically without prospective multi-center validation.

Contact the corresponding author for anonymized data access: `oussama.elothmani@ept.u-carthage.tn`

---

## License

MIT License — see [LICENSE](LICENSE).
