# Reproducibility: Baseline Hyperparameters, Search Budget, and Seeds

This document responds directly to the review comment that the exact
optimizer hyperparameters, hyperparameter-search budget, and number of
random seeds used for the baseline methods were not reported.

## Where each piece of information lives

| Requested item | Source of truth |
|---|---|
| Exact optimizer hyperparameters (final, trained model) per baseline | [`configs/baselines.yaml`](../configs/baselines.yaml) — one block per baseline, `optimizer:` field |
| Hyperparameter search protocol + search space | `configs/baselines.yaml` — `hyperparameter_search:` field per baseline (`method`, `search_space`) |
| Hyperparameter search **budget** (number of configurations evaluated) | `configs/baselines.yaml` — `hyperparameter_search.budget` (declared) **and** the auto-generated `results/baselines/<name>_manifest.json` (actual — one entry per trial run) |
| Number of random seeds / seed values | `configs/baselines.yaml` — `common.seeds`, `common.n_seeds`, shared by every baseline for a fair comparison |
| Lightweight / frozen-encoder baseline full spec | [`configs/baseline_frozen_encoder.yaml`](../configs/baseline_frozen_encoder.yaml) |

## Why both a config file *and* a manifest

`configs/baselines.yaml` states the *intended* protocol. Running
[`scripts/train_baselines.py`](../scripts/train_baselines.py) additionally
writes a **run manifest** (`results/baselines/<baseline_name>_manifest.json`,
implemented in [`src/baselines/manifest.py`](../src/baselines/manifest.py))
that records what was *actually executed*: every hyperparameter
configuration evaluated during search (with its score), the exact
hyperparameters selected for the final model at each seed, which seeds
completed, the git commit, and timestamps. This closes the gap reviewers
often flag — a static table of numbers that can silently drift from the code
that produced them — by making the manifest the artifact that is actually
cited.

**Before submitting the camera-ready / rebuttal**, run the baselines end to
end and cite numbers from the manifests, not from memory:

```bash
python scripts/train_baselines.py \
    --data-root ./data \
    --output-dir results/baselines
```

This produces `results/baselines/summary.csv` (one row per baseline, with
mean ± std AUC and a pointer to its manifest) which can be pasted directly
into the comparison table.

## Lightweight / frozen-encoder baseline

The previously unaddressed frozen-encoder comparison is implemented in
[`src/baselines/frozen_encoder_baseline.py`](../src/baselines/frozen_encoder_baseline.py):
off-the-shelf pretrained encoders (ImageNet ViT / ImageNet-transferred
ResNet-18 / randomly-initialized physio CNN) are **frozen** — no contrastive
pretraining on UNBC-McMaster/BioVid, no fine-tuning — and only a lightweight
linear or 1-layer-MLP probe on top of concatenated pooled features is
trained, under the same k-shot / LOSO protocol as the proposed model. This
isolates how much of the reported gain is attributable to the contrastive
pretraining + fine-tuning stages versus simply reusing generic pretrained
features. Results are reported per k-shot setting (`k ∈ {1, 3, 5, 10}`) to
match the existing Key Results table.
