# Exploratory Preterm Subgroup: Per-Subject Information

This document responds to the review comment requesting that the pending
per-subject information for the exploratory preterm subgroup analysis be
completed.

## What is required

For every neonate included in the preterm subgroup analysis, the paper
should report (in a table, typically in the Appendix/Supplementary
Material) the subject-level attributes that let a reader judge whether the
subgroup is well-characterized and whether the LOSO folds are balanced
across confounders such as gestational age and NICU site.

A fillable template is provided at
[`docs/preterm_subgroup_template.csv`](../docs/preterm_subgroup_template.csv)
with one row per subject and the following columns:

| Column | Description |
|---|---|
| `subject_id` | De-identified subject code, matching the ID used in `src/data/datasets.py` / the LOSO fold assignment in `src/evaluation/loso.py` |
| `gestational_age_weeks_birth` | Gestational age at birth, in completed weeks |
| `postmenstrual_age_weeks_assessment` | Postmenstrual age at the time of the pain assessment recording(s), in weeks |
| `birth_weight_g` | Birth weight in grams |
| `sex` | `M` / `F` |
| `nicu_site` | Which of the 2 NICUs the subject was recruited from (matches the paper's "2 NICUs" description) |
| `n_recordings` | Number of recordings/episodes contributed by this subject |
| `n_pain_labeled` | Number of pain-positive labeled samples |
| `n_no_pain_labeled` | Number of pain-negative (baseline/no-pain) labeled samples |
| `modalities_available` | Which of video/audio/physio were available for this subject, e.g. `video+audio+physio`, `video+physio` |
| `loso_fold_notes` | Any fold-specific notes (e.g. excluded from a particular ablation, missing modality reason) |

## Suggested subgroup summary statistics to report in-text

Once the table is complete, the main text description of the exploratory
preterm subgroup should state, at minimum:
- n subjects in the preterm subgroup (out of the full 34-neonate cohort)
- gestational age range and mean ± SD at birth
- postmenstrual age range at assessment
- how many of the 2 NICU sites are represented, and the per-site split
- any modality-availability caveats specific to this subgroup (e.g. if
  audio or physio coverage is lower among the most preterm infants)
