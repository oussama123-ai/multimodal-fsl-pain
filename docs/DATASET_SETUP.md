# Dataset Setup Guide

This document explains how to prepare the three datasets used in the paper.

---

## 1. UNBC-McMaster Shoulder Pain Expression Archive Database

**Access**: Request access at [https://www.cs.cmu.edu/afs/cs/project/face/www/database.htm](https://www.cs.cmu.edu/afs/cs/project/face/www/database.htm)

The dataset contains:
- 48,398 video frames from 25 subjects undergoing shoulder physiotherapy
- PSAM pain scores (0–10)
- **Video + physiological signals only — no audio**

### Preparing the dataset

After downloading, organize into:

```
data/unbc_mcmaster/
├── metadata.json
├── videos/
│   ├── 042_clip01.avi
│   └── ...
└── physio/
    ├── 042_clip01.npy   ← shape (3, L): [HR, SpO2, RR]
    └── ...
```

**`metadata.json` format:**
```json
[
  {
    "subject":  "042",
    "video":    "videos/042_clip01.avi",
    "physio":   "physio/042_clip01.npy",
    "label":    1,
    "psam":     3
  }
]
```

Label binarization: `label = 1 if psam > 0 else 0`.

---

## 2. BioVid Heat Pain Database

**Access**: Request at [https://www.nit.ovgu.de/BioVid.html](https://www.nit.ovgu.de/BioVid.html)

The dataset contains:
- 8,700 multimodal recordings from 87 healthy subjects
- Calibrated heat pain stimuli at 4 intensity levels
- **All three modalities: video, audio (cry/vocalization), physiological signals**

### Preparing the dataset

```
data/biovid/
├── metadata.json
├── videos/
├── audio/
└── physio/
```

**`metadata.json` format:**
```json
[
  {
    "subject":  "101",
    "video":    "videos/101_pa1_BL1.avi",
    "audio":    "audio/101_pa1_BL1.wav",
    "physio":   "physio/101_pa1_BL1.npy",
    "label":    0
  }
]
```

Labels: `0 = baseline, 1 = low, 2 = moderate, 3 = high`.
With `--binary True`, labels are collapsed to `0 = no pain, 1 = pain (>0)`.

---

## 3. Neonatal Pilot Cohort

**Access**: The neonatal dataset is **not publicly available** due to ethical and privacy
restrictions (IRB Protocol #2023-IRB-045). Anonymized data may be available upon
reasonable request to `oussama.elothmani@ept.u-carthage.tn` with appropriate ethical approval.

### Dataset statistics

| Property             | Value                                |
|----------------------|--------------------------------------|
| Subjects (neonates)  | 34                                   |
| NICUs                | 2 (Military Hospital of Tunisia)     |
| Recordings           | 1,247                                |
| Pain instances       | 689 (55.2%)                          |
| No-pain instances    | 558 (44.8%)                          |
| Gestational age      | 28–42 weeks                          |
| Pain stimuli         | Heel stick (41%), venipuncture (34%), postoperative (25%) |
| Inter-rater κ        | 0.82 (NIPS + COMFORT-B scales)       |

### Expected layout

```
data/neonatal/
├── metadata.json
├── videos/
├── audio/
└── physio/
```

**`metadata.json` format:**
```json
[
  {
    "subject":         "N001",
    "gestational_age": 35,
    "nicu":            "NICU_A",
    "procedure":       "heel_stick",
    "video":           "videos/N001_001.mp4",
    "audio":           "audio/N001_001.wav",
    "physio":          "physio/N001_001.npy",
    "label":           1,
    "nips_score":      6
  }
]
```

---

## Physiological signal format (`.npy`)

All `.npy` arrays have shape `(3, L)` where:
- Row 0: Heart Rate (HR) in bpm
- Row 1: Oxygen Saturation (SpO₂) in %
- Row 2: Respiration Rate (RR) in breaths/min

Windows are 30 seconds at 1 Hz sampling → `L = 30` for the default configuration.

---

## Generating synthetic data for testing

To verify your installation without real data, use the provided mock dataset:

```python
from tests.test_data import TestEpisodeSampler
mock = TestEpisodeSampler.MockDataset(n_subjects=34, n_per_subject=37)
# This produces a synthetic dataset with the same API as NeonatalDataset
```

Or generate placeholder metadata files:

```bash
python -c "
import json, os
os.makedirs('data/neonatal', exist_ok=True)
records = [{'subject': f'N{i:03d}', 'gestational_age': 35, 'nicu': 'A',
            'procedure': 'heel_stick', 'video': 'v.mp4', 'audio': 'a.wav',
            'physio': 'p.npy', 'label': i%2, 'nips_score': 4} for i in range(34)]
json.dump(records, open('data/neonatal/metadata.json', 'w'), indent=2)
print('Placeholder metadata created.')
"
```
