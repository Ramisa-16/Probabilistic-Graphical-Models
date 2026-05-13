# Human Activity Recognition: EfficientNetB3 vs PGM Models

A comparative study of **deep learning** and **probabilistic graphical models** for Human Activity Recognition (HAR), using the same dataset and feature pipeline to enable a fair, side-by-side evaluation.

---

## Overview

This notebook trains four models on a HAR image dataset and produces a full comparison across accuracy, F1 score, interpretability, and inference speed:

| Model | Paradigm | Key Property |
|---|---|---|
| **EfficientNetB3** | Deep Learning | Highest accuracy; black-box |
| **Naive Bayes** | PGM | Ultra-fast; fully interpretable |
| **TAN Bayesian Network** | PGM | Explicit body-part dependencies |
| **Hidden Markov Model** | PGM | Temporal pose modeling |

---

## Dataset

- **Source:** [Human Action Recognition (HAR) Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) on Kaggle
- **Format:** RGB images with class labels in a CSV file
- **Split:** 80% train / 20% validation (stratified)
- **Sample size:** Up to 10,000 images (stratified subsample for speed)

---

## Architecture & Pipeline

### Deep Learning — EfficientNetB3

- ImageNet-pretrained EfficientNetB3 backbone
- Two-stage training: frozen warmup (3 epochs) → fine-tuning of last 160 layers (15 epochs)
- MixUp augmentation (α = 0.2, 20% probability)
- Optimizer: AdamW with cosine decay schedule
- Mixed precision (`float16`) for GPU efficiency
- Input size: 300×300

### PGM Shared Feature Extraction

All three PGM models share a **MediaPipe Pose** keypoint pipeline:

1. **33 body landmarks** (x, y, z, visibility) → 132 raw features
2. **8 joint angles** (elbows, knees, hips, torso)
3. **9 vertical positions** (nose, shoulders, hips, knees, ankles, wrists)
4. **2 limb-span ratios** (arm span / torso, leg / torso)
5. **Total: 151 features per sample**

### Naive Bayes
- Gaussian NB with Yeo-Johnson power transform + standard scaling
- `var_smoothing = 1e-6`

### TAN Bayesian Network
- **Tree-Augmented Naïve Bayes (TAN)** structure learned via `pgmpy`'s `TreeSearch`
- KMeans discretization (4 bins) of 13 selected pose features
- Symmetry features added (elbow/knee difference)
- BDeu prior for parameter estimation (`equivalent_sample_size = 30`)
- Inference via Variable Elimination

### Hidden Markov Model
- One **GMM-HMM** trained per activity class (generative approach)
- 6 hidden states, 2 GMM mixtures, diagonal covariance
- Sequences created by sliding window (length = 10) within each class
- Delta features appended (first-order differences)
- Classification: argmax of log-likelihood across class models

---

## Requirements

```
tensorflow >= 2.x
mediapipe == 0.10.14
pgmpy
hmmlearn
opencv-python-headless
scikit-learn
numpy
pandas
matplotlib
seaborn
```

Install:
```bash
pip install mediapipe==0.10.14 --force-reinstall --no-deps
pip install pgmpy hmmlearn opencv-python-headless
```

> **Note:** After installing mediapipe, restart the kernel before proceeding.

---

## Configuration

Key hyperparameters (defined in the Config cell):

| Parameter | Value |
|---|---|
| `IMAGE_SIZE` | (300, 300) |
| `BATCH_SIZE` | 32 |
| `SAMPLE_SIZE` | 10,000 |
| `WARMUP_EPOCHS` | 3 |
| `FINETUNE_EPOCHS` | 15 |
| `FINE_TUNE_LAST_N` | 160 layers |
| `LR_HEAD` | 5e-4 |
| `LR_FINE` | 5e-5 (cosine decay) |
| `MIXUP_ALPHA` | 0.2 |
| `PGM_SAMPLE_SIZE` | 10,000 |
| `N_HMM_COMPONENTS` | 6 |
| `N_MIX` (GMM) | 2 |

---

## Output Files

All outputs saved to `/kaggle/working/`:

| File | Description |
|---|---|
| `EfficientNetB3.weights.h5` | Best checkpoint weights |
| `EfficientNetB3_HAR.keras` | Full saved model |
| `comparison_1_overall_metrics.png` | Bar chart: Acc / Prec / Recall / F1 |
| `comparison_2_radar.png` | Radar chart (6-axis) |
| `comparison_3_f1_heatmap.png` | Per-class F1 heatmap across all models |
| `comparison_4_confusion_matrices.png` | 2×2 normalized confusion matrices |
| `comparison_5_training_curve.png` | EfficientNetB3 training history |
| `comparison_6_speed_efficiency.png` | Training time / inference time / interpretability |
| `comparison_7_dl_vs_pgm_perclass.png` | EfficientNetB3 vs best PGM per class |
| `comparison_8_pgm_perclass_f1.png` | PGM models per-class F1 grouped bar chart |
| `interpretability_proof.png` | Probability distributions from all PGM models |

---

## Interpretability Demo

A dedicated section demonstrates why PGMs are not black boxes:

- **Bayesian Network** shows full `P(Activity | observed pose features)` with reasoning chain
- **Naive Bayes** shows per-feature log-likelihood contributions and Z-score match indicators
- **HMM** shows state transition matrices and identifies the most stable "core pose" per activity
- **EfficientNetB3** is shown to provide only a final prediction with no reasoning pathway

---

## Key Findings

> **EfficientNetB3** achieves the highest raw accuracy but offers no explanation.
>
> **Bayesian Network** explicitly models how body-part positions relate to activities and can answer *why* a prediction was made.
>
> **HMM** is the only model that captures temporal pose structure — uniquely valuable for video-based HAR.
>
> **Naive Bayes** is the fastest PGM — suitable for resource-constrained or real-time deployments.

PGMs trade some accuracy for full transparency, principled uncertainty quantification, and auditability — making them preferable in healthcare, safety-critical, and low-resource settings.

---

## Running on Kaggle

1. Add the [HAR dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) to your notebook
2. Enable GPU accelerator
3. Run all cells in order (restart kernel after the pip install cell)
4. Outputs appear in `/kaggle/working/`
