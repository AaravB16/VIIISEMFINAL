# Project Details: Phase I and Phase II

## Project Objective

Project Aarav provides a reproducible framework for evaluating intrusion detection models on automotive CAN traffic (CICIoV2024), with emphasis on:

- model comparison under multiple split strategies,
- robustness under realistic perturbations,
- deployment-oriented feature reduction,
- transparent reporting and reproducibility.

---

## Phase I (All Work Prior to Today)

Everything not listed in Phase II is treated as Phase I.

### 1) Core Framework and Dataset Pipeline

- Repository and framework structure for repeatable IDS experimentation.
- Configuration-driven experiment control via `config/config.yaml`.
- Dataset loading for CICIoV2024 decimal/hex formats.
- Unified schema handling (`ID`, `DATA_0..7`, labels, scenario metadata).

### 2) Data Preprocessing Foundations

- Missing-value handling.
- Duplicate-removal strategy on CAN frame fields.
- Optional normalization pipelines (`minmax` / `standard`).
- Inter-arrival proxy indexing and related preprocessing.

### 3) Baseline Feature and Split Workflows

- Frame-level feature extraction pipeline.
- Split strategies:
  - random split,
  - scenario split,
  - attack-holdout split across major attack families.

### 4) Modeling and Evaluation Baseline

- Classical models integrated:
  - Random Forest,
  - SVM,
  - XGBoost,
  - LightGBM.
- Training, inference, and metric evaluation pipeline.
- Metrics include:
  - accuracy, precision, recall, F1,
  - detection rate / false-positive rate,
  - ROC-AUC (where valid),
  - confusion matrix outputs.

### 5) Reporting, Reproducibility, and Existing Visualization

- Result serialization (`experiment_results.json` and related artifacts).
- Experiment metadata and checksum logging for reproducibility.
- Existing plotting utilities in `src/visualization.py`.
- Baseline report generation script (`scripts/generate_report.py`).

### 6) Baseline Robustness and Documentation Set

- Robustness analysis framework and comparative scoring support.
- Initial robustness perturbation strategies (timing jitter, feature noise, attack scaling).
- Existing documentation package (`README.md`, `PROJECT_OVERVIEW.md`, `RESEARCH_PAPER.md`, etc.).

---

## Phase II (Today’s Work)

All tasks completed today are Phase II.

### 1) Deep Learning Integration Expansion

- Integrated deep-learning model pathing into experiment runner for head-to-head comparability.
- Added/connected:
  - MLP,
  - LSTM,
  - 1D-CNN scaffolding/runner compatibility.
- Added safeguards for sequence-model requirements (e.g., sliding-window dependency).

### 2) Sequence and Sliding-Window Feature Upgrade

- Extended feature engineering with configurable sliding-window generation.
- Added representation modes:
  - `flatten`,
  - `aggregate`,
  - `hybrid`,
  - sequence tensor mode for DL sequence models.
- Added configurable window aggregation statistics.

### 3) Cross-Dataset Validation Pipeline

- Added external dataset canonicalization and adapter logic.
- Implemented `scripts/run_cross_dataset_validation.py` for CIC-trained-to-external testing flow.
- Added config schema for external datasets and label mapping.

### 4) Adversarial Robustness Extension

- Added adversarial evasion generation module (`src/adversarial.py`).
- Integrated adversarial-evasion strategy hook into augmentation engine.
- Added constrained perturbation logic (bounded perturbation, benign-direction pull, controlled noise).

### 5) Feature-Selection Benchmark Workflow

- Added `scripts/benchmark_feature_selection.py`.
- Implemented benchmark sweeps for:
  - baseline,
  - correlation filter,
  - PCA,
  - mutual information (and combinations).
- Added recommendation logic for minimal deployable feature sets under F1 constraints.

### 6) Documentation and Reporting Consolidation

- Created:
  - `ResultsII.md` (artifact-backed run summary),
  - `InferenceII.md` (what outcomes imply),
  - `Active.md` (implementation + validity + viability status).
- Updated paper/report-oriented sections and maintained checklist status alignment across docs.

### 7) Fresh Execution and Artifact Refresh

- Re-ran major scripts to refresh result artifacts.
- Confirmed successful large run completion and regenerated metrics artifacts.

### 8) New Timestamped Visualization Script (Today)

- Added new script: `scripts/visualize_results.py`.
- Purpose: generate easy-to-consume charts for project outcomes.
- Output behavior: saves graphs to timestamped run directories under:
  - `results/runs/<timestamp>/figures/`
- Script also writes run metadata summary:
  - `results/runs/<timestamp>/run_summary.json`

---

## Current State Summary

- Phase I provides the original experiment foundation and baseline framework.
- Phase II adds major new capability layers (DL pathing, temporal features, cross-dataset tooling, adversarial extension, FS benchmarking), updated narrative/report files, and a new timestamped visualization workflow.
- Together, these phases now provide a stronger end-to-end research and deployment-oriented evaluation stack.
