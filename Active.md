# Active Implementation Dossier

## 1. Scope of What Is Implemented

This project currently includes implementation for all six requested workstreams:

1. Deep-learning integration into experiment flow
2. Sequence/sliding-window feature construction
3. Cross-dataset validation pipeline
4. Adversarial robustness generation/evaluation support
5. Feature-selection benchmarking and recommendation
6. Documentation/reporting polish with maintained status checklists

## 2. How Each Workstream Is Implemented

## 2.1 Deep Learning Integration

Implemented in:

- `src/deep_learning.py`
- `src/models.py`
- `scripts/run_experiment.py`
- `config/config.yaml`

Implementation mechanics:

- Added deep-learning model builders (`mlp`, `lstm`, `cnn1d`) with sklearn-like wrapper for uniform fit/predict flow.
- `get_enabled_models()` and `get_model()` support DL model registration alongside classical models.
- `run_experiment.py` includes `--enable-dl` and per-model feature extraction pathing so sequence models can consume 3D tensors.
- Defensive logic skips sequence DL runs when sliding windows are not enabled.

Current runtime status:

- **Implemented in code**.
- **Not fully exercised in latest full run** because TensorFlow is unavailable in active environment.

## 2.2 Sequence and Sliding-Window Features

Implemented in:

- `src/feature_engineering.py`
- `config/config.yaml`

Implementation mechanics:

- Added `_aggregate_window()` with configurable statistics (`mean`, `std`, `min`, `max`, `first`, `last`, `delta`, `median`).
- Added representation modes: `flatten`, `aggregate`, `hybrid`.
- Added sequence-ready tensor mode (`for_sequence_model=True`) to return `(n_windows, window_size, n_features)` for LSTM/CNN.
- `prepare_features()` dispatches to frame-level or window-level paths according to config.

Current runtime status:

- **Implemented and validated syntactically**.
- **Window path is conditional** on `features.use_sliding_window`; default run used frame-level path.

## 2.3 Cross-Dataset Validation

Implemented in:

- `src/cross_dataset.py`
- `scripts/run_cross_dataset_validation.py`
- `config/config.yaml`

Implementation mechanics:

- Added canonicalization of external schemas to project schema (`ID`, `DATA_0..7`, canonical labels).
- Added robust mixed numeric parsing (decimal/hex/binary-like token handling).
- Added label normalization and optional mapping.
- Added script pipeline: train on CICIoV2024, normalize with train-fit/test-transform, evaluate on external dataset(s).

Current runtime status:

- **Script executes successfully**.
- **No cross-dataset result rows produced in latest run** because external dataset entries are disabled in config.

## 2.4 Adversarial Robustness Evaluation

Implemented in:

- `src/adversarial.py`
- `src/augmentation_engine.py`
- `config/config.yaml`

Implementation mechanics:

- Added constrained adversarial-evasion variant generation:
  - selects attack rows,
  - moves features toward benign centroid,
  - adds stochastic noise,
  - projects perturbations to epsilon-bounded range and clips to valid global bounds.
- Added augmentation strategy hook `adversarial_evasion` in augmentation engine.

Current runtime status:

- **Implemented in code**.
- **Not active in latest full run** because `augmentation.strategies.adversarial_evasion.enabled` is currently `false`.
- Functional smoke validation of adversarial generator succeeded on a minimal synthetic dataframe.

## 2.5 Feature-Selection Benchmarking

Implemented in:

- `scripts/benchmark_feature_selection.py`

Implementation mechanics:

- Builds candidate configurations across:
  - baseline (no FS),
  - correlation-only,
  - PCA-only,
  - correlation + PCA,
  - MI and correlation + MI for multiple `k` values.
- Evaluates each setting per selected model and records both quality and efficiency metrics.
- Recommends minimal-feature setting meeting effective F1 target (`baseline - drop_tolerance` unless explicit target provided).

Current runtime status:

- **Implemented and executed successfully**.
- Output artifact generated: `results/feature_selection_benchmark_ii.json`.

## 2.6 Documentation and Reporting

Implemented in:

- `README.md`
- `PROJECT_OVERVIEW.md`
- `RESEARCH_PAPER.md`
- `ResultsII.md`
- `InferenceII.md`
- `Active.md`

Implementation mechanics:

- Updated project docs and paper sections to reflect implemented workstreams and current status.
- Added maintained checklists and explicit status statements.
- Added artifact-backed report/inference files from latest script runs.

## 3. Validity Assessment of Current Results

## 3.1 What supports validity

- End-to-end experiment execution completed (`112` runs, `0` failed).
- Robustness deltas computed over `696` comparisons.
- Feature-selection benchmarking completed (`68` benchmark rows).
- Syntax checks (`compileall`) passed previously for `src` and `scripts`.

## 3.2 Validity limits

- Deep-learning runtime claims are limited until TensorFlow-backed runs are executed.
- Cross-dataset generalization cannot be concluded from latest run because external datasets were disabled.
- Adversarial-evasion stress results are not included in the latest full-run metrics because the strategy was configured off.
- ROC-AUC class-mismatch warnings appear in some settings, so ROC-AUC should be interpreted cautiously where unavailable/None.

## 4. Viability Assessment

## 4.1 Technically viable now

- Classical ML pipeline execution (data load → preprocessing → split → train/eval → robustness → metadata) is operational and reproducible.
- Feature-selection reduction path appears deployment-friendly (3-feature recommendations with strong retained F1).

## 4.2 Conditionally viable (requires environment/config updates)

- Deep learning head-to-head viability: requires TensorFlow installation and DL-enabled runs.
- Cross-dataset viability: requires enabled external datasets and successful canonicalization on real files.
- Adversarial-evasion viability in full benchmarking: requires enabling strategy in config and rerunning experiments.

## 4.3 Practical deployment note

Given current evidence, a classical-model deployment path is viable today.  
A broader production claim (especially for unseen datasets and advanced adversarial behavior) requires the pending conditional runs above.
