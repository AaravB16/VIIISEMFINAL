# ResultsII: Consolidated Script Run Output

## 1. Execution Summary

The following scripts were executed to generate fresh artifacts:

1. `scripts/run_experiment.py --config config/config.yaml --no-viz --log-level INFO`
2. `scripts/run_cross_dataset_validation.py --config config/config.yaml --allow-missing --output results/cross_dataset_results_ii.json --log-level INFO`
3. `scripts/benchmark_feature_selection.py --config config/config.yaml --split random --output results/feature_selection_benchmark_ii.json --log-level INFO`

Run metadata snapshot (from `results/experiment_metadata.json`):

- `run_timestamp`: `2026-05-11T11:00:47.111445+00:00`
- `n_experiments`: `112`
- `models_evaluated`: `lightgbm`, `random_forest`, `svm`, `xgboost`
- `splits_used`: `random`, `scenario`, `attack_holdout_DoS`, `attack_holdout_GAS`, `attack_holdout_RPM`, `attack_holdout_SPEED`, `attack_holdout_STEERING_WHEEL`
- `dataset_variants`: `original`, `jitter_augmented`, `noise_augmented`, `scaled_attack`

## 2. Core Experiment Outcomes (from `results/experiment_results.json`)

### 2.1 Best F1 by split (original dataset variant)

- `attack_holdout_DoS`: **lightgbm** `F1=0.963086`, `DR=0.600000`
- `attack_holdout_GAS`: **xgboost** `F1=0.994213`, `DR=0.900000`
- `attack_holdout_RPM`: **xgboost** `F1=0.980288`, `DR=0.500000`
- `attack_holdout_SPEED`: **xgboost** `F1=0.990404`, `DR=0.916667`
- `attack_holdout_STEERING_WHEEL`: **xgboost** `F1=0.992828`, `DR=0.700000`
- `random`: **xgboost** `F1=0.996983`, `DR=0.875000`
- `scenario`: **lightgbm** `F1=0.997222`, `DR=1.000000`

### 2.2 Best Detection Rate (DR) by split (original dataset variant)

- `attack_holdout_DoS`: **svm** `DR=1.000000`
- `attack_holdout_GAS`: **svm** `DR=1.000000`
- `attack_holdout_RPM`: **svm** `DR=0.625000`
- `attack_holdout_SPEED`: **xgboost** `DR=0.916667`
- `attack_holdout_STEERING_WHEEL`: **random_forest** `DR=0.800000`
- `random`: **svm** `DR=1.000000`
- `scenario`: **lightgbm** `DR=1.000000`

### 2.3 Aggregate model averages across all 112 runs (all variants + all splits)

- **lightgbm**: `mean F1=0.987527`, `mean DR=0.787381`, `mean latency=0.007654 ms/sample`, `mean train time=3.345607 s`
- **xgboost**: `mean F1=0.986516`, `mean DR=0.692500`, `mean latency=0.021146 ms/sample`, `mean train time=0.312286 s`
- **random_forest**: `mean F1=0.985944`, `mean DR=0.655179`, `mean latency=0.126464 ms/sample`, `mean train time=0.514571 s`
- **svm**: `mean F1=0.982809`, `mean DR=0.873810`, `mean latency=0.009121 ms/sample`, `mean train time=0.196643 s`

### 2.4 Hardest holdout per model (original variant)

- **random_forest**: `attack_holdout_DoS` (`DR=0.280000`, `F1=0.958521`)
- **svm**: `attack_holdout_RPM` (`DR=0.625000`, `F1=0.972191`)
- **xgboost**: `attack_holdout_DoS` (`DR=0.200000`, `F1=0.957559`)
- **lightgbm**: `attack_holdout_RPM` (`DR=0.562500`, `F1=0.980281`)

## 3. Robustness Outcomes (from `results/robustness_analysis.json`)

Robustness computed over `696` metric deltas.

Ranking by robustness score (lower is better):

1. **svm**: `0.000232` (most robust)
2. **random_forest**: `0.000355`
3. **xgboost**: `0.000512`
4. **lightgbm**: `0.000535`

`most_robust` summary:

- model: `svm`
- robustness_score: `0.000232`
- avg absolute delta:
  - `accuracy=0.000265`
  - `precision=0.000146`
  - `recall=0.000265`
  - `f1_score=0.000213`
  - `roc_auc=0.000373`

## 4. Feature-Selection Benchmark Outcomes (from `results/feature_selection_benchmark_ii.json`)

Benchmark context:

- `split_strategy`: `random`
- `drop_tolerance`: `0.002`
- `results rows`: `68`

Per-model recommendation (minimal features satisfying target criterion):

- **random_forest**
  - baseline `F1=0.996364`
  - recommended: `mi_k_3` with `3` features
  - recommended `F1=0.996983`
- **svm**
  - baseline `F1=0.992437`
  - recommended: `mi_k_3` with `3` features
  - recommended `F1=0.992870`
- **xgboost**
  - baseline `F1=0.996983`
  - recommended: `mi_k_3` with `3` features
  - recommended `F1=0.996983`
  - absolute best-F1 setting observed: `mi_k_6` with `F1=0.997911`
- **lightgbm**
  - baseline `F1=0.996364`
  - recommended: `mi_k_3` with `3` features
  - recommended `F1=0.996286`

## 5. Cross-Dataset Validation Outcome

Cross-dataset script ran successfully, but no result rows were produced because all configured external datasets are currently disabled in `config/config.yaml`:

- `cross_dataset_validation.enabled: false`
- `car_hacking.enabled: false`
- `otids.enabled: false`
- `road.enabled: false`

Therefore, `results/cross_dataset_results_ii.json` was not created.

## 6. Operational Notes and Caveats

- TensorFlow is not installed in the active Python environment, so deep-learning models are unavailable at runtime in this execution.
- Several runs emitted ROC-AUC class-mismatch warnings (expected in some split/class-availability combinations), while classification and detection metrics were still produced.
- Report focuses on generated artifacts in:
  - `results/experiment_results.json`
  - `results/robustness_analysis.json`
  - `results/experiment_metadata.json`
  - `results/feature_selection_benchmark_ii.json`
