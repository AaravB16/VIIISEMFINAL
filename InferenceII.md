# InferenceII: Outcomes and Learnings from ResultsII

## 1. High-Level Outcome

The framework executed successfully for classical ML evaluation at scale (`112` experiment runs), and produced stable, reproducible performance under configured augmentations. Overall detection quality is strong in standard settings, but zero-day generalization remains split- and attack-dependent.

## 2. What the Results Indicate

### 2.1 Accuracy is high, but not the whole story

- F1 is consistently high (roughly `0.96` to `0.997` across original split/model combinations).
- However, detection rate (DR) on held-out attacks varies significantly by model and attack type, indicating different generalization behavior.

### 2.2 Clear model trade-off surfaced

- **LightGBM/XGBoost**: strongest top-end F1 in many splits.
- **SVM**: strongest robustness and often strongest DR on unseen attack families (notably DoS and GAS holdouts).

Interpretation: high aggregate F1 can coexist with weaker novel-attack recall; model choice should align with risk posture.

### 2.3 Zero-day weak spots are concentrated

- Hardest region remains DoS/RPM holdout behavior for tree ensembles (e.g., RF DoS DR `0.28`, XGBoost DoS DR `0.20`).
- This suggests attack-family-specific bias in learned boundaries when true unseen-pattern shift occurs.

### 2.4 Robustness to tested perturbations is good

- Robustness deltas are very small overall.
- SVM is most robust (`0.000232` score), implying stable behavior under timing jitter / noise / attack scaling perturbations used here.

### 2.5 Feature minimization is promising for deployment

- Benchmark recommends `mi_k_3` (3 features) for all four classical models under configured tolerance.
- Performance remains close to baseline, supporting embedded deployment feasibility with reduced feature footprint.

## 3. Key Learnings

1. **Generalization metrics (DR in holdout splits) must be treated as first-class deployment criteria**, not just weighted F1.
2. **Hybrid deployment strategy is justified**: use high-precision gradient boosting with a robustness-oriented detector path (e.g., SVM) for safety coverage.
3. **Feature budget can likely be reduced substantially** without major performance loss for current data/task setup.
4. **Cross-dataset claims are not yet evidenced in this run**, because external datasets were disabled in config.
5. **Deep learning integration is implemented but not currently evidenced at runtime** in this environment due missing TensorFlow.

## 4. Validity Boundaries of Current Inference

The above inferences are valid for:

- CICIoV2024-derived training/evaluation workflow,
- currently enabled augmentations and config settings,
- classical models exercised in the latest run.

The above inferences are not yet fully validated for:

- externally sourced datasets (cross-platform transfer),
- adversarial-evasion-enabled experiment sweeps,
- deep-learning model runtime comparisons in this environment.

## 5. Immediate Next Actions Suggested by the Results

1. Enable and run cross-dataset dataset entries in `config/config.yaml` to quantify transferability.
2. Enable adversarial-evasion augmentation in config and rerun robustness deltas for stronger stress testing.
3. Install TensorFlow and run `--enable-dl` experiments for true head-to-head MLP/LSTM/CNN comparison.
4. Prioritize improvement work on DoS and RPM holdout detection for tree ensembles.
