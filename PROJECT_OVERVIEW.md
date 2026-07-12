# Project Aarav — Detailed Overview

**A Standardized Evaluation Framework for IoV Intrusion Detection**

---

## 1. Project Requirement

### 1.1 Problem Statement

The Internet of Vehicles (IoV) ecosystem relies on the Controller Area Network (CAN) bus for in-vehicle communication. CAN was designed without security mechanisms, leaving vehicles vulnerable to cyberattacks including denial-of-service (DoS), message spoofing, and replay attacks. Machine learning-based intrusion detection systems (IDS) are a promising countermeasure, but the research community lacks standardized evaluation frameworks, making it difficult to compare approaches and assess real-world readiness.

### 1.2 Objectives

1. **Build a reproducible evaluation framework** for benchmarking ML-based IDS on the CICIoV2024 dataset
2. **Evaluate multiple models** (Random Forest, SVM, XGBoost, LightGBM) across diverse split strategies
3. **Assess zero-day generalization** using attack-holdout evaluation where entire attack categories are withheld from training
4. **Measure deployment readiness** including inference latency, false positive rates, and robustness to data perturbations
5. **Introduce augmentation-based robustness evaluation** to validate model stability under realistic variations
6. **Produce comprehensive documentation** including results, analysis, and research-format reporting

### 1.3 Dataset

- **CICIoV2024** from the Canadian Institute for Cybersecurity, University of New Brunswick
- CAN bus traffic in decimal format with 6 classes: BENIGN, DoS, Spoofing-GAS, Spoofing-RPM, Spoofing-SPEED, Spoofing-STEERING_WHEEL
- 1,408,219 raw frames → 3,588 unique frames after deduplication

### 1.4 Key Requirements

- Configurable via YAML (models, splits, augmentation, features)
- Deterministic execution (seed=42) for reproducibility
- Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC, detection rate, FPR, latency
- Automated report generation with visualizations
- Experiment metadata logging with dataset checksums and library versions
- Support for data augmentation and robustness analysis

---

## 2. Implementation

### 2.1 Project Structure

```
Project_Aarav/
├── config/
│   └── config.yaml              # Central experiment configuration
├── src/
│   ├── data_loader.py           # Dataset loading, merging, label assignment
│   ├── preprocessing.py         # Deduplication, normalization, feature engineering
│   ├── feature_engineering.py   # Frame/temporal feature construction
│   ├── feature_selection.py      # Mutual information, correlation filter, PCA
│   ├── splitting.py             # Random, scenario, attack-holdout split strategies
│   ├── models.py                # Model factory (RF, SVM, XGBoost, LightGBM)
│   ├── deep_learning.py          # MLP, LSTM, 1D-CNN architectures (TensorFlow)
│   ├── cross_dataset.py          # External dataset canonicalization and loading
│   ├── adversarial.py            # Evasion-oriented CAN perturbation generation
│   ├── evaluation.py            # Metric computation, result formatting
│   ├── visualization.py         # Confusion matrices, ROC curves, comparisons
│   ├── augmentation_engine.py   # Jitter, noise, intensity scaling, adversarial
│   ├── robustness.py            # Delta computation, model ranking
│   └── reproducibility.py       # Metadata capture, checksums, environment logging
├── scripts/
│   ├── run_experiment.py        # Main experiment runner with tqdm progress
│   ├── generate_report.py       # Markdown report generator
│   ├── run_cross_dataset_validation.py # CIC-trained models on external datasets
│   └── benchmark_feature_selection.py  # FS benchmark + minimal-set recommendation
├── results/                     # Output directory (JSON, reports, plots)
├── CICIoV2024/                  # Dataset directory
├── RESULTS.md                   # Comprehensive numerical results
├── INFERENCE.md                 # Analysis and interpretation
├── RESEARCH_PAPER.md            # Academic paper format
├── PROJECT_OVERVIEW.md          # This file
└── README.md                    # Quick-start guide
```

### 2.2 Core Modules

#### Data Pipeline (`data_loader.py`, `preprocessing.py`)
- Loads 6 CSV files from CICIoV2024 (decimal format)
- Merges into unified DataFrame with label column
- Removes duplicates on `[ID, DATA_0...DATA_7]` (99.7% reduction)
- Applies Min-Max normalization
- Adds inter-arrival index feature

#### Split Strategies (`splitting.py`)
- **Random:** Stratified 80/20 train/test split
- **Scenario:** Splits by driving scenario/session groups
- **Attack Holdout:** For each of 5 attack types, excludes that type from training entirely. Tests model's ability to detect completely unseen attack categories (zero-day evaluation)

#### Model Training (`models.py`)
- Factory pattern with YAML-driven configuration
- **Random Forest:** 200 trees, max_depth=20, balanced class weights
- **SVM:** RBF kernel, C=1.0, gamma=scale, balanced weights, max_iter=5000
- **XGBoost:** 200 estimators, max_depth=8, LR=0.1, subsample=0.8
- **LightGBM:** 200 estimators, max_depth=8, 31 leaves, balanced weights

#### Evaluation (`evaluation.py`)
- Weighted-average precision, recall, F1-score
- ROC-AUC (one-vs-rest, where applicable)
- Detection Rate: recall specifically on attack classes
- False Positive Rate: rate of benign samples misclassified as attacks
- Training time and per-sample inference latency
- Label remapping helper for non-contiguous labels in holdout splits

#### Augmentation Engine (`augmentation_engine.py`)
Four augmentation strategies with deterministic seeded RNG:
1. **Timing Jitter (2%):** Gaussian perturbation of inter-arrival index
2. **Feature Noise (1%):** Additive Gaussian noise on CAN data bytes
3. **Attack Intensity Scaling (0.85×–1.15×):** Multiplicative scaling of attack frame features with automatic dtype upcasting to prevent int→float truncation
4. **Adversarial Evasion:** Constrained perturbations that move attack frames toward benign feature statistics while keeping values in valid ranges

#### Robustness Analysis (`robustness.py`)
- Computes per-metric absolute deltas between original and augmented results
- Aggregates deltas into per-model robustness scores
- Ranks models by overall stability
- Outputs structured JSON with rankings and per-metric breakdowns

#### Visualization (`visualization.py`)
- Confusion matrices per model/split
- ROC curves (where applicable)
- Model comparison bar charts
- Training time comparisons
- Robustness heatmaps and performance drop charts

#### Reproducibility (`reproducibility.py`)
- Captures full experiment metadata: config, Python version, platform, library versions
- Computes SHA-256 checksums of all dataset files
- Logs augmentation parameters and dataset variant information
- Ensures bit-for-bit reproducible results with fixed seed

#### Cross-Dataset Validation (`cross_dataset.py`, `scripts/run_cross_dataset_validation.py`)
- Canonicalizes external CAN IDS datasets (Car-Hacking, OTIDS, ROAD) into Project Aarav schema
- Supports mixed numeric encodings (decimal/hex/binary-style payload tokens)
- Evaluates CIC-trained models on external datasets with consistent preprocessing and metrics

#### Feature Selection Benchmarking (`scripts/benchmark_feature_selection.py`)
- Sweeps baseline, correlation filtering, MI k-selection, and PCA combinations
- Records feature-count vs. performance trade-offs per model
- Recommends minimal feature configuration meeting target F1 constraints for embedded deployment

### 2.3 Experiment Runner (`run_experiment.py`)

The main script orchestrates the full pipeline:
1. Load and preprocess dataset
2. Generate augmented dataset variants (if enabled)
3. For each dataset variant × split strategy × model:
   - Split data, train model, evaluate, record results
4. Run robustness analysis comparing original vs. augmented results
5. Save all results, metadata, and reports to `results/`
6. Progress tracking via tqdm with logging integration

**Total experiments:** 4 models × 7 splits × (4 or 5 variants, depending on adversarial-evasion enablement)

### 2.4 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Framework | scikit-learn 1.8.0 |
| Gradient Boosting | XGBoost 3.2.0, LightGBM 4.6.0 |
| Deep Learning | TensorFlow 2.20.0, Keras 3.13.2 |
| Data Processing | pandas 3.0.1, NumPy 2.4.2 |
| Visualization | matplotlib 3.10.8, seaborn 0.13.2 |
| Configuration | PyYAML 6.0.3 |
| Progress | tqdm 4.67.3 |

---

## 3. Results Summary

### 3.1 Baseline Performance

All four models achieve exceptional classification performance:

| Model | Best F1 | Best Split | Detection Rate | AUC |
|---|---|---|---|---|
| LightGBM | 0.9972 | Scenario | 1.000 | 0.9999 |
| XGBoost | 0.9970 | Random | 0.875 | 0.9999 |
| Random Forest | 0.9964 | Random | 0.875 | 0.9991 |
| SVM | 0.9951 | Scenario | 0.900 | 0.9999 |

### 3.2 Zero-Day Generalization

Attack holdout evaluation reveals critical gaps:

- **DoS Holdout:** Most challenging — XGBoost DR = 20%, RF DR = 28%. SVM uniquely maintains 100% DR.
- **RPM Holdout:** Second hardest — DR ranges from 43.75% (RF) to 62.5% (SVM)
- **GAS, SPEED, STEERING_WHEEL:** Better generalization — DR ranges from 70% to 100%

**Key Finding:** Tree-based models learn attack-specific patterns, while SVM learns generalized anomaly boundaries.

### 3.3 Robustness

Models are highly resilient to data perturbations:

| Model | Robustness Score | Interpretation |
|---|---|---|
| SVM | 0.000232 | Most stable — kernel smoothing resists perturbation |
| Random Forest | 0.000355 | Second most stable — ensemble averaging reduces variance |
| XGBoost | 0.000512 | Moderate — boosted trees show slight sensitivity |
| LightGBM | 0.000535 | Most sensitive — histogram binning affected by noise |

Maximum F1 delta across all 84 augmented experiments: **< 0.004**

### 3.4 Real-Time Viability

All models achieve sub-millisecond inference latency, well within CAN bus timing constraints (1–10 ms frame intervals):
- Fastest: XGBoost (0.0042 ms/sample)
- Slowest: Random Forest (0.0972 ms/sample)

### 3.5 Recommendation

A **hybrid IDS architecture** is recommended:
- **LightGBM** as primary classifier (best accuracy, lowest FPR)
- **SVM** as secondary anomaly detector (best generalization, highest DR for novel attacks)

### 3.6 Remaining Work Checklist (Maintained)

- **DONE** — Deep Learning Integration (MLP/LSTM/1D-CNN integrated into `scripts/run_experiment.py`)
- **DONE** — Sequence & Sliding-Window Features (flatten/aggregate/hybrid representations in `src/feature_engineering.py`)
- **DONE** — Cross-Dataset Validation (`src/cross_dataset.py` + `scripts/run_cross_dataset_validation.py`)
- **DONE** — Adversarial Robustness Evaluation (`src/adversarial.py` + adversarial variant in `src/augmentation_engine.py`)
- **DONE** — Feature Selection Benchmarking (`scripts/benchmark_feature_selection.py`)
- **DONE (ongoing maintenance)** — Final Report & Research Paper documentation polish

---

## 4. References

### 4.1 Dataset

1. Canadian Institute for Cybersecurity, "CICIoV2024 Dataset," University of New Brunswick, 2024. https://www.unb.ca/cic/datasets/iov-dataset-2024.html

### 4.2 CAN Bus Security

2. C. Miller and C. Valasek, "Remote exploitation of an unaltered passenger vehicle," *Black Hat USA*, 2015.
3. W. Wu et al., "Sliding window optimized information entropy analysis method for intrusion detection on in-vehicle networks," *IEEE Access*, vol. 6, 2018.

### 4.3 ML-Based Intrusion Detection

4. H. M. Song, J. Woo, and H. K. Kim, "In-vehicle network intrusion detection using deep convolutional neural network," *Vehicular Communications*, vol. 21, 2020.
5. S. Rajapaksha et al., "AI-based intrusion detection systems for in-vehicle networks: A survey," *ACM Computing Surveys*, vol. 55, no. 11, 2023.
6. M. D. Hossain et al., "LSTM-based intrusion detection system for in-vehicle CAN bus communications," *IEEE Access*, vol. 8, 2020.
7. S. Tariq, S. Lee, and S. S. Woo, "CAN-ADF: The controller area network attack detection framework," *Computers & Security*, vol. 94, 2020.

### 4.4 IoV Security Surveys

8. M. R. Aliyu, M. Usman, and A. Abba, "A survey on Internet of Vehicles: Applications, security issues and solutions," *Internet of Things*, vol. 22, 2023.
9. J. Cui et al., "A review on safety failures, security attacks, and available countermeasures for autonomous vehicles," *Ad Hoc Networks*, vol. 90, 2019.

### 4.5 Robustness and Adversarial ML

10. M. Kneib and C. Huth, "Scission: Signal characteristic-based sender identification and intrusion detection in automotive networks," in *ACM CCS*, 2018.
11. Y. Yan et al., "Adversarial attacks and defenses in deep learning-based CAN intrusion detection systems: A survey," *IEEE TVT*, 2023.

### 4.6 Benchmark Datasets

12. H. M. Song, J. Woo, and H. K. Kim, "Car-Hacking Dataset," *IEEE DataPort*, 2018.
13. H. Lee, S. H. Jeong, and H. K. Kim, "OTIDS: A novel intrusion detection system dataset for in-vehicle network," 2017.
14. M. Hanselmann et al., "CANet: An unsupervised intrusion detection system for high dimensional CAN bus data," *IEEE Access*, vol. 8, 2020.

### 4.7 Tools and Libraries

15. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, 2011.
16. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *KDD*, 2016.
17. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *NeurIPS*, 2017.
18. M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in *OSDI*, 2016.

---

*Project Aarav — Built for standardized, reproducible evaluation of IoV intrusion detection systems.*
