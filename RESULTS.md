# Experiment Results — Project Aarav

**CICIoV2024 Intrusion Detection Evaluation Framework**
**Experiment Date:** 2026-03-05 | **Total Experiments:** 112

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Raw Frames | 1,408,219 |
| After Deduplication | 3,588 (99.7% duplicates removed) |
| Features | ID, DATA_0–DATA_7 (9 CAN frame fields) |
| Normalization | Min-Max scaling |
| Additional Feature | Inter-arrival index |

### Class Distribution (Post-Dedup)

| Class | Count | Proportion |
|---|---|---|
| BENIGN | 3,117 | 86.87% |
| DoS | 191 | 5.32% |
| RPM (Spoofing) | 139 | 3.87% |
| SPEED (Spoofing) | 66 | 1.84% |
| STEERING_WHEEL (Spoofing) | 50 | 1.39% |
| GAS (Spoofing) | 25 | 0.70% |

---

## 2. Experimental Setup

- **Models:** Random Forest, SVM (RBF kernel), XGBoost, LightGBM
- **Split Strategies:** Random (80/20), Scenario-based, Attack Holdout (5 attack types)
- **Dataset Variants:** Original, Jitter Augmented (2% timing jitter), Noise Augmented (1% Gaussian noise), Scaled Attack (0.85–1.15× intensity scaling)
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Detection Rate, False Positive Rate, Training Time, Per-sample Latency

---

## 3. Baseline Results (Original Dataset)

### 3.1 Random Split

| Model | Accuracy | Precision | Recall | F1-Score | Detection Rate | FPR | Train Time (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9972 | 0.9961 | 0.9972 | 0.9964 | 0.8750 | 0.0000 | 0.376 | 0.0973 |
| SVM | 0.9916 | 0.9941 | 0.9916 | 0.9924 | 1.0000 | 0.0056 | 0.197 | 0.0098 |
| XGBoost | 0.9972 | 0.9972 | 0.9972 | 0.9970 | 0.8750 | 0.0000 | 1.811 | 0.0042 |
| LightGBM | 0.9972 | 0.9961 | 0.9972 | 0.9964 | 0.8750 | 0.0000 | 2.250 | 0.0097 |

### 3.2 Scenario-Based Split

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Detection Rate | FPR | Train Time (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9944 | 0.9903 | 0.9944 | 0.9924 | 0.9991 | 0.7000 | 0.0000 | 0.378 | 0.0962 |
| SVM | 0.9958 | 0.9944 | 0.9958 | 0.9951 | 0.9999 | 0.9000 | 0.0000 | 0.202 | 0.0108 |
| XGBoost | 0.9972 | 0.9949 | 0.9972 | 0.9960 | 0.9999 | 0.9000 | 0.0000 | 1.730 | 0.0405 |
| LightGBM | 0.9972 | 0.9972 | 0.9972 | 0.9972 | 0.9999 | 1.0000 | 0.0000 | 2.576 | 0.0090 |

### 3.3 Attack Holdout Splits

#### DoS Holdout

| Model | Accuracy | Precision | Recall | F1-Score | Detection Rate | FPR | Train Time (s) |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9714 | 0.9465 | 0.9714 | 0.9585 | 0.2800 | 0.0000 | 0.358 |
| SVM | 0.9551 | 0.9678 | 0.9551 | 0.9599 | 1.0000 | 0.0169 | 0.169 |
| XGBoost | 0.9714 | 0.9443 | 0.9714 | 0.9576 | 0.2000 | 0.0000 | 0.396 |
| LightGBM | 0.9714 | 0.9561 | 0.9714 | 0.9631 | 0.6000 | 0.0000 | 1.942 |

#### GAS Holdout

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Detection Rate | FPR |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9944 | 0.9913 | 0.9944 | 0.9924 | 0.9986 | 0.8000 | 0.0000 |
| SVM | 0.9792 | 0.9907 | 0.9792 | 0.9842 | 0.9969 | 1.0000 | 0.0155 |
| XGBoost | 0.9958 | 0.9931 | 0.9958 | 0.9942 | 0.9986 | 0.9000 | 0.0000 |
| LightGBM | 0.9944 | 0.9919 | 0.9944 | 0.9930 | 0.9985 | 0.9000 | 0.0000 |

#### RPM Holdout

| Model | Accuracy | Precision | Recall | F1-Score | Detection Rate | FPR |
|---|---|---|---|---|---|---|
| Random Forest | 0.9862 | 0.9733 | 0.9862 | 0.9796 | 0.4375 | 0.0000 |
| SVM | 0.9725 | 0.9739 | 0.9725 | 0.9722 | 0.6250 | 0.0141 |
| XGBoost | 0.9862 | 0.9746 | 0.9862 | 0.9803 | 0.5000 | 0.0000 |
| LightGBM | 0.9862 | 0.9751 | 0.9862 | 0.9803 | 0.5625 | 0.0000 |

#### SPEED Holdout

| Model | Accuracy | Precision | Recall | F1-Score | Detection Rate | FPR |
|---|---|---|---|---|---|---|
| Random Forest | 0.9917 | 0.9881 | 0.9917 | 0.9897 | 0.8333 | 0.0000 |
| SVM | 0.9889 | 0.9857 | 0.9889 | 0.9868 | 0.8333 | 0.0028 |
| XGBoost | 0.9917 | 0.9895 | 0.9917 | 0.9904 | 0.9167 | 0.0000 |
| LightGBM | 0.9917 | 0.9895 | 0.9917 | 0.9904 | 0.9167 | 0.0000 |

#### STEERING_WHEEL Holdout

| Model | Accuracy | Precision | Recall | F1-Score | Detection Rate | FPR |
|---|---|---|---|---|---|---|
| Random Forest | 0.9944 | 0.9912 | 0.9944 | 0.9924 | 0.8000 | 0.0000 |
| SVM | 0.9917 | 0.9893 | 0.9917 | 0.9902 | 0.8000 | 0.0028 |
| XGBoost | 0.9944 | 0.9917 | 0.9944 | 0.9928 | 0.7000 | 0.0000 |
| LightGBM | 0.9944 | 0.9917 | 0.9944 | 0.9928 | 0.7000 | 0.0000 |

---

## 4. Augmented Dataset Results (F1-Score / Detection Rate)

### 4.1 Jitter Augmented (2% Timing Jitter)

| Model | Random | Scenario | DoS Hold. | GAS Hold. | RPM Hold. | SPEED Hold. | STEER Hold. |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9963 / 0.75 | 0.9924 / 0.70 | 0.9622 / 0.52 | 0.9924 / 0.80 | 0.9796 / 0.44 | 0.9897 / 0.83 | 0.9924 / 0.80 |
| SVM | 0.9916 / 1.00 | 0.9951 / 0.90 | 0.9599 / 1.00 | 0.9842 / 1.00 | 0.9722 / 0.63 | 0.9868 / 0.83 | 0.9902 / 0.80 |
| XGBoost | 0.9970 / 0.88 | 0.9958 / 0.80 | 0.9574 / 0.16 | 0.9930 / 0.90 | 0.9796 / 0.44 | 0.9897 / 0.83 | 0.9928 / 0.70 |
| LightGBM | 0.9970 / 0.88 | 0.9981 / 1.00 | 0.9624 / 0.56 | 0.9931 / 1.00 | 0.9808 / 0.63 | 0.9904 / 0.92 | 0.9928 / 0.70 |

### 4.2 Noise Augmented (1% Gaussian Noise)

| Model | Random | Scenario | DoS Hold. | GAS Hold. | RPM Hold. | SPEED Hold. | STEER Hold. |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9963 / 0.75 | 0.9924 / 0.70 | 0.9555 / 0.16 | 0.9924 / 0.70 | 0.9796 / 0.44 | 0.9892 / 0.75 | 0.9921 / 0.60 |
| SVM | 0.9916 / 1.00 | 0.9951 / 0.90 | 0.9595 / 1.00 | 0.9842 / 1.00 | 0.9722 / 0.63 | 0.9868 / 0.83 | 0.9902 / 0.80 |
| XGBoost | 0.9963 / 0.75 | 0.9960 / 0.90 | 0.9586 / 0.36 | 0.9942 / 0.90 | 0.9796 / 0.44 | 0.9897 / 0.83 | 0.9921 / 0.60 |
| LightGBM | 0.9963 / 0.75 | 0.9956 / 1.00 | 0.9619 / 0.56 | 0.9930 / 0.90 | 0.9807 / 0.56 | 0.9887 / 0.67 | 0.9921 / 0.60 |

### 4.3 Scaled Attack (0.85×–1.15× Intensity)

| Model | Random | Scenario | DoS Hold. | GAS Hold. | RPM Hold. | SPEED Hold. | STEER Hold. |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9963 / 0.75 | 0.9924 / 0.70 | 0.9597 / 0.36 | 0.9924 / 0.80 | 0.9796 / 0.44 | 0.9897 / 0.83 | 0.9924 / 0.80 |
| SVM | 0.9908 / 1.00 | 0.9951 / 0.90 | 0.9599 / 1.00 | 0.9842 / 1.00 | 0.9722 / 0.63 | 0.9860 / 0.67 | 0.9902 / 0.80 |
| XGBoost | 0.9951 / 0.88 | 0.9960 / 0.90 | 0.9553 / 0.12 | 0.9942 / 0.90 | 0.9803 / 0.50 | 0.9904 / 0.92 | 0.9910 / 0.70 |
| LightGBM | 0.9970 / 0.88 | 0.9981 / 1.00 | 0.9644 / 0.76 | 0.9930 / 0.90 | 0.9808 / 0.63 | 0.9904 / 0.92 | 0.9910 / 0.70 |

---

## 5. Robustness Analysis

### 5.1 F1-Score Deltas (Augmented − Original)

#### Jitter Augmented

| Model | Random | Scenario | DoS Hold. | GAS Hold. | RPM Hold. | SPEED Hold. | STEER Hold. |
|---|---|---|---|---|---|---|---|
| Random Forest | −0.0001 | 0.0000 | +0.0037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM | −0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 0.0000 | −0.0001 | −0.0002 | −0.0012 | −0.0007 | −0.0007 | 0.0000 |
| LightGBM | +0.0006 | +0.0008 | −0.0007 | +0.0001 | +0.0005 | 0.0000 | 0.0000 |

#### Noise Augmented

| Model | Random | Scenario | DoS Hold. | GAS Hold. | RPM Hold. | SPEED Hold. | STEER Hold. |
|---|---|---|---|---|---|---|---|
| Random Forest | −0.0001 | 0.0000 | −0.0030 | −0.0001 | 0.0000 | −0.0006 | −0.0003 |
| SVM | −0.0008 | 0.0000 | −0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | −0.0007 | 0.0000 | +0.0011 | 0.0000 | −0.0007 | −0.0007 | −0.0007 |
| LightGBM | −0.0001 | −0.0017 | −0.0012 | 0.0000 | +0.0005 | −0.0018 | −0.0007 |

#### Scaled Attack

| Model | Random | Scenario | DoS Hold. | GAS Hold. | RPM Hold. | SPEED Hold. | STEER Hold. |
|---|---|---|---|---|---|---|---|
| Random Forest | −0.0001 | 0.0000 | +0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SVM | −0.0016 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | −0.0008 | 0.0000 |
| XGBoost | −0.0019 | 0.0000 | −0.0022 | 0.0000 | 0.0000 | 0.0000 | −0.0019 |
| LightGBM | +0.0006 | +0.0008 | +0.0013 | 0.0000 | +0.0005 | 0.0000 | −0.0019 |

### 5.2 Model Robustness Ranking

| Rank | Model | Robustness Score | Avg |Δ Accuracy| | Avg |Δ Precision| | Avg |Δ F1| |
|---|---|---|---|---|---|
| 1 | **SVM** | **0.000232** | 0.000265 | 0.000146 | 0.000213 |
| 2 | Random Forest | 0.000355 | 0.000065 | 0.000867 | 0.000428 |
| 3 | XGBoost | 0.000512 | 0.000263 | 0.001055 | 0.000604 |
| 4 | LightGBM | 0.000535 | 0.000265 | 0.001098 | 0.000652 |

> **Most Robust Model: SVM** — Lowest average absolute delta across all metrics, variants, and splits (robustness score = 0.000232).

---

## 6. Training Time & Inference Latency (Baseline)

| Model | Avg Train Time (s) | Avg Latency (ms/sample) |
|---|---|---|
| SVM | 0.164 | 0.0084 |
| Random Forest | 0.367 | 0.0972 |
| XGBoost | 0.763 | 0.0095 |
| LightGBM | 2.088 | 0.0092 |

---

*Results generated by Project Aarav — CICIoV2024 Evaluation Framework. Seed: 42.*
