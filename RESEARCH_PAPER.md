# A Standardized Evaluation Framework for Machine Learning-Based Intrusion Detection in the Internet of Vehicles Using CICIoV2024

---

## Abstract

The Internet of Vehicles (IoV) faces escalating cybersecurity threats targeting the Controller Area Network (CAN) bus, the backbone of in-vehicle communication. This paper presents a standardized evaluation framework for benchmarking machine learning-based intrusion detection systems (IDS) using the CICIoV2024 dataset. We evaluate four classifiers — Random Forest, Support Vector Machine (SVM), XGBoost, and LightGBM — across seven split strategies including random, scenario-based, and five attack-holdout configurations. To assess deployment resilience, we introduce an augmentation-based robustness evaluation pipeline employing timing jitter, Gaussian feature noise, and attack intensity scaling. Our experiments (112 total configurations) reveal that all models achieve F1-scores exceeding 0.99 under standard evaluation, with LightGBM achieving the best scenario-based performance (F1 = 0.9972, AUC = 0.9999). However, attack holdout evaluation exposes significant generalization gaps, particularly for DoS (detection rates as low as 20%) and RPM spoofing. SVM emerges as the most robust model (robustness score: 0.000232) with consistent detection rates across all perturbation types. All models maintain sub-millisecond inference latency, confirming real-time viability for CAN bus monitoring.

**Keywords:** Internet of Vehicles, Intrusion Detection System, CAN Bus Security, Machine Learning, CICIoV2024, Robustness Evaluation

---

## 1. Introduction

The rapid proliferation of connected and autonomous vehicles has transformed the automotive landscape into a networked ecosystem known as the Internet of Vehicles (IoV). While IoV enables advanced features such as V2X communication, over-the-air updates, and cooperative driving, it simultaneously expands the attack surface for malicious actors [1, 2].

The Controller Area Network (CAN) bus, standardized in ISO 11898, remains the dominant in-vehicle communication protocol. CAN was designed in the 1980s without authentication, encryption, or access control mechanisms, making it inherently vulnerable to injection attacks [3]. Demonstrated attack vectors include denial-of-service (DoS) flooding, spoofing of sensor readings (speed, RPM, steering angle, gas pedal position), and fuzzy attacks that inject random CAN frames [4].

Machine learning-based intrusion detection systems (IDS) have emerged as a promising defense mechanism, capable of learning normal CAN traffic patterns and detecting anomalous frames in real time [5, 6]. However, the IDS research community faces several methodological challenges:

1. **Inconsistent evaluation protocols** — Studies use different datasets, split strategies, and metrics, making cross-comparison difficult [7].
2. **Overfitting to known attacks** — Models trained and tested on the same attack types may not generalize to novel threats [8].
3. **Lack of robustness evaluation** — Few studies assess how models perform under realistic data perturbations [9].

This paper addresses these gaps by presenting a comprehensive, reproducible evaluation framework that:
- Benchmarks four widely-used ML classifiers on the CICIoV2024 dataset
- Employs seven split strategies including attack-holdout evaluation for zero-day generalization testing
- Introduces augmentation-based robustness analysis with three perturbation strategies
- Measures deployment-critical metrics including inference latency and false positive rates

---

## 2. Related Work

### 2.1 CAN Bus Intrusion Detection

Early CAN IDS approaches relied on rule-based and statistical methods, including entropy analysis [10] and time-interval monitoring [11]. ML-based approaches have since demonstrated superior performance, with studies employing decision trees [12], random forests [13], SVMs [14], and deep learning architectures including CNNs and LSTMs [15, 16].

### 2.2 Evaluation Datasets

Several CAN bus intrusion detection datasets have been published:
- **Car-Hacking Dataset (2018)** — Contains DoS, fuzzy, RPM, and gear spoofing attacks [17]
- **OTIDS (2018)** — Focused on fabrication, suspension, and masquerade attacks [18]
- **SynCAN (2020)** — Synthetic dataset with signal-level features [19]
- **CICIoV2024** — The most recent benchmark from the Canadian Institute for Cybersecurity, featuring CAN frames in both hexadecimal and decimal formats with five attack categories [20]

### 2.3 Robustness Evaluation

Robustness evaluation of CAN IDS remains underexplored. Adversarial machine learning studies have shown that small perturbations to CAN frame features can evade detection [21]. Our framework addresses this by systematically evaluating model stability under controlled augmentations.

---

## 3. Methodology

### 3.1 Dataset Description

We use the CICIoV2024 dataset provided by the Canadian Institute for Cybersecurity [20]. The dataset contains CAN bus traffic captured from a real vehicle and includes:

- **Benign traffic:** Normal driving CAN frames
- **Attack categories:** DoS, Spoofing-GAS, Spoofing-RPM, Spoofing-SPEED, Spoofing-STEERING_WHEEL

The raw dataset comprises 1,408,219 CAN frames. Each frame includes a CAN arbitration ID and 8 data bytes (DATA_0 through DATA_7).

### 3.2 Data Preprocessing

1. **Deduplication:** Duplicate frames (sharing identical ID and data bytes) are removed, reducing the dataset to 3,588 unique frames (99.7% duplicate removal rate). This prevents data leakage in train/test splits.
2. **Feature engineering:** An inter-arrival index feature is computed to capture temporal ordering.
3. **Normalization:** Min-Max scaling is applied to all features, mapping values to [0, 1].

Post-deduplication class distribution: BENIGN (86.87%), DoS (5.32%), RPM (3.87%), SPEED (1.84%), STEERING_WHEEL (1.39%), GAS (0.70%).

### 3.3 Models

We evaluate four classifiers selected for their diversity in learning paradigms:

1. **Random Forest (RF):** Ensemble of 200 decision trees (max_depth=20, balanced class weights)
2. **Support Vector Machine (SVM):** RBF kernel (C=1.0, gamma=scale, balanced class weights)
3. **XGBoost:** Gradient-boosted trees (200 estimators, max_depth=8, learning_rate=0.1)
4. **LightGBM:** Histogram-based gradient boosting (200 estimators, max_depth=8, 31 leaves)

### 3.4 Evaluation Strategies

#### 3.4.1 Random Split
Standard 80/20 train/test split with stratified sampling (seed=42).

#### 3.4.2 Scenario-Based Split
Data is split by driving scenario/session to evaluate cross-scenario generalization.

#### 3.4.3 Attack Holdout
For each of the five attack types, all samples of that attack are excluded from training and placed in the test set. This evaluates zero-day detection capability — whether a model can detect an entirely unseen attack category.

### 3.5 Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1-Score (weighted), ROC-AUC
- **Security:** Detection Rate (recall on attack classes), False Positive Rate
- **Efficiency:** Training time, per-sample inference latency

### 3.6 Augmentation-Based Robustness Evaluation

We introduce three augmentation strategies to simulate real-world data variation:

1. **Timing Jitter (σ = 2%):** Gaussian perturbation of the inter-arrival index to simulate clock drift and network delays.
2. **Feature Noise (σ = 1%):** Additive Gaussian noise on CAN data bytes to simulate sensor noise and quantization effects.
3. **Attack Intensity Scaling (0.85×–1.15×):** Multiplicative scaling of attack frame features to simulate varied attack intensities.

Robustness is quantified as the average absolute delta in classification metrics between original and augmented evaluations.

---

## 4. Results

### 4.1 Baseline Performance

Under random evaluation, all models achieve F1-scores ≥ 0.9924 (Table 1). XGBoost achieves the highest random-split F1 (0.9970), while LightGBM leads on scenario-based evaluation (F1 = 0.9972, DR = 100%, AUC = 0.9999).

**Table 1: Baseline Results — Random and Scenario Splits**

| Model | Random F1 | Random DR | Scenario F1 | Scenario DR | Scenario AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9964 | 0.875 | 0.9924 | 0.700 | 0.9991 |
| SVM | 0.9924 | 1.000 | 0.9951 | 0.900 | 0.9999 |
| XGBoost | 0.9970 | 0.875 | 0.9960 | 0.900 | 0.9999 |
| LightGBM | 0.9964 | 0.875 | 0.9972 | 1.000 | 0.9999 |

### 4.2 Attack Holdout Performance

The attack holdout evaluation (Table 2) reveals significant generalization gaps. DoS holdout is the most challenging, with tree-based models achieving detection rates of 20–60%. SVM uniquely maintains 100% DR across DoS and GAS holdouts.

**Table 2: Attack Holdout Results (F1-Score / Detection Rate)**

| Model | DoS | GAS | RPM | SPEED | STEER |
|---|---|---|---|---|---|
| RF | 0.9585 / 0.28 | 0.9924 / 0.80 | 0.9796 / 0.44 | 0.9897 / 0.83 | 0.9924 / 0.80 |
| SVM | 0.9599 / 1.00 | 0.9842 / 1.00 | 0.9722 / 0.63 | 0.9868 / 0.83 | 0.9902 / 0.80 |
| XGBoost | 0.9576 / 0.20 | 0.9942 / 0.90 | 0.9803 / 0.50 | 0.9904 / 0.92 | 0.9928 / 0.70 |
| LightGBM | 0.9631 / 0.60 | 0.9930 / 0.90 | 0.9803 / 0.56 | 0.9904 / 0.92 | 0.9928 / 0.70 |

### 4.3 Robustness Analysis

All augmented experiments produce F1 deltas below 0.004 (Table 3). SVM is identified as the most robust model with the lowest average absolute delta.

**Table 3: Model Robustness Ranking**

| Rank | Model | Robustness Score | Avg |ΔF1| |
|---|---|---|---|
| 1 | SVM | 0.000232 | 0.000213 |
| 2 | Random Forest | 0.000355 | 0.000428 |
| 3 | XGBoost | 0.000512 | 0.000604 |
| 4 | LightGBM | 0.000535 | 0.000652 |

### 4.4 Inference Latency

All models achieve sub-millisecond per-sample inference (Table 4), confirming real-time viability.

**Table 4: Training and Inference Efficiency**

| Model | Avg Train Time (s) | Avg Latency (ms/sample) |
|---|---|---|
| SVM | 0.164 | 0.0084 |
| Random Forest | 0.367 | 0.0972 |
| XGBoost | 0.763 | 0.0095 |
| LightGBM | 2.088 | 0.0092 |

---

## 5. Discussion

### 5.1 Model Selection Trade-offs

Our results reveal a fundamental tension between classification accuracy and generalization capability. LightGBM achieves the highest overall performance metrics but ranks lowest in robustness. Conversely, SVM shows the best generalization to unseen attacks and data perturbations while sacrificing marginal accuracy.

This trade-off has practical implications: in safety-critical IoV deployments, the cost of missing an attack far exceeds the cost of a false positive. SVM's consistent high detection rates make it the preferred choice for such scenarios, while LightGBM's precision suits monitoring applications where false alarm fatigue is a concern.

### 5.2 The Deduplication Challenge

The 99.7% duplicate removal rate is characteristic of CAN bus data, where periodic messages are transmitted at fixed intervals. While necessary for valid evaluation, it produces a small unique-frame dataset (3,588 samples) that limits statistical confidence, particularly for minority classes (GAS: 25 samples). Future work should explore sequence-based or window-based features to retain temporal structure while avoiding duplication artifacts.

### 5.3 Zero-Day Detection Limitations

The attack holdout results highlight a critical challenge: tree-based models learn attack-type-specific patterns rather than general anomaly signatures. The DoS holdout is particularly instructive — DoS attacks flood the bus with high-frequency messages, producing a fundamentally different traffic pattern from spoofing attacks. Models trained only on spoofing variants cannot reliably detect this distinct attack vector.

### 5.4 Augmentation as a Validation Tool

The near-zero F1 deltas under augmentation validate that models learn robust feature representations rather than overfitting to exact numerical values. However, detection rate shows more sensitivity to perturbation (e.g., RF's DR on DoS holdout drops from 28% to 16% with noise), suggesting that borderline attack samples are sensitive to small feature changes.

### 5.5 Limitations

1. The small post-dedup dataset limits statistical confidence for minority classes.
2. Cross-dataset studies require schema harmonization; label/feature mismatches can introduce evaluation bias.
3. Current adversarial testing is based on constrained data-space evasion perturbations; broader model-aware attacks remain to be benchmarked.
4. Full model × split × variant × feature-selection sweeps are computationally expensive and may require staged execution.

---

## 6. Conclusion

We presented a comprehensive, reproducible evaluation framework for CAN bus intrusion detection using the CICIoV2024 dataset. Our key findings are:

1. All four ML models achieve excellent classification performance (F1 > 0.99) under standard evaluation protocols.
2. Attack holdout evaluation reveals critical generalization gaps, with DoS detection rates as low as 20% for tree-based models trained without DoS examples.
3. SVM demonstrates the strongest robustness (score: 0.000232) and generalization capability, maintaining high detection rates for unseen attack types.
4. All models achieve sub-millisecond inference, confirming real-time deployment viability.
5. Augmentation-based robustness evaluation validates model stability under realistic perturbations (max F1 delta < 0.004).

We recommend a hybrid IDS architecture combining LightGBM for precision and SVM for novel attack detection. The framework and all evaluation code are openly available for reproducibility.

---

## 7. Future Work

1. **Online adaptation:** Integrate incremental and continual learning methods to adapt IDS behavior as attack patterns evolve.
2. **Deployment-aware optimization:** Quantize and prune top-performing models for low-power automotive ECUs and benchmark latency/energy trade-offs.
3. **Federated CAN IDS:** Explore privacy-preserving collaborative learning across fleets without sharing raw vehicle telemetry.
4. **Formal robustness certification:** Pair empirical adversarial testing with certifiable robustness bounds for safety-critical assurance.
5. **Protocol and architecture expansion:** Extend support beyond CAN to CAN-FD, FlexRay, and Automotive Ethernet in mixed-bus environments.

---

## 8. Implementation Status Checklist (Maintained)

- [x] **1. Deep Learning Integration (High Priority):** MLP, LSTM, and 1D-CNN models integrated into the experiment runner for head-to-head comparison with classical ML baselines.
- [x] **2. Sequence & Sliding-Window Features (High Priority):** Configurable sliding-window feature generation implemented with aggregate, flatten, hybrid, and sequence-ready outputs.
- [x] **3. Cross-Dataset Validation (Medium Priority):** Cross-dataset evaluation workflow added for CICIoV2024-trained models against Car-Hacking, OTIDS, and ROAD via dataset adapters.
- [x] **4. Adversarial Robustness Evaluation (Medium Priority):** Crafted adversarial evasion augmentation added and integrated into robustness evaluation pipelines.
- [x] **5. Feature Selection Benchmarking (Medium Priority):** Benchmark script added to compare MI, correlation filtering, and PCA and recommend minimal deployable feature settings.
- [x] **6. Final Report & Research Paper (High Priority/Ongoing):** Documentation, overview, and paper sections updated to reflect completed implementation and results framing.

---

## 9. References

[1] M. R. Aliyu, M. Usman, and A. Abba, "A survey on Internet of Vehicles: Applications, security issues and solutions," *Internet of Things*, vol. 22, 2023.

[2] J. Cui, L. S. Liew, G. Sabaliauskaite, and F. Meng, "A review on safety failures, security attacks, and available countermeasures for autonomous vehicles," *Ad Hoc Networks*, vol. 90, 2019.

[3] C. Miller and C. Valasek, "Remote exploitation of an unaltered passenger vehicle," *Black Hat USA*, 2015.

[4] W. Wu, R. Kurachi, G. Zeng, et al., "Sliding window optimized information entropy analysis method for intrusion detection on in-vehicle networks," *IEEE Access*, vol. 6, pp. 45233–45245, 2018.

[5] H. M. Song, J. Woo, and H. K. Kim, "In-vehicle network intrusion detection using deep convolutional neural network," *Vehicular Communications*, vol. 21, 2020.

[6] M. L. Han, B. I. Kwak, and H. K. Kim, "Anomaly intrusion detection method for vehicular networks based on survival analysis," *Vehicular Communications*, vol. 14, pp. 52–63, 2018.

[7] S. Rajapaksha, H. Kalutarage, M. O. Al-Kadri, et al., "AI-based intrusion detection systems for in-vehicle networks: A survey," *ACM Computing Surveys*, vol. 55, no. 11, 2023.

[8] M. D. Hossain, H. Inoue, H. Ochiai, D. Fall, and Y. Kadobayashi, "LSTM-based intrusion detection system for in-vehicle CAN bus communications," *IEEE Access*, vol. 8, pp. 185489–185502, 2020.

[9] M. Kneib and C. Huth, "Scission: Signal characteristic-based sender identification and intrusion detection in automotive networks," in *ACM CCS*, 2018.

[10] M. Müter and N. Asaj, "Entropy-based anomaly detection for in-vehicle networks," in *IEEE Intelligent Vehicles Symposium*, 2011.

[11] H. Lee, S. H. Jeong, and H. K. Kim, "OTIDS: A novel intrusion detection system for in-vehicle network by using remote frame," in *PST*, 2017.

[12] W. Choi, K. Joo, H. J. Jo, M. C. Park, and D. H. Lee, "VoltageIDS: Low-level communication characteristics for automotive intrusion detection system," *IEEE TIFS*, vol. 13, no. 8, 2018.

[13] S. Tariq, S. Lee, and S. S. Woo, "CAN-ADF: The controller area network attack detection framework," *Computers & Security*, vol. 94, 2020.

[14] M. Markovitz and A. Wool, "Field classification, modeling and anomaly detection in unknown CAN bus networks," *Vehicular Communications*, vol. 9, pp. 43–52, 2017.

[15] H. M. Song, H. R. Kim, and H. K. Kim, "Intrusion detection system based on the analysis of time intervals of CAN messages for in-vehicle network," in *ICOIN*, 2016.

[16] A. Taylor, S. Leblanc, and N. Bhatt, "Anomaly detection in automobile control network data with long short-term memory networks," in *DSAA*, 2016.

[17] H. M. Song, J. Woo, and H. K. Kim, "Car-Hacking Dataset," *IEEE DataPort*, 2018.

[18] H. Lee, S. H. Jeong, and H. K. Kim, "OTIDS: A novel intrusion detection system dataset for in-vehicle network," 2017.

[19] M. Hanselmann, T. Strauss, K. Dormann, and H. Ulmer, "CANet: An unsupervised intrusion detection system for high dimensional CAN bus data," *IEEE Access*, vol. 8, 2020.

[20] Canadian Institute for Cybersecurity, "CICIoV2024 Dataset," University of New Brunswick, 2024. Available: https://www.unb.ca/cic/datasets/iov-dataset-2024.html

[21] Y. Yan, G. Yang, Q. Yu, and L. Li, "Adversarial attacks and defenses in deep learning-based CAN intrusion detection systems: A survey," *IEEE TVT*, 2023.

---

*Manuscript prepared as part of Project Aarav — CICIoV2024 Evaluation Framework.*
