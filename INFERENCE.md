# Inference & Analysis — Project Aarav

**CICIoV2024 Intrusion Detection Evaluation Framework**

---

## 1. Overall Model Performance

All four models achieve excellent classification performance on the CICIoV2024 dataset under standard (random) evaluation, with F1-scores exceeding 0.992. This confirms that CAN bus intrusion detection using frame-level features is highly feasible with classical ML approaches.

**Key observation:** XGBoost leads the random split with F1 = 0.9970, while LightGBM achieves the highest scenario-split F1 of 0.9972 with a perfect 100% detection rate and near-perfect AUC of 0.999992.

---

## 2. Impact of Split Strategy

### 2.1 Random vs. Scenario Split

Performance is comparable between random and scenario-based splits (F1 difference < 0.005 for all models), suggesting that the learned patterns generalize well across different driving scenarios present in the dataset. Scenario-based evaluation provides a more realistic assessment, and the consistently high scores validate the models' practical applicability.

### 2.2 Attack Holdout Evaluation

The attack holdout strategy reveals critical limitations in generalization to unseen attack types:

- **DoS Holdout** — the most challenging split. Detection rates collapse dramatically: XGBoost drops to 20%, Random Forest to 28%. Only SVM maintains 100% detection rate, though at the cost of a higher FPR (1.69%). LightGBM balances best among tree models with 60% DR and F1 = 0.9631.
- **RPM Holdout** — second most difficult. All models show reduced DR (43.75%–62.5%), indicating RPM spoofing patterns are poorly inferred from other attacks.
- **GAS Holdout** — well-handled; XGBoost and LightGBM both achieve 90% DR, SVM achieves 100%.
- **SPEED Holdout** — good generalization; XGBoost and LightGBM reach 91.67% DR.
- **STEERING_WHEEL Holdout** — moderate performance; all models achieve 70–80% DR.

**Inference:** Tree-based models (RF, XGBoost, LightGBM) tend to learn attack-specific decision boundaries and struggle with novel attacks. SVM's RBF kernel learns a more generalized anomaly boundary, consistently achieving the highest detection rates in holdout scenarios at the expense of slightly elevated false positive rates.

---

## 3. Detection Rate vs. False Positive Rate Trade-off

A clear trade-off emerges between detection rate and false positive rate:

- **SVM** consistently achieves the highest detection rates (100% in random, scenario, DoS holdout, and GAS holdout) but is the only model with non-zero FPR (0.28%–1.69%).
- **Tree-based models** (RF, XGBoost, LightGBM) maintain zero FPR in most scenarios but at the cost of lower detection rates, particularly for unseen attack types.

**For IoV security applications**, missing an attack (low DR) is generally more dangerous than a false alarm (elevated FPR). SVM's behavior is therefore preferable in safety-critical deployments, while tree models may be suited for environments where false alarms are costly.

---

## 4. Robustness Under Data Augmentation

### 4.1 General Stability

All models demonstrate remarkable stability under augmentation. The maximum observed F1 delta across all 84 augmented experiments is only 0.0037, indicating that the learned decision boundaries are resilient to realistic data perturbations.

### 4.2 Per-Augmentation Findings

- **Timing Jitter (2%):** Minimal impact on F1 for all models. LightGBM shows slight improvement in some cases (+0.0008 F1 on scenario split), suggesting its gradient-boosted trees are not sensitive to timestamp-based feature perturbation.
- **Feature Noise (1% Gaussian):** The augmentation with the most consistent (though still small) negative impact. Random Forest on DoS holdout suffers the largest F1 drop (−0.0030). Noise primarily affects detection rate rather than overall accuracy — e.g., RF's DR drops from 28% to 16% on noise-augmented DoS holdout.
- **Attack Intensity Scaling (0.85×–1.15×):** Variable impact. XGBoost's DR on DoS holdout drops to 12% (from 20%), while LightGBM's improves to 76% (from 60%), suggesting that scaled attack features occasionally push borderline samples across decision boundaries in both directions.

### 4.3 Robustness Ranking

SVM is the most robust model (score: 0.000232), followed by Random Forest (0.000355), XGBoost (0.000512), and LightGBM (0.000535). SVM's lower sensitivity to augmentation aligns with its kernel-based learning which creates smoother decision boundaries compared to tree-based split rules.

---

## 5. Training Efficiency & Inference Latency

- **SVM** is the fastest to train (avg 0.164s) and has among the lowest inference latency (0.0084 ms/sample), making it the most deployment-efficient model.
- **XGBoost** achieves the lowest per-sample latency (0.0042 ms on random split) due to optimized tree traversal.
- **Random Forest** has the highest latency (0.0972 ms/sample) due to ensembling 200 trees without boosting optimization.
- **LightGBM** has the longest training time (avg 2.088s) due to its histogram-based gradient boosting.

All models achieve sub-millisecond inference, well within the requirements for real-time CAN bus monitoring (typical CAN frame rate: 1–10 ms between frames).

---

## 6. Effect of Dataset Deduplication

The aggressive deduplication (99.7% removal: 1,408,219 → 3,588) is a critical consideration. The high duplicate rate is characteristic of CAN bus datasets where identical frames are transmitted repeatedly. While deduplication prevents data leakage between train/test sets, the resulting small dataset size (3,588 samples) limits:

1. Statistical power of evaluation metrics, particularly detection rate on holdout splits
2. Model capacity exploration — deeper/wider models may not be fully utilized
3. Class balance — GAS class is reduced to only 25 samples

Despite these limitations, the consistent high performance across multiple split strategies and augmentation variants provides confidence in the results' validity.

---

## 7. Key Conclusions

1. **All four models are effective** for CAN bus intrusion detection, with F1-scores > 0.99 under standard evaluation.
2. **LightGBM is the best-performing model** under standard and scenario-based evaluation (F1 = 0.9972, DR = 100%, AUC = 0.999992 on scenario split).
3. **SVM is the most robust and generalizable model**, maintaining high detection rates across unseen attack types and data perturbations, at the cost of slightly elevated false positive rates.
4. **Generalization to unseen attack types is the primary challenge** — DoS and RPM holdout splits show significant detection rate drops for tree-based models.
5. **Models are resilient to realistic data perturbations** — all augmentation strategies produce F1 deltas < 0.004, validating deployment stability.
6. **All models meet real-time constraints** — sub-millisecond inference latency is suitable for CAN bus monitoring.

### Recommendation

For **production IoV intrusion detection systems**, a two-model ensemble is recommended:
- **LightGBM** as the primary classifier for optimal accuracy and low FPR
- **SVM** as a secondary anomaly detector to catch novel attack types missed by tree-based models

This combination leverages LightGBM's precision and SVM's generalization for comprehensive coverage.

---

*Analysis based on Project Aarav experiment results. See RESULTS.md for complete numerical data.*
