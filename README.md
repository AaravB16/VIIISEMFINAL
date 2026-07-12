# Developing a Standardized and Reproducible Evaluation Framework for IoV Intrusion Detection Using the CICIoV2024 Dataset

## Overview

This project implements a standardized evaluation framework for benchmarking Machine Learning-based Intrusion Detection Systems (IDS) on the CICIoV2024 dataset. It addresses methodological inconsistencies in existing IoV intrusion detection research by providing:

- **Consistent preprocessing** pipeline with configurable deduplication and normalization
- **Multiple data splitting strategies**: random, scenario-based, and attack-holdout
- **Unified feature construction** with frame-level, sliding-window flattening, sliding-window aggregation, and hybrid representations
- **Benchmark model evaluation**: Random Forest, SVM, XGBoost, LightGBM, plus optional deep learning models (MLP/LSTM/1D-CNN)
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and Inference Latency
- **Cross-dataset validation pipeline** for CIC-trained models against external CAN IDS datasets (Car-Hacking, OTIDS, ROAD)
- **Robustness evaluation** across realistic augmentations and adversarial-evasion perturbations
- **Feature-selection benchmarking** to identify minimal deployable feature sets
- **Full reproducibility** via YAML configuration, fixed random seeds, and documented experimental parameters

## Project Structure

```
Project_Aarav/
├── CICIoV2024/              # Dataset (decimal, hexadecimal, binary formats)
├── config/
│   └── config.yaml          # Experiment configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading utilities
│   ├── preprocessing.py     # Preprocessing pipeline
│   ├── feature_engineering.py # Feature extraction
│   ├── feature_selection.py # Feature selection pipeline (MI/correlation/PCA)
│   ├── splitting.py         # Data splitting strategies
│   ├── models.py            # Model implementations
│   ├── deep_learning.py     # MLP/LSTM/1D-CNN model wrappers
│   ├── cross_dataset.py     # External dataset canonicalization and loading
│   ├── adversarial.py       # Adversarial CAN perturbation utilities
│   ├── evaluation.py        # Metrics and reporting
│   ├── augmentation_engine.py # Jitter/noise/scaling/adversarial variants
│   ├── robustness.py        # Robustness delta analysis
│   └── utils.py             # Configuration and helpers
├── scripts/
│   ├── run_experiment.py    # Main experiment runner
│   ├── generate_report.py   # Report generation
│   ├── run_cross_dataset_validation.py # Train on CIC, test on external datasets
│   └── benchmark_feature_selection.py  # FS strategy benchmark + recommendation
├── notebooks/               # Jupyter notebooks for exploration
├── results/                 # Experiment outputs
├── requirements.txt
├── Synopsis.docx
└── README.md
```

## Setup

```bash
# Activate virtual environment
source .env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Experiment

```bash
python scripts/run_experiment.py
```

### Run Head-to-Head Including Deep Learning Models

```bash
python scripts/run_experiment.py --enable-dl
```

### Run Specific Model and Split

```bash
# Only Random Forest with random split
python scripts/run_experiment.py --model random_forest --split random

# Only XGBoost with attack holdout
python scripts/run_experiment.py --model xgboost --split attack_holdout
```

### Custom Configuration

```bash
python scripts/run_experiment.py --config config/custom_config.yaml
```

### Generate Report

```bash
python scripts/generate_report.py
python scripts/generate_report.py --output results/report.txt
```

### Cross-Dataset Validation (CICIoV2024 → External Datasets)

```bash
python scripts/run_cross_dataset_validation.py --allow-missing
```

### Feature-Selection Benchmarking

```bash
python scripts/benchmark_feature_selection.py --split random
```

## Configuration

All experimental parameters are defined in `config/config.yaml`:

- **seed**: Random seed for reproducibility (default: 42)
- **dataset**: File paths and format selection
- **preprocessing**: Duplicate removal, normalization method
- **features**: Frame-level columns, sliding window settings
- **splitting**: Test size, strategies (random/scenario/attack_holdout)
- **models**: Enable/disable models and set hyperparameters
- **cross_dataset_validation**: External dataset paths and label mapping
- **augmentation**: Includes adversarial-evasion variant settings
- **evaluation**: Metric selection, averaging method

## Dataset

The CICIoV2024 dataset from the Canadian Institute for Cybersecurity captures CAN bus traffic from a 2019 Ford vehicle. It includes:

- **Benign**: Normal driving behavior (~1.2M frames)
- **DoS**: Denial-of-Service attacks (~74K frames)
- **Spoofing-GAS**: Gas pedal spoofing (~10K frames)
- **Spoofing-RPM**: Engine RPM spoofing (~55K frames)
- **Spoofing-SPEED**: Speed spoofing (~25K frames)
- **Spoofing-STEERING_WHEEL**: Steering wheel spoofing (~20K frames)

## Reproducibility

Every experiment run produces:
- `results/experiment_results.json` — Full metrics for all model/split combinations
- Console summary table with key metrics
- All random seeds, split indices, and configuration archived for transparency

## Remaining Work Checklist (Maintained)

- **DONE** — Deep Learning Integration (MLP/LSTM/1D-CNN integrated into main runner with per-model shape handling)
- **DONE** — Sequence & Sliding-Window Features (flatten/aggregate/hybrid window representations implemented)
- **DONE** — Cross-Dataset Validation (external dataset loader + validation runner implemented)
- **DONE** — Adversarial Robustness Evaluation (adversarial-evasion augmentation strategy implemented)
- **DONE** — Feature Selection Benchmarking (automated benchmark + minimal-feature recommendation script implemented)
- **DONE (ongoing maintenance)** — Final Report & Research Paper polish/documentation updates
