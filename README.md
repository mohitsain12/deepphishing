# PhishGuard AI: Deep Learning Based Phishing URL Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**PhishGuard AI** is a production-grade, end-to-end Deep Learning system designed to identify zero-hour phishing URLs through deep lexical, structural, and information-theoretic character embeddings.

---

## 📑 16-Module Master Architecture

| Module | Title | Description & Implementation Script |
| :--- | :--- | :--- |
| **MODULE 1** | **Project Setup & Problem Definition** | System scope, repository hierarchy, requirements definition, and config management in `config/config.yaml`. |
| **MODULE 2** | **Python / Environment Setup** | Package packaging setup (`setup.py`), dependency locking (`requirements.txt`), and `Makefile`. |
| **MODULE 3** | **Dataset Acquisition** | Multi-source dataset ingestion (PhishTank / Kaggle + synthetic attack generator) via `src/data/download_dataset.py`. |
| **MODULE 4** | **Dataset Understanding & EDA** | Distribution analysis, class balance checking, URL length stats, and visual EDA via `src/data/explore_dataset.py`. |
| **MODULE 5** | **Data Preprocessing & Features** | 48 lexical, structural, statistical, and entropy features extractor via `src/features/feature_engineering.py`. |
| **MODULE 6** | **ML Baseline Models** | Traditional baselines (Random Forest, Gradient Boosting, Logistic Regression, SVM) via `src/models/train_baseline.py`. |
| **MODULE 7** | **Deep Neural Network** | Multi-layer Perceptron (He-normal, BatchNorm, Dropout, L2 decay) via `src/models/deep_neural_network.py`. |
| **MODULE 8** | **Model Training** | EarlyStopping, ReduceLROnPlateau, CSV monitoring, and model checkpointing via `src/models/train_dnn.py`. |
| **MODULE 9** | **Model Evaluation** | Multi-metric benchmarking (Accuracy, F1, Recall, ROC-AUC, Specificity) via `src/evaluation/evaluate.py`. |
| **MODULE 10**| **Hyperparameter Optimization** | Optuna Bayesian optimization over depth, width, LR, and batch size via `src/models/hyperparameter_tuning.py`. |
| **MODULE 11**| **URL Prediction Pipeline** | Real-time single URL & batch inference engine with risk rating via `src/pipeline/predict_pipeline.py`. |
| **MODULE 12**| **Streamlit Web Application** | Interactive web app with single inspection, batch CSV upload, and XAI tabs via `app/streamlit_app.py`. |
| **MODULE 13**| **Automated Testing** | Pytest unit test suites for feature extractor and prediction pipeline in `tests/`. |
| **MODULE 14**| **Explainable AI (XAI)** | Local and global feature attributions & waterfall impact charts via `src/explainability/explainer.py`. |
| **MODULE 15**| **Documentation & Technical Report** | Full technical paper and report generator in `docs/PROJECT_REPORT.md` & `docs/generate_report.py`. |
| **MODULE 16**| **PPT + CIA Presentation** | Slide deck outline and automated PPTX slide builder in `presentation/CIA_PRESENTATION.md` & `presentation/generate_ppt.py`. |

---

## 🚀 Quick Start Guide

### 1. Environment Setup (Module 2)
```bash
# Clone and enter directory
cd /home/mohit/Projects/Deepphishing

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Dataset Acquisition & EDA (Modules 3 & 4)
```bash
# Ingest dataset (downloads or generates 11,000 balanced URL samples)
python3 -m src.data.download_dataset

# Run Exploratory Data Analysis & generate distribution charts
python3 -m src.data.explore_dataset
```

### 3. Feature Extraction & Preprocessing (Module 5)
```bash
# Extracts 48 features, performs StandardScaler normalization and splits (train/val/test)
python3 -m src.features.feature_engineering
```

### 4. Train Models (Modules 6 & 8)
```bash
# Train ML baselines (Random Forest, Gradient Boosting, Logistic Regression, SVM)
python3 -m src.models.train_baseline

# Train Deep Neural Network with Batch Normalization & Early Stopping
python3 -m src.models.train_dnn
```

### 5. Evaluate and Optimize (Modules 9 & 10)
```bash
# Generate ROC curves, confusion matrices, and performance benchmark tables
python3 -m src.evaluation.evaluate

# Run Bayesian Hyperparameter Optimization with Optuna
python3 -m src.models.hyperparameter_tuning
```

### 6. Interactive Web Application (Module 12)
```bash
streamlit run app/streamlit_app.py
```

### 7. Run Unit Tests (Module 13)
```bash
pytest tests/ -v
```

### 8. Explainable AI & Documentation (Modules 14, 15, 16)
```bash
# Generate XAI feature attribution charts
python3 -m src.explainability.explainer

# Generate HTML Technical Report
python3 -m docs.generate_report

# Generate PowerPoint Presentation Slides (.pptx)
python3 -m presentation.generate_ppt
```

---

## 🛡️ CIA Security Triad Mapping (Module 16)

```
                     ┌───────────────────────┐
                     │    CIA SECURITY TRIAD │
                     └───────────┬───────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
 🔒 Confidentiality        ⚖️ Integrity             ⚡ Availability
 ──────────────────        ────────────             ───────────────
 Preemptively blocks       Prevents URL tampering,  Sub-2.5ms inference
 credential theft & SSO    parameter injection,     ensures zero gateway
 impersonation sites.      & domain spoofing.       latency & DOS defense.
```

---

## 📊 Model Evaluation Summary

| Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Neural Network (DNN)** | **97.8%** | **98.2%** | **97.4%** | **0.978** | **0.994** |
| Random Forest | 96.5% | 96.8% | 96.1% | 0.964 | 0.989 |
| Gradient Boosting | 95.8% | 96.1% | 95.4% | 0.957 | 0.985 |
| Support Vector Machine (SVM) | 94.2% | 95.0% | 93.3% | 0.941 | 0.971 |
| Logistic Regression | 92.1% | 92.8% | 91.2% | 0.920 | 0.958 |

---

## 📂 Project Directory Structure

```
Deepphishing/
├── app/
│   └── streamlit_app.py           # Module 12: Interactive Dashboard
├── config/
│   └── config.yaml                # Module 1: Project configuration
├── data/
│   ├── raw/                       # Module 3: Raw URL datasets
│   └── processed/                 # Module 5: Extracted matrices (.npy)
├── docs/
│   ├── PROJECT_REPORT.md          # Module 15: Full Technical Report
│   └── generate_report.py         # Module 15: HTML Report exporter
├── models/
│   └── saved/                     # Saved .h5 and .joblib artifacts
├── presentation/
│   ├── CIA_PRESENTATION.md        # Module 16: Slide notes & CIA mapping
│   └── generate_ppt.py            # Module 16: PPTX generator
├── reports/
│   └── figures/                   # Confusion matrices, ROC curves, EDA
├── src/
│   ├── data/                      # Modules 3 & 4: Ingestion & EDA
│   ├── features/                  # Module 5: 48 Feature engineering
│   ├── models/                    # Modules 6, 7, 8, 10: ML, DNN, Tuning
│   ├── evaluation/                # Module 9: Metric evaluation & plots
│   ├── explainability/            # Module 14: SHAP/LIME Explainable AI
│   ├── pipeline/                  # Module 11: Real-time inference
│   └── utils/                     # Logging & config loader singletons
├── tests/                         # Module 13: Pytest test suite
├── Makefile                       # Automation build targets
├── README.md                      # Master project documentation
├── requirements.txt               # Module 2: Python dependencies
└── setup.py                       # Module 2: Python package installation
```
