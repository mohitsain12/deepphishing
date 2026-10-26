# PhishGuard AI

PhishGuard AI is a Python and Streamlit application for detecting phishing URLs from lexical, structural, and information-theoretic features. It supports a deep neural network, conventional machine-learning baselines, model evaluation, and local feature attribution.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the complete data-preparation flow:

```bash
make data
```

Train and evaluate models:

```bash
make train
make evaluate
```

Start the local web application:

```bash
make app
```

The interface provides single-URL scanning, model performance metrics, and explainable-AI feature attribution.

## Testing

```bash
make test
```

## Repository layout

- `src/`: data preparation, features, models, evaluation, inference, and explainability.
- `app/streamlit_app.py`: interactive single-URL detector.
- `config/config.yaml`: paths, feature flags, and training configuration.
- `data/`: raw inputs and processed arrays.
- `models/saved/`: trained-model and evaluation artifacts.
- `reports/`: EDA, training, evaluation, and XAI figures.
- `tests/`: pytest coverage for URL feature extraction and prediction contracts.
