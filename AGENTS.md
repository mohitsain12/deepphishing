# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/`, organized by pipeline stage: `data/` for acquisition and EDA, `features/` for URL feature extraction, `models/` for baseline and neural-network training, `evaluation/`, `pipeline/`, `explainability/`, and shared `utils/`. The Streamlit UI is `app/streamlit_app.py`. Configuration belongs in `config/config.yaml`; keep datasets under `data/raw/` or `data/processed/`, model artifacts in `models/saved/`, and generated figures/reports in `reports/`. Tests live in `tests/`.

Run modules from the repository root, for example `python -m src.features.feature_engineering`, so package imports resolve consistently.

## Build, Test, and Development Commands

- `make setup` creates `venv` and installs dependencies; use `pip install -e .` when developing the package.
- `make data` downloads/prepares data, runs EDA, and writes processed feature matrices.
- `make train` trains baseline models and the DNN; `make evaluate` generates evaluation outputs.
- `make app` starts the local Streamlit interface.
- `make test` runs `pytest tests/ -v --cov=src --cov-report=html`. For a faster check, use `pytest tests/ -v`.

## Coding Style & Naming Conventions

Target Python 3.9+ and use four-space indentation. Follow the existing style: `snake_case` for variables, functions, modules, and test names; `PascalCase` for classes (for example, `URLFeatureExtractor`); and `UPPER_SNAKE_CASE` for constants. Keep type and return contracts stable for pipeline-facing code. Add focused docstrings for public classes or non-obvious feature calculations. No formatter or linter is configured, so preserve surrounding formatting and avoid unrelated rewrites.

## Testing Guidelines

Use pytest. Add tests in `tests/test_<area>.py` with functions named `test_<behavior>`. Cover normal URLs plus malformed, adversarial, and phishing-like inputs when changing feature extraction or prediction behavior. Keep tests deterministic and avoid requiring network downloads or retraining. Run the relevant test file first, then `make test` before submitting changes.

## Commit & Pull Request Guidelines

This checkout contains no Git metadata, so no repository-specific historical convention can be verified. Use short imperative commit subjects such as `Add suspicious TLD feature`. Keep commits focused. Pull requests should explain the behavioral change, identify affected pipeline stages/configuration, link the relevant issue when available, include test results, and attach Streamlit screenshots for UI changes. Do not commit datasets, trained-model binaries, credentials, or generated coverage output unless explicitly requested.

## Security & Configuration

Treat URLs and uploaded CSVs as untrusted input. Keep secrets out of `config/config.yaml` and source control; use environment variables for any future credentials. Review feature and model-path changes against `config/config.yaml` to prevent mismatched artifacts.
