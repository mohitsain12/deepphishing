.PHONY: setup data train evaluate predict app test clean all

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete. Activate with: source venv/bin/activate"

# Download and prepare dataset
data:
	python -m src.data.download_dataset
	python -m src.data.explore_dataset
	python -m src.features.feature_engineering

# Train all models
train:
	python -m src.models.train_baseline
	python -m src.models.train_dnn

# Evaluate models
evaluate:
	python -m src.evaluation.evaluate

# Run hyperparameter optimization
optimize:
	python -m src.models.hyperparameter_tuning

# Run prediction pipeline
predict:
	python -m src.pipeline.predict_pipeline

# Launch Streamlit app
app:
	streamlit run app/streamlit_app.py

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Generate explainability reports
explain:
	python -m src.explainability.explainer

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run everything
all: data train evaluate explain
	@echo "Full pipeline complete!"
