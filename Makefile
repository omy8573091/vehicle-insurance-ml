# Makefile
.PHONY: init setup data train serve test clean

init:
	@echo "Initializing project..."
	git init
	dvc init
	mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts --host 0.0.0.0 &
	python -m pip install -e .

setup:
	@echo "Setting up environment..."
	python -m pip install -r requirements.txt
	pre-commit install

data:
	@echo "Running data pipeline..."
	dvc repro data_processing

train:
	@echo "Training model..."
	dvc repro train_model

serve:
	@echo "Serving model..."
	mlflow models serve -m "models:/insurance-risk/production" -p 5000 --no-conda

test:
	@pytest tests/ -v --cov=src --cov-report=html

clean:
	@echo "Cleaning up..."
	dvc gc -f
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

monitor:
	@echo "Starting monitoring..."
	docker-compose -f infrastructure/docker-compose.yml up -d