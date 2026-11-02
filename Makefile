.PHONY: help install dev train train-full api ui test lint format docker-up docker-down clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

dev:  ## Install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

train:  ## Train all models (uses data/Fake.csv + data/True.csv if present, else demo sample)
	$(PY) -m ml.train

api:  ## Run the FastAPI backend with auto-reload
	uvicorn backend.main:app --reload --port 8000

ui:  ## Run the Streamlit frontend
	streamlit run frontend/streamlit_app.py --server.port 8501

test:  ## Run the test suite with coverage
	pytest --cov

lint:  ## Lint and check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-format and fix lint issues
	ruff format .
	ruff check --fix .

docker-up:  ## Build and start the full stack
	docker compose up --build -d

docker-down:  ## Stop the stack
	docker compose down

clean:  ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
