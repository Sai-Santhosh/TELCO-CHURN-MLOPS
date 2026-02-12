# Makefile - Telco Churn + Experimentation Decision Studio

PYTHON := python
PIP := pip
PROJECT_ROOT := .

.PHONY: help install train experiment-demo experiment-dashboard experiment-analyze test

help:
	@echo "Telco Churn + Experimentation Decision Studio"
	@echo "=============================================="
	@echo "  make install            - Install dependencies"
	@echo "  make train              - Train churn model"
	@echo "  make experiment-demo    - Run simulator + analysis + report"
	@echo "  make experiment-dashboard - Start Streamlit dashboard"
	@echo "  make experiment-analyze - Run analysis (use EXP_ID=xxx)"
	@echo "  make test               - Run pytest"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

train:
	$(PYTHON) pipelines/sklearn_pipeline.py

experiment-demo:
	$(PYTHON) scripts/run_experiment_demo.py

experiment-dashboard:
	$(PYTHON) -m streamlit run apps/experiment_dashboard.py --server.port 8501

experiment-analyze:
	@if [ -z "$(EXP_ID)" ]; then \
		echo "Usage: make experiment-analyze EXP_ID=your_experiment_id"; \
		exit 1; \
	fi
	$(PYTHON) -c "from src.experimentation.analyze import run_analysis; from src.experimentation.report import render_exec_summary; r=run_analysis(experiment_id='$(EXP_ID)'); render_exec_summary(r.to_dict(), '$(EXP_ID)')"

test:
	$(PYTHON) -m pytest tests -v --tb=short -q
