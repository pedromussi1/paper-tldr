# Convenience targets. Works in WSL / Git Bash on Windows; `make` is preinstalled
# on the GitHub Actions ubuntu-latest runner.
#
# All Python invocations use `python` from the active venv. Activate with
# `.venv/Scripts/Activate.ps1` (PowerShell) or `source .venv/bin/activate` (bash).

PYTHON ?= python
SUMMARIZE ?= scripts/summarize_results.py

.PHONY: help test lint data eda train summarize curves all

help:
	@echo "Targets:"
	@echo "  make test       - run unit tests"
	@echo "  make lint       - run ruff lint"
	@echo "  make data       - download SciTLDR and write data/processed/*.jsonl"
	@echo "  make eda        - length distributions and sample pairs"
	@echo "  make train      - QLoRA training with configs/base.yaml"
	@echo "  make summarize  - print the canonical val results table"
	@echo "  make curves     - plot train/eval loss for the canonical r=16 run"
	@echo "  make all        - lint + test"

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m ruff check src scripts tests

data:
	$(PYTHON) scripts/prepare_data.py

eda:
	$(PYTHON) scripts/eda.py

train:
	$(PYTHON) scripts/run_train.py --config configs/base.yaml

summarize:
	$(PYTHON) $(SUMMARIZE)

curves:
	$(PYTHON) scripts/plot_training_curves.py

all: lint test
