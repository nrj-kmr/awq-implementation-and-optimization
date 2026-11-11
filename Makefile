.PHONY: help install test run ablations figures presentation slides paper all clean

# Environment detection (defaults to miniforge awq env if present, else active python)
CONDA_ENV_PY = /opt/homebrew/Caskroom/miniforge/base/envs/awq/bin/python
CONDA_ENV_PIP = /opt/homebrew/Caskroom/miniforge/base/envs/awq/bin/pip

PYTHON := $(shell if [ -f $(CONDA_ENV_PY) ]; then echo $(CONDA_ENV_PY); else which python3; fi)
PIP := $(shell if [ -f $(CONDA_ENV_PIP) ]; then echo $(CONDA_ENV_PIP); else which pip3; fi)
PDFLATEX := pdflatex

help:
	@echo "========================================================================"
	@echo " AWQ Empirical Verification Project - Automation Tasks"
	@echo " Python Interpreter: $(PYTHON)"
	@echo "========================================================================"
	@echo "  make run           Run full AWQ live demo (SmolLM-135M, W4A16, G=64)"
	@echo "  make ablations     Run 9-configuration grid ablation (N=4,3,2 x G=32,64,128)"
	@echo "  make paper         Compile IEEE Conference Paper (paper/ieee_awq_paper.pdf)"
	@echo "  make presentation  Compile LaTeX Beamer presentation (presentation/lab_presentation.pdf)"
	@echo "  make figures       Regenerate publication vector figures in figures/"
	@echo "  make all           Compile both IEEE paper and Beamer presentation"
	@echo "  make test          Execute unit test suite (5 test cases)"
	@echo "  make clean         Purge __pycache__, .pyc, and LaTeX auxiliary files"
	@echo "========================================================================"

install:
	@echo "--> Installing awq-lab in editable mode..."
	$(PIP) install -e .

test:
	@echo "--> Running unit test suite..."
	$(PYTHON) -m unittest discover tests

run:
	@echo "--> Running live AWQ experiment benchmark on Apple Silicon..."
	$(PYTHON) main_experiment.py --bits 4 --group_size 64 --num_layers 2

ablations:
	@echo "--> Running ablation study across bit-widths and group sizes..."
	$(PYTHON) run_ablations.py

figures:
	@echo "--> Re-generating academic presentation figures..."
	$(PYTHON) helpers/generate_presentation_figures.py

paper:
	@echo "--> Compiling IEEE conference research paper in paper/..."
	cd paper && $(PDFLATEX) -interaction=nonstopmode ieee_awq_paper.tex
	@echo "--> IEEE paper compiled cleanly to paper/ieee_awq_paper.pdf"

presentation: slides

slides:
	@echo "--> Compiling academic Beamer presentation in presentation/..."
	cd presentation && $(PDFLATEX) -interaction=nonstopmode lab_presentation.tex
	@echo "--> Presentation compiled cleanly to presentation/lab_presentation.pdf"

all: paper presentation

clean:
	@echo "--> Cleaning up __pycache__, bytecode, and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf build dist *.egg-info
	@echo "--> Cleaning up LaTeX auxiliary files (preserving final PDFs)..."
	rm -f presentation/*.aux presentation/*.log presentation/*.nav \
	      presentation/*.out presentation/*.snm presentation/*.toc \
	      presentation/*.vrb paper/*.aux paper/*.log paper/*.out \
	      *.aux *.log *.nav *.out *.snm *.toc *.vrb
	@echo "--> Done! Workspace is clean."
