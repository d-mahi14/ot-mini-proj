# ==============================================================================
# Makefile: Fair and Load-Balanced Student-Alumni Matching at VJTI
# Multi-Objective Capacitated Bipartite Assignment with Rawlsian Fairness & Parity
# ==============================================================================

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
SEEDS ?=

.PHONY: all install test data baselines exact benchmark sensitivity interpret figures clean help

# Default target
all: data baselines test exact benchmark sensitivity interpret figures
	@echo ==============================================================================
	@echo  FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY!
	@echo ==============================================================================

help:
	@echo Available Make Targets:
	@echo   install      - Install project dependencies from requirements.txt
	@echo   test         - Run pytest verification suite on tests/
	@echo   data         - Generate/verify synthetic benchmark datasets in data/
	@echo   baselines    - Run baseline heuristics (Random, Greedy, Min-Cost Flow)
	@echo   exact        - Run exact HiGHS MILP Pareto frontier and secondary scenario
	@echo   benchmark    - Run multi-objective evolutionary algorithms benchmark (opt: SEEDS=1)
	@echo   sensitivity  - Run parameter sensitivity and coverage-gap analysis
	@echo   interpret    - Run Pareto frontier interpretation and institutional policy report
	@echo   figures      - Generate visualizations and compile publication-grade PDFs
	@echo   all          - Execute full end-to-end research and verification pipeline
	@echo   clean        - Remove temporary build files, bytecode, and test caches

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

data:
	$(PYTHON) -m src.data_generator

baselines:
	$(PYTHON) main.py --mode baselines

exact:
	$(PYTHON) main.py --mode exact

benchmark:
	$(PYTHON) main.py --mode run $(if $(SEEDS),--seeds $(SEEDS),)

sensitivity:
	$(PYTHON) main.py --mode sensitivity

interpret:
	$(PYTHON) main.py --mode interpretation --instance small_50x30

figures:
	$(PYTHON) generate_figures.py

clean:
	@$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; [p.unlink() for p in pathlib.Path('.').rglob('.coverage*')]; print('Build artifacts and caches cleaned.')"
