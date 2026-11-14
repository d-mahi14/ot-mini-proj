"""Build script to generate a complete, publication-grade LaTeX project report
(approx 25-30+ pages) for Fair & Load-Balanced Student-Alumni Matching at VJTI.
"""

import os
from pathlib import Path

LATEX_DIR = Path("report/latex")
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# We will create modular, rigorous LaTeX files:
# 1. main.tex: Document preamble, packages, title page, table of contents, list of figures, list of tables
# 2. ch1_introduction.tex: Institutional context, motivation, research questions, scope & ethics
# 3. ch2_formulation.tex: Detailed MO-CBAP mathematical formulation, variables, objectives, polyhedral constraints
# 4. ch3_algorithms.tex: Algorithmic suite (Random, Greedy, MCF & repair, MILP HiGHS, NSGA-II, SPEA2, MOEA/D, Hybrid Memetic)
# 5. ch4_experiments.tex: Benchmark setup, calibration of theta & delta, shared HV reference, multi-seed statistical methodology
# 6. ch5_results.tex: Comprehensive empirical results across Small, Medium, Large instances, Pareto analysis, boxplots, tables
# 7. ch6_ablation_sensitivity.tex: Component ablation (seeding vs local search), delta sensitivity, corporate diversity, scalability
# 8. ch7_discussion_conclusion.tex: Engineering implications for VJTI, critical limitations, future directions
# 9. references.bib: Comprehensive bibliography

print("Generating LaTeX report files...")

