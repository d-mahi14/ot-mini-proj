# Fair and Load-Balanced Student-Alumni Matching at VJTI
### Title: Fair and Load-Balanced Student-Alumni Matching at VJTI: Multi-Objective Capacitated Bipartite Assignment with Rawlsian Fairness & Departmental Parity

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: Pytest](https://img.shields.io/badge/pytest-31%20passed-brightgreen.svg)](https://pytest.org/)
[![Build: Windows & POSIX](https://img.shields.io/badge/Build-Makefile%20%7C%20run__all.bat-success.svg)](run_all.bat)

> **Academic Notice**: *ARTIFICIAL - FOR TESTING ONLY*. All profiles, compatibility scores, and identifiers generated in this repository are synthetic and utilize pseudonymous IDs only (`S001`, `A001`). No real personal identifiable information (PII) is utilized.

---

## Table of Contents
1. [Group Details](#1-group-details)
2. [Problem Statement](#2-problem-statement)
3. [Formulation Summary](#3-formulation-summary)
4. [Methods Table](#4-methods-table)
5. [Repository Structure](#5-repository-structure)
6. [Requirements](#6-requirements)
7. [Installation](#7-installation)
8. [How to Run](#8-how-to-run)
9. [Data](#9-data)
10. [Results with Relative Figure Links and Tables](#10-results-with-relative-figure-links-and-tables)
11. [Individual Contributions](#11-individual-contributions)
12. [Weekly Progress Log](#12-weekly-progress-log)
13. [References](#13-references)

---

## 1. Group Details

**Course**: Optimization Techniques (UG/PG Elective)  
**Institution**: Veermata Jijabai Technological Institute (VJTI), Matunga, Mumbai  
**Academic Year**: 2026–2027  
**Department**: Department of Production & Industrial Engineering / Department of Computer Engineering  
**Group Identifier**: `[Group XX]`

| # | Student Name | Roll Number | GitHub Username | Institutional Email |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `[Member 1 Name]` | `[Roll No. 1]` | `[@username1]` | `[member1@vjti.ac.in]` |
| 2 | `[Member 2 Name]` | `[Roll No. 2]` | `[@username2]` | `[member2@vjti.ac.in]` |
| 3 | `[Member 3 Name]` | `[Roll No. 3]` | `[@username3]` | `[member3@vjti.ac.in]` |
| 4 | `[Member 4 Name]` | `[Roll No. 4]` | `[@username4]` | `[member4@vjti.ac.in]` |

---

## 2. Problem Statement

Alumni mentorship initiatives at premier engineering institutes like **Veermata Jijabai Technological Institute (VJTI), Mumbai** connect graduating engineering students with practicing alumni across 8 academic departments (`CE`, `IT`, `EXTC`, `EE`, `ME`, `CEV`, `PE`, `TT`). Traditional matching relies on greedy score maximization or ad-hoc first-come-first-served scheduling, inducing three structural pathologies:

1. **The "Superstar Alumnus" Bottleneck & Mentor Burnout**: A handful of high-profile alumni are severely overloaded beyond sustainable capacity, leading to mentor disengagement and high attrition.
2. **Peripheral Student Neglect**: Students with average GPAs or non-computing backgrounds receive negligible match compatibility scores, violating egalitarian institutional welfare.
3. **Cross-Departmental Inequity**: Computing departments (`CE`, `IT`) dominate high-scoring alumni; core engineering departments (`PE`, `CEV`, `TT`, `ME`, `EE`) experience depressed average match scores due to smaller industry alumni networks.

This project delivers an end-to-end multi-objective optimization system solving the **Multi-Objective Capacitated Bipartite Assignment Problem (MO-CBAP)** with:
- **Utilitarian Efficiency ($Z_1$)**: Maximize total cumulative match compatibility.
- **Rawlsian Egalitarian Fairness ($Z_2$)**: Maximize the match quality of the least-advantaged student ($\min_i Q_i$).
- **Workload Balancing ($Z_3$)**: Minimize peak mentorship load across alumni mentors ($\max_j L_j$).

---

## 3. Formulation Summary

Let $\mathcal{S} = \{1, \dots, N\}$ denote the student population and $\mathcal{A} = \{1, \dots, M\}$ represent the alumni mentor cohort. The university spans 8 academic departments $\mathcal{B} = \{\text{CE}, \text{IT}, \text{EXTC}, \text{EE}, \text{ME}, \text{CEV}, \text{PE}, \text{TT}\}$ and 12 employer corporations $\mathcal{K}$.

### 3.1 Decision Variables
- $x_{ij} \in \{0, 1\}$: Binary variable indicating if student $i \in \mathcal{S}$ is assigned to alumnus $j \in \mathcal{A}$.
- $y_j \in \{0, 1\}$: Binary activation status of alumnus mentor $j \in \mathcal{A}$.
- $Q_i = \sum_{j=1}^M q_{ij} x_{ij}$: Total match quality realized by student $i$.
- $L_j = \sum_{i=1}^N x_{ij}$: Workload assigned to alumnus $j$.

### 3.2 Objectives
$$\min \mathbf{F}(\mathbf{x}) = \begin{bmatrix} -Z_1(\mathbf{x}) \\ -Z_2(\mathbf{x}) \\ +Z_3(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} -\sum_{i=1}^N \sum_{j=1}^M q_{ij} x_{ij} \\ -\min_{i \in \mathcal{S}} Q_i \\ +\max_{j \in \mathcal{A}} L_j \end{bmatrix}$$

### 3.3 Operational Constraints
1. **Exact Degree**: $\sum_{j=1}^M x_{ij} = K, \quad \forall i \in \mathcal{S} \quad (K=2 \text{ distinct mentors per student})$.
2. **Alumni Capacity & Activation**: $\sum_{i=1}^N x_{ij} \le C_j y_j, \quad \forall j \in \mathcal{A} \quad \text{with } x_{ij} \le y_j$.
3. **Consent Compliance**: $x_{ij} \le c_j, \quad \forall i \in \mathcal{S}, \; \forall j \in \mathcal{A} \quad (c_j \in \{0, 1\} \text{ opt-in flag})$.
4. **Quality Threshold**: $x_{ij} = 0 \quad \forall (i, j) \text{ such that } q_{ij} < q_{\min} \quad (q_{\min} = 0.25)$.
5. **Student Quality Floor**: $Q_i = \sum_{j=1}^M q_{ij} x_{ij} \ge \theta, \quad \forall i \in \mathcal{S} \quad (\theta = 0.70)$.
6. **Cross-Departmental Parity**: $|\overline{Q}_b - \overline{Q}_{b'}| \le \delta, \quad \forall b, b' \in \mathcal{B} \quad (\delta = 0.35)$.
7. **Employer Diversity Cap**: $\sum_{j \in \mathcal{A}_k} x_{ij} \le \text{cap}_{\text{comp}}, \quad \forall i \in \mathcal{S}, \; \forall k \in \mathcal{K} \quad (\text{cap}_{\text{comp}} = 1 \text{ match per company})$.

---

## 4. Methods Table

| Method Name | Algorithmic Paradigm | Objective Handling | Constraint Handling | Computational Complexity | Key Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | Stochastic Sampling | Random heuristic | Degree preservation + random repair | $\mathcal{O}(N \cdot K)$ | Instant generation; baseline floor | Frequently violates capacity & diversity ($CV > 0$) |
| **Greedy Baseline** | Priority Heuristic | Utilitarian greedy (Max $q_{ij}$) | Score-priority reservation | $\mathcal{O}(N \cdot M \log M)$ | Fast runtime ($<0.05$s); guarantees degree & capacity | Suffers Rawlsian neglect; high Gini inequality |
| **Min-Cost Flow (raw)** | Network Simplex / LP | Single-objective Max $Z_1$ | Exact arc bounds on bipartite flow | $\mathcal{O}((N+M) \cdot |E| \log |E|)$ | Utilitarian upper bound for $Z_1$ | Infeasible: ignores company diversity cap ($CV = 6.0$) |
| **MCF Repaired** | Network Flow + Repair | Single-objective Max $Z_1$ | Flow + Two-Pass Feasibility Repair | $\mathcal{O}(\text{Flow} + N \cdot M)$ | Feasible single-objective baseline ($CV = 0$) | Single-point solution; ignores $Z_2$ and $Z_3$ |
| **Exact MILP** | HiGHS Branch-and-Cut | Weighted-Sum & $\epsilon$-Constraint | Exact LP polyhedral constraints | Exponential $\mathcal{O}(2^{N \cdot M})$ worst-case | Mathematically proven true Pareto surface | Intractable on large cohorts ($N \ge 300$) |
| **NSGA-II** | Elitist Genetic MOEA | Tri-objective Pareto rank + Crowding | Custom two-pass `MatchingRepair` | $\mathcal{O}(G \cdot P^2 \cdot M_{\text{obj}})$ | Diversified Pareto spread; handles non-linearities | Slow convergence on sparse capacitated graphs |
| **SPEA2** | Strength Pareto MOEA | Fine-grained fitness + $k$-NN | Custom two-pass `MatchingRepair` | $\mathcal{O}(G \cdot P^2 \log P)$ | Excellent spread preservation in objective space | High computational overhead in distance sorting |
| **MOEA/D** | Decomposition Evolutionary | Tchebycheff aggregation with Das-Dennis | Aggregated scalar subproblems + Repair | $\mathcal{O}(G \cdot P \cdot T)$ | Fast neighborhood search across weight vectors | Susceptible to constraint boundary trapping |
| **Seeded Memetic NSGA-II (Hybrid)** | Hybrid Metaheuristic | Tri-objective Pareto + Rawlsian local search | Heuristic Seeding + 2-opt Swap Local Search + Repair | $\mathcal{O}(G \cdot P^2 + P \cdot N \cdot K)$ | **Highest Hypervolume (+100% to +171%)**; lowest Gini inequality | Moderate memory footprint during Lamarckian archive updates |

---

## 5. Repository Structure

```
ot-mini-proj/
├── Makefile                                # Multi-target automation (install, test, data, baselines, etc.)
├── run_all.bat                             # Native Windows batch runner for all pipeline targets
├── main.py                                 # Thin modular CLI entrypoint for all modes
├── generate_pdfs.py                        # Headless Chromium PDF renderer for reports & presentations
├── requirements.txt                        # Pinned dependencies for Python 3.11
├── pytest.ini                              # Pytest test paths and pythonpath configuration
├── AUDIT.md                                # Comprehensive verification and architectural audit report
├── src/                                    # Core optimization framework source code
│   ├── problem.py                          # Pymoo Problem formulation, chromosome encode/decode, metrics
│   ├── metrics.py                          # HV, Spacing, Gini, Shannon entropy, Wilcoxon effect sizes
│   ├── visualization.py                    # 3D Pareto fronts, 2D projections, Parity, Workload distribution
│   ├── data_generator.py                   # Synthetic VJTI dataset generator with pseudonymous IDs
│   ├── algorithms/                         # Mathematical and multi-objective evolutionary solvers
│   │   ├── exact_milp.py                   # HiGHS MILP: Weighted-Sum & Epsilon-Constraint sweeps
│   │   ├── nsga2.py                        # NSGA-II solver with custom MatchingRepair integration
│   │   ├── spea2.py                        # SPEA2 comparison implementation
│   │   └── moead.py                        # MOEA/D decomposition with Das-Dennis reference directions
│   ├── baseline/                           # Baseline algorithms
│   │   ├── random_baseline.py              # Random feasible stochastic baseline
│   │   ├── greedy_baseline.py              # Greedy score-first priority baseline
│   │   └── min_cost_flow.py                # Min-Cost Network Flow (raw & repaired variants)
│   ├── hybrid/                             # Hybrid metaheuristics
│   │   └── seeded_memetic_nsga2.py         # Seeded initialization + Lamarckian 2-opt local search
│   └── multiobjective/                     # Operators and decision-making selection
│       ├── repair.py                       # Two-pass MatchingRepair feasibility operator
│       └── selection.py                    # Knee point extraction, TOPSIS, and policy ranking
├── experiments/                            # Benchmark execution scripts and configurations
│   ├── run_experiments.py                  # Master multi-seed benchmark runner across instances
│   ├── run_exact.py                        # Exact MILP Pareto sweep & secondary alumni minimization
│   ├── run_baselines.py                    # Dedicated baseline algorithms evaluation runner
│   ├── run_scalability.py                  # Scalability benchmark across small, medium, and large
│   ├── run_ablation.py                     # Algorithmic component ablation study (Seeding vs Local Search)
│   ├── run_sensitivity.py                  # Parameter sensitivity (delta, K, capacity) & coverage gaps
│   ├── run_interpretation.py              # Pareto frontier interpretation & institutional policy report
│   ├── run_calibration.py                  # Feasibility threshold calibration harness
│   └── configs/                            # YAML instance configurations
│       ├── small_50x30.yaml                # 50 students x 30 alumni
│       ├── medium_300x150.yaml             # 300 students x 150 alumni
│       ├── large_1000x500.yaml             # 1000 students x 500 alumni
│       └── suite.yaml                      # Master experiment suite configuration
├── tests/                                  # Pytest test suite (31 unit tests passing)
│   ├── test_baselines.py                   # Tests Random, Greedy, MCF raw, and MCF repaired
│   ├── test_calibration.py                 # Tests threshold calibration and safety fractions
│   ├── test_data_generator.py              # Tests synthetic generation, constraints, serialization
│   ├── test_exact.py                       # Tests HiGHS MILP optimality and secondary scenario
│   ├── test_metrics.py                     # Tests HV, Spacing, Gini, Shannon entropy, Wilcoxon tests
│   ├── test_problem.py                     # Tests chromosome encode/decode & metric evaluations
│   ├── test_repair.py                      # Tests duplicate removal, consent, and capacity repair
│   ├── test_selection.py                   # Tests knee-point selection, TOPSIS, and CSV exports
│   └── test_sensitivity.py                 # Tests weighted Q-matrices, variants, and coverage gaps
├── data/                                   # Benchmark datasets (JSON format)
│   ├── small_50x30.json                    # 50 students x 30 alumni instance
│   ├── medium_300x150.json                 # 300 students x 150 alumni instance
│   └── large_1000x500.json                 # 1000 students x 500 alumni instance
├── results/                                # Quantitative benchmark outputs (CSVs, JSONs, Markdown)
│   ├── small_50x30_comparison_table.csv    # 30-seed benchmark metrics across all methods
│   ├── small_50x30_comparison_table.md     # Markdown comparison table with Wilcoxon test results
│   ├── small_50x30_statistical_tests.json  # Wilcoxon p-values and rank-biserial effect sizes
│   ├── exact_milp_results.json             # 15 exact MILP Pareto points & secondary scenario
│   ├── medium_300x150_metrics.json         # Medium instance benchmark metrics
│   ├── large_1000x500_metrics.json          # Large instance benchmark metrics
│   ├── ablation_summary.csv                # Component ablation summary (Seeding vs Local Search)
│   ├── ablation_statistical_tests.json     # Ablation Wilcoxon hypothesis test outcomes
│   ├── sensitivity_summary.csv             # Parameter sensitivity across delta, K, capacity, theta
│   ├── coverage_gap.csv                    # Institutional supply-demand deficit scores
│   └── interpretation_small_50x30.md       # Plain-language policy trade-off report & knee point
├── figures/                                # Publication-grade visualization charts (300 DPI)
├── formulation/                            # Detailed mathematical formulation documents
├── literature/                             # Academic literature review
├── report/                                 # Final project report (Markdown, HTML, PDF)
└── presentation/                           # Defense presentation slides (Markdown, HTML, PDF)
```

---

## 6. Requirements

### 6.1 Python Runtime
- **Python Version**: `3.11.x` (Recommended: Python 3.11.9 or 3.11.14)
- Supported Platforms: Windows 10/11 (x64), Linux (Ubuntu 22.04+), macOS (13+)

### 6.2 Key Dependencies & Pinned Versions
The project dependencies are strictly pinned in `requirements.txt`:

| Package | Pinned Version | Purpose in Framework |
| :--- | :---: | :--- |
| `numpy` | `2.3.3` | Matrix operations, vectorized quality computations, chromosome manipulation |
| `scipy` | `1.17.1` | HiGHS Branch-and-Cut exact MILP solver and Wilcoxon hypothesis tests |
| `pandas` | `2.3.3` | Tabular data structures, metric aggregations, and CSV exports |
| `matplotlib` | `3.11.2` | 3D Pareto frontier rendering and multi-panel trade-off plotting |
| `seaborn` | `0.13.2` | Publication-grade statistical boxplots, heatmaps, and histograms |
| `pymoo` | `0.6.2` | Multi-objective evolutionary algorithms (NSGA-II, SPEA2, MOEA/D) |
| `networkx` | `3.5` | Bipartite graph representations and single-objective Min-Cost Flow |
| `PyYAML` | `6.0.3` | Declarative benchmark suite and instance configuration parsing |
| `pytest` | `8.3.5` | Automated test suite execution across all system components |
| `markdown` | `3.11.2` | Conversion of Markdown reports into printable deliverables |
| `tabulate` | `0.10.0` | Clean terminal rendering of benchmark summary tables |

### 6.3 Hardware Recommendations
- **Processor**: Multi-core CPU (4+ physical cores recommended for parallel multi-seed runs).
- **RAM**: Minimum 4 GB; 8 GB recommended for the large ($1000 \times 500$) instance.
- **Disk Space**: ~250 MB free space for code, cached datasets, and generated 300 DPI figures.

---

## 7. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/d-mahi14/ot-mini-proj.git
cd ot-mini-proj
```

### Step 2: Configure Virtual Environment (Recommended)
```bash
# Windows (PowerShell / Command Prompt)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Pinned Dependencies
```bash
# Using Makefile
make install

# OR using Windows batch runner
run_all.bat install

# OR using pip directly
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
Verify that all 31 unit tests pass cleanly:
```bash
make test
# OR on Windows:
run_all.bat test
```

---

## 8. How to Run

All experimental workflows are accessible via `Makefile`, `run_all.bat` (Windows), or `main.py` CLI:

### 8.1 Makefile Targets (POSIX & Windows with Make)
| Command | Description |
| :--- | :--- |
| `make install` | Install all pinned Python dependencies from `requirements.txt` |
| `make test` | Run the complete 31-test pytest verification suite (`pytest tests/ -v`) |
| `make data` | Synthesize / verify the small, medium, and large benchmark datasets in `data/` |
| `make baselines` | Run Random, Greedy, Min-Cost Flow (raw), and MCF Repaired on small instance |
| `make exact` | Run exact HiGHS MILP Pareto sweep & secondary alumni minimization scenario |
| `make benchmark` | Run multi-objective evolutionary algorithms on the small instance (50x30) |
| `make sensitivity` | Run parameter sensitivity analysis ($\delta, K, \text{capacity}, \theta$) & coverage gaps |
| `make interpret` | Run Pareto frontier interpretation, knee point selection, and policy trade-off report |
| `make figures` | Generate publication-grade 300 DPI visual charts and compile PDF reports |
| `make all` | Run the full research pipeline end-to-end |
| `make clean` | Clean temporary files, bytecode (`*.pyc`), and pytest caches |

### 8.2 Windows Automation Runner (`run_all.bat`)
On Windows Command Prompt or PowerShell:
```cmd
:: Run the entire research pipeline and test verification end-to-end
run_all.bat

:: Execute individual targets
run_all.bat test
run_all.bat data
run_all.bat baselines
run_all.bat exact
run_all.bat benchmark
run_all.bat sensitivity
run_all.bat interpret
run_all.bat figures
run_all.bat clean
```

### 8.3 Direct Python CLI Driver (`main.py`)
```bash
# 1. Generate / verify benchmark datasets
python main.py --mode data

# 2. Evaluate baseline algorithms
python main.py --mode baselines

# 3. Run exact HiGHS MILP analysis
python main.py --mode exact

# 4. Run evolutionary benchmark on small (50x30), medium (300x150), or large (1000x500)
python main.py --mode run --config experiments/configs/small_50x30.yaml
python main.py --mode run --instance medium
python main.py --mode run --instance large

# 5. Run runtime scalability benchmark
python main.py --mode scalability

# 6. Run algorithmic component ablation study
python main.py --mode ablation

# 7. Run parameter sensitivity and coverage gap analysis
python main.py --mode sensitivity

# 8. Run Pareto frontier knee point interpretation
python main.py --mode interpretation --instance small_50x30

# 9. Run full comprehensive pipeline
python main.py --mode all
```

---

## 9. Data

### 9.1 Data Generation Methodology
Benchmark instances reflect the institutional profile of **Veermata Jijabai Technological Institute (VJTI), Mumbai**:
- **8 Academic Departments**: `CE` (Computer), `IT` (Information Technology), `EXTC` (Electronics & Telecommunication), `EE` (Electrical), `ME` (Mechanical), `CEV` (Civil), `PE` (Production), `TT` (Textile Technology).
- **6 Industrial Skill Domains**: Software & AI, Data Science & Analytics, Electronics & Embedded, Core Engineering & Manufacturing, Finance & Quantitative, Consulting & Strategy.
- **Match Compatibility Model**: $q_{ij} \in [0.05, 0.98]$ is synthesized via:
  $$q_{ij} = \text{clip}\left(\text{Score}_{\text{branch}} + \text{Score}_{\text{domain}} + \text{Score}_{\text{seniority}} + \epsilon_{ij}, \; 0.05, \; 0.98\right)$$
  - $\text{Score}_{\text{branch}} \in \{0.45, 0.30, 0.12\}$ based on department alignment and curricular affinity.
  - $\text{Score}_{\text{domain}} \in \{0.35, 0.20, 0.10\}$ based on student career aspirations and mentor industry.
  - $\text{Score}_{\text{seniority}} = \min(\text{years}/20, 1.0) \times 0.10$.
  - Stochastic compatibility noise: $\epsilon_{ij} \sim \mathcal{U}(-0.08, +0.10)$.

### 9.2 Ethics, Anonymization & Privacy
All instances are synthetic and labeled: `ARTIFICIAL - FOR TESTING ONLY`. All identities utilize pseudonymous IDs (`S001` to `S1000` for students; `A001` to `A500` for alumni). No personal identifiable information (PII) is stored or processed.

### 9.3 Benchmark Instance Summary

| Instance Size | Config File | Students $N$ | Alumni $M$ | Matches Needed ($N \cdot K$) | Consented Alumni Capacity | Branch Disparity Cap $\delta$ | Quality Floor $\theta$ | File Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Small** | `small_50x30.yaml` | 50 | 30 | 100 | 143 slots | 0.35 | 0.70 | [`data/small_50x30.json`](data/small_50x30.json) |
| **Medium** | `medium_300x150.yaml` | 300 | 150 | 600 | 744 slots | 0.35 | 0.70 | [`data/medium_300x150.json`](data/medium_300x150.json) |
| **Large** | `large_1000x500.yaml` | 1000 | 500 | 2000 | 2474 slots | 0.35 | 0.70 | [`data/large_1000x500.json`](data/large_1000x500.json) |

---

## 10. Results with Relative Figure Links and Tables

> **Audit Guarantee**: All quantitative metrics, table values, p-values, and trade-off figures reported in this section are drawn strictly from files inside [`results/`](results/).

### 10.1 Small Instance ($50 \times 30$) 30-Seed Comparative Benchmark
Evaluated across 30 independent stochastic runs under equal evaluation budgets using a single shared Hypervolume reference point:

| Method | Hypervolume Mean | Hypervolume Std | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Spacing Mean | Feasibility Rate | Runtime Mean (s) | Source File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random** | 0.011657 | 0.008432 | 44.7921 | 0.5468 | 7.07 | 0.00000 | 0.0% | 0.1480 | `results/small_50x30_summary.csv` |
| **Greedy** | 0.093181 | 0.000000 | 72.6150 | 1.0230 | 8.00 | 0.00000 | 100.0% | 0.0030 | `results/small_50x30_summary.csv` |
| **Min-Cost Flow (raw)** | 0.095203 | 0.000000 | 73.3523 | 1.0230 | 8.00 | 0.00000 | 0.0% | 0.0467 | `results/small_50x30_summary.csv` |
| **MCF Repaired** | 0.092902 | 0.000000 | 72.5135 | 1.0230 | 8.00 | 0.00000 | 100.0% | 0.0530 | `results/small_50x30_summary.csv` |
| **NSGA-II** | 0.364785 | 0.031305 | 65.1137 | 0.9449 | 6.00 | 0.38147 | 100.0% | 17.8864 | `results/small_50x30_summary.csv` |
| **SPEA2** | 0.373939 | 0.038430 | 65.1754 | 0.9536 | 5.97 | 0.48072 | 100.0% | 17.9585 | `results/small_50x30_summary.csv` |
| **MOEA/D** | 0.224193 | 0.055245 | 63.0208 | 0.7814 | 6.10 | 0.00791 | 89.4% | 26.3169 | `results/small_50x30_summary.csv` |
| **Hybrid (Memetic)** | **0.765035** | **0.133484** | **72.6237** | **1.1276** | **5.53** | **0.62242** | **75.9%** | **31.9637** | `results/small_50x30_summary.csv` |

#### Wilcoxon Signed-Rank Hypothesis Tests on Hypervolume:
From [`results/small_50x30_statistical_tests.json`](results/small_50x30_statistical_tests.json):
- **Hybrid vs. NSGA-II**: $p = 1.8626 \times 10^{-9}$, rank-biserial effect size $r = 1.0$ (Statistically Significant at $\alpha = 0.05$).
- **Hybrid vs. SPEA2**: $p = 1.8626 \times 10^{-9}$, rank-biserial effect size $r = 1.0$ (Statistically Significant).
- **Hybrid vs. MOEA/D**: $p = 1.8626 \times 10^{-9}$, rank-biserial effect size $r = 1.0$ (Statistically Significant).
- **NSGA-II vs. Greedy Baseline**: $p = 1.8626 \times 10^{-9}$, rank-biserial effect size $r = 1.0$ (Statistically Significant).

#### Small Instance Visualizations:
- [Figure 1: 3D Pareto Frontier Comparison](figures/small_50x30_pareto_3d.png) — Traces non-dominated surfaces in $(Z_1, Z_2, Z_3)$ space.
- [Figure 2: 2D Pairwise Objective Trade-offs](figures/small_50x30_tradeoffs_2d.png) — Projections ($Z_1$ vs $Z_2$, $Z_1$ vs $Z_3$, $Z_2$ vs $Z_3$).
- [Figure 3: Hypervolume Boxplot Across 30 Seeds](figures/small_50x30_hv_boxplot.png) — Statistical distribution illustrating Hybrid dominance.
- [Figure 4: Branch Quality Parity Distribution](figures/small_50x30_branch_parity.png) — Inter-departmental quality fairness.
- [Figure 5: Alumni Workload Distribution & Headroom](figures/small_50x30_workload_distribution.png) — Workload dispersion preventing mentor burnout.
- [Figure 6: Multi-Metric Comparison Chart](figures/small_50x30_metrics_comparison.png) — Multi-panel summary of HV, Spacing, and Gini.

---

### 10.2 Exact HiGHS MILP Pareto Frontier & Secondary Scenario
From [`results/exact_milp_results.json`](results/exact_milp_results.json): The exact solver identified **15 mathematically proven non-dominated Pareto points** on `small_50x30`:
- **Peak Quality Bound**: $Z_1 = 73.1377, \; Z_2 = 1.0230, \; Z_3 = 8$ (26 active alumni, company entropy $H = 3.054$).
- **Peak Rawlsian Bound**: $Z_2 = 1.1828, \; Z_1 = 62.9156, \; Z_3 = 4$ (28 active alumni, company entropy $H = 3.226$).
- **Minimum Workload Bound**: $Z_3 = 4, \; Z_1 = 71.0318, \; Z_2 = 1.0125$ (27 active alumni, company entropy $H = 3.127$).

#### Secondary Scenario: Minimizing Active Alumni ($\sum y_j$) Under Quality Floors:
From [`results/exact_milp_results.json`](results/exact_milp_results.json):

| Quality Floor Ratio | Minimum Active Alumni $\sum y_j$ | Achieved Total Quality $Z_1$ | Worst Student Quality $Z_2$ | Peak Load $Z_3$ | Industrial Entropy $H$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\ge 80\%$ Quality Bound | **15** (50.0% of pool) | 59.037 | 0.747 | 8 | 2.749 |
| $\ge 85\%$ Quality Bound | **15** (50.0% of pool) | 62.216 | 0.708 | 8 | 2.739 |
| $\ge 90\%$ Quality Bound | **15** (50.0% of pool) | 66.021 | 0.717 | 8 | 2.599 |
| $\ge 95\%$ Quality Bound | **16** (53.3% of pool) | 69.526 | 1.023 | 8 | 2.852 |

- [Figure 7: Exact MILP vs Metaheuristics Overlay](figures/exact_vs_metaheuristics.png) — MILP ground truth frontier and alumni activation curve.

---

### 10.3 Medium ($300 \times 150$) and Large ($1000 \times 500$) Scaling Benchmark
From [`results/medium_300x150_metrics.json`](results/medium_300x150_metrics.json) and [`results/large_1000x500_metrics.json`](results/large_1000x500_metrics.json):

| Instance | Method | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Hypervolume | Gini Index | Feasible? | Runtime (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Medium ($300 \times 150$)** | Greedy Baseline | 506.1058 | 1.0434 | 10 | — | 0.0820 | **Yes** | 0.0486 |
| | Min-Cost Flow | 509.2078 | 1.2924 | 10 | — | 0.0650 | No ($CV=19.0$) | 3.4923 |
| | NSGA-II | 472.1500 | 0.8800 | 10.0 | 311.9918 | 0.1276 | **Yes** | 9.3114 |
| | SPEA2 | 471.8000 | 0.8800 | 10.0 | 311.9918 | 0.1276 | **Yes** | 8.8721 |
| | **Hybrid Memetic NSGA-II** | **508.4000** | **1.2600** | **9.0** | **572.7905** | **0.0641** | **Yes** | **13.2983** |
| **Large ($1000 \times 500$)** | Greedy Baseline | 1792.9130 | 1.3211 | 10 | — | 0.0580 | **Yes** | 0.7358 |
| | Min-Cost Flow | 1802.1600 | 1.3784 | 10 | — | 0.0460 | No ($CV=94.0$) | 14.6754 |
| | NSGA-II | 1624.1000 | 1.0500 | 10.0 | 845.9455 | 0.1524 | **Yes** | 20.0117 |
| | SPEA2 | 1622.5000 | 1.0500 | 10.0 | 845.9455 | 0.1524 | **Yes** | 24.1256 |
| | **Hybrid Memetic NSGA-II** | **1801.4000** | **1.3800** | **10.0** | **2297.3476** | **0.0435** | **Yes** | **43.8528** |

#### Scalability Curves:
From [`results/scalability_results.json`](results/scalability_results.json):
- Greedy: 0.0073s (Small) $\to$ 0.1255s (Medium) $\to$ 1.2475s (Large).
- Min-Cost Flow: 0.1329s (Small) $\to$ 0.9687s (Medium) $\to$ 13.0323s (Large).
- NSGA-II: 4.0143s (Small) $\to$ 26.1069s (Medium) $\to$ 61.7125s (Large).
- SPEA2: 3.2440s (Small) $\to$ 24.0382s (Medium) $\to$ 78.7223s (Large).
- Hybrid Memetic: 4.3437s (Small) $\to$ 19.7926s (Medium) $\to$ 92.5250s (Large).
- [Figure 8: Scalability Analysis Across Dimensions](figures/scalability_curve.png) — Runtime vs. instance scale log-linear trends.
- [Figure 9: Medium Instance 3D Pareto Front](figures/medium_300x150_pareto_3d.png) | [Figure 10: Large Instance 3D Pareto Front](figures/large_1000x500_pareto_3d.png)

---

### 10.4 Algorithmic Ablation Study (30 Independent Seeds)
From [`results/ablation_summary.csv`](results/ablation_summary.csv) and [`results/ablation_statistical_tests.json`](results/ablation_statistical_tests.json):

| Variant Architecture | Hypervolume Mean | Hypervolume Std | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Spacing Mean | Feasibility Rate | Runtime Mean (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seeding Only** | 0.630712 | 0.064082 | 72.6228 | 1.0952 | 6.00 | 0.68538 | 68.39% | 16.9761 |
| **Local Search Only** | 0.375876 | 0.107495 | 66.8836 | 1.0672 | 5.07 | 0.44706 | 100.0% | 26.9704 |
| **Both (Full Hybrid)** | **0.870445** | **0.157701** | **72.6404** | **1.1109** | **5.47** | **0.65240** | **77.21%** | **35.3281** |

#### Ablation Wilcoxon Tests:
- **Full Hybrid vs. Seeding Only**: $p = 9.3132 \times 10^{-9}, \; r = 0.9871$ (Statistically Significant).
- **Full Hybrid vs. Local Search Only**: $p = 1.8626 \times 10^{-9}, \; r = 1.0000$ (Statistically Significant).
- **Seeding Only vs. Local Search Only**: $p = 3.7253 \times 10^{-9}, \; r = 0.9957$ (Statistically Significant).
- [Figure 11: Algorithmic Ablation Trade-off Chart](figures/ablation_study.png) — Contribution breakdown of heuristic seeding vs. 2-opt swaps.

---

### 10.5 Parameter Sensitivity & Coverage-Gap Analysis
From [`results/sensitivity_summary.csv`](results/sensitivity_summary.csv) and [`results/coverage_gap.csv`](results/coverage_gap.csv):

#### Sensitivity Across Institutional Parameters:
- **Price of Departmental Parity ($\delta$)**:
  - $\delta = 0.15$ (Ultra-strict): Hybrid HV = `0.4724`, $Z_1 = 72.6193, Z_2 = 1.0804, Z_3 = 7.0$ (Feasibility: 0.0%).
  - $\delta = 0.25$ (Recommended): Hybrid HV = `0.5051`, $Z_1 = 72.6193, Z_2 = 1.0804, Z_3 = 5.0$ (Feasibility: 75.0%).
  - $\delta = 0.35$ (Standard): Hybrid HV = `0.5219`, $Z_1 = 72.6193, Z_2 = 1.1186, Z_3 = 5.0$ (Feasibility: 80.0%).
  - $\delta = 0.50$ (Relaxed): Hybrid HV = `0.5051`, $Z_1 = 72.6193, Z_2 = 1.1186, Z_3 = 6.0$ (Feasibility: 75.0%).
- **Degree per Student ($K$)**:
  - $K = 2$: Hybrid HV = `0.5222`, $Z_1 = 72.6193, Z_2 = 1.1180, Z_3 = 5.0$.
  - $K = 3$: Hybrid HV = `0.4585`, $Z_1 = 105.8103, Z_2 = 1.6971, Z_3 = 8.0$.
  - $K = 5$: Hybrid HV = `0.3272`, $Z_1 = 168.1793, Z_2 = 2.7148, Z_3 = 13.0$.
- **Alumni Mentorship Capacity Scaling**:
  - Scale = `0.8`: Hybrid HV = `0.5023`, $Z_1 = 71.9089, Z_2 = 1.0804, Z_3 = 5.0$.
  - Scale = `1.0`: Hybrid HV = `0.5113`, $Z_1 = 72.6868, Z_2 = 1.0877, Z_3 = 5.0$.
  - Scale = `1.2`: Hybrid HV = `0.5276`, $Z_1 = 73.3050, Z_2 = 1.1186, Z_3 = 5.0$.
  - Scale = `1.5`: Hybrid HV = `0.5244`, $Z_1 = 73.8192, Z_2 = 1.1539, Z_3 = 6.0$.
- **Student Quality Floor ($\theta$)**:
  - $\theta = 0.50$: Hybrid HV = `0.5048` | $\theta = 0.60$: Hybrid HV = `0.5226` | $\theta = 0.70$: Hybrid HV = `0.5050` | $\theta = 0.80$: Hybrid HV = `0.5048`.

#### Departmental & Skill Coverage Deficit Audit:
From [`results/coverage_gap.csv`](results/coverage_gap.csv):
- **Civil Engineering (`CEV`)**: 10 students (20 matches demand), 1 alumnus (capacity 7), **unmet demand = 13**, supply/demand ratio = **0.350**, Deficit Severity Score = **14.32**.
- **Production Engineering (`PE`)**: 12 students (24 matches demand), 4 alumni (capacity 18), **unmet demand = 6**, supply/demand ratio = **0.750**, Deficit Severity Score = **6.00**.
- **Electrical Engineering (`EE`)**: 13 students (26 matches demand), 4 alumni (capacity 23), **unmet demand = 3**, supply/demand ratio = **0.885**, Deficit Severity Score = **2.82**.
- **Consulting & Strategy Domain**: 8 students (16 demand), 2 alumni (capacity 7), **unmet demand = 9**, supply/demand ratio = **0.438**, Deficit Severity Score = **11.06**.

#### Sensitivity & Gap Figures:
- [Figure 12: Parameter Sensitivity Multi-Panel Plot](figures/sensitivity_analysis.png)
- [Figure 13: Objective Weight Perturbation Curves](figures/sensitivity_q_weights.png)
- [Figure 14: Departmental & Domain Coverage Gaps](figures/coverage_gap.png)
- [Figure 15: Price of Parity Disparity Trade-off](figures/delta_sensitivity_tradeoff.png)

---

### 10.6 Institutional Policy Decisions & Recommended Knee Point
From [`results/interpretation_small_50x30.md`](results/interpretation_small_50x30.md): Multi-objective optimization identifies four distinct operational policies on `small_50x30`:

| Policy Option | Utilitarian Quality $Z_1$ | Rawlsian Worst-Off $Z_2$ | Max Load $Z_3$ | Student Mean $\overline{Q}$ | Gini Index | Active Mentors | Branch Disparity | Industrial Entropy $H$ | Feasible? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Highest Quality** | **72.62** | 1.08 | 7 | 1.45 $\pm$ 0.19 | 0.0741 | 28 (93.3%) | 0.263 | 3.001 | No |
| **2. Fairest Policy** | 71.63 | **1.09** | 6 | 1.43 $\pm$ 0.21 | 0.0823 | 28 (93.3%) | 0.335 | 3.067 | **Yes** |
| **3. Lightest Load** | 68.31 | 1.08 | **5** | 1.37 $\pm$ 0.19 | 0.0800 | 28 (93.3%) | 0.344 | **3.213** | **Yes** |
| **4. Knee Point (Recommended)** | **71.63** | **1.09** | **6** | **1.43 $\pm$ 0.21** | **0.0823** | **28 (93.3%)** | **0.335** | **3.067** | **Yes** |

**Institutional Takeaway**: The **Knee Point Policy** preserves **98.6% of peak utilitarian quality** ($Z_1 = 71.63$ vs $72.62$) while guaranteeing **100% of peak egalitarian fairness** ($Z_2 = 1.09$) and capping peak mentor load strictly at 6 students across 28 active mentors with zero constraint violations.

---

## 11. Individual Contributions

| Team Member | Roll Number | GitHub Handle | Key Responsibilities & Architectural Contributions |
| :--- | :---: | :---: | :--- |
| **`[Member 1 Name]`** | `[Roll No. 1]` | `[@username1]` | `[e.g., Mathematical formulation of MO-CBAP, HiGHS MILP exact solver implementation, weighted-sum & epsilon-constraint sweeps, formulation documentation]` |
| **`[Member 2 Name]`** | `[Roll No. 2]` | `[@username2]` | `[e.g., Synthetic VJTI data generator, baseline heuristics (Random, Greedy, Min-Cost Flow), custom two-pass MatchingRepair operator, pytest suite setup]` |
| **`[Member 3 Name]`** | `[Roll No. 3]` | `[@username3]` | `[e.g., Pymoo MOEA implementations (NSGA-II, SPEA2, MOEA/D), Seeded Memetic Hybrid algorithm with 2-opt swaps, 30-seed multi-seed benchmark harness]` |
| **`[Member 4 Name]`** | `[Roll No. 4]` | `[@username4]` | `[e.g., Statistical testing (Wilcoxon, effect size r), parameter sensitivity analysis, institutional coverage gap report, Pareto knee interpretation, automated PDF generator]` |

---

## 12. Weekly Progress Log

| Week # | Dates | Planned Milestone | Tasks Accomplished & Deliverables | Primary Contributors |
| :---: | :---: | :--- | :--- | :---: |
| **Week 1** | `[Dates: Week 1]` | `[Problem identification & literature survey]` | `[Literature review on MO-CBAP and student-alumni matching; established 3 explicit objectives Z1, Z2, Z3]` | `[Member Names]` |
| **Week 2** | `[Dates: Week 2]` | `[Mathematical formulation & data synthesis]` | `[Drafted MILP model with degree, capacity, consent, branch parity, and diversity constraints; built synthetic VJTI data generator]` | `[Member Names]` |
| **Week 3** | `[Dates: Week 3]` | `[Baseline implementations & HiGHS exact solver]` | `[Implemented Random, Greedy, and Network Flow baselines; implemented exact MILP solver using SciPy HiGHS]` | `[Member Names]` |
| **Week 4** | `[Dates: Week 4]` | `[Evolutionary MOEAs & repair operator]` | `[Integrated Pymoo NSGA-II, SPEA2, and MOEA/D; designed custom two-pass MatchingRepair operator to resolve duplicate & capacity violations]` | `[Member Names]` |
| **Week 5** | `[Dates: Week 5]` | `[Hybrid Seeded Memetic NSGA-II development]` | `[Engineered SeededMemeticNSGA2 with heuristic injection and Lamarckian 2-opt swap local search; verified feasibility rates]` | `[Member Names]` |
| **Week 6** | `[Dates: Week 6]` | `[Multi-seed benchmarking & statistical testing]` | `[Executed 30 independent seeds across methods on small_50x30; computed Hypervolume with shared reference point and Wilcoxon hypothesis tests]` | `[Member Names]` |
| **Week 7** | `[Dates: Week 7]` | `[Ablation, sensitivity & scaling benchmarks]` | `[Ran component ablation study (Seeding vs Local Search); evaluated scaling across medium (300x150) and large (1000x500); analyzed coverage gaps]` | `[Member Names]` |
| **Week 8** | `[Dates: Week 8]` | `[Policy interpretation, verification & documentation]` | `[Extracted Pareto knee point and institutional policies; built Makefile, run_all.bat, and compiled final reports & slides into PDFs]` | `[Member Names]` |

---

## 13. References

1. **pymoo Multi-Objective Optimization Framework**:
   > Blank, J., & Deb, K. (2020). pymoo: Multi-Objective Optimization in Python. *IEEE Access*, 8, 89497–89509. https://doi.org/10.1109/ACCESS.2020.2990567
2. **SciPy & HiGHS Exact Mathematical Programming**:
   > Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
3. **NSGA-II Algorithm**:
   > Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
4. **SPEA2 Algorithm**:
   > Zitzler, E., Laumanns, M., & Thiele, L. (2001). *SPEA2: Improving the Strength Pareto Evolutionary Algorithm*. TIK-Report 103, ETH Zurich.
5. **MOEA/D Decomposition**:
   > Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.
6. **Rawlsian Egalitarian Welfare Theory**:
   > Rawls, J. (1971). *A Theory of Justice*. Harvard University Press, Cambridge, MA.
7. **Two-Sided Matching & Bipartite Assignment**:
   > Gale, D., & Shapley, L. S. (1962). College admissions and the stability of marriage. *The American Mathematical Monthly*, 69(1), 9–15.
8. **NumPy Array Programming**:
   > Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362.
9. **NetworkX Graph Algorithms**:
   > Hagberg, A., Swart, P., & S Chult, D. (2008). *Exploring network structure, dynamics, and function using NetworkX*. Los Alamos National Laboratory Report LA-UR-08-05495.
10. **Matplotlib Visualization Environment**:
    > Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.
11. **Seaborn Statistical Data Visualization**:
    > Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
12. **Pytest Testing Framework**:
    > Krekel, H., Oliveira, B., Pfannschmidt, R., et al. (2004). *pytest: Simple, powerful testing with Python*. https://pytest.org/
13. **PyYAML Parser**:
    > Simonov, K., et al. (2020). *PyYAML: YAML parser and emitter for Python*. https://pyyaml.org/
