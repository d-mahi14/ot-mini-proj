# Fair and Load-Balanced Student-Alumni Matching at VJTI
### Multi-Objective Capacitated Bipartite Assignment with Rawlsian Fairness & Departmental Parity

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: Pytest](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)](https://pytest.org/)

> **Academic Notice**: *ARTIFICIAL - FOR TESTING ONLY*. All profiles, compatibility scores, and identifiers generated in this repository are synthetic and utilize pseudonymous IDs only (`S001`, `A001`). No real personal identifiable information (PII) is utilized.

---

## 1. Overview & Institutional Motivation

Mentorship programs at technical institutions like **Veermata Jijabai Technological Institute (VJTI), Mumbai** connect graduating engineering students with practicing alumni across 8 academic departments (`CE`, `IT`, `EXTC`, `EE`, `ME`, `CEV`, `PE`, `TT`). Traditional matching relies on greedy score maximization or ad-hoc first-come-first-served scheduling, inducing three structural flaws:
1. **Mentor Burnout**: Highly rated alumni are overloaded beyond sustainable capacity.
2. **Peripheral Student Neglect**: Students with average GPAs or cross-domain career interests receive subpar matches.
3. **Departmental Inequity**: Computing departments dominate high-scoring alumni; core engineering departments suffer structural deficits.

This project delivers an end-to-end multi-objective optimization system solving the **Multi-Objective Capacitated Bipartite Assignment Problem (MO-CBAP)** with:
- **Utilitarian Efficiency ($Z_1$)**: Maximize total cumulative match compatibility.
- **Rawlsian Egalitarian Fairness ($Z_2$)**: Maximize the match quality of the least-advantaged student ($\min_i Q_i$).
- **Workload Balancing ($Z_3$)**: Minimize peak mentorship load across alumni mentors ($\max_j L_j$).

---

## 2. Mathematical Model Summary

### 2.1 Objectives
$$\min \mathbf{F}(\mathbf{x}) = \begin{bmatrix} -Z_1(\mathbf{x}) \\ -Z_2(\mathbf{x}) \\ +Z_3(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} -\sum_{i=1}^N \sum_{j=1}^M q_{ij} x_{ij} \\ -\min_{i \in \mathcal{S}} Q_i \\ +\max_{j \in \mathcal{A}} L_j \end{bmatrix}$$

### 2.2 Constraints
1. **Exact Degree**: $\sum_{j=1}^M x_{ij} = K, \quad \forall i \in \mathcal{S}$ ($K=2$ distinct matches per student).
2. **Alumni Capacity & Activation**: $\sum_{i=1}^N x_{ij} \le C_j y_j, \quad \forall j \in \mathcal{A}$ with $x_{ij} \le y_j$.
3. **Consent Compliance**: $x_{ij} \le c_j, \quad \forall i, j$ ($c_j \in \{0, 1\}$ opt-in flag).
4. **Quality Threshold**: $x_{ij} = 0 \quad \forall (i, j) \text{ such that } q_{ij} < q_{\min}$.
5. **Student Quality Floor**: $Q_i = \sum_j q_{ij} x_{ij} \ge \theta, \quad \forall i \in \mathcal{S}$.
6. **Cross-Departmental Parity**: $|\overline{Q}_b - \overline{Q}_{b'}| \le \delta, \quad \forall b, b' \in \mathcal{B}$.
7. **Employer Diversity Cap**: $\sum_{j \in \mathcal{A}_k} x_{ij} \le \text{cap}_{\text{comp}}, \quad \forall i, k$ (at most 1 match per firm).

---

## 3. Repository Architecture

```
ot-proj/
├── main.py                     # Thin driver entrypoint (CLI for all modes)
├── requirements.txt            # Pinned dependencies for Python 3.11
├── src/
│   ├── problem.py              # Pymoo Problem formulation & solution evaluation
│   ├── metrics.py              # HV, Spacing, Gini index, Shannon entropy, Load
│   ├── visualization.py        # 3D Pareto fronts, 2D projections, Parity, Workload
│   ├── data_generator.py       # Synthetic VJTI instance generator with pseudonyms
│   ├── baseline/               # Baseline heuristics
│   │   ├── random_baseline.py  # Random feasible stochastic baseline
│   │   ├── greedy_baseline.py  # Greedy score-first priority baseline
│   │   └── min_cost_flow.py    # Single-objective Min-Cost Network Flow (Max Z1)
│   ├── algorithms/             # Exact mathematical solvers
│   │   └── exact_milp.py       # HiGHS MILP: Weighted-Sum & Epsilon-Constraint
│   ├── multiobjective/         # Multi-objective evolutionary algorithms
│   │   ├── nsga2.py            # NSGA-II with custom repair
│   │   ├── spea2.py            # SPEA2 comparison
│   │   ├── moead.py            # MOEA/D decomposition with Das-Dennis weights
│   │   └── repair.py           # Two-pass MatchingRepair feasibility operator
│   └── hybrid/                 # Hybrid metaheuristics
│       └── seeded_memetic_nsga2.py # Seeded initialization + Lamarckian local search
├── experiments/
│   ├── configs/                # YAML instance & suite configurations
│   │   ├── small_50x30.yaml    # 50 students x 30 alumni
│   │   ├── medium_300x150.yaml # 300 students x 150 alumni
│   │   ├── large_1000x500.yaml # 1000 students x 500 alumni
│   │   └── suite.yaml          # Master benchmark suite configuration
│   ├── run_experiments.py     # Main benchmark runner across algorithms
│   ├── run_exact.py           # Exact MILP Pareto sweep & secondary scenario
│   ├── run_scalability.py     # Scalability benchmark across small, medium, large
│   └── run_ablation.py        # Component ablation & Price of Parity study
├── tests/                      # Pytest verification suite (15 unit tests)
│   ├── test_data_generator.py  # Tests synthetic generation & serialization
│   ├── test_problem.py         # Tests chromosome encode/decode & metrics
│   ├── test_repair.py          # Tests capacity & duplicate repair
│   ├── test_baselines.py       # Tests Random, Greedy, MinCostFlow
│   ├── test_exact.py           # Tests HiGHS MILP optimality
│   └── test_metrics.py         # Tests HV, Spacing, Gini, Shannon entropy
├── data/                       # Benchmark instances (JSON)
├── results/                    # Quantitative JSON metric outputs
├── figures/                    # Publication-grade PNG charts (300 DPI)
├── formulation/                # Mathematical formulation documents
├── literature/                 # Academic literature survey
├── report/                     # Complete final project report
└── presentation/               # University defense presentation slides
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.11.x
- Git

### Setup
```bash
git clone https://github.com/d-mahi14/ot-mini-proj.git
cd ot-proj
pip install -r requirements.txt
```

---

## 5. Usage & CLI Driver

All operations are coordinated through `main.py`:

### Run Standard Benchmark on Small Instance (50x30)
```bash
python main.py --mode run --config experiments/configs/small_50x30.yaml
```

### Run Exact MILP Pareto Sweep & Secondary Scenario
```bash
python main.py --mode exact
```

### Run Medium (300x150) or Large (1000x500) Instance
```bash
python main.py --mode run --instance medium
python main.py --mode run --instance large
```

### Run Scalability Across Dimensions
```bash
python main.py --mode scalability
```

### Run Component Ablation & Price of Parity Analysis
```bash
python main.py --mode ablation
```

### Run All Experiments & Re-generate All Figures
```bash
python main.py --mode all
```

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 6. Key Empirical Results

### 6.1 Multi-Objective Benchmark Summary

| Scale | Method | Total Quality $Z_1$ | Worst-off $Z_2$ | Max Load $Z_3$ | Hypervolume | Gini Index | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small (50x30)** | Exact MILP (Ground Truth) | 73.35 | 1.02 | 8 | 443.98 | 0.073 | 14.25s |
| | NSGA-II | 70.42 | 1.12 | 7 | 115.75 | 0.084 | 3.05s |
| | **Hybrid Memetic NSGA-II** | **72.80** | **1.21** | **6** | **232.17** | **0.078** | **2.96s** |
| **Medium (300x150)** | Greedy Baseline | 506.11 | 1.04 | 10 | — | 0.082 | 0.05s |
| | NSGA-II | 472.15 | 0.88 | 10 | 311.99 | 0.128 | 9.31s |
| | **Hybrid Memetic NSGA-II** | **508.40** | **1.26** | **9** | **572.79** | **0.064** | **13.30s** |
| **Large (1000x500)** | Greedy Baseline | 1792.91 | 1.32 | 10 | — | 0.058 | 0.74s |
| | NSGA-II | 1624.10 | 1.05 | 10 | 845.95 | 0.152 | 20.01s |
| | **Hybrid Memetic NSGA-II** | **1801.40** | **1.38** | **10** | **2297.35** | **0.044** | **43.85s** |

### 6.2 Price of Departmental Parity ($\delta$)
- $\delta = 0.15$ (Ultra-strict equality): $Z_1 = 65.80$, Disparity $= 0.148$ (-9.0% quality penalty)
- $\delta = 0.25$ (Recommended balance): $Z_1 = 69.47$, Disparity $= 0.244$ (-3.9% quality penalty)
- $\delta = 0.35$ (Standard): $Z_1 = 72.32$, Disparity $= 0.326$

### 6.3 Secondary Scenario: Minimum Alumni Utilization
- Operating at $\ge 90\%$ of peak quality requires activating only **15 alumni mentors** (50% of pool) while maintaining high industrial entropy ($H = 2.60$).

---

## 7. Generated Visualizations

All figures are automatically generated by code at 300 DPI and saved in [`figures/`](figures/):
1. [`small_50x30_pareto_3d.png`](figures/small_50x30_pareto_3d.png): 3D Pareto frontier comparing Exact MILP, NSGA-II, SPEA2, and Hybrid.
2. [`small_50x30_tradeoffs_2d.png`](figures/small_50x30_tradeoffs_2d.png): 2D trade-offs ($Z_1$ vs $Z_2$, $Z_1$ vs $Z_3$, $Z_2$ vs $Z_3$).
3. [`small_50x30_branch_parity.png`](figures/small_50x30_branch_parity.png): VJTI branch-wise average quality parity.
4. [`small_50x30_workload_distribution.png`](figures/small_50x30_workload_distribution.png): Alumni workload histogram and headroom.
5. [`small_50x30_metrics_comparison.png`](figures/small_50x30_metrics_comparison.png): Multi-panel metric comparison (HV, Spacing, Non-Dominated Count, Gini).
6. [`exact_vs_metaheuristics.png`](figures/exact_vs_metaheuristics.png): True MILP frontier overlay and secondary alumni minimization curve.
7. [`ablation_study.png`](figures/ablation_study.png): Algorithmic component ablation and Price of Parity sensitivity curve.
8. [`scalability_curve.png`](figures/scalability_curve.png): Runtime scaling curve across Small, Medium, and Large dimensions.

---

## 8. Library Citations

This project relies on the following open-source scientific computing libraries:

1. **pymoo**:
   > Blank, J., & Deb, K. (2020). pymoo: Multi-Objective Optimization in Python. *IEEE Access*, 8, 89497-89509.
2. **SciPy** (HiGHS MILP Solver):
   > Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261-272.
3. **NumPy**:
   > Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.
4. **NetworkX**:
   > Hagberg, A., Swart, P., & S Chult, D. (2008). *Exploring network structure, dynamics, and function using NetworkX*. Los Alamos National Laboratory.
5. **Matplotlib**:
   > Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.
6. **Seaborn**:
   > Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
7. **Pytest**:
   > Krekel, H., Oliveira, B., Pfannschmidt, R., et al. (2004). *pytest: Simple, powerful testing with Python*. https://pytest.org/
8. **PyYAML**:
   > Simonov, K., et al. *PyYAML: YAML parser and emitter for Python*. https://pyyaml.org/
