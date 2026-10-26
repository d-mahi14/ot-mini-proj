# Project Audit Report: Fair and Load-Balanced Student-Alumni Matching at VJTI

**Date of Audit**: October 26, 2026  
**Auditor**: Antigravity AI Pair Programmer  
**Repository**: `c:\dev\ot-proj`  
**Evaluation Target**: Compliance with Project Guide, Multi-Objective Optimization Requirements, and Rigorous Operational Standards.

---

## 1. Summary of Pytest Verification

```
platform win32 -- Python 3.11.14, pytest-8.3.5, pluggy-1.6.0
rootdir: C:\dev\ot-proj

tests/test_baselines.py::test_random_baseline PASSED                     [  6%]
tests/test_baselines.py::test_greedy_baseline PASSED                     [ 13%]
tests/test_baselines.py::test_min_cost_flow_baseline PASSED              [ 20%]
tests/test_data_generator.py::test_synthetic_data_generation_constraints PASSED [ 26%]
tests/test_data_generator.py::test_instance_serialization PASSED         [ 33%]
tests/test_exact.py::test_exact_milp_weighted_sum PASSED                 [ 40%]
tests/test_exact.py::test_exact_milp_alumni_minimization PASSED          [ 46%]
tests/test_metrics.py::test_extract_non_dominated_front PASSED           [ 53%]
tests/test_metrics.py::test_spacing_metric PASSED                        [ 60%]
tests/test_metrics.py::test_gini_coefficient PASSED                      [ 66%]
tests/test_metrics.py::test_shannon_entropy PASSED                       [ 73%]
tests/test_problem.py::test_chromosome_encode_decode PASSED              [ 80%]
tests/test_problem.py::test_evaluate_solution_metrics PASSED             [ 86%]
tests/test_repair.py::test_repair_fixes_duplicates_and_consent PASSED    [ 93%]
tests/test_repair.py::test_repair_enforces_capacity PASSED               [100%]

============================= 15 passed in 7.53s ==============================
```

All 15 unit tests pass cleanly covering data synthesis, matrix encoding/decoding, constraint violation evaluation, custom repair operator, all baselines (Random, Greedy, Min-Cost Flow), exact HiGHS MILP, and fairness metrics (Gini, Spacing, Non-dominated extraction, Shannon entropy).

---

## 2. Comprehensive Requirements Audit

### 2.1 Requirements Met (With Exact File Paths)

| Category | Requirement | File Path(s) | Status | Details |
| :--- | :--- | :--- | :---: | :--- |
| **Python Version** | Python 3.11 Environment | `requirements.txt` | **MET** | Pinned dependencies running on Python 3.11.14 |
| **Modular Layout** | Multiple modules, no mono-file or notebook-only | `main.py`, `src/`, `experiments/`, `tests/` | **MET** | Strict separation of concerns across 12+ Python modules |
| **Problem Model** | Tri-objective: max $Z_1$, max $Z_2$, min $Z_3$ | `src/problem.py` | **MET** | Evaluates $-Z_1$ (total quality), $-Z_2$ (worst student quality), $+Z_3$ (max alumni load) |
| **Degree Constraint** | Exactly $K$ matches per student ($\sum_j x_{ij} = K$) | `src/problem.py`, `src/multiobjective/repair.py` | **MET** | Strictly enforced in model, repair operator, and MILP |
| **Capacity & Consent** | $\sum_i x_{ij} \le C_j y_j$ and $x_{ij} \le c_j$ | `src/problem.py`, `src/algorithms/exact_milp.py` | **MET** | Verified in evaluation, repair pass, and MILP constraints |
| **Match Threshold** | $x_{ij} = 0$ if $q_{ij} < q_{\min}$ | `src/problem.py`, `src/algorithms/exact_milp.py` | **MET** | Hard bounds in MILP ($ub = 0$) and repair filtering |
| **Quality Floor** | $Q_i = \sum_j q_{ij} x_{ij} \ge \theta$ | `src/problem.py`, `src/algorithms/exact_milp.py` | **MET** | Linear constraints in MILP and shortfall penalties |
| **Department Parity** | $|\overline{Q}_b - \overline{Q}_{b'}| \le \delta$ | `src/problem.py`, `src/algorithms/exact_milp.py` | **MET** | Linearized pairwise departmental constraints |
| **Company Diversity** | Max share per company $\le \text{cap}_{\text{comp}}$ | `src/problem.py`, `src/algorithms/exact_milp.py`, `src/multiobjective/repair.py` | **MET** | Enforced in MILP, repair operator, and greedy heuristic |
| **Baselines** | Random, Greedy, Min-Cost Flow | `src/baseline/random_baseline.py`, `greedy_baseline.py`, `min_cost_flow.py` | **MET** | Implemented, benchmarked, and tested |
| **Exact MILP** | Weighted-Sum & $\epsilon$-Constraint on Small Instance | `src/algorithms/exact_milp.py`, `experiments/run_exact.py` | **MET** | SciPy HiGHS solves both sweeps to mathematical optimality |
| **Secondary Scenario** | Min $\sum y_j$ under $\epsilon$-bound on quality | `src/algorithms/exact_milp.py`, `experiments/run_exact.py` | **MET** | Tested across 80%, 85%, 90%, 95% quality bounds |
| **MOEAs** | NSGA-II, SPEA2, MOEA/D | `src/multiobjective/nsga2.py`, `spea2.py`, `moead.py` | **MET** | Pymoo implementations with custom repair |
| **Custom Repair** | Two-pass feasibility repair operator | `src/multiobjective/repair.py` | **MET** | Subclasses `pymoo.core.repair.Repair` |
| **Hybrid Metaheuristic**| Seeded + Memetic NSGA-II | `src/hybrid/seeded_memetic_nsga2.py` | **MET** | Injects Greedy/MCF seeds + Lamarckian local search |
| **Evaluation Metrics** | HV, Spacing, $|PF^*|$, Gini, Entropy, Load, Runtime | `src/metrics.py` | **MET** | Comprehensive suite implemented and tested |
| **Three Instances** | Small ($50 \times 30$), Medium ($300 \times 150$), Large ($1000 \times 500$) | `data/small_50x30.json`, `medium_300x150.json`, `large_1000x500.json` | **MET** | Generated, cached, and benchmarked |
| **Ethics & Privacy** | Pseudonymous IDs (`S001`, `A001`), artificial label | `src/data_generator.py`, `data/*.json` | **MET** | Strictly synthetic data labelled "ARTIFICIAL - FOR TESTING ONLY" |
| **Relative Paths** | Relative paths only | `main.py`, `experiments/configs/*.yaml` | **MET** | No hardcoded absolute local paths in application code |
| **Docstrings** | Docstrings on every module | All modules in `src/`, `experiments/`, `tests/` | **MET** | Full PEP 257 module, class, and method docstrings |
| **Driver** | `main.py` is a thin driver | `main.py` | **MET** | Thin CLI parser delegating to modular functions |

---

## 3. Specific Detailed Audit Inquiries

### a) Does each result table use 20-30 independent seeds with mean/std stored in results/, and are Wilcoxon tests present?
- **Status**: **GAP IDENTIFIED**
- **Findings**:
  - The master configuration [`experiments/configs/suite.yaml`](file:///c:/dev/ot-proj/experiments/configs/suite.yaml) declares 3 seeds (`[42, 123, 999]`), but the current experiment execution runner [`experiments/run_experiments.py`](file:///c:/dev/ot-proj/experiments/run_experiments.py) executes a single seed (`seed=42`) per instance to save runtime during initial pipeline validation.
  - The JSON outputs in [`results/`](file:///c:/dev/ot-proj/results/) store single-run point metrics, not distributions over 20-30 independent runs.
  - No Wilcoxon signed-rank or Mann-Whitney U test function is currently implemented in [`src/metrics.py`](file:///c:/dev/ot-proj/src/metrics.py) to assess statistical significance across multiple seeds.
- **Required Remediation**: Add a multi-seed execution harness in `experiments/run_experiments.py` running 20 or 30 seeds, compute `mean` and `std` for all indicators, store aggregated arrays in `results/*_metrics.json`, and implement `scipy.stats.wilcoxon` tests comparing Hybrid Memetic vs NSGA-II.

---

### b) Do all methods share one hypervolume reference point and normalization, and where is it defined?
- **Status**: **GAP IDENTIFIED**
- **Findings**:
  - In [`src/metrics.py`](file:///c:/dev/ot-proj/src/metrics.py#L67-L105), `compute_hypervolume(points, ref_point=None, ideal_point=None, nadir_point=None)` supports reference points and min-max normalization.
  - However, when called from [`experiments/run_experiments.py`](file:///c:/dev/ot-proj/experiments/run_experiments.py#L143-L148), `ref_point` was omitted (`ref_point=None`).
  - Consequently, line 92 in `src/metrics.py` dynamically falls back to:
    ```python
    r_point = np.max(nd_points, axis=0) * 1.1
    ```
    This means **each algorithm currently evaluates Hypervolume against its own local bounding reference point** rather than a shared, fixed reference point derived from global Nadir bounds or problem specifications.
- **Required Remediation**: Explicitly compute a shared global reference point (e.g., $1.1 \times \text{global\_nadir}$ or fixed theoretical bounds $[-Z_1^{\min}, -Z_2^{\min}, Z_3^{\max}]$) per instance and normalize all algorithm fronts to $[0, 1]^3$ prior to calculating hypervolume.

---

### c) Why is Min-Cost Flow infeasible on the small instance (CV=6) and why is exact MILP's Z2 lower than the heuristics?
- **Status**: **EXPLAINED & DOCUMENTED**
- **Technical Explanation**:
  1. **Why Min-Cost Flow has $CV = 6.0$**:
     - Min-Cost Flow operates on a classical bipartite network where nodes are students and alumni.
     - Arc capacities enforce $x_{ij} \in \{0, 1\}$, student supply enforces $\sum_j x_{ij} = K$, and alumni sink capacity enforces $\sum_i x_{ij} \le C_j$.
     - However, **the corporate employer diversity cap** ($\sum_{j \in \mathcal{A}_k} x_{ij} \le \text{cap}_{\text{comp}}$) and **branch parity** ($|\overline{Q}_b - \overline{Q}_{b'}| \le \delta$) cannot be represented in a standard bipartite flow network without multi-commodity side constraints (which destroys the total unimodularity property).
     - Because Min-Cost Flow greedily routes flow along the highest match scores $q_{ij}$, it assigned multiple mentors from the same corporate employer to several students, violating constraint (C7) 6 times.
  2. **Why Exact MILP's $Z_2$ appears lower on certain Pareto points**:
     - Exact MILP simultaneously enforces **all 7 hard operational constraints** (degree $K$, capacity $C_j$, consent $c_j$, quality threshold $q_{\min}$, quality floor $\theta = 0.70$, branch parity $\delta = 0.35$, and company diversity cap $\text{cap}_{\text{comp}} = 1$).
     - In the Pareto sweep, when the MILP is instructed to minimize maximum load $Z_3 = \max_j L_j$ to its absolute minimum ($L_{\max} = 4$), the solver is forced to activate alumni with lower compatibility scores to avoid concentrating load on popular mentors.
     - Consequently, for load-balanced solutions with $L_{\max} = 4$, the worst-off student quality $Q_{\text{worst}}$ drops to $0.7011$ (satisfying the hard quality floor $\theta = 0.70$).
     - Conversely, Greedy and Min-Cost Flow ignored load balancing and allowed high maximum load ($L_{\max} = 8$ or $10$), allowing them to keep $Z_2$ higher at the expense of severe mentor congestion. On the fairness-focused MILP solutions ($w_2 = 1.0$), MILP actually achieves $Z_2 = 1.25$, which is higher than Greedy ($1.02$).

---

### d) Are $\theta$ and $\delta$ calibrated from data?
- **Status**: **GAP IDENTIFIED**
- **Findings**:
  - In [`src/data_generator.py`](file:///c:/dev/ot-proj/src/data_generator.py#L137-L143) and [`experiments/configs/small_50x30.yaml`](file:///c:/dev/ot-proj/experiments/configs/small_50x30.yaml), $\theta$ and $\delta$ are statically hardcoded as parameters (`theta_min_quality: 0.70`, `delta_branch_parity: 0.35`).
  - They are not dynamically calibrated from the generated matrix $q_{ij}$ (e.g., setting $\theta = K \cdot \text{quantile}_{15}(q_{ij})$ or calibrating $\delta$ as a percentage of the unconstrained baseline departmental spread $\Delta_{\text{baseline}}$).
- **Required Remediation**: Add a calibration utility in `src/data_generator.py` or `src/problem.py` that computes empirical threshold baselines from the data distribution.

---

### e) Is the diversity cap implemented?
- **Status**: **MET (With Caveat)**
- **Findings**:
  - **In Model Evaluation**: Fully implemented in [`src/problem.py`](file:///c:/dev/ot-proj/src/problem.py#L225-L235) (counts violations whenever student receives $> \text{cap}_{\text{comp}}$ mentors from the same employer).
  - **In Exact MILP**: Fully formulated as linear constraints in [`src/algorithms/exact_milp.py`](file:///c:/dev/ot-proj/src/algorithms/exact_milp.py#L182-L198) (`LinearConstraint` over corporate subsets).
  - **In Custom Repair**: Strictly enforced in [`src/multiobjective/repair.py`](file:///c:/dev/ot-proj/src/multiobjective/repair.py#L90-L95).
  - **In Greedy Baseline**: Enforced via `student_companies` tracking dictionary.
  - **Caveat**: Not enforced in Min-Cost Flow due to network flow structure limits, causing its $CV=6$ violation.

---

### f) Are all paths relative and are seeds and parameters read from YAML?
- **Status**: **MET (With Minor Exception)**
- **Findings**:
  - All file references across YAML configs, `main.py`, and `src/` use relative paths (`data/`, `results/`, `figures/`).
  - Algorithm parameters (`pop_size`, `n_gen`, `seed`, `q_min`, `k_matches`) are parsed from YAML configs using `PyYAML`.
  - **Minor Exception**: In [`experiments/run_scalability.py`](file:///c:/dev/ot-proj/experiments/run_scalability.py#L74-L75), generations were adjusted in code (`n_gen = 25 if instance.n_students > 500 else 30`) instead of being read strictly from YAML.

---

### g) Is there a Makefile or run_all script?
- **Status**: **GAP IDENTIFIED**
- **Findings**:
  - While `python main.py --mode all` executes the full sequence of experiments, there is no root `Makefile` (`make test`, `make run`, `make all`) or shell script (`run_all.sh` / `run_all.bat`).
- **Required Remediation**: Create a cross-platform `Makefile` and `run_all.bat` / `run_all.sh` supporting `make install`, `make test`, `make benchmark`, `make exact`, `make ablation`, `make all`.

---

### h) Does README.md follow the 13-section format?
- **Status**: **GAP IDENTIFIED**
- **Findings**:
  - [`README.md`](file:///c:/dev/ot-proj/README.md) currently contains 8 numbered sections:
    1. Overview & Institutional Motivation
    2. Mathematical Model Summary
    3. Repository Architecture
    4. Installation & Environment Setup
    5. Usage & CLI Driver
    6. Key Empirical Results
    7. Generated Visualizations
    8. Library Citations
  - It lacks the complete 13-section academic template (e.g. separate sections for Methodology & Algorithm Formulations, Theoretical Complexity Analysis, Experimental Setup & Cohort Sizing, Ethical & Privacy Considerations, Limitations & Future Work).
- **Required Remediation**: Restructure `README.md` into the formal 13-section format.

---

### i) Are any numbers in report/final_report.md, presentation_slides.md or literature_review.md not traceable to files in results/, and are all citations real and verifiable?
- **Status**: **VERIFIED & TRACEABLE**
- **Findings**:
  - All quantitative numbers in [`report/final_report.md`](file:///c:/dev/ot-proj/report/final_report.md) and [`presentation/presentation_slides.md`](file:///c:/dev/ot-proj/presentation/presentation_slides.md) map directly to values in `results/`:
    - Small instance table $\to$ `results/small_50x30_metrics.json`
    - Medium instance table $\to$ `results/medium_300x150_metrics.json`
    - Large instance table $\to$ `results/large_1000x500_metrics.json`
    - Exact MILP points and secondary scenario $\to$ `results/exact_milp_results.json`
    - Price of parity values ($\delta \in [0.15, 0.50]$) $\to$ `results/ablation_results.json`
    - Scalability runtimes $\to$ `results/scalability_results.json`
  - All 9 literature citations (Ahuja et al. 1993, Bertsimas et al. 2011, Budish 2011, Deb et al. 2002, Gale & Shapley 1962, Kuhn 1955, Rawls 1971, Zhang & Li 2007, Zitzler et al. 2001) are genuine peer-reviewed publications.
  - All 8 software citations (`pymoo`, `scipy`, `numpy`, `networkx`, `matplotlib`, `seaborn`, `pytest`, `pyyaml`) are real and correct.

---

## 4. Prioritized Action Plan / Gap Summary

| Priority | Gap Description | Targeted Component | Proposed Action |
| :---: | :--- | :--- | :--- |
| **P1** | **Multi-Seed Benchmark & Statistical Tests** | `experiments/run_experiments.py`, `src/metrics.py` | Run 20-30 independent random seeds, store mean $\pm$ std in `results/`, implement Wilcoxon signed-rank tests. |
| **P1** | **Shared Global Hypervolume Reference Point** | `src/metrics.py`, `experiments/run_experiments.py` | Compute a single fixed reference point per instance from global Nadir/bounds and normalize fronts to $[0, 1]^3$. |
| **P2** | **Data-Driven $\theta$ and $\delta$ Calibration** | `src/data_generator.py` | Calibrate $\theta$ via $K \times \text{quantile}_{15}(q_{ij})$ and $\delta$ via baseline departmental variance. |
| **P2** | **Makefile & Automation Scripts** | Root repository | Provide a clean `Makefile` and `run_all.bat` / `run_all.sh`. |
| **P2** | **13-Section README Formatting** | `README.md` | Expand `README.md` into the formal 13-section specification. |
