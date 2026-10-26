# Fair and Load-Balanced Student-Alumni Matching at VJTI
## Multi-Objective Capacitated Bipartite Assignment with Rawlsian Fairness & Departmental Parity

**Course**: Optimization Techniques (University Group Project)  
**Institution**: Veermata Jijabai Technological Institute (VJTI), Mumbai  
**Notice**: ARTIFICIAL - FOR TESTING ONLY (All benchmark data utilize pseudonymous IDs: S001, A001)

---

## Slide 1: The Problem & Institutional Context

### The Challenge at VJTI:
- Each year, hundreds of graduating engineering students across **8 academic departments** seek alumni mentorship.
- **Traditional matching** relies on first-come-first-served or greedy score maximization.

### Critical Failures of Greedy Approaches:
1. **Mentor Burnout**: Top alumni receive overwhelming request loads, leading to dropouts.
2. **Peripheral Student Neglect**: Students with average GPAs or cross-domain aspirations receive poor matches.
3. **Departmental Inequity**: Computing branches (CE/IT) dominate high-scoring alumni; core branches (Production, Textile, Civil) face structural deficits.

---

## Slide 2: Mathematical Formulation

### Decision Variables:
- $x_{ij} \in \{0, 1\}$: Binary match between student $i$ and alumnus $j$.
- $y_j \in \{0, 1\}$: Alumni activation status ($\sum_i x_{ij} \ge 1$).

### The Tri-Objective Trade-off:
$$\min \mathbf{F}(\mathbf{x}) = \begin{bmatrix} -Z_1 \\ -Z_2 \\ +Z_3 \end{bmatrix}$$
1. **$\max Z_1 = \sum_{i,j} q_{ij} x_{ij}$**: Utilitarian Match Quality (Total Human Capital).
2. **$\max Z_2 = \min_i Q_i$**: Rawlsian Egalitarian Fairness (Protect the least-advantaged student).
3. **$\min Z_3 = \max_j L_j$**: Workload Balancing (Prevent mentor overload).

---

## Slide 3: Operational Constraints

| Constraint | Mathematical Form | Operational Purpose |
| :--- | :--- | :--- |
| **Degree ($K$)** | $\sum_j x_{ij} = K, \forall i$ | Exactly $K$ mentors per student ($K=2$) |
| **Capacity ($C_j$)** | $\sum_i x_{ij} \le C_j y_j, \forall j$ | Mentor cannot exceed capacity cap |
| **Consent ($c_j$)** | $x_{ij} \le c_j, \forall i,j$ | Prior opt-in mandatory |
| **Threshold ($q_{\min}$)** | $x_{ij} = 0 \text{ if } q_{ij} < q_{\min}$ | No low-quality matches allowed |
| **Quality Floor ($\theta$)** | $Q_i = \sum_j q_{ij} x_{ij} \ge \theta$ | Guarantees minimum mentorship quality |
| **Branch Parity ($\delta$)** | $|\overline{Q}_b - \overline{Q}_{b'}| \le \delta, \forall b, b'$ | Egalitarian parity across departments |
| **Diversity Cap** | $\sum_{j \in \text{Comp}(k)} x_{ij} \le 1$ | Prevent single-firm clustering |

---

## Slide 4: Algorithmic Architecture

Our methodology benchmarks three methodological tiers:

```
+-----------------------------------------------------------------+
|                        TIER 1: BASELINES                        |
|  - Random Feasible Matching (Stochastic)                        |
|  - Greedy Score-First (Prioritizes highest q_ij)                |
|  - Min-Cost Network Flow (Exact single-objective Max Z1)        |
+-----------------------------------------------------------------+
                                |
+-----------------------------------------------------------------+
|                     TIER 2: EXACT METHOD                        |
|  - HiGHS MILP Solver via SciPy                                  |
|  - Weighted-Sum Simplex Grid + Epsilon-Constraint Sweep         |
|  - Solves Small Instance (50x30) to mathematical ground truth   |
+-----------------------------------------------------------------+
                                |
+-----------------------------------------------------------------+
|                 TIER 3: MULTI-OBJECTIVE MOEAs                   |
|  - NSGA-II (Deb et al., 2002) with Custom Repair Operator       |
|  - SPEA2 (Zitzler et al., 2001) Strength Pareto Evolutionary    |
|  - MOEA/D (Zhang & Li, 2007) Decomposition with Das-Dennis Ref |
|  - HYBRID: Seeded + Memetic NSGA-II (Lamarckian Local Search)   |
+-----------------------------------------------------------------+
```

---

## Slide 5: Custom Feasibility Repair Operator

### Why Generic Genetic Operators Fail:
- Discrete integer chromosomes $X \in [0, M-1]^{N \times K}$ easily produce duplicate alumni, capacity overflows, and diversity violations during crossover/mutation.

### Two-Pass MatchingRepair Mechanism:
- **Pass 1 (Student Level)**:
  - Filters non-consenting alumni ($c_j = 0$) and sub-threshold scores ($q_{ij} < q_{\min}$).
  - Enforces corporate diversity cap (at most 1 alumnus per company).
  - Fills shortfalls greedily from sorted feasible candidate pools.
- **Pass 2 (System Level Load Re-balancing)**:
  - Identifies bottleneck alumni ($L_j > C_j$).
  - Systematically swaps students from overloaded mentors to underutilized candidates with highest available $q_{ij}$, preserving quality floors and branch parity.

---

## Slide 6: Small Instance Benchmark ($50 \times 30$)

True Pareto Frontier traced by HiGHS MILP vs Evolutionary Solvers:

| Algorithm | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Hypervolume | Spacing | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 45.95 | 0.60 | 8 | — | — | 0.012 |
| **Greedy Baseline** | 72.62 | 1.02 | 8 | — | — | 0.002 |
| **Min-Cost Flow** | 73.35 | 1.02 | 8 | — | — | 0.053 |
| **Exact MILP** | [62.4 - 73.3] | [1.02 - 1.25] | [4 - 8] | **443.98** | 2.25 | 14.25 |
| **NSGA-II** | [68.1 - 70.4] | [0.98 - 1.12] | [6 - 7] | 115.75 | 0.01 | 3.05 |
| **SPEA2** | [68.0 - 69.8] | [0.99 - 1.10] | [6 - 7] | 118.29 | 0.00 | 3.21 |
| **Hybrid Memetic** | [69.2 - 72.8] | [1.02 - 1.21] | [5 - 7] | **232.17** | 0.15 | **2.96** |

**Observation**: Hybrid Memetic NSGA-II doubles the Hypervolume of standard NSGA-II and discovers high-quality compromise solutions in under 3 seconds.

---

## Slide 7: Scalability Across Cohort Dimensions

Benchmarking across Small ($50 \times 30$), Medium ($300 \times 150$), and Large ($1000 \times 500$):

| Scale | Demand ($N \times K$) | Greedy (s) | Min-Cost Flow (s) | NSGA-II (s) | Hybrid Memetic (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Small (50x30)** | 100 | 0.007 | 0.133 | 4.01 | 4.34 |
| **Medium (300x150)**| 600 | 0.126 | 0.969 | 26.11 | 19.79 |
| **Large (1000x500)** | 2000 | 1.248 | 13.03 | 61.71 | 92.53 |

- **Hypervolume Performance at Large Scale**:
  - Raw NSGA-II: $HV = 845.95$, Gini = 0.152
  - **Hybrid Memetic NSGA-II**: **$HV = 2297.35$ (+171.6%)**, **Gini = 0.0435 (3.5x lower inequality)**

---

## Slide 8: The Price of Branch Parity ($\delta$)

How much total efficiency does VJTI sacrifice to guarantee equal departmental outcomes?

| Disparity Tolerance $\delta$ | Max Quality $Z_1$ | Achieved Disparity | Efficiency Impact |
| :---: | :---: | :---: | :---: |
| **0.15** (Ultra-Strict) | 65.80 | 0.148 | -9.0% |
| **0.25** (Balanced Parity) | 69.47 | 0.244 | -3.9% |
| **0.35** (Standard Parity) | 72.32 | 0.326 | Reference (0%) |
| **0.50** (Relaxed Parity) | 70.61 | 0.428 | -2.4% |

**Strategic Recommendation**: $\delta = 0.25$ provides an ideal sweet spot, eliminating perceived bias against core engineering departments at an efficiency cost of only 3.9%.

---

## Slide 9: Secondary Scenario: Minimizing Active Alumni

*Can we achieve top-tier student matching while contacting fewer alumni?*

- **At $\ge 80\%$ Max Quality**: Only **15 alumni** needed ($H = 2.75$)
- **At $\ge 90\%$ Max Quality**: Still only **15 alumni** needed ($H = 2.60$)
- **At $\ge 95\%$ Max Quality**: **16 alumni** required ($H = 2.85$)

**Administrative Value**: VJTI can achieve 90% of maximum mentorship value by engaging just **50% of its alumni roster**, saving significant volunteer coordination effort while maintaining high company entropy.

---

## Slide 10: Conclusion & Implementation Roadmap

### Key Findings:
1. **Hybrid Metaheuristics Excel**: Seeded Memetic NSGA-II consistently dominates standard MOEAs in Hypervolume and fairness.
2. **Equitable Allocations**: Gini inequality dropped to 0.043, proving high equity across all students.
3. **Scalability**: Solves 1,000 students x 500 alumni in <95 seconds.

### Roadmap for VJTI:
- **Phase 1**: Embed Seeded Memetic NSGA-II engine into the VJTI Alumni Portal backend.
- **Phase 2**: Standardize departmental disparity policy at $\delta = 0.25$.
- **Phase 3**: Enforce employer diversity cap to ensure multifaceted career mentorship.
