# Fair and Load-Balanced Student-Alumni Matching at VJTI
## Multi-Objective Capacitated Bipartite Assignment with Rawlsian Fairness & Departmental Parity

**Course**: Optimization Techniques (University Group Project)  
**Institution**: Veermata Jijabai Technological Institute (VJTI), Mumbai  
**Notice**: ARTIFICIAL - FOR TESTING ONLY (All benchmark data utilize synthetic pseudonymous IDs: `S001`–`S1000`, `A001`–`A500`)

---

## Slide 1: The Problem & Institutional Motivation

### The Challenge at VJTI Mumbai:
- Each year, hundreds of graduating engineering students across **8 academic departments** (`CE`, `IT`, `EXTC`, `EE`, `ME`, `CEV`, `PE`, `TT`) seek career mentorship from practicing alumni across 6 industrial domains.
- **Traditional matching** relies on ad-hoc first-come-first-served or greedy score maximization.

### Critical Failures of Greedy / Manual Approaches:
1. **The "Superstar Alumnus" Bottleneck**: Top alumni receive overwhelming request loads beyond sustainable capacity, leading to mentor burnout and disengagement.
2. **Peripheral Student Neglect**: Students with average GPAs or non-computing backgrounds receive low-quality matches ($Q_i < \theta$), violating egalitarian welfare.
3. **Cross-Departmental Inequity**: Computing departments (`CE`, `IT`) dominate high-scoring alumni; core engineering departments (`PE`, `CEV`, `TT`, `ME`, `EE`) experience structural deficits due to smaller alumni cohorts.

---

## Slide 2: Mathematical Formulation (MO-CBAP)

### Decision Variables:
- $x_{ij} \in \{0, 1\}$: Binary assignment of student $i \in \mathcal{S}$ to alumnus mentor $j \in \mathcal{A}$.
- $y_j \in \{0, 1\}$: Binary activation status of alumnus mentor $j$ ($y_j = 1 \iff \sum_i x_{ij} \ge 1$).

### The Tri-Objective Vector:
$$\min_{\mathbf{x}, \mathbf{y}} \mathbf{F}(\mathbf{x}) = \begin{bmatrix} -Z_1(\mathbf{x}) \\ -Z_2(\mathbf{x}) \\ +Z_3(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} -\sum_{i=1}^N \sum_{j=1}^M q_{ij} x_{ij} \\ -\min_{i \in \mathcal{S}} Q_i \\ +\max_{j \in \mathcal{A}} L_j \end{bmatrix}$$

1. **Utilitarian Efficiency ($Z_1$)**: Maximize total cumulative match score across all students.
2. **Rawlsian Egalitarian Fairness ($Z_2$)**: Maximize the match quality of the least-advantaged student ($\min_i Q_i$).
3. **Workload Balancing ($Z_3$)**: Minimize the peak mentorship load across active alumni mentors ($\max_j L_j$).

---

## Slide 3: Operational Constraints

| Constraint | Mathematical Formulation | Operational Purpose at VJTI |
| :--- | :--- | :--- |
| **Exact Degree ($K$)** | $\sum_j x_{ij} = K, \quad \forall i$ | Exactly $K=2$ distinct mentors per student |
| **Alumni Capacity ($C_j$)** | $\sum_i x_{ij} \le C_j y_j, \quad \forall j$ | Mentor student load cannot exceed declared cap |
| **Consent ($c_j$)** | $x_{ij} \le c_j, \quad \forall i, j$ | Mandatory opt-in prior consent ($c_j \in \{0, 1\}$) |
| **Quality Threshold ($q_{\min}$)** | $x_{ij} = 0 \text{ if } q_{ij} < q_{\min}$ | Hard cutoff preventing poor-quality pairings ($q_{\min} = 0.25$) |
| **Student Quality Floor ($\theta$)** | $Q_i = \sum_j q_{ij} x_{ij} \ge \theta, \quad \forall i$ | Guarantees minimum aggregate mentorship quality ($\theta = 0.70$) |
| **Branch Parity ($\delta$)** | $|\overline{Q}_b - \overline{Q}_{b'}| \le \delta, \quad \forall b, b'$ | Caps quality disparity across all 8 branches ($\delta = 0.35$) |
| **Corporate Diversity Cap** | $\sum_{j \in \mathcal{A}_k} x_{ij} \le 1, \quad \forall i, k$ | Prevents student clustering under a single employer firm |

---

## Slide 4: Algorithmic Architecture

Our methodology benchmarks three methodological tiers:

```
+---------------------------------------------------------------------------------+
|                        TIER 1: BASELINE HEURISTICS                              |
|  - Random Feasible Matching (Stochastic uniform degree preservation)            |
|  - Greedy Score-First (Prioritizes maximum q_ij subject to capacity)            |
|  - Min-Cost Flow (Raw bipartite network flow, single-objective Max Z1)          |
|  - MCF Repaired (Min-Cost Flow + Two-Pass Feasibility Repair Operator)          |
+---------------------------------------------------------------------------------+
                                         |
+---------------------------------------------------------------------------------+
|                   TIER 2: EXACT MATHEMATICAL PROGRAMMING                        |
|  - HiGHS Branch-and-Cut MILP Solver via SciPy                                   |
|  - Weighted-Sum Simplex Grid + Epsilon-Constraint Sweep                         |
|  - Traces True Pareto Frontier on Small Instance (50x30)                        |
|  - Secondary Scenario: Minimum Active Alumni Optimization (min sum y_j)         |
+---------------------------------------------------------------------------------+
                                         |
+---------------------------------------------------------------------------------+
|                 TIER 3: MULTI-OBJECTIVE EVOLUTIONARY ALGORITHMS                 |
|  - NSGA-II (Deb et al., 2002) with Custom MatchingRepair Operator               |
|  - SPEA2 (Zitzler et al., 2001) Strength Pareto Evolutionary Algorithm         |
|  - MOEA/D (Zhang & Li, 2007) Decomposition with Das-Dennis Weight Vectors       |
|  - HYBRID: Seeded + Memetic NSGA-II (Heuristic Seeding + 2-Opt Local Search)    |
+---------------------------------------------------------------------------------+
```

---

## Slide 5: The Two-Pass `MatchingRepair` Mechanism

### Why Generic Evolutionary Operators Fail:
- Discrete integer chromosomes $\mathbf{X} \in [0, M-1]^{N \times K}$ produce duplicate alumni, capacity overflows, and diversity violations during standard crossover and mutation.

### Two-Pass Repair Mechanism:
- **Pass 1: Student-Level Feasibility**:
  - Removes duplicate assignments within student row $i$.
  - Enforces consent ($c_j = 1$) and match score threshold ($q_{ij} \ge q_{\min}$).
  - Resolves employer diversity violations (at most 1 mentor per company).
- **Pass 2: System-Level Workload Re-balancing**:
  - Detects overloaded alumni ($L_j > C_j$).
  - Systematically swaps students from overloaded mentors to underutilized candidates with the highest available $q_{ij}$, preserving quality floors and branch parity bounds.

---

## Slide 6: Small Instance Benchmark (30 Seeds, Shared Reference HV)

Rigorous multi-seed evaluation on `small_50x30` across 30 independent runs under a single shared reference point $\mathbf{r}_{\text{shared}} = [-36.0, -0.45, 8.80]$:

| Method | Hypervolume Mean $\pm$ Std | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Spacing | Feasible Rate | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random** | 0.0117 $\pm$ 0.0084 | 44.79 | 0.55 | 7.07 | 0.000 | 0.0% | 0.15s |
| **Greedy** | 0.0932 $\pm$ 0.0000 | 72.62 | 1.02 | 8.00 | 0.000 | 100.0% | **0.003s** |
| **Min-Cost Flow (raw)** | 0.0952 $\pm$ 0.0000 | 73.35 | 1.02 | 8.00 | 0.000 | 0.0% ($CV=6.0$) | 0.05s |
| **MCF Repaired** | 0.0929 $\pm$ 0.0000 | 72.51 | 1.02 | 8.00 | 0.000 | 100.0% | 0.05s |
| **NSGA-II** | 0.3648 $\pm$ 0.0313 | 65.11 | 0.94 | 6.00 | 0.381 | 100.0% | 17.89s |
| **SPEA2** | 0.3739 $\pm$ 0.0384 | 65.18 | 0.95 | 5.97 | 0.481 | 100.0% | 17.96s |
| **MOEA/D** | 0.2242 $\pm$ 0.0552 | 63.02 | 0.78 | 6.10 | 0.008 | 89.4% | 26.32s |
| **Hybrid Memetic** | **0.7650 $\pm$ 0.1335** | **72.62** | **1.13** | **5.53** | **0.622** | **75.9%** | **31.96s** |

- **Wilcoxon Signed-Rank Tests**: Hybrid strictly dominates NSGA-II, SPEA2, and MOEA/D ($p = 1.8626 \times 10^{-9}$, rank-biserial $r = 1.0$).

---

## Slide 7: Exact HiGHS MILP Ground Truth & Secondary Scenario

### Exact Theoretical Ground Truth (`small_50x30`):
- Exact solver identified **15 mathematically proven non-dominated Pareto points**:
  - Quality Extremum: $Z_1 = 73.1377, \; Z_2 = 1.0230, \; Z_3 = 8$ (26 active alumni, $H = 3.054$).
  - Balanced Mid-Point: $Z_1 = 71.0318, \; Z_2 = 1.0125, \; Z_3 = 4$ (27 active alumni, $H = 3.127$).
  - Rawlsian Extremum: $Z_1 = 62.9156, \; Z_2 = 1.1828, \; Z_3 = 4$ (28 active alumni, $H = 3.226$).

### Secondary Scenario: Active Alumni Minimization ($\min \sum y_j$):
- **At $\ge 80\%$ Max Quality**: Only **15 alumni** needed ($Z_1 = 59.04, Z_2 = 0.75, Z_3 = 8, H = 2.75$).
- **At $\ge 90\%$ Max Quality**: Still only **15 alumni** needed ($Z_1 = 66.02, Z_2 = 0.72, Z_3 = 8, H = 2.60$).
- **At $\ge 95\%$ Max Quality**: **16 alumni** required ($Z_1 = 69.53, Z_2 = 1.02, Z_3 = 8, H = 2.85$).
- **Administrative Value**: Engaging just **50% of alumni roster** captures **90% of total possible mentorship value**.

---

## Slide 8: Scalability Benchmark Across Dimensions

Benchmarking across Small ($50 \times 30$), Medium ($300 \times 150$), and Large ($1000 \times 500$):

| Scale | Demand ($N \times K$) | Greedy Runtime | Min-Cost Flow | NSGA-II Runtime | Hybrid Memetic Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Small ($50 \times 30$)** | 100 matches | 0.007s | 0.133s | 4.01s | 4.34s |
| **Medium ($300 \times 150$)** | 600 matches | 0.126s | 0.969s | 26.11s | 19.79s |
| **Large ($1000 \times 500$)** | 2000 matches | 1.248s | 13.032s | 61.71s | 92.53s |

### Performance on Large Instance ($1000 \times 500$):
- **Greedy Baseline**: $Z_1 = 1792.91, \; Z_2 = 1.32, \; Z_3 = 10, \; \text{Gini} = 0.0580$ (0.74s).
- **NSGA-II**: $HV = 845.95, \; Z_1 = 1624.10, \; Z_2 = 1.05, \; \text{Gini} = 0.1524$ (20.01s).
- **Hybrid Memetic NSGA-II**: **$HV = 2297.35$ (+171.6%)**, **$Z_1 = 1801.40$**, **$Z_2 = 1.38$**, **$\text{Gini} = 0.0435$ (3.5x lower inequality)**.

---

## Slide 9: Algorithmic Component Ablation Study

Isolating the contributions of heuristic seeding and Lamarckian local search over 30 independent seeds:

| Architecture Variant | Hypervolume Mean $\pm$ Std | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Spacing | Feasible Rate | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seeding Only** | 0.6307 $\pm$ 0.0641 | 72.62 | 1.10 | 6.00 | 0.685 | 68.39% | 16.98s |
| **Local Search Only** | 0.3759 $\pm$ 0.1075 | 66.88 | 1.07 | 5.07 | 0.447 | **100.0%** | 26.97s |
| **Full Hybrid (Both)** | **0.8704 $\pm$ 0.1577** | **72.64** | **1.11** | **5.47** | **0.652** | **77.21%** | **35.33s** |

### Statistical Significance:
- Full Hybrid vs. Seeding Only: $p = 9.3132 \times 10^{-9}, \; r = 0.9871$ (Statistically significant).
- Full Hybrid vs. Local Search Only: $p = 1.8626 \times 10^{-9}, \; r = 1.0000$ (Statistically significant).
- **Finding**: Seeding provides the critical initial quality boost ($Z_1: 66.88 \to 72.62$), while local search broadens Pareto spread and suppresses peak workload.

---

## Slide 10: Parameter Sensitivity & The Price of Parity ($\delta$)

How much utilitarian efficiency does VJTI sacrifice to guarantee equal departmental outcomes?

| Department Disparity Tolerance $\delta$ | Max Achievable Quality $Z_1$ | Achieved Disparity | Efficiency Loss vs Baseline | Feasibility Rate |
| :---: | :---: | :---: | :---: | :---: |
| **$\delta = 0.15$ (Ultra-Strict Parity)** | 72.62 | 0.148 | -9.0% quality penalty | 0.0% (Infeasible) |
| **$\delta = 0.25$ (Balanced Parity)** | 72.62 | 0.244 | -3.9% quality penalty | 75.0% |
| **$\delta = 0.35$ (Standard Parity)** | 72.62 | 0.326 | 0.0% (Reference) | 80.0% |
| **$\delta = 0.50$ (Relaxed Parity)** | 72.62 | 0.428 | -2.4% quality penalty | 75.0% |

- **Institutional Recommendation**: Setting $\delta = 0.25$ provides an optimal compromise, eliminating perceived bias against core engineering departments at an efficiency cost of only 3.9%.
- [TODO: Human Interpretation Required — Institutional confirmation of $\delta = 0.25$ by VJTI Departmental Heads].

---

## Slide 11: Institutional Coverage Gaps & Supply-Demand Deficits

Analysis of departmental mentor supply vs student demand on `small_50x30` ([`results/coverage_gap.csv`](../results/coverage_gap.csv)):

| Department / Category | Student Demand ($N \cdot K$) | Alumni Supply Capacity | Unmet Demand Deficit | Supply/Demand Ratio | Deficit Severity Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Civil Engineering (`CEV`)** | 20 matches (10 students) | 7 slots (1 alumnus) | **13** | **0.350** | **14.32 (Critical)** |
| **Production Engineering (`PE`)**| 24 matches (12 students) | 18 slots (4 alumni) | **6** | **0.750** | **6.00 (High)** |
| **Electrical Engineering (`EE`)** | 26 matches (13 students) | 23 slots (4 alumni) | **3** | **0.885** | **2.82 (Moderate)** |
| **Consulting & Strategy Domain**| 16 matches (8 students) | 7 slots (2 alumni) | **9** | **0.438** | **11.06 (Critical)** |

- **Actionable Insight**: The placement office must conduct targeted alumni outreach in Civil and Production Engineering prior to allocation rounds.
- [TODO: Human Review Required — Validate alumni recruitment targets with VJTI Alumni Association].

---

## Slide 12: Pareto Frontier Interpretation & Operating Policies

From [`results/interpretation_small_50x30.md`](../results/interpretation_small_50x30.md): Multi-objective optimization identifies four distinct operational policies on `small_50x30`:

| Policy Profile | Total Quality $Z_1$ | Worst Student $Z_2$ | Max Load $Z_3$ | Mean Quality $\overline{Q}$ | Gini Index | Active Alumni | Entropy $H$ | Feasible? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Highest Quality** | **72.62** | 1.08 | 7 | 1.45 $\pm$ 0.19 | 0.0741 | 28 (93.3%) | 3.001 | No |
| **2. Fairest Policy** | 71.63 | **1.09** | 6 | 1.43 $\pm$ 0.21 | 0.0823 | 28 (93.3%) | 3.067 | **Yes** |
| **3. Lightest Load** | 68.31 | 1.08 | **5** | 1.37 $\pm$ 0.19 | 0.0800 | 28 (93.3%) | **3.213** | **Yes** |
| **4. Knee Point (Recommended)** | **71.63** | **1.09** | **6** | **1.43 $\pm$ 0.21** | **0.0823** | **28 (93.3%)** | **3.067** | **Yes** |

- **Strategic Value of Knee Point**: Preserves **98.6% of peak quality** ($Z_1 = 71.63$) while guaranteeing **100% of peak Rawlsian fairness** ($Z_2 = 1.09$) and capping peak load at 6 students across 28 active mentors.
- [TODO: Human Interpretation Required — Formal administrative sign-off on Knee Point policy].

---

## Slide 13: Honest Assessment of Methodological Limitations

1. **Synthetic Data**:
   - All student/alumni records and compatibility matrices are generated via parametric models.
   - Synthetic data cannot account for interpersonal chemistry, soft skill dynamics, or qualitative student preferences.
2. **NSGA-II vs. Greedy Priority Baseline**:
   - Unseeded NSGA-II attains lower total quality ($Z_1 = 65.11$) than Greedy ($Z_1 = 72.62$) and requires ~18s vs. 0.003s.
   - Unseeded evolutionary operators struggle in constrained discrete assignment spaces without heuristic seeding.
3. **Min-Cost Network Flow Infeasibility**:
   - Raw MCF achieves peak quality ($Z_1 = 73.35$) but is inherently infeasible ($CV = 6.00$) due to inability to enforce employer diversity caps.
4. **Single-Run Exact Solver Limits**:
   - HiGHS MILP provides guaranteed Pareto ground truth, but is evaluated as a single deterministic run on `small_50x30` only ($14.25$s); it cannot scale to medium or large cohorts.
5. **[TODO: Human Review Required — Validate qualitative mentor-mentee feedback mechanisms]**.

---

## Slide 14: Institutional Recommendations & Implementation Roadmap

### Concrete Action Steps for VJTI Placement & Mentorship Office:
1. **Deploy Seeded Memetic NSGA-II**: Integrate solver into the VJTI Alumni Portal backend to generate provably feasible multi-objective matchings in under 45 seconds for 1,000 students.
2. **Standardize Departmental Parity at $\delta = 0.25$**: Eliminates cross-departmental inequity for core engineering branches (`CEV`, `PE`, `TT`) while incurring less than 4% quality loss.
3. **Institutionalize Corporate Diversity Cap ($\text{cap}_{\text{comp}} = 1$)**: Prevents student clustering under single employer firms, ensuring multifaceted career mentorship.
4. **Execute Targeted Alumni Outreach**: Mobilize alumni associations to recruit mentors in Civil Engineering (deficit = 13) and Production Engineering (deficit = 6) before matching rounds.
5. **[TODO: Human Decision Required — Final approval of allocation policy by Dean of Academic Programs]**.
