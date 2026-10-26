# Mathematical Formulation: Fair and Load-Balanced Student-Alumni Matching at VJTI

**Course**: Optimization Techniques (University Group Project)  
**Institution**: Veermata Jijabai Technological Institute (VJTI), Mumbai  
**Classification**: Multi-Objective Capacitated Bipartite Assignment Problem (MO-CBAP)  

---

## 1. Problem Context & Motivation

Mentorship allocation programs in technical universities such as VJTI involve assigning graduating students to practicing alumni across engineering disciplines. Traditional assignment models suffer from three fundamental deficiencies:
1. **Utilitarian Myopia**: Maximizing aggregate match score often concentrates high-quality alumni among top-tier departments or high-scoring students, leaving under-represented departments or lower-scoring students with subpar matches.
2. **Mentor Burnout / Over-Concentration**: Popular alumni receive unmanageable mentoring loads, leading to high drop-out rates and degraded mentorship quality.
3. **Departmental Inequity**: Variance in alumni base across engineering branches (e.g., Computer Engineering vs. Textile Technology or Production Engineering) induces severe disparity in average mentorship quality.

To resolve these challenges, this study presents a tri-objective mathematical programming and evolutionary optimization model enforcing **Rawlsian egalitarian fairness**, **cross-departmental parity**, **workload balancing**, and **employer diversity caps**.

---

## 2. Sets, Indices, and Parameters

### 2.1 Sets and Indices
- $i \in \mathcal{S} = \{1, 2, \dots, N\}$: Set of students.
- $j \in \mathcal{A} = \{1, 2, \dots, M\}$: Set of alumni mentors.
- $b \in \mathcal{B} = \{\text{CE}, \text{IT}, \text{EXTC}, \text{EE}, \text{ME}, \text{CEV}, \text{PE}, \text{TT}\}$: Set of academic departments (branches) at VJTI.
- $\mathcal{S}_b \subseteq \mathcal{S}$: Subset of students matriculated in department $b$, where $\bigcup_{b \in \mathcal{B}} \mathcal{S}_b = \mathcal{S}$ and $\mathcal{S}_b \cap \mathcal{S}_{b'} = \emptyset$ for $b \ne b'$.
- $k \in \mathcal{K}$: Set of unique corporate employers / companies.
- $\mathcal{A}_k \subseteq \mathcal{A}$: Subset of alumni employed by company $k$.

### 2.2 Parameters
| Parameter | Domain | Description |
| :--- | :--- | :--- |
| $q_{ij}$ | $[0, 1]$ | Compatibility / match quality score between student $i$ and alumnus $j$ |
| $K$ | $\mathbb{Z}^+$ | Exact number of distinct alumni matched to each student (e.g., $K=2$) |
| $C_j$ | $\mathbb{Z}^+$ | Maximum student mentorship capacity of alumnus $j$ |
| $c_j$ | $\{0, 1\}$ | Consent / opt-in availability flag of alumnus $j$ ($c_j = 1$ if opted in) |
| $q_{\min}$ | $[0, 1]$ | Minimum admissible bilateral match quality threshold |
| $\theta$ | $\mathbb{R}^+$ | Minimum cumulative match quality floor per student ($Q_i \ge \theta$) |
| $\delta$ | $\mathbb{R}^+$ | Maximum allowable disparity tolerance in average quality between academic branches |
| $\text{cap}_{\text{comp}}$ | $\mathbb{Z}^+$ | Maximum allowable matches for student $i$ from the same employer / company |

---

## 3. Decision Variables

### 3.1 Primary Decision Variables
- $x_{ij} \in \{0, 1\}$: Binary assignment variable, equal to $1$ if student $i$ is assigned to alumnus $j$, and $0$ otherwise.
- $y_j \in \{0, 1\}$: Binary activation variable, equal to $1$ if alumnus $j$ is assigned at least one student, and $0$ otherwise.

### 3.2 Auxiliary / State Variables
- $Q_i = \sum_{j=1}^M q_{ij} x_{ij}$: Cumulative match quality realized by student $i$.
- $L_j = \sum_{i=1}^N x_{ij}$: Total student mentoring load allocated to alumnus $j$.
- $Q_{\text{worst}} \in \mathbb{R}^+$: Rawlsian minimum student quality ($Q_{\text{worst}} \le Q_i, \forall i \in \mathcal{S}$).
- $L_{\max} \in \mathbb{Z}^+$: Peak alumni mentorship load ($L_{\max} \ge L_j, \forall j \in \mathcal{A}$).
- $\overline{Q}_b = \frac{1}{|\mathcal{S}_b|} \sum_{i \in \mathcal{S}_b} Q_i$: Departmental average cumulative quality for branch $b \in \mathcal{B}$.

---

## 4. Multi-Objective Formulation

The problem is governed by three conflicting optimization criteria:

### Objective 1: Maximize Total Utilitarian Match Quality ($Z_1$)
$$\max Z_1 = \sum_{i=1}^N \sum_{j=1}^M q_{ij} x_{ij}$$
*Economic Interpretation*: Maximizes aggregate human capital and career alignment across the university cohort.

### Objective 2: Maximize Rawlsian Egalitarian Fairness ($Z_2$)
$$\max Z_2 = \min_{i \in \mathcal{S}} Q_i = Q_{\text{worst}}$$
*Social Welfare Interpretation*: In accordance with John Rawls' *Difference Principle*, optimizes the welfare of the least-advantaged student in the university cohort.

### Objective 3: Minimize Maximum Mentor Workload ($Z_3$)
$$\min Z_3 = \max_{j \in \mathcal{A}} L_j = L_{\max}$$
*Operational Balance Interpretation*: Prevents burnout of highly rated alumni and distributes mentorship responsibility evenly across the alumni network.

In multi-objective optimization (MOO) canonical minimization form:
$$\min \mathbf{F}(\mathbf{x}) = \begin{bmatrix} -Z_1(\mathbf{x}) \\ -Z_2(\mathbf{x}) \\ +Z_3(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} -\sum_{i=1}^N \sum_{j=1}^M q_{ij} x_{ij} \\ -Q_{\text{worst}} \\ +L_{\max} \end{bmatrix}$$

---

## 5. Constraint Specifications

### 5.1 Student Degree Constraints (Exact $K$ Matches)
Each student must receive exactly $K$ distinct alumni mentors:
$$\sum_{j=1}^M x_{ij} = K, \quad \forall i \in \mathcal{S} \tag{C1}$$

### 5.2 Alumni Capacity and Activation Constraints
An alumnus can only be assigned students if consented and active, and total assignments cannot exceed capacity:
$$\sum_{i=1}^N x_{ij} \le C_j y_j, \quad \forall j \in \mathcal{A} \tag{C2}$$
$$x_{ij} \le y_j, \quad \forall i \in \mathcal{S}, \; j \in \mathcal{A} \tag{C2'}$$

### 5.3 Consent Compliance
Matches are strictly restricted to opted-in alumni:
$$x_{ij} \le c_j, \quad \forall i \in \mathcal{S}, \; j \in \mathcal{A} \tag{C3}$$

### 5.4 Minimum Match Quality Threshold
No assignment is permitted if the bilateral compatibility is below threshold $q_{\min}$:
$$x_{ij} = 0, \quad \forall (i, j) \text{ such that } q_{ij} < q_{\min} \tag{C4}$$

### 5.5 Minimum Student Quality Floor
Each student's cumulative quality must satisfy an absolute minimum baseline $\theta$:
$$Q_i = \sum_{j=1}^M q_{ij} x_{ij} \ge \theta, \quad \forall i \in \mathcal{S} \tag{C5}$$

### 5.6 Cross-Departmental Branch Parity
The difference in average student match quality between any two departments cannot exceed tolerance $\delta$:
$$|\overline{Q}_b - \overline{Q}_{b'}| \le \delta, \quad \forall b, b' \in \mathcal{B} \tag{C6}$$
Expressed linearly:
$$\overline{Q}_b - \overline{Q}_{b'} \le \delta, \quad \forall b, b' \in \mathcal{B}$$
$$\overline{Q}_{b'} - \overline{Q}_b \le \delta, \quad \forall b, b' \in \mathcal{B}$$
where:
$$\overline{Q}_b = \frac{1}{|\mathcal{S}_b|} \sum_{i \in \mathcal{S}_b} \sum_{j=1}^M q_{ij} x_{ij}$$

### 5.7 Corporate Employer Diversity Cap
To avoid single-firm clustering, at most $\text{cap}_{\text{comp}}$ of a student's matches may originate from the same employer:
$$\sum_{j \in \mathcal{A}_k} x_{ij} \le \text{cap}_{\text{comp}}, \quad \forall i \in \mathcal{S}, \; \forall k \in \mathcal{K} \tag{C7}$$

### 5.8 Variable Domain Constraints
$$x_{ij} \in \{0, 1\}, \quad \forall i \in \mathcal{S}, \; j \in \mathcal{A}$$
$$y_j \in \{0, 1\}, \quad \forall j \in \mathcal{A}$$
$$Q_{\text{worst}} \ge 0, \quad L_{\max} \in \mathbb{Z}^+$$

---

## 6. Secondary Objective: Alumni Utilization Minimization

In resource-constrained alumni engagement operations, activating an alumnus involves outreach overhead, onboarding materials, and administrative overhead. In the secondary scenario, we formulate an $\epsilon$-constraint model:
$$\min \sum_{j=1}^M y_j$$
Subject to:
$$Z_1(\mathbf{x}) \ge \epsilon_1 \cdot Z_1^*$$
$$Z_2(\mathbf{x}) \ge \epsilon_2$$
$$Z_3(\mathbf{x}) \le \epsilon_3$$
and constraints $(\text{C1}) - (\text{C7})$, where $Z_1^*$ is the unconstrained maximum utilitarian quality and $\epsilon_1 \in [0.80, 0.95]$.

Additionally, the **Shannon Entropy** of employer representation across active assignments is monitored:
$$H(\mathcal{K}) = - \sum_{k \in \mathcal{K}} p_k \log_2(p_k)$$
where $p_k = \frac{\sum_{i=1}^N \sum_{j \in \mathcal{A}_k} x_{ij}}{N \cdot K}$. Higher entropy indicates healthy dispersion across the industrial sector.
