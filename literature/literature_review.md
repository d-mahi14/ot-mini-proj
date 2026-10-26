# Literature Review: Multi-Objective Capacitated Assignment and Algorithmic Fairness in University Matching

**Course**: Optimization Techniques (University Group Project)  
**Institution**: Veermata Jijabai Technological Institute (VJTI), Mumbai  

---

## 1. Classical Assignment and Bipartite Matching Foundations

The study of bipartite assignment constitutes one of the bedrock domains of Operations Research (OR).
- **The Linear Sum Assignment Problem (LSAP)**: Formally introduced by Kuhn (1955) through the Hungarian Method, the LSAP solves $\max \sum_{i,j} q_{ij} x_{ij}$ subject to one-to-one matching constraints in $O(N^3)$ polynomial time.
- **Capacitated Assignment Problem (CAP)**: When nodes on one side of the bipartite graph have capacity $C_j > 1$, the problem generalizes to the Hitchcock Transportation Problem and Minimum-Cost Network Flow (Ford & Fulkerson, 1956; Ahuja et al., 1993). Under integer node demands and arc capacities, the constraint matrix is totally unimodular (TUM), ensuring integral extreme points via linear programming.
- **Limitations in Higher Education**: Standard LSAP and network flows optimize exclusively for aggregate utilitarian welfare ($Z_1$). In student mentorship, this yields severe distribution skew: peripheral students receive bottom-tier assignments, and popular mentors are loaded to exhaustion.

---

## 2. Stability vs. Multi-Objective Multi-Attribute Assignment

Matching theory in economics is largely framed around the Gale-Shapley Deferred Acceptance algorithm (Gale & Shapley, 1962; Roth, 1984) for Stable Marriage and Hospital-Resident Allocation:
- **Ordinal vs. Cardinal Valuations**: Stable matching relies on ordinal rankings. In institutional university-alumni matching, bilateral compatibility is inherently multi-attribute and cardinal (combining academic department alignment, domain interest overlap, years of experience, and skillsets into $q_{ij} \in [0, 1]$).
- **Hard Degree Requirements**: In university mentorship, each student requires exactly $K > 1$ distinct mentors, and alumni possess individual capacity caps $C_j$.
- **Fairness Beyond Non-Wastefulness**: While deferred acceptance guarantees pairwise stability, it does not guarantee equitable spread of quality or balanced mentor workload (Abdulkadiroğlu & Sönmez, 2003; Budish, 2011).

---

## 3. Algorithmic Fairness and Social Choice in Optimization

Recent advances in algorithmic fairness categorize allocation criteria into three competing philosophies:
1. **Utilitarianism (Bentham, 1789)**: Maximizes sum of utilities $\sum_i Q_i$. Efficient but indifferent to distributional inequality.
2. **Rawlsian Egalitarianism (Rawls, 1971)**: The Difference Principle dictates maximizing the welfare of the worst-off individual: $\max \min_i Q_i$. Bertsimas, Farias, and Trichakis (2011, 2012) formalized the *Price of Fairness*, quantifying the reduction in total efficiency when enforcing Rawlsian or proportional fairness.
3. **Group Parity & Demographical Fairness**: In technical universities with diverse departments, ensuring parity across branches $\mathcal{B}$ prevents structural marginalization of non-computing departments (e.g., Civil, Textile, Production) compared to high-demand computing fields (Chierichetti et al., 2017).

---

## 4. Multi-Objective Evolutionary Algorithms (MOEAs)

When simultaneous criteria (Total Quality, Rawlsian Floor, Workload Balance) compete under non-convex operational constraints, Multi-Objective Evolutionary Algorithms provide Pareto frontier approximations:
- **NSGA-II (Deb, Pratap, Agarwal, & Meyarivan, 2002)**: Utilizes non-dominated sorting and crowding distance niching to preserve diversity along the trade-off surface.
- **SPEA2 (Zitzler, Laumanns, & Thiele, 2001)**: Incorporates fine-grained fitness assignment based on dominance strength and $k$-th nearest neighbor density estimation.
- **MOEA/D (Zhang & Li, 2007)**: Decomposes a multi-objective problem into scalar subproblems using uniform reference directions (Das & Dennis, 1998) and optimizes them collaboratively.
- **Challenge of Infeasibility**: Direct application of genetic operators (crossover, mutation) on discrete assignment chromosomes frequently produces duplicate matches, capacity overflow, and threshold violations. This necessitates custom repair operators (Michalewicz, 1996; Gottlieb, 2001).

---

## 5. Memetic Algorithms and Hybrid Metaheuristics

Moscato (1989) and Krasnogor & Smith (2005) introduced *Memetic Algorithms*—hybrids of evolutionary search and problem-specific local search operators:
- **Heuristic Seeding**: Seeding genetic populations with heuristic baselines (Greedy, Min-Cost Flow) accelerates convergence toward the optimal hypervolume region.
- **Lamarckian Local Search**: Performing targeted neighborhood exchanges on non-dominated fronts (specifically targeting the minimum quality student $i^*$ and the maximum load alumnus $j^*$) enables escaping local Pareto plateaus.

---

## 6. Synthesis and Project Contribution

This project integrates mathematical programming (Exact MILP with HiGHS) and hybrid metaheuristics (Seeded Memetic NSGA-II) to provide a rigorous, empirical multi-objective optimization framework for VJTI's student-alumni ecosystem.

### Key References
1. Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.
2. Bertsimas, D., Farias, V. F., & Trichakis, N. (2011). The price of fairness. *Operations Research*, 59(1), 17-31.
3. Budish, E. (2011). The combinatorial assignment problem: Approximate competitive equilibrium from equal incomes. *Journal of Political Economy*, 119(6), 1061-1103.
4. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
5. Gale, D., & Shapley, L. S. (1962). College admissions and the stability of marriage. *The American Mathematical Monthly*, 69(1), 9-15.
6. Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1‐2), 83-97.
7. Rawls, J. (1971). *A Theory of Justice*. Harvard University Press.
8. Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712-731.
9. Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the strength Pareto evolutionary algorithm. *TIK-Report*, 103.
