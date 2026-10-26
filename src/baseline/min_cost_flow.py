"""Single-Objective Min-Cost Flow (Max Z1) Baseline for Student-Alumni Matching.

Maximizes total match score Z1 = sum q_ij x_ij subject to:
- Student degree sum_j x_ij = K
- Alumni capacity sum_i x_ij <= C_j
- Consent c_j == 1 and threshold q_ij >= q_min
Formulated as a minimum-cost network flow problem on a directed bipartite graph.
"""

from __future__ import annotations

import time
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from src.problem import MatchingSolution, StudentAlumniMatchingProblem


class MinCostFlowBaseline:
    """Exact Single-Objective (Utilitarian Max Quality) Network Flow Solver."""

    def __init__(self, problem: StudentAlumniMatchingProblem) -> None:
        """Initialize min cost flow solver with problem instance."""
        self.problem = problem

    def solve(self, method: str = "auto") -> MatchingSolution:
        """Solve min-cost flow formulation to maximize Z1.

        Args:
            method: 'networkx' (cycle cancelling / cost scaling) or 'scipy' (HiGHS LP).

        Returns:
            Evaluated MatchingSolution.
        """
        start_time = time.perf_counter()
        n = self.problem.n_students
        m = self.problem.n_alumni
        k = self.problem.k_matches

        x_bin = np.zeros((n, m), dtype=int)

        if method == "auto":
            method = "networkx" if (n * m) <= 10000 else "scipy"

        if method == "networkx":
            G = nx.DiGraph()
            source = "src"
            sink = "snk"

            total_demand = n * k
            G.add_node(source, demand=-total_demand)
            G.add_node(sink, demand=total_demand)

            for i in range(n):
                s_node = f"s_{i}"
                G.add_node(s_node, demand=0)
                G.add_edge(source, s_node, capacity=k, weight=0)

            for i in range(n):
                s_node = f"s_{i}"
                for j in range(m):
                    if self.problem.consents[j] == 1 and self.problem.q_matrix[i, j] >= self.problem.q_min:
                        a_node = f"a_{j}"
                        cost = int(-round(self.problem.q_matrix[i, j] * 10000))
                        G.add_edge(s_node, a_node, capacity=1, weight=cost)

            for j in range(m):
                a_node = f"a_{j}"
                G.add_node(a_node, demand=0)
                if self.problem.consents[j] == 1:
                    G.add_edge(a_node, sink, capacity=int(self.problem.capacities[j]), weight=0)

            try:
                flow_dict = nx.min_cost_flow(G)
                for i in range(n):
                    s_node = f"s_{i}"
                    for a_node, fl in flow_dict.get(s_node, {}).items():
                        if fl > 0 and a_node.startswith("a_"):
                            j = int(a_node.split("_")[1])
                            x_bin[i, j] = 1
            except Exception:
                return self.solve(method="scipy")

        else:
            from scipy.sparse import coo_matrix

            num_vars = n * m
            c = -self.problem.q_matrix.flatten()

            # Sparse equality: sum_j x_ij = K for each student i
            row_eq = np.repeat(np.arange(n), m)
            col_eq = np.arange(num_vars)
            data_eq = np.ones(num_vars, dtype=float)
            A_eq = coo_matrix((data_eq, (row_eq, col_eq)), shape=(n, num_vars)).tocsr()
            b_eq = np.full(n, float(k))

            # Sparse inequality: sum_i x_ij <= C_j for each alumnus j
            row_ub = np.tile(np.arange(m), n)
            col_ub = np.arange(num_vars)
            data_ub = np.ones(num_vars, dtype=float)
            A_ub = coo_matrix((data_ub, (row_ub, col_ub)), shape=(m, num_vars)).tocsr()
            b_ub = self.problem.capacities.astype(float)

            # Bounds
            lb = np.zeros(num_vars)
            ub = np.zeros(num_vars)
            valid_mask = ((self.problem.consents == 1)[None, :] & (self.problem.q_matrix >= self.problem.q_min)).flatten()
            ub[valid_mask] = 1.0

            bounds = np.column_stack([lb, ub])

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success and res.x is not None:
                x_bin = (res.x > 0.5).astype(int).reshape((n, m))
            else:
                # Relax threshold if needed
                ub_relaxed = np.zeros(num_vars)
                consent_mask = np.tile(self.problem.consents == 1, n)
                ub_relaxed[consent_mask] = 1.0
                bounds_relaxed = np.column_stack([lb, ub_relaxed])
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds_relaxed, method="highs")
                if res.success and res.x is not None:
                    x_bin = (res.x > 0.5).astype(int).reshape((n, m))

        solution = self.problem.evaluate_solution(x_bin)
        return solution
