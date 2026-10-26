"""Exact Mixed-Integer Linear Programming (MILP) Solvers for Matching.

Implements exact mathematical programming using SciPy HiGHS:
1. Weighted-Sum MILP: Solves scalarized combinations of (Z1, Z2, Z3).
2. Epsilon-Constraint MILP: Traces the exact Pareto frontier by constraining (Z2, Z3) while optimizing Z1.
3. Secondary Epsilon-Constraint Scenario: Minimizes alumni activation sum_j y_j subject to performance bounds.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, milp

from src.problem import MatchingSolution, StudentAlumniMatchingProblem


class ExactMILP:
    """Exact MILP solver implementing Weighted-Sum, Epsilon-Constraint, and Alumni Minimization."""

    def __init__(self, problem: StudentAlumniMatchingProblem) -> None:
        """Initialize exact MILP solver with problem instance."""
        self.problem = problem
        self.n = problem.n_students
        self.m = problem.n_alumni
        self.k = problem.k_matches
        self.q = problem.q_matrix
        self.capacities = problem.capacities
        self.consents = problem.consents
        self.q_min = problem.q_min
        self.theta = problem.theta
        self.delta = problem.delta
        self.companies = problem.alumni_companies
        self.max_company_share = problem.max_company_share

        # Variable indexing:
        # x_{ij}: indices 0 to n*m - 1
        # y_j: indices n*m to n*m + m - 1
        # Q_worst: index n*m + m
        # L_max: index n*m + m + 1
        self.idx_x = 0
        self.idx_y = self.n * self.m
        self.idx_q_worst = self.idx_y + self.m
        self.idx_l_max = self.idx_q_worst + 1
        self.n_vars = self.idx_l_max + 1

        # Variable types: 1 = integer/binary, 0 = continuous
        self.integrality = np.zeros(self.n_vars)
        self.integrality[: self.idx_y + self.m] = 1  # x and y are binary
        self.integrality[self.idx_q_worst] = 0  # continuous
        self.integrality[self.idx_l_max] = 1  # integer max load

        # Variable bounds (lb, ub)
        self.lb = np.zeros(self.n_vars)
        self.ub = np.ones(self.n_vars)
        # Q_worst bounds: 0 to k * 1.0
        self.lb[self.idx_q_worst] = 0.0
        self.ub[self.idx_q_worst] = float(self.k)
        # L_max bounds: 1 to n * k
        self.lb[self.idx_l_max] = 1.0
        self.ub[self.idx_l_max] = float(self.n * self.k)

        # Enforce consent and q_min bounds directly on x_ij variables
        for i in range(self.n):
            for j in range(self.m):
                var_idx = i * self.m + j
                if self.consents[j] != 1 or self.q[i, j] < self.q_min:
                    self.ub[var_idx] = 0.0

        self._base_constraints: Optional[List[LinearConstraint]] = None

    def _build_base_constraints(self) -> List[LinearConstraint]:
        """Build core operational constraints (C1-C7 + auxiliary definitions)."""
        if self._base_constraints is not None:
            return self._base_constraints

        constraints: List[LinearConstraint] = []

        # (C1) Degree constraints: sum_j x_{ij} = K for each student i
        A_deg = np.zeros((self.n, self.n_vars))
        for i in range(self.n):
            for j in range(self.m):
                A_deg[i, i * self.m + j] = 1.0
        constraints.append(LinearConstraint(A_deg, np.full(self.n, self.k), np.full(self.n, self.k)))

        # (C2) Capacity and activation constraints: sum_i x_{ij} <= C_j * y_j
        # sum_i x_{ij} - C_j * y_j <= 0
        A_cap = np.zeros((self.m, self.n_vars))
        for j in range(self.m):
            for i in range(self.n):
                A_cap[j, i * self.m + j] = 1.0
            A_cap[j, self.idx_y + j] = -float(self.capacities[j])
        constraints.append(LinearConstraint(A_cap, np.full(self.m, -np.inf), np.zeros(self.m)))

        # Linking: x_{ij} <= y_j
        # x_{ij} - y_j <= 0 for all (i, j)
        A_link = np.zeros((self.n * self.m, self.n_vars))
        row = 0
        for i in range(self.n):
            for j in range(self.m):
                A_link[row, i * self.m + j] = 1.0
                A_link[row, self.idx_y + j] = -1.0
                row += 1
        constraints.append(LinearConstraint(A_link, np.full(self.n * self.m, -np.inf), np.zeros(self.n * self.m)))

        # (C5) Quality floor: sum_j q_{ij} x_{ij} >= theta for each student i
        A_floor = np.zeros((self.n, self.n_vars))
        for i in range(self.n):
            for j in range(self.m):
                A_floor[i, i * self.m + j] = self.q[i, j]
        constraints.append(LinearConstraint(A_floor, np.full(self.n, self.theta), np.full(self.n, np.inf)))

        # Auxiliary 1: Q_worst <= sum_j q_{ij} x_{ij}
        # Q_worst - sum_j q_{ij} x_{ij} <= 0
        A_q_worst = np.zeros((self.n, self.n_vars))
        for i in range(self.n):
            A_q_worst[i, self.idx_q_worst] = 1.0
            for j in range(self.m):
                A_q_worst[i, i * self.m + j] = -self.q[i, j]
        constraints.append(LinearConstraint(A_q_worst, np.full(self.n, -np.inf), np.zeros(self.n)))

        # Auxiliary 2: L_max >= sum_i x_{ij}
        # L_max - sum_i x_{ij} >= 0
        A_l_max = np.zeros((self.m, self.n_vars))
        for j in range(self.m):
            A_l_max[j, self.idx_l_max] = 1.0
            for i in range(self.n):
                A_l_max[j, i * self.m + j] = -1.0
        constraints.append(LinearConstraint(A_l_max, np.zeros(self.m), np.full(self.m, np.inf)))

        # (C6) Branch parity constraints: |avgQ_b - avgQ_b'| <= delta
        # avgQ_b = (1 / |S_b|) sum_{i in S_b} sum_j q_ij x_ij
        branches = self.problem.unique_branches
        b_rows: List[np.ndarray] = []
        b_lb: List[float] = []
        b_ub: List[float] = []

        for idx_b1 in range(len(branches)):
            b1 = branches[idx_b1]
            s1 = self.problem.branch_to_student_indices[b1]
            if len(s1) == 0:
                continue
            for idx_b2 in range(idx_b1 + 1, len(branches)):
                b2 = branches[idx_b2]
                s2 = self.problem.branch_to_student_indices[b2]
                if len(s2) == 0:
                    continue

                row_vec = np.zeros(self.n_vars)
                # avgQ_b1
                w1 = 1.0 / len(s1)
                for i in s1:
                    for j in range(self.m):
                        row_vec[i * self.m + j] += w1 * self.q[i, j]
                # - avgQ_b2
                w2 = 1.0 / len(s2)
                for i in s2:
                    for j in range(self.m):
                        row_vec[i * self.m + j] -= w2 * self.q[i, j]

                b_rows.append(row_vec)
                b_lb.append(-self.delta)
                b_ub.append(self.delta)

        if b_rows:
            constraints.append(LinearConstraint(np.array(b_rows), np.array(b_lb), np.array(b_ub)))

        # (C7) Company Diversity: max_company_share matches per student
        comp_rows: List[np.ndarray] = []
        unique_companies = list(set(self.companies))
        for i in range(self.n):
            for comp in unique_companies:
                alumni_with_comp = [j for j in range(self.m) if self.companies[j] == comp]
                if len(alumni_with_comp) > self.max_company_share:
                    r_vec = np.zeros(self.n_vars)
                    for j in alumni_with_comp:
                        r_vec[i * self.m + j] = 1.0
                    comp_rows.append(r_vec)

        if comp_rows:
            constraints.append(
                LinearConstraint(
                    np.array(comp_rows),
                    np.full(len(comp_rows), -np.inf),
                    np.full(len(comp_rows), float(self.max_company_share)),
                )
            )

        self._base_constraints = constraints
        return constraints

    def solve_weighted_sum(
        self,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 0.5,
        time_limit_sec: float = 60.0,
    ) -> Optional[MatchingSolution]:
        """Solve weighted-sum scalarization: min w1*(-Z1) + w2*(-Z2) + w3*(Z3)."""
        c = np.zeros(self.n_vars)
        # - w1 * sum_ij q_ij x_ij
        for i in range(self.n):
            for j in range(self.m):
                c[i * self.m + j] = -w1 * self.q[i, j]
        # - w2 * Q_worst
        c[self.idx_q_worst] = -w2
        # + w3 * L_max
        c[self.idx_l_max] = w3

        constraints = self._build_base_constraints()
        res = milp(
            c=c,
            integrality=self.integrality,
            bounds=(self.lb, self.ub),
            constraints=constraints,
            options={"time_limit": time_limit_sec},
        )

        if res.success and res.x is not None:
            x_vals = np.round(res.x[: self.n * self.m]).astype(int)
            x_bin = x_vals.reshape((self.n, self.m))
            return self.problem.evaluate_solution(x_bin)
        return None

    def solve_epsilon_constraint(
        self,
        eps_worst_quality: float,
        eps_max_load: float,
        time_limit_sec: float = 60.0,
    ) -> Optional[MatchingSolution]:
        """Solve epsilon-constraint model: Maximize Z1 subject to Z2 >= eps2 and Z3 <= eps3."""
        # Objective: minimize - Z1
        c = np.zeros(self.n_vars)
        for i in range(self.n):
            for j in range(self.m):
                c[i * self.m + j] = -self.q[i, j]

        base_constraints = list(self._build_base_constraints())

        # Additional epsilon constraints:
        # Q_worst >= eps_worst_quality
        A_eps_q = np.zeros((1, self.n_vars))
        A_eps_q[0, self.idx_q_worst] = 1.0
        base_constraints.append(LinearConstraint(A_eps_q, np.array([eps_worst_quality]), np.array([np.inf])))

        # L_max <= eps_max_load
        A_eps_l = np.zeros((1, self.n_vars))
        A_eps_l[0, self.idx_l_max] = 1.0
        base_constraints.append(LinearConstraint(A_eps_l, np.array([-np.inf]), np.array([eps_max_load])))

        res = milp(
            c=c,
            integrality=self.integrality,
            bounds=(self.lb, self.ub),
            constraints=base_constraints,
            options={"time_limit": time_limit_sec},
        )

        if res.success and res.x is not None:
            x_vals = np.round(res.x[: self.n * self.m]).astype(int)
            x_bin = x_vals.reshape((self.n, self.m))
            return self.problem.evaluate_solution(x_bin)
        return None

    def solve_minimize_alumni_usage(
        self,
        min_quality_ratio: float = 0.90,
        max_load_bound: Optional[int] = None,
        time_limit_sec: float = 60.0,
    ) -> Optional[MatchingSolution]:
        """Secondary Scenario: Minimize active alumni sum_j y_j subject to performance bounds."""
        # First obtain unconstrained max quality
        best_quality_sol = self.solve_weighted_sum(w1=1.0, w2=0.0, w3=0.0, time_limit_sec=time_limit_sec)
        if best_quality_sol is None:
            min_target_z1 = self.n * self.theta
        else:
            min_target_z1 = best_quality_sol.z1_total_quality * min_quality_ratio

        # Objective: minimize sum_j y_j
        c = np.zeros(self.n_vars)
        for j in range(self.m):
            c[self.idx_y + j] = 1.0

        constraints = list(self._build_base_constraints())

        # Bound on total quality Z1 >= min_target_z1
        A_z1 = np.zeros((1, self.n_vars))
        for i in range(self.n):
            for j in range(self.m):
                A_z1[0, i * self.m + j] = self.q[i, j]
        constraints.append(LinearConstraint(A_z1, np.array([min_target_z1]), np.array([np.inf])))

        # Optional bound on maximum load
        if max_load_bound is not None:
            A_l = np.zeros((1, self.n_vars))
            A_l[0, self.idx_l_max] = 1.0
            constraints.append(LinearConstraint(A_l, np.array([-np.inf]), np.array([float(max_load_bound)])))

        res = milp(
            c=c,
            integrality=self.integrality,
            bounds=(self.lb, self.ub),
            constraints=constraints,
            options={"time_limit": time_limit_sec},
        )

        if res.success and res.x is not None:
            x_vals = np.round(res.x[: self.n * self.m]).astype(int)
            x_bin = x_vals.reshape((self.n, self.m))
            return self.problem.evaluate_solution(x_bin)
        return None

    def trace_pareto_front(
        self,
        grid_steps: int = 5,
        time_limit_per_point: float = 30.0,
    ) -> List[MatchingSolution]:
        """Trace the Pareto frontier on small instances using weighted-sum and epsilon-constraint sweeps."""
        solutions: List[MatchingSolution] = []

        # 1. Anchor solutions
        anchors = [
            (1.0, 0.05, 0.0),   # Quality focus
            (0.1, 1.0, 0.0),    # Fairness focus (max Q_worst)
            (0.05, 0.05, 1.0),  # Load balance focus (min L_max)
        ]
        for w1, w2, w3 in anchors:
            sol = self.solve_weighted_sum(w1, w2, w3, time_limit_sec=time_limit_per_point)
            if sol is not None:
                solutions.append(sol)

        # 2. Weighted-sum simplex grid sweep
        for i in range(grid_steps + 1):
            for j in range(grid_steps + 1 - i):
                k = grid_steps - i - j
                w1 = i / float(grid_steps)
                w2 = j / float(grid_steps)
                w3 = k / float(grid_steps)
                if w1 == 0 and w2 == 0 and w3 == 0:
                    continue
                sol = self.solve_weighted_sum(w1, w2, w3, time_limit_sec=time_limit_per_point)
                if sol is not None:
                    solutions.append(sol)

        # Filter duplicates based on objective vector
        unique_solutions: List[MatchingSolution] = []
        seen = set()
        for s in solutions:
            key = (round(s.z1_total_quality, 2), round(s.z2_worst_quality, 2), int(s.z3_max_load))
            if key not in seen:
                seen.add(key)
                unique_solutions.append(s)

        return unique_solutions
