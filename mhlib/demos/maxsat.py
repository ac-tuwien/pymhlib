"""Demo application solving the MAXSAT problem."""

import numpy as np
import random

from mhlib.solution import BoolVectorSolution


class MAXSATInstance:
    """MAXSAT problem instance.

    Attributes
        - n: number of variables, i.e., size of incidence vector
        - m: number of clauses
        - clauses: list of clauses, where each clause is represented by a list of integers;
            a positive integer v refers to the v-th variable, while a negative integer v refers
            to the negated form of the v-th variable; note that variable indices start with 1 (-1)
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
        self.m = 0
        self.clauses = list()

        with open(file_name, "r") as file:
            for line in file:
                if line.startswith("c"):
                    # ignore comments
                    continue
                fields = line.split()
                if len(fields) == 4 and fields[0] == "p" and fields[1] == "cnf":
                    try:
                        self.n = int(fields[2])
                        self.m = int(fields[3])
                    except ValueError:
                        raise ValueError(f"Invalid values in line 'p cnf': {line}")
                elif len(fields) >= 1:
                    # read clause
                    if not fields[-1].startswith("0"):
                        raise ValueError(f"Last field in clause line must be 0, but is not: {line}, {fields[-1]!r}")
                    try:
                        self.clauses.append([int(s) for s in fields[:-1]])
                    except ValueError:
                        raise ValueError(f"Invalid clause: {line}")

        # make basic check if instance is meaningful
        if not 1 <= self.n <= 1000000:
            raise ValueError(f"Invalid n: {self.n}")
        if not 1 <= self.m <= 1000000:
            raise ValueError(f"Invalid m: {self.m}")
        if len(self.clauses) != self.m:
            raise ValueError(f"Number of clauses should be {self.m}, but {len(self.clauses)} read")

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n},\nclauses={self.clauses!r}\n"


class MAXSATSolution(BoolVectorSolution):
    """Solution to a MAXSAT instance.

    Attributes
        - inst: associated MAXSATInstance
        - x: binary incidence vector
    """

    def __init__(self, inst: MAXSATInstance):
        super().__init__(inst.n, inst=inst)

    def copy(self):
        sol = MAXSATSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        fulfilled_clauses = 0
        for clause in self.inst.clauses:
            for v in clause:
                if self.x[abs(v)-1] == (1 if v > 0 else 0):
                    fulfilled_clauses += 1
                    break
        return fulfilled_clauses

    def check(self):
        """Check if valid solution.

        Raises ValueError if problem detected.
        """
        if len(self.x) != self.inst.n:
            raise ValueError("Invalid length of solution")
        super().check()

    def construct(self, par, result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        del result
        self.initialize(par)

    def local_improve(self, par, result):
        """Perform k_flip_local_search."""
        del result
        self.k_flip_local_search(par, False)

    def shaking(self, par, result):
        """Scheduler method that performs shaking by flipping par random positions."""
        del result
        for i in range(par):
            p = random.randrange(0, self.inst.n)
            self.x[p] = not self.x[p]
        self.invalidate()

    def k_flip_local_search(self, k: int, best_improvement) -> bool:
        """Perform one major iteration of a k-flip local search.

        If best_improvement is set, the neighborhood is completely searched and a best neighbor is kept;
        otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
        better solution.

        Returns True if an improved solution has been found.
        """
        x = self.x
        assert 0 < k <= len(x)
        better_found = False
        best_sol = self.copy()
        p = np.full(k, -1)  # flipped positions
        # initialize
        i = 0  # current index in p to consider
        while i >= 0:
            # evaluate solution
            if i == k:
                self.invalidate()
                if self.is_better(best_sol):
                    if not best_improvement:
                        return True
                    best_sol.copy_from(self)
                    better_found = True
                i -= 1  # backtrack
            else:
                if p[i] == -1:
                    # this index has not yet been placed
                    p[i] = (p[i-1] if i > 0 else -1) + 1
                    x[p[i]] = not x[p[i]]
                    i += 1  # continue with next position (if any)
                elif p[i] < len(x) - (k - i):
                    # further positions to explore with this index
                    x[p[i]] = not x[p[i]]
                    p[i] += 1
                    x[p[i]] = x[p[i]]
                    i += 1
                else:
                    # we are at the last position with the i-th index, backtrack
                    x[p[i]] = not x[p[i]]
                    p[i] = -1  # unset position
                    i -= 1
        if better_found:
            self.copy_from(best_sol)
            self.invalidate()
            return better_found


if __name__ == '__main__':
    from mhlib.demos.common import run_gvns_demo, data_dir
    run_gvns_demo('MAXSAT', MAXSATInstance, MAXSATSolution, data_dir+"advanced.cnf")
