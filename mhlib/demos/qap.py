"""Demo application solving the Quadratic Assignment Problem (QAP)."""

import numpy as np
import random

from ..permutation_solution import PermutationSolution


class QAPInstance:
    """Quadratic Assignment Problem (QAP) instance.

    Attributes
        - n: instance size
        - a: distance matrix
        - b: flow matrix
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
        self.a: np.array
        self.b: np.array
        with open(file_name, "r") as file:
            self.n = int(file.readline())
            if not 2 <= self.n <= 1000:
                raise ValueError(f"Invalid n read from file {file_name}: {self.n}")
            self.a = np.empty([self.n, self.n], dtype=int)
            self.b = np.empty([self.n, self.n], dtype=int)
            file.readline()  # skip empty line
            for i in range(self.n):
                line = file.readline()
                self.a[i] = [int(aij) for aij in line.split()]
            file.readline()
            for i in range(self.n):
                line = file.readline()
                self.b[i] = [int(bij) for bij in line.split()]

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n},\na={self.a},\nb={self.b}\n"


class QAPSolution(PermutationSolution):
    """Solution to a QAP instance.

    Attributes
        - inst: associated QAPInstance
        - x: integer vector representing a permutation
    """

    def __init__(self, inst: QAPInstance, **kwargs):
        """Initializes the solution with 0,...,n-1 if init is set."""
        super().__init__(inst.n, inst=inst, **kwargs)

    def copy(self):
        sol = QAPSolution(self.inst, init=False)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'QAPSolution'):
        super().copy_from(other)

    def calc_objective(self):
        obj = np.einsum('ij,ij', self.inst.a, self.inst.b[self.x][:, self.x])
        return obj

    def construct(self, par, result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        del result
        self.initialize(par)

    def local_improve(self, par, result):
        """Perform local search."""
        del par, result
        self.two_exchange_neighborhood_search(False)

    def shaking(self, par, result):
        """Scheduler method that performs shaking by par random 2-exchange moves."""
        del result
        for i in range(par):
            p1 = random.randrange(0, self.inst.n)
            p2 = random.randrange(0, self.inst.n)
            self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
        self.invalidate()

    def two_exchange_delta_eval(self, p1: int, p2: int, update_obj_val=True, allow_infeasible=False) -> bool:
        if update_obj_val:
            x = self.x
            a = self.inst.a
            b = self.inst.b
            d = np.inner(a[:, p1] - a[:, p2], b[x, x[p2]] - b[x, x[p1]]) - \
                (a[p1, p1] - a[p1, p2]) * (b[x[p1], x[p2]] - b[x[p1], x[p1]]) - \
                (a[p2, p1] - a[p2, p2]) * (b[x[p2], x[p2]] - b[x[p2], x[p1]])
            d += np.inner(a[p1, :] - a[p2, :], b[x[p2], x] - b[x[p1], x]) - \
                (a[p1, p1] - a[p2, p1]) * (b[x[p2], x[p1]] - b[x[p1], x[p1]]) - \
                (a[p1, p2] - a[p2, p2]) * (b[x[p2], x[p2]] - b[x[p1], x[p2]])
            d += (a[p1, p1] - a[p2, p2]) * (b[x[p2], x[p2]] - b[x[p1], x[p1]]) + \
                 (a[p1, p2] - a[p2, p1]) * (b[x[p2], x[p1]] - b[x[p1], x[p2]])
            self.obj_val -= d
        return True


if __name__ == '__main__':
    import os
    from .common import run_gvns_demo
    run_gvns_demo('QAP', QAPInstance, QAPSolution, os.path.join('mhlib', 'demos', 'bur26a.dat'))
