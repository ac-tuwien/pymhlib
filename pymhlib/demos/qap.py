"""Demo application solving the Quadratic Assignment Problem (QAP).

There are a set of n facilities and a set of n locations. For each pair of locations,
a distance is specified and for each pair of facilities a flow is given
(e.g., the amount of supplies transported between the two facilities).
The task is to assign all facilities to different locations with the goal of minimizing
the sum of the distances multiplied by the corresponding flows.
"""

import numpy as np
import random
from typing import Any, Tuple

from pymhlib.permutation_solution import PermutationSolution
from pymhlib.scheduler import Result
from pymhlib.solution import TObj


class QAPInstance:
    """Quadratic Assignment Problem (QAP) instance.

    There are a set of n facilities and a set of n locations. For each pair of locations,
    a distance is specified and for each pair of facilities a flow is given
    (e.g., the amount of supplies transported between the two facilities).
    The task is to assign all facilities to different locations with the goal of minimizing
    the sum of the distances multiplied by the corresponding flows.

    Attributes
        - n: instance size
        - a: distance matrix
        - b: flow matrix
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
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
        return f"n={self.n}\n"  #,\na={self.a},\nb={self.b}\n"


class QAPSolution(PermutationSolution):
    """Solution to a QAP instance.

    Attributes
        - inst: associated QAPInstance
        - x: integer vector representing a permutation
    """

    to_maximize = False

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

    def construct(self, par: Any, _result: Result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def local_improve(self, _par: Any, _result: Result):
        """Perform one major iteration of local search in the 2-exchange neighborhood."""
        self.two_exchange_neighborhood_search(False)

    def shaking(self, par: Any, _result: Result):
        """Scheduler method that performs shaking by par random 2-exchange moves."""
        for i in range(par):
            p1 = random.randrange(0, self.inst.n)
            p2 = random.randrange(0, self.inst.n)
            self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
        self.invalidate()

    def two_exchange_move_delta_eval(self, p1: int, p2: int) -> TObj:
        """Return delta value in objective when exchanging positions p1 and p2 in self.x.

        The solution is not changed.
        """
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
        return d

    def random_move_delta_eval(self) -> Tuple[Any, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        return self.random_two_exchange_move_delta_eval()

    def apply_neighborhood_move(self, move):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        p1, p2 = move
        x = self.x
        x[p1], x[p2] = x[p2], x[p1]

    def crossover(self, other: 'QAPSolution') -> 'QAPSolution':
        """Perform cycle crossover."""
        # return self.partially_mapped_crossover(other)
        return self.cycle_crossover(other)


if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    run_optimization('QAP', QAPInstance, QAPSolution, data_dir+'bur26a.dat')
