"""Demo application solving the traveling salesman problem."""

import numpy as np
import random

from mhlib.permutation_solution import PermutationSolution


class TSPInstance:
    """TSP problem instance.

    Attributes
        - n: number of cities, i.e., size of incidence vector
        - distances: square matrix of intergers representing the distances between two cities;
            zero means there is not connection between the two cities
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 4
        self.distances = [
            [0, 2, 1, 3],
            [2, 0, 3, 2],
            [1, 3, 0, 1],
            [3, 2, 1, 0]
        ]

        # make basic check if instance is meaningful
        if not 1 <= self.n <= 1000000:
            raise ValueError(f"Invalid n: {self.n}")

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n},\ndistances={self.distances!r}\n"


class TSPSolution(PermutationSolution):
    """Solution to a TSP instance.

    Attributes
        - inst: associated TSPInstance
        - x: order cities are visited
    """

    def __init__(self, inst: TSPInstance):
        super().__init__(inst.n, inst=inst)

    def copy(self):
        sol = TSPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        distance = 0

        for i in range(self.inst.n - 1):
            distance = distance + self.inst.distances[self.x[i]][self.x[i + 1]]

        distance = distance + self.inst.distances[self.x[-1]][self.x[0]]

        return distance

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
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

    def shaking(self, par, result):
        """Scheduler method that performs shaking by flipping par random positions."""
        del result
        for i in range(par):
            p = random.randrange(0, self.inst.n)
            self.x[p] = not self.x[p]
        self.invalidate()


if __name__ == '__main__':
    from mhlib.demos.common import run_pbig_demo, data_dir

    run_pbig_demo('TSP', TSPInstance, TSPSolution, data_dir + "advanced.cnf")
