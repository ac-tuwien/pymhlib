"""Demo application solving the symmetric traveling salesman problem."""

import random
import numpy as np
import math

from mhlib.permutation_solution import PermutationSolution


class TSPInstance:
    """TSP problem instance.

    Attributes
        - n: number of cities, i.e., size of incidence vector
        - distances: square matrix of integers representing the distances between two cities;
            zero means there is not connection between the two cities
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        coordinates = {}
        dimension = None

        with open(file_name, "r") as f:
            for line in f:
                if line.startswith("NAME") or line.startswith("COMMENT") or line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    break
                elif line.startswith("TYPE"):
                    assert (line.split()[-1] == "TSP")
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    assert (line.split()[-1] == "EUC_2D")
                elif line.startswith("DIMENSION"):
                    dimension = int(line.split()[-1])
                else:
                    split_line = line.split()
                    num = int(split_line[0]) - 1  # starts at 1
                    x = int(split_line[1])
                    y = int(split_line[2])

                    coordinates[num] = (x, y)

        assert (len(coordinates) == dimension)

        # building adjacency matrix
        distances = np.zeros((dimension, dimension))

        for i in range(0, dimension):
            for j in range(i + 1, dimension):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                dist = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
                distances[i][j] = distances[j][i] = int(dist)

        self.distances = distances
        self.n = dimension

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
        - x: order in which cities are visited
    """

    def __init__(self, inst: TSPInstance):
        super().__init__(inst.n, inst=inst)
        self.obj_val_valid = False

    def copy(self):
        sol = TSPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        distance = 0

        for i in range(self.inst.n - 1):
            distance += self.inst.distances[self.x[i]][self.x[i + 1]]

        distance += self.inst.distances[self.x[-1]][self.x[0]]

        return distance

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
        """
        if len(self.x) != self.inst.n:
            raise ValueError("Invalid length of solution")
        super().check()

    def construct(self, par, _result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def shaking(self, _par, _result):
        """Scheduler method that performs shaking by flipping par random positions."""
        a = random.randint(0, self.inst.n - 1)
        b = random.randint(0, self.inst.n - 1)

        self.x[a], self.x[b] = self.x[b], self.x[a]
        self.invalidate()

    def local_improve(self, _par, _result):
        self.two_opt_neighborhood_search(True)

    def two_opt_neighborhood_search(self, best_improvement) -> bool:
        """Perform the systematic search of the 2-opt neighborhood, in which two edges are exchanged.

        The neighborhood is searched in a randomized ordering.
        Note that frequently, a more problem-specific neighborhood search with delta-evaluation is
        much more efficient!

        :param best_improvement:  if set, the neighborhood is completely searched and a best neighbor is kept;
            otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
            better solution.

        :return: True if an improved solution has been found
        """
        n = self.inst.n
        best_obj = orig_obj = self.obj()
        best_p1 = None
        best_p2 = None
        order = np.arange(n)
        np.random.shuffle(order)
        for idx, p1 in enumerate(order[:n - 1]):
            for p2 in order[idx + 1:]:

                if p1 > p2:
                    p1, p2 = p2, p1

                self.x[p1:(p2+1)] = self.x[p1:(p2+1)][::-1]
                if self.two_opt_delta_eval(p1, p2):
                    if self.is_better_obj(self.obj(), best_obj):
                        if not best_improvement:
                            return True
                        best_obj = self.obj()
                        best_p1 = p1
                        best_p2 = p2
                    self.x[p1:(p2+1)] = self.x[p1:(p2+1)][::-1]
                    self.obj_val = orig_obj
                    assert self.two_opt_delta_eval(p1, p2, False)
        if best_p1:
            self.x[best_p1:(best_p2+1)] = self.x[best_p1:(best_p2+1)][::-1]
            self.obj_val = best_obj
            return True
        self.obj_val = orig_obj
        return False

    def two_opt_delta_eval(self, p1: int, p2: int, update_obj_val=True, _allow_infeasible=False) -> bool:
        assert(p1 < p2)

        if not update_obj_val:
            # All Permutations are valid, nothing to do here.
            return True

        if p1 == 0 and p2 == len(self.x) - 1:
            # Reversing the whole solution has no effect
            return True

        # The solution looks as follows:
        # .... prev, p1, ... p2, nxt ...
        prev = p1 - 1
        nxt = p2 + 1 if p2 + 1 < len(self.x) else 0

        p1_city = self.x[p1]
        p2_city = self.x[p2]
        prev_city = self.x[prev]
        next_city = self.x[nxt]

        # Current order
        dist_now_a = self.inst.distances[prev_city][p1_city]
        dist_now_b = self.inst.distances[p2_city][next_city]
        dist_now = dist_now_a + dist_now_b

        # Reversed order
        dist_rev_a = self.inst.distances[prev_city][p2_city]
        dist_rev_b = self.inst.distances[p1_city][next_city]
        dist_rev = dist_rev_a + dist_rev_b

        # Update objective value
        self.obj_val += dist_now
        self.obj_val -= dist_rev

        if __debug__:
            # Check delta evaluation
            probe = self.obj_val
            self.invalidate()
            assert(probe == self.obj())

        return True


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir

    run_optimization('TSP', TSPInstance, TSPSolution, data_dir + "xqf131.tsp")
