"""Demo application solving the symmetric traveling salesman problem."""

import random
import numpy
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
        distances = numpy.zeros((dimension, dimension))

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
        self.two_exchange_neighborhood_search(True)

    def two_exchange_delta_eval(self, p1: int, p2: int, update_obj_val=True, allow_infeasible=False) -> bool:
        """A 2-exchange move was performed, if feasible update other solution data accordingly, else revert.

        It can be assumed that the solution was in a correct state with a valid objective value before the move.
        The default implementation just calls invalidate() and returns True.

        :param p1: first position
        :param p2: second position
        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
            the update of other data done
        """

        if p1 > p2:
            p1, p2 = p2, p1

        assert(p1 < p2)

        if not update_obj_val:
            # All Permutations are valid, nothing to do here.
            return True

        if p1 == 0 and p2 == len(self.x) - 1:
            # Reversing the whole solution has no effect
            return True

        prev = p1 - 1
        next = p2 + 1 if p2 + 1 < len(self.x) else 0

        p1_city = self.x[p1]
        p2_city = self.x[p2]
        prev_city = self.x[prev]
        next_city = self.x[next]

        # print(f"p1_city: {p1_city}")
        # print(f"p2_city: {p2_city}")
        # print(f"next_city: {next_city}")
        # print(f"prev_city: {prev_city}")

        # Old order
        dist_1a = self.inst.distances[prev_city][p1_city]
        # print(f"distance from {prev_city} to {p1_city} is {dist_1a}")
        dist_1b = self.inst.distances[p2_city][next_city]
        # print(f"distance from {p2_city} to {next_city} is {dist_1b}")
        dist_1 = dist_1a + dist_1b

        # Reversed order
        dist_2a = self.inst.distances[prev_city][p2_city]
        # print(f"distance from {prev_city} to {p2_city} is {dist_2a}")
        dist_2b = self.inst.distances[p1_city][next_city]
        # print(f"distance from {p1_city} to {next_city} is {dist_2b}")
        dist_2 = dist_2a + dist_2b

        # Check if values are correct to begin with
        if self.obj_val_valid:
            probe_val = self.obj_val
            self.invalidate()
            assert(probe_val == self.obj())
        else:
            self.obj()

        self.obj_val -= dist_1
        self.obj_val += dist_2

        numpy.set_printoptions(linewidth=numpy.inf)
        # print(self.x)
        self.x[p1:(p2+1)] = self.x[p1:(p2+1)][::-1]

        assert(self.x[prev] == prev_city)  # prev city did not change
        assert(self.x[next] == next_city)  # next city did not change
        assert(self.x[p1] == p2_city)  # p1 city was reversed
        assert(self.x[p2] == p1_city)  # p2 city was reversed
        # print(self.x)

        return True


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir

    run_optimization('TSP', TSPInstance, TSPSolution, data_dir + "xqf131.tsp")
