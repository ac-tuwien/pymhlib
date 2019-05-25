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
                    num = int(split_line[0])
                    x = int(split_line[1])
                    y = int(split_line[2])

                    coordinates[num] = (x, y)

        assert (len(coordinates) == dimension)

        # building adjacency matrix
        distances = numpy.zeros((dimension, dimension))

        for i in range(1, dimension):
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

        if not update_obj_val:
            # All Permutations are valid, nothing to do here.
            return True

        # Note: p1 and p2 have already been moved in x

        x_new = self.x
        # new
        a_new = x_new[p1 - 1]
        b_new = x_new[p1]
        c_new = x_new[p1 + 1] if p1 + 1 < len(self.x) else x_new[0]

        d_new = x_new[p2 - 1]
        e_new = x_new[p2]
        f_new = x_new[p2 + 1] if p2 + 1 < len(self.x) else x_new[0]

        # old
        x_old = self.x
        x_old[p1], x_old[p2] = x_old[p2], x_old[p1]  # swap to get old state

        a_old = x_old[p2 - 1]
        b_old = x_old[p2]
        c_old = x_old[p2 + 1] if p2 + 1 < len(self.x) else x_old[0]

        d_old = x_old[p1 - 1]
        e_old = x_old[p1]
        f_old = x_old[p1 + 1] if p1 + 1 < len(self.x) else x_old[0]

        x_old[p1], x_old[p2] = x_old[p2], x_old[p1]  # swap back to new state

        dist = self.inst.distances

        first_new = dist[a_new][b_new] + dist[b_new][c_new]
        first_old = dist[a_old][b_old] + dist[b_old][c_old]

        second_new = dist[d_new][e_new] + dist[e_new][f_new]
        second_old = dist[d_old][e_old] + dist[e_old][f_old]

        old = first_old + second_old
        new = first_new + second_new

        self.obj_val -= old
        self.obj_val += new

        # Check
        # fast = self.obj()
        # self.invalidate()
        # full = self.obj()
        # assert (fast == full)

        return True


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir

    run_optimization('TSP', TSPInstance, TSPSolution, data_dir + "xqf131.tsp")
