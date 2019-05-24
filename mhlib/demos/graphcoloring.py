"""Demo application solving the graph coloring problem."""

import numpy as np
import networkx as nx
import random

from mhlib.subset_solution import VectorSolution

from mhlib.settings import get_settings_parser

parser = get_settings_parser()
parser.add("--mh_gcp_colors", type=int, default=100, help='number of colors available')


class GCInstance:
    """Graph coloring problem instance.

    Attributes
        - n: number of nodes
        - m number of edges
        - colors: number of colors
        - graph: the graph we want to color
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
        self.m = 0
        self.colors = settings.mh_gcp_colors
        self.graph = nx.Graph()
        with open(file_name, "r") as f:
            for line in f:
                flag = line[0]
                if flag == 'p':
                    split_line = line.split(' ')
                    self.n = int(split_line[2])
                    self.m = int(split_line[3])
                    self.graph.add_nodes_from(range(self.n))
                elif flag == 'e':
                    split_line = line.split(' ')
                    u = int(split_line[1]) - 1
                    v = int(split_line[2]) - 1
                    self.graph.add_edge(u, v)
        self.p = np.ones(self.n, dtype=int)  # here we only read unweighted VCP instances

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n} m={self.m} c={self.colors}\n"


class GCSolution(VectorSolution):
    """Solution to a graph coloring problem instance.

    Additional attributes
        - x: for each node the color that is assigned to it
    """

    def __init__(self, inst: GCInstance):
        super().__init__(inst.n, inst=inst)

    def copy(self):
        sol = GCSolution(self.inst)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'GCSolution'):
        super().copy_from(other)

    def calc_objective(self):
        violations = 0

        for u, v in self.inst.graph.edges:
            if self.x[u] == self.x[v]:
                violations += 1

        return violations

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

    def local_improve(self, par, result):
        """Scheduler method that performs one iteration of the exchange neighborhood."""

        result.changed = False

        for i in range(par):
            p = random.randint(0, len(self.x) - 1)

            used = [0] * self.inst.colors

            for adj in self.inst.graph.adj[p]:
                col = self.x[adj]
                used[col] += 1

            violations = used[self.x[p]]

            if min(used) < violations:
                # we can improve by changing to a different color
                minimals = [i for i, x in enumerate(used) if x == min(used)]
                new_col = random.choice(minimals)

                self.x[p] = new_col
                self.obj_val -= violations
                self.obj_val += min(used)
                result.changed = True


    def shaking(self, par, result):
        """Scheduler method that performs shaking by remove_some(par) and random_fill()."""

        for i in range(par):
            p = random.randint(0, len(self.x) - 1)

            col_old = self.x[p]
            change = 0

            col_new = random.randint(0, self.inst.colors - 1)

            for adj in self.inst.graph.adj[p]:
                if col_old == self.x[adj]:
                    change -= 1

                if col_new == self.x[adj]:
                    change += 1

            self.x[p] = col_new
            self.obj_val += change
            result.changed = True


    def initialize(self, k):
        pass


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir
    from mhlib.settings import settings, get_settings_parser

    settings.mh_maxi = False
    run_optimization('GCP', GCInstance, GCSolution, data_dir + "misp-simple.clq")
