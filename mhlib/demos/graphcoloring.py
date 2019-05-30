"""Demo application solving the graph coloring problem."""

import networkx as nx
import random

from mhlib.subset_solution import VectorSolution

from mhlib.settings import get_settings_parser

parser = get_settings_parser()
parser.add("--mh_gcp_colors", type=int, default=3, help='number of colors available')


class GCInstance:
    """Graph coloring problem instance.
    This instance contains the graph to color as well as a fixed number of available colors.
    Starting from a solution in which all nodes are colored the same, we try to reduce the number of conflicts.

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

    def construct(self, par, _result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def local_improve(self, _par, result):
        """Scheduler method that performs one iteration of a local search following a first improvement strategy.
        The neighborhood used is defined by all solutions that can be created by changing the color
        of a vertex involved in a conflict.
        """

        for p in range(len(self.x)):
            nbhcol = {}
            for col in range(self.inst.colors):
                nbhcol[col] = 0

            for adj in self.inst.graph.adj[p]:
                nbhcol[self.x[adj]] += 1

            oldcol = self.x[p]
            if nbhcol[oldcol] > 0:
                # Violation found

                for newcol in range(self.inst.colors):
                    if nbhcol[newcol] < nbhcol[oldcol]:
                        # Possible improvement found
                        self.x[p] = newcol
                        self.obj_val -= nbhcol[oldcol]
                        self.obj_val += nbhcol[newcol]
                        result.changed = True
                        return

        result.changed = False

    def shaking(self, par, result):
        """Scheduler method that performs shaking by randomly assigning a different color
        to 'par' many random vertices that are involved in a conflict.
        """

        conflicted = []
        result.changed = False

        for u in range(len(self.x)):
            for v in self.inst.graph.adj[u]:
                if self.x[u] == self.x[v]:
                    # Conflict found
                    conflicted.append(u)
                    break

        for _ in range(par):
            if len(conflicted) == 0:
                return

            u = random.choice(conflicted)
            # Pick random color (different from current)
            randcol = random.randint(0, self.inst.colors - 2)

            if randcol >= self.x[u]:
                randcol += 1

            self.x[u] = randcol
            self.invalidate()
            result.changed = True

            # Prevent this vertex from getting changed again
            conflicted.remove(u)

    def initialize(self, _k):
        pass


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir
    from mhlib.settings import settings, get_settings_parser

    settings.mh_maxi = False
    run_optimization('GCP', GCInstance, GCSolution, data_dir + "misp-simple.clq")
