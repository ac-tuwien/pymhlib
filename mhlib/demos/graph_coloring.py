"""Demo application solving the graph coloring problem.

Given a graph and an number of colors, color each node with one color so that
the number of adjacent nodes having the same color is minimized.
"""

import networkx as nx
import numpy as np

from mhlib.subset_solution import VectorSolution

from mhlib.settings import get_settings_parser

parser = get_settings_parser()
parser.add("--mh_gcp_colors", type=int, default=3, help='number of colors available')


class GCInstance:
    """Graph coloring problem instance.

    Given a graph and an number of colors, color each node with one color so that
    the number of adjacent nodes having the same color is minimized.

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

    Attributes
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
            nbh_col = {}
            for col in range(self.inst.colors):
                nbh_col[col] = 0

            for adj in self.inst.graph.adj[p]:
                nbh_col[self.x[adj]] += 1

            old_col = self.x[p]
            if nbh_col[old_col] > 0:
                # Violation found

                for new_col in range(self.inst.colors):
                    if nbh_col[new_col] < nbh_col[old_col]:
                        # Possible improvement found
                        self.x[p] = new_col
                        self.obj_val -= nbh_col[old_col]
                        self.obj_val += nbh_col[new_col]
                        result.changed = True
                        return

        result.changed = False

    def shaking(self, par, result):
        """Scheduler method that performs shaking by randomly assigning a different color
        to 'par' many random vertices that are involved in conflicts.
        """

        under_conflict = []
        result.changed = False

        for u in range(len(self.x)):
            for v in self.inst.graph.adj[u]:
                if self.x[u] == self.x[v]:
                    # Conflict found
                    under_conflict.append(u)
                    break

        for _ in range(par):
            if len(under_conflict) == 0:
                return

            u = np.random.choice(under_conflict)
            # Pick random color (different from current)
            rand_col = np.random.randint(0, self.inst.colors - 1)

            if rand_col >= self.x[u]:
                rand_col += 1

            self.x[u] = rand_col
            self.invalidate()
            result.changed = True

            # Prevent this vertex from getting changed again
            under_conflict.remove(u)

    def initialize(self, _k):
        """Initialize solution vector with random colors."""
        self.x = np.random.randint(self.inst.colors, size=len(self.x))
        self.invalidate()


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir
    from mhlib.settings import settings, get_settings_parser

    settings.mh_maxi = False
    run_optimization('Graph Coloring', GCInstance, GCSolution, data_dir + "fpsol2.i.1.col")
