"""Demo application solving the graph coloring problem.

Given a graph and an number of colors, color each node with one color so that
the number of adjacent nodes having the same color is minimized.
"""

import numpy as np
from typing import Any, Tuple

from pymhlib.solution import VectorSolution, TObj
from pymhlib.scheduler import Result
from pymhlib.settings import get_settings_parser, settings
from pymhlib.demos.graphs import create_or_read_simple_graph

parser = get_settings_parser()
parser.add_argument("--mh_gcp_colors", type=int, default=3, help='number of colors available')


class GCInstance:
    """Graph coloring problem instance.

    Given a graph and an number of colors, color each node with one color so that
    the number of adjacent nodes having the same color is minimized.

    Attributes
        - graph: the graph we want to color
        - n: number of nodes
        - m number of edges
        - colors: number of colors
    """

    def __init__(self, name: str):
        """Create or read graph with given name."""
        self.graph = create_or_read_simple_graph(name)
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()
        self.colors = settings.mh_gcp_colors

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n} m={self.m} c={self.colors}\n"


class GCSolution(VectorSolution):
    """Solution to a graph coloring problem instance.

    Attributes
        - x: for each node the color that is assigned to it
    """

    to_maximize = False

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

    def local_improve(self, _par: Any, result: Result):
        """Scheduler method that performs one iteration of a local search following a first improvement strategy.
        The neighborhood used is defined by all solutions that can be created by changing the color
        of a vertex involved in a conflict.
        """
        n = len(self.x)
        order = np.arange(n)
        np.random.shuffle(order)
        for p in order:
            nbh_col = {}
            for col in range(self.inst.colors):
                nbh_col[col] = 0
            for adj in self.inst.graph.adj[p]:
                nbh_col[self.x[adj]] += 1
            old_col = self.x[p]
            if nbh_col[old_col] > 0:
                # violation found
                for new_col in range(self.inst.colors):
                    if nbh_col[new_col] < nbh_col[old_col]:
                        # Possible improvement found
                        self.x[p] = new_col
                        self.obj_val -= nbh_col[old_col]
                        self.obj_val += nbh_col[new_col]
                        result.changed = True
                        return
        result.changed = False

    def shaking(self, par: Any, result: Result):
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

    def random_move_delta_eval(self) -> Tuple[int, int, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        raise NotImplementedError

    def apply_neighborhood_move(self, pos, color: int):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        self.x[pos] = color

    def crossover(self, other: 'GCSolution') -> 'GCSolution':
        """ Preform uniform crossover."""
        return self.uniform_crossover(other)


if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import settings, get_settings_parser

    settings.mh_maxi = False
    run_optimization('Graph Coloring', GCInstance, GCSolution, data_dir + "fpsol2.i.1.col")
