"""Demo application solving the maximum (weighted) independent set problem (MISP).

Give an undirected (weighted) graph, find a maximum cardinality subset of nodes where
no pair of nodes is adjacent in the graph.
"""

import numpy as np
from typing import Any, Tuple

from pymhlib.solution import TObj
from pymhlib.subsetvec_solution import SubsetVectorSolution
from pymhlib.scheduler import Result
from pymhlib.demos.graphs import create_or_read_simple_graph


class MISPInstance:
    """Maximum (weighted) independent set problem (MISP) instance.

    Give an undirected (weighted) graph, find a maximum cardinality subset of nodes where
    no pair of nodes is adjacent in the graph.

    Attributes
        - graph: undirected unweighted graph to consider
        - n: number of nodes
        - m number of edges
        - p: prices (weights) of items
    """

    def __init__(self, name: str):
        """Create or read graph with given name.

        So far we only create unweighted MISP instances here.
        """
        self.graph = create_or_read_simple_graph(name)
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()
        self.p = np.ones(self.n, dtype=int)

    def __repr__(self):
        return f"n={self.n} m={self.m}\n"


class MISPSolution(SubsetVectorSolution):
    """Solution to a MISP instance.

    Additional attributes
        - covered: for each node the number of selected neighbor nodes plus one if the node itself is selected
    """

    to_maximize = True

    def __init__(self, inst: MISPInstance):
        super().__init__(range(inst.n), inst=inst)
        self.covered = np.zeros(inst.n, dtype=int)

    @classmethod
    def unselected_elems_in_x(cls) -> bool:
        return False

    def copy(self):
        sol = MISPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'MISPSolution'):
        super().copy_from(other)
        self.covered[:] = other.covered

    def calc_objective(self):
        return np.sum(self.inst.p[self.x[:self.sel]]) if self.sel else 0

    def check(self, unsorted=False):
        super().check(unsorted)
        selected = set(self.x[:self.sel])
        for u, v in self.inst.graph.edges:
            if u in selected and v in selected:
                raise ValueError(f"Invalid solution - adjacent nodes selected: {u}, {v}")
        new_covered = np.zeros(self.inst.n, dtype=int)
        for u in self.x[:self.sel]:
            new_covered[u] += 1
            for v in self.inst.graph.neighbors(u):
                new_covered[v] += 1
        if np.any(self.covered != new_covered):
            raise ValueError(f"Invalid covered values in solution: {self.covered}")

    def clear(self):
        super().clear()
        self.covered.fill(0)

    def construct(self, par: Any, _result: Result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def local_improve(self, _par: Any, result: Result):
        """Scheduler method that performs one iteration of the exchange neighborhood."""
        if not self.two_exchange_random_fill_neighborhood_search(False):
            result.changed = False

    def shaking(self, par: Any, _result: Result):
        """Scheduler method that performs shaking by remove_some(par) and random_fill()."""
        self.remove_some(par)
        self.fill(list(np.nonzero(self.covered == 0)[0]))

    def may_be_extendible(self) -> bool:
        return np.any(self.covered == 0)

    def element_removed_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        u = self.x[self.sel]
        self.covered[u] -= 1
        for v in self.inst.graph.neighbors(u):
            self.covered[v] -= 1
        if update_obj_val:
            self.obj_val -= self.inst.p[u]
        return True

    def element_added_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        u = self.x[self.sel-1]
        if allow_infeasible or not self.covered[u]:
            # accept
            self.covered[u] += 1
            for v in self.inst.graph.neighbors(u):
                self.covered[v] += 1
            if update_obj_val:
                self.obj_val += self.inst.p[u]
            return self.covered[u] == 1
        # revert
        self.sel -= 1
        return False

    def random_move_delta_eval(self) -> Tuple[int, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        raise NotImplementedError

    def apply_neighborhood_move(self, pos: int):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        raise NotImplementedError

    def crossover(self, other: 'MISPSolution') -> 'MISPSolution':
        """Apply subset_crossover."""
        return self.subset_crossover(other)


if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    run_optimization('MISP', MISPInstance, MISPSolution, data_dir + "frb40-19-1.mis")
