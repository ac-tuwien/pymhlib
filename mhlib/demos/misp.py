"""Demo application solving the maximum (weighted) independent set problem (MISP)."""

import numpy as np
import networkx as nx

from mhlib.subset_solution import SubsetSolution


class MISPInstance:
    """MISP problem instance.

    Attributes
        - n: number of nodes
        - m number of edges
        - p: prices (weights) of items
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
        self.m = 0
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
        self.p = np.ones(self.n, dtype=int)  # here we only read unweighted MISP instances

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n} m={self.m}\n"


class MISPSolution(SubsetSolution):
    """Solution to a MISP instance.

    Additional attributes
        - covered: for each node the number of selected neighbor nodes plus one if the node itself is selected
    """

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

    def construct(self, par, result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        del result
        self.initialize(par)

    def local_improve(self, par, result):
        """Scheduler method that performs one iteration of the exchange neighborhood."""
        del par
        if not self.two_exchange_random_fill_neighborhood_search(False):
            result.changed = False

    def shaking(self, par, result):
        """Scheduler method that performs shaking by remove_some(par) and random_fill()."""
        del result
        self.check()  # TODO finally remove
        self.remove_some(par)
        self.check()  # TODO finally remove
        self.random_fill(list(np.nonzero(self.covered == 0)[0]))
        self.check()  # TODO finally remove

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

    # def two_exchange_delta_eval(self, p1, p2, update_obj_val=True, allow_infeasible=False) -> bool:
    #     # TODO adapt from MKP
    #     elem_added = self.x[p1]
    #     elem_removed = self.x[p2]
    #     y_new = self.y + self.inst.r[:, elem_added] - self.inst.r[:, elem_removed]
    #     feasible = np.all(y_new <= self.inst.b)
    #     if allow_infeasible or feasible:
    #         # accept
    #         self.y = y_new
    #         if update_obj_val:
    #             self.obj_val += self.inst.p[elem_added] - self.inst.p[elem_removed]
    #         return feasible
    #     # revert
    #     self.x[p1], self.x[p2] = elem_removed, elem_added
    #     return False


if __name__ == '__main__':
    from mhlib.demos.common import run_gvns_demo, data_dir
    from mhlib.settings import settings
    settings.meths_li = 0  # TODO finally remove
    run_gvns_demo('MISP', MISPInstance, MISPSolution, data_dir + "misp-simple.clq")
