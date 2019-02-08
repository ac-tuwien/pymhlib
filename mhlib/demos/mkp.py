"""Demo application solving the multi-dimensional knapsack problem (MKP)."""

import numpy as np

from mhlib.subset_solution import SubsetSolution


class MKPInstance:
    """MKP problem instance.

    Attributes
        - n: number of items
        - m: number of resources, i.e., constraints
        - p: prices of items
        - r: resource consumption values
        - b: resource limits
        - obj_opt: optimal objective value or 0 if not known
        - r_min: minimal resource consumption value over all elements for each resource
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file in Chu and Beasley's format."""
        self.n = 0
        self.m = 0
        self.p = None
        self.r = None
        self.b = None
        self.obj_opt = 0

        all_values = []
        with open(file_name, "r") as file:
            for line in file:
                for word in line.split():
                    all_values.append(int(word))
        self.n = all_values[0]
        self.m = all_values[1]
        if len(all_values) != 3+self.n+self.m*self.n+self.m:
            raise ValueError(f"Invalid number of values in MKP instance file {file_name}")
        self.obj_opt = all_values[2]
        self.p = np.array(all_values[3:3+self.n])
        self.r = np.array(all_values[3+self.n:3+self.n+self.m*self.n]).reshape([self.m, self.n])
        self.b = np.array(all_values[3+self.n+self.m*self.n:3+self.n+self.m*self.n+self.m])
        self.r_min = np.min(self.r, 1)

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n} m={self.m},\np={self.p},\nr={self.r},\nb={self.b}\n"


class MKPSolution(SubsetSolution):
    """Solution to an MKP instance.

    Additional attributes
        - y: amount of each resource used
    """

    def __init__(self, inst: MKPInstance):
        super().__init__(range(inst.n), inst=inst)
        self.y = np.zeros([self.inst.m], dtype=int)

    def copy(self):
        sol = MKPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'MKPSolution'):
        super().copy_from(other)
        self.y[:] = other.y

    def calc_objective(self):
        return np.sum(self.inst.p[self.x[:self.sel]])

    def calc_y(self):
        """Calculates z from scratch."""
        self.y = np.sum(self.inst.r[:, self.x[:self.sel]], axis=1)

    def check(self, unsorted=False):
        super().check(unsorted)
        y_old = self.y
        self.calc_y()
        if np.any(y_old != self.y):
            raise ValueError(f"Solution had invalid y values: {self.y!s} {y_old!s}")
        if np.any(self.y > self.inst.b):
            raise ValueError(f"Solution exceeds capacity limits:  {self.y}, {self.inst.b}")

    def clear(self):
        self.y.fill(0)
        super().clear()

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
        self.remove_some(par)
        self.random_fill(self.x[self.sel:])
        # self.check()

    def may_be_extendible(self) -> bool:
        return np.all(self.y + self.inst.r_min <= self.inst.b) and self.sel < len(self.x)

    def element_removed_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        elem = self.x[self.sel]
        self.y -= self.inst.r[:, elem]
        if update_obj_val:
            self.obj_val -= self.inst.p[elem]
        return True

    def element_added_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        elem = self.x[self.sel-1]
        y = self.y + self.inst.r[:, elem]
        feasible = np.all(y <= self.inst.b)
        if allow_infeasible or feasible:
            # accept
            self.y = y
            if update_obj_val:
                self.obj_val += self.inst.p[elem]
            return feasible
        # revert
        self.sel -= 1
        return False

    def two_exchange_delta_eval(self, p1, p2, update_obj_val=True, allow_infeasible=False) -> bool:
        elem_added = self.x[p1]
        elem_removed = self.x[p2]
        y_new = self.y + self.inst.r[:, elem_added] - self.inst.r[:, elem_removed]
        feasible = np.all(y_new <= self.inst.b)
        if allow_infeasible or feasible:
            # accept
            self.y = y_new
            if update_obj_val:
                self.obj_val += self.inst.p[elem_added] - self.inst.p[elem_removed]
            return feasible
        # revert
        self.x[p1], self.x[p2] = elem_removed, elem_added
        return False


if __name__ == '__main__':
    from mhlib.demos.common import run_gvns_demo, data_dir
    run_gvns_demo('MKP', MKPInstance, MKPSolution, data_dir + "mknapcb5-01.txt")
