"""Demo application solving the MAXSAT problem.

The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.
"""

import numpy as np
import random
from typing import Any

from mhlib.binvec_solution import BinaryVectorSolution
from mhlib.alns import ALNS
from mhlib.scheduler import Result


class MAXSATInstance:
    """MAXSAT problem instance.

    The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.

    Attributes
        - n: number of variables, i.e., size of incidence vector
        - m: number of clauses
        - clauses: list of clauses, where each clause is represented by a list of integers;
            a positive integer v refers to the v-th variable, while a negative integer v refers
            to the negated form of the v-th variable; note that variable indices start with 1 (-1)
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
        self.m = 0
        self.clauses = list()

        with open(file_name, "r") as file:
            for line in file:
                if line.startswith("c"):
                    # ignore comments
                    continue
                fields = line.split()
                if len(fields) == 4 and fields[0] == "p" and fields[1] == "cnf":
                    try:
                        self.n = int(fields[2])
                        self.m = int(fields[3])
                    except ValueError:
                        raise ValueError(f"Invalid values in line 'p cnf': {line}")
                elif len(fields) >= 1:
                    # read clause
                    if not fields[-1].startswith("0"):
                        raise ValueError(f"Last field in clause line must be 0, but is not: {line}, {fields[-1]!r}")
                    try:
                        self.clauses.append(np.array([int(s) for s in fields[:-1]]))
                    except ValueError:
                        raise ValueError(f"Invalid clause: {line}")

        # make basic check if instance is meaningful
        if not 1 <= self.n <= 1000000:
            raise ValueError(f"Invalid n: {self.n}")
        if not 1 <= self.m <= 1000000:
            raise ValueError(f"Invalid m: {self.m}")
        if len(self.clauses) != self.m:
            raise ValueError(f"Number of clauses should be {self.m}, but {len(self.clauses)} read")

    def __repr__(self):
        return f"m={self.m}, n={self.n}\n"  # , clauses={self.clauses!r}\n"


class MAXSATSolution(BinaryVectorSolution):
    """Solution to a MAXSAT instance.

    Attributes
        - inst: associated MAXSATInstance
        - x: binary incidence vector
        - destroyed: list of indices of variables that have been destroyed by the ALNS's destroy operator
    """

    def __init__(self, inst: MAXSATInstance):
        super().__init__(inst.n, inst=inst)
        self.destroyed = None

    def copy(self):
        sol = MAXSATSolution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        fulfilled_clauses = 0
        for clause in self.inst.clauses:
            for v in clause:
                if self.x[abs(v)-1] == (1 if v > 0 else 0):
                    fulfilled_clauses += 1
                    break
        return fulfilled_clauses

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
        """
        if len(self.x) != self.inst.n:
            raise ValueError("Invalid length of solution")
        super().check()

    def construct(self, par: Any, _result: Result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def local_improve(self, par: Any, _result: Result):
        """Perform one k_flip_neighborhood_search."""
        self.k_flip_neighborhood_search(par, False)

    def shaking(self, par, _result):
        """Scheduler method that performs shaking by flipping par random positions."""
        for i in range(par):
            p = random.randrange(self.inst.n)
            self.x[p] = not self.x[p]
        self.invalidate()

    def destroy(self, par: Any, _result: Result):
        """Destroy operator for ALNS selects par*ALNS.get_number_to_destroy positions uniformly at random for removal.

        Selected positions are stored with the solution in list self.destroyed.
        """
        num = min(ALNS.get_number_to_destroy(len(self.x)) * par, len(self.x))
        self.destroyed = np.random.choice(range(len(self.x)), num, replace=False)
        self.invalidate()

    def repair(self, _par: Any, _result: Result):
        """Repair operator for ALNS assigns new random values to all positions in self.destroyed."""
        assert self.destroyed is not None
        for p in self.destroyed:
            self.x[p] = random.randrange(2)
        self.destroyed = None
        self.invalidate()

    def crossover(self, other: 'MAXSATSolution'):
        """ Perform uniform crossover as crossover."""
        return self.uniform_crossover(other)


if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir
    run_optimization('MAXSAT', MAXSATInstance, MAXSATSolution, data_dir+"advanced.cnf")
