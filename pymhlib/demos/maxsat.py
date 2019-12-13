"""Demo application solving the MAXSAT problem.

The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.
"""

import numpy as np
import random
from typing import Any, Tuple

from pymhlib.solution import TObj
from pymhlib.binvec_solution import BinaryVectorSolution
from pymhlib.alns import ALNS
from pymhlib.scheduler import Result


class MAXSATInstance:
    """MAXSAT problem instance.

    The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.

    Attributes
        - n: number of variables, i.e., size of incidence vector
        - m: number of clauses
        - clauses: list of clauses, where each clause is represented by an array of integers;
            a positive integer v refers to the v-th variable, while a negative integer v refers
            to the negated form of the v-th variable; note that variable indices start with 1 (-1)
        - variable_usage: array containing for each variable a list with the indices of the clauses in
            which the variable appears; needed for efficient incremental evaluation
    """

    def __init__(self, file_name: str):
        """Read an instance from the specified file."""
        self.n = 0
        self.m = 0
        self.clauses = list()
        self.variable_usage: np.ndarray

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
                    self.variable_usage = [list() for _ in range(self.n)]
                elif len(fields) >= 1:
                    # read clause
                    if not fields[-1].startswith("0"):
                        raise ValueError(f"Last field in clause line must be 0, but is not: {line}, {fields[-1]!r}")
                    try:
                        clause = [int(s) for s in fields[:-1]]
                        for v in clause:
                            self.variable_usage[abs(v)-1].append(len(self.clauses))
                        self.clauses.append(np.array(clause))
                    except ValueError:
                        raise ValueError(f"Invalid clause: {line}")

        for v, usage in enumerate(self.variable_usage):
            self.variable_usage[v] = np.array(usage)

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

    to_maximize = True

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
        """Scheduler method that constructs a new random solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def local_improve(self, par: Any, _result: Result):
        """Perform one k_flip_neighborhood_search."""
        self.k_flip_neighborhood_search(par, False)

    def shaking(self, par, _result):
        """Scheduler method that performs shaking by flipping par random positions."""
        self.k_random_flips(par)

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

    def flip_variable(self, pos: int):
        delta_obj = self.flip_move_delta_eval(pos)
        self.obj_val += delta_obj
        self.x[pos] = not self.x[pos]

    def flip_move_delta_eval(self, pos: int) -> TObj:
        """Determine delta in objective value when flipping position pos."""
        assert self.obj_val_valid
        val = not self.x[pos]
        delta = 0
        for clause in self.inst.variable_usage[pos]:
            val_fulfills_now = False
            for v in self.inst.clauses[clause]:
                if abs(v)-1 == pos:
                    val_fulfills_now = (val if v > 0 else not val)
                elif self.x[abs(v) - 1] == (1 if v > 0 else 0):
                    break  # clause fulfilled by other variable, no change
            else:
                delta += 1 if val_fulfills_now else -1
        return delta

    def random_move_delta_eval(self) -> Tuple[int, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        return self.random_flip_move_delta_eval()

    def apply_neighborhood_move(self, pos):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        self.x[pos] = not self.x[pos]

    def crossover(self, other: 'MAXSATSolution'):
        """ Perform uniform crossover as crossover."""
        return self.uniform_crossover(other)


if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)
    run_optimization('MAXSAT', MAXSATInstance, MAXSATSolution, data_dir+"maxsat-adv1.cnf")
