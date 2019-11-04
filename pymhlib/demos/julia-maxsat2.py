"""Demo application for showing the integration with the Julia language, solving the MAXSAT problem.

Julia and Python's julia package must be installed properly.
The Julia module julia-maxsat.jl is used via Python's julia interface package.
This variant of the demo uses an own Python solution class in which Julia is called just for the
objective function evaluation and the local improvement.

The goal in the MAXSAT problem is to maximize the number of clauses satisfied in a boolean function given in
conjunctive normal form.
"""

import numpy as np
import random
from typing import Any

# from julia import Julia
# jl = Julia(sysimage="/home/guenther/s.so")  # only use when compiled Julia system image available
from julia import Base, Main
import os

from pymhlib.binvec_solution import BinaryVectorSolution
from pymhlib.alns import ALNS
from pymhlib.scheduler import Result
from pymhlib.demos.maxsat import MAXSATInstance

Main.eval(r'include("'+os.path.dirname(__file__)+r'/julia-maxsat.jl")')


class JuliaMAXSAT2Solution(BinaryVectorSolution):
    """Solution to a MAXSAT instance.

    Attributes
        - inst: associated MAXSATInstance
        - x: binary incidence vector
        - destroyed: list of indices of variables that have been destroyed by the ALNS's destroy operator
    """

    to_maximize = True

    def __init__(self, inst: Main.JuliaMAXSAT.JuliaMAXSATInstance):
        super().__init__(inst.n, inst=inst)
        self.destroyed = None

    def copy(self):
        sol = JuliaMAXSAT2Solution(self.inst)
        sol.copy_from(self)
        return sol

    def calc_objective(self):
        return Main.JuliaMAXSAT.obj(self.x, self.inst.julia_inst)

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
        obj_val = self.obj()
        x = self.x
        new_obj_val = Main.JuliaMAXSAT.k_flip_neighborhood_search_b(x, obj_val, self.inst.julia_inst, par, False)
        if new_obj_val > obj_val:
            self.x = x
            self.obj_val = new_obj_val
            return True
        return False

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

    def crossover(self, other: 'JuliaMAXSAT2Solution'):
        """ Perform uniform crossover as crossover."""
        return self.uniform_crossover(other)


if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)
    run_optimization('MAXSAT', Main.JuliaMAXSAT.JuliaMAXSATInstance, JuliaMAXSAT2Solution, data_dir+"maxsat-adv1.cnf")
