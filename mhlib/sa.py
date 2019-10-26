"""A simulated annealing class

It extends the more general scheduler module/class by distinguishing between construction heuristics and
neighborhood structures. Allows for callbacks after each iteration. Parameters for geometric cooling and iterations
until equilibrium can be provided.
"""

from typing import List
import time
import numpy as np
import random
from math import exp

from mhlib.scheduler import Method, Scheduler
from mhlib.settings import get_settings_parser
from mhlib.solution import Solution

parser = get_settings_parser()
parser.add_argument("--mh_sa_T_init", type=float, default=30,
                    help='SA initial temperature')
parser.add_argument("--mh_sa_alpha", type=float, default=0.95,
                    help='SA alpha for geometric cooling')
parser.add_argument("--mh_sa_equi_iter", type=int, default=10000,
                    help='SA iterations until equilibrium')


class SA(Scheduler):
    """A simulated annealing (SA).

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meths_np: list of neighbor proposal methods
        - iter_cb: callback for each iteration passing iteration number, proposed sol, accepted sol, and temperature
        - temperature: current temperature
        - equi_iter: iterations until equilibrium
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_np: List[Method], iter_cb,
                 own_settings: dict = None, consider_initial_sol=False):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meths_np: list of neighbor proposal methods
        :param iter_cb: callback for each iteration passing iteration number, proposed sol, accepted sol, and temperature
        :param own_settings: optional dictionary with specific settings
        :param consider_initial_sol: if true consider sol as valid solution that should be improved upon; otherwise
            sol is considered just a possibly uninitialized of invalid solution template
        """
        super().__init__(sol, meths_ch + meths_np, own_settings, consider_initial_sol)
        self.meths_ch = meths_ch
        self.meths_np = meths_np
        self.iter_cb = iter_cb
        self.temperature = self.own_settings.mh_sa_T_init
        self.equi_iter = self.own_settings.mh_sa_equi_iter

    def metropolis_criterion(self, sol_new: Solution, sol_current: Solution) -> bool:
        """Apply Metropolis criterion as acceptance decision, return True when sol_new should be accepted."""
        if sol_new.is_better(sol_current):
            return True
        return np.random.random_sample() <= exp(-abs(sol_new.obj() - sol_current.obj()) / self.temperature)

    def cool_down(self):
        """Apply geometric cooling."""
        self.temperature *= self.own_settings.mh_sa_alpha

    def sa(self, sol: Solution):
        """Perform simulated annealing with geometric cooling on given solution."""
        sol2 = sol.copy()

        while True:
            for it_ in range(self.equi_iter):
                neighbor_proposal_method = random.sample(self.meths_np, 1)[0]
                res = self.perform_method(neighbor_proposal_method, sol2)
                terminate = res.terminate
                if self.metropolis_criterion(sol2, sol):
                    if self.iter_cb is not None:
                        self.iter_cb(self.iteration, sol2, sol2, self.temperature)
                    sol.copy_from(sol2)
                    if terminate or res.terminate:
                        return
                else:
                    if self.iter_cb is not None:
                        self.iter_cb(self.iteration, sol2, sol, self.temperature)
                    if terminate or res.terminate:
                        return
                    sol2.copy_from(sol)
            self.cool_down()

    def run(self) -> None:
        """Actually performs the construction heuristics followed by the SA."""
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.sa(sol)
