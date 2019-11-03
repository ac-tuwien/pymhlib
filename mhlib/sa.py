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

from mhlib.scheduler import Method, Scheduler, MethodStatistics
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
        - meth_propose_neighborhood_move: propose neighborhood move method
        - meth_apply_neighborhood_move: apply neighborhood move method return by propose method
        - iter_cb: callback for each iteration passing iteration number, proposed sol, accepted sol, temperature, and acceptance
        - temperature: current temperature
        - equi_iter: iterations until equilibrium
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meth_propose_neighborhood_move,
                 meth_apply_neighborhood_move, iter_cb, own_settings: dict = None, consider_initial_sol=False):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meth_propose_neighborhood_move: list of neighbor proposal methods
        :param meth_apply_neighborhood_move: apply neighborhood move method return by propose method
        :param iter_cb: callback for each iteration passing iteration number, proposed sol, accepted sol, temperature, and acceptance
        :param own_settings: optional dictionary with specific settings
        :param consider_initial_sol: if true consider sol as valid solution that should be improved upon; otherwise
            sol is considered just a possibly uninitialized of invalid solution template
        """
        super().__init__(sol, meths_ch, own_settings, consider_initial_sol)
        self.meths_ch = meths_ch
        self.meth_propose_neighborhood_move = meth_propose_neighborhood_move
        self.meth_apply_neighborhood_move = meth_apply_neighborhood_move
        self.method_stats['sa'] = MethodStatistics()
        self.iter_cb = iter_cb
        self.temperature = self.own_settings.mh_sa_T_init
        self.equi_iter = self.own_settings.mh_sa_equi_iter

    def metropolis_criterion(self, delta_f) -> bool:
        """Apply Metropolis criterion as acceptance decision determined by delta_f and current temperature."""
        if Solution.is_better_obj(delta_f, 0):
            return True
        return np.random.random_sample() <= exp(-abs(delta_f) / self.temperature)

    def cool_down(self):
        """Apply geometric cooling."""
        self.temperature *= self.own_settings.mh_sa_alpha

    def sa(self, sol: Solution):
        """Perform simulated annealing with geometric cooling on given solution."""

        def sa_iteration(sol: Solution, _par, result):
            neighborhood_move, delta_f = self.meth_propose_neighborhood_move(sol)
            acceptance = self.metropolis_criterion(delta_f)
            if acceptance:
                self.meth_apply_neighborhood_move(sol, neighborhood_move)
                sol.obj_val = sol.obj() + delta_f
                result.changed = True
            if self.iter_cb is not None:
                self.iter_cb(self.iteration, sol, self.temperature, acceptance)
        sa_method = Method("sa", sa_iteration, 0)

        while True:
            for it_ in range(self.equi_iter):
                res = self.perform_method(sa_method, sol)
                if res.terminate:
                    return True
            self.cool_down()

    def run(self) -> None:
        """Actually performs the construction heuristics followed by the SA."""
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.sa(sol)
