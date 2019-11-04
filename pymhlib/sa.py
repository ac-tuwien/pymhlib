"""A class implementing a simulated annealing (SA) metaheuristic.

It extends the more general scheduler module/class. Allows for callbacks after each iteration.
Parameter mh_sa_equi_iter controls how many random neighbor moves are investigated at each
temperature level, and such a series of moves is considered one method call of the scheduler.

From the demo applications, only the TSP, QAP and MAXSAT support SA so far.
"""

from typing import List, Callable
import time
import numpy as np
from math import exp

from pymhlib.scheduler import Method, Scheduler, MethodStatistics
from pymhlib.settings import get_settings_parser
from pymhlib.solution import Solution, TObj

parser = get_settings_parser()
parser.add_argument("--mh_sa_T_init", type=float, default=30,
                    help='SA initial temperature')
parser.add_argument("--mh_sa_alpha", type=float, default=0.95,
                    help='SA alpha for geometric cooling')
parser.add_argument("--mh_sa_equi_iter", type=int, default=10000,
                    help='SA iterations until equilibrium')


class SA(Scheduler):
    """A simulated annealing metaheuristic (SA).

    Parameter mh_sa_equi_iter controls how many random neighbor moves are investigated at each
    temperature level, and such a series of moves is considered one method call of the scheduler.

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - random_move_delta_eval: propose neighborhood move method
        - apply_neighborhood_move: apply neighborhood move method return by propose method
        - iter_cb: callback for each iteration passing iteration number, proposed sol, accepted sol, temperature,
            and acceptance
        - temperature: current temperature
        - equi_iter: iterations until equilibrium
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], random_move_delta_eval: Callable,
                 apply_neighborhood_move: Callable, iter_cb: Callable, own_settings: dict = None,
                 consider_initial_sol=False):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param random_move_delta_eval: function that chooses a random move and determines the delta in the obj_val
        :param apply_neighborhood_move: apply neighborhood move method return by propose method
        :param iter_cb: callback for each iteration passing iteration number, proposed sol, accepted sol, temperature,
            and acceptance
        :param own_settings: optional dictionary with specific settings
        :param consider_initial_sol: if true consider sol as valid solution that should be improved upon; otherwise
            sol is considered just a possibly uninitialized of invalid solution template
        """
        super().__init__(sol, meths_ch, own_settings, consider_initial_sol)
        self.meths_ch = meths_ch
        self.random_move_delta_eval = random_move_delta_eval
        self.apply_neighborhood_move = apply_neighborhood_move
        self.method_stats['sa'] = MethodStatistics()
        self.iter_cb = iter_cb
        self.temperature = self.own_settings.mh_sa_T_init
        self.equi_iter = self.own_settings.mh_sa_equi_iter

    def metropolis_criterion(self, sol, delta_obj:TObj) -> bool:
        """Apply Metropolis criterion as acceptance decision determined by delta_obj and current temperature."""
        if sol.is_better_obj(delta_obj, 0):
            return True
        return np.random.random_sample() <= exp(-abs(delta_obj) / self.temperature)

    def cool_down(self):
        """Apply geometric cooling."""
        self.temperature *= self.own_settings.mh_sa_alpha

    def sa(self, sol: Solution):
        """Perform simulated annealing with geometric cooling on given solution."""

        def sa_iteration(sol: Solution, _par, result):
            neighborhood_move, delta_obj = self.random_move_delta_eval(sol)
            acceptance = self.metropolis_criterion(sol, delta_obj)
            if acceptance:
                self.apply_neighborhood_move(sol, neighborhood_move)
                sol.obj_val = sol.obj_val + delta_obj
                result.changed = True
            if self.iter_cb is not None:
                self.iter_cb(self.iteration, sol, self.temperature, acceptance)
        sa_method = Method("sa", sa_iteration, 0)

        while True:
            for it_ in range(self.equi_iter):
                t_start = time.process_time()
                obj_old = self.incumbent.obj()
                res = self.perform_method(sa_method, sol, delayed_success=True)
                self.delayed_success_update(sa_method, obj_old, t_start, sol)
                if res.terminate:
                    return True
            self.cool_down()

    def run(self) -> None:
        """Actually performs the construction heuristics followed by the SA."""
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.sa(sol)
