"""Adaptive Large Neighborhood Search (ALNS).

The module realizes a classical ALNS based on the scheduler module.
"""

from typing import List, Tuple
import numpy as np
from math import exp
from itertools import chain
from dataclasses import dataclass

from pymhlib.settings import get_settings_parser, settings, boolArg
from pymhlib.solution import Solution
from pymhlib.scheduler import Scheduler, Method
from pymhlib.log import LogLevel


parser = get_settings_parser()
parser.add_argument("--mh_alns_segment_size", type=int, default=200, help='ALNS segment size')
parser.add_argument("--mh_alns_sigma1", type=int, default=33, help='ALNS score for new global best solution')
parser.add_argument("--mh_alns_sigma2", type=int, default=9, help='ALNS score for better than current solution')
parser.add_argument("--mh_alns_sigma3", type=int, default=13, help='ALNS score for worse accepted solution')
parser.add_argument("--mh_alns_gamma", type=float, default=0.1, help='ALNS ')
parser.add_argument("--mh_alns_init_temp_factor", type=float, default=1.05,
                    help='ALNS factor for determining initial temperature')
parser.add_argument("--mh_alns_temp_dec_factor", type=float, default=0.99975,
                    help='ALNS factor for decreasing the temperature')
parser.add_argument("--mh_alns_dest_min_abs", type=int, default=4, help='ALNS minimum number of elements to destroy')
parser.add_argument("--mh_alns_dest_max_abs", type=int, default=60, help='ALNS maximum number of elements to destroy')
parser.add_argument("--mh_alns_dest_min_ratio", type=float, default=0.05,
                    help='ALNS minimum ratio of elements to destroy')
parser.add_argument("--mh_alns_dest_max_ratio", type=float, default=0.35,
                    help='ALNS maximum ratio of elements to destroy')
parser.add_argument("--mh_alns_logscores", type=boolArg, default=True, help='ALNS write out log information on scores')


@dataclass
class ScoreData:
    """Weight of a method and all data relevant to calculate the score and update the weight.

    Properties
        - weight: weight to be used for selecting methods
        - score: current score in current segment
        - applied: number of applications in current segment
    """
    weight: float = 1.0
    score: int = 0
    applied: int = 0


class ALNS(Scheduler):
    """An adaptive large neighborhood search (ALNS).

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meths_de: list of destroy methods
        - meths_repair: list of repair methods
        - score_data: dictionary yielding ScoreData for a method
        - temperature: temperature for Metropolis criterion
        - next_segment: iteration number of next segment for updating operator weights
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_destroy: List[Method], meths_repair: List[Method],
                 own_settings: dict = None, consider_initial_sol=False):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meths_destroy: list of destroy methods
        :param meths_repair: list of repair methods
        :param own_settings: optional dictionary with specific settings
        :param consider_initial_sol: if true consider sol as valid solution that should be improved upon; otherwise
            sol is considered just a possibly uninitialized of invalid solution template
        """
        super().__init__(sol, meths_ch + meths_destroy + meths_repair, own_settings, consider_initial_sol)
        self.meths_ch = meths_ch
        assert meths_destroy and meths_repair
        self.meths_destroy = meths_destroy
        self.meths_repair = meths_repair
        self.score_data = {m.name: ScoreData() for m in chain(self.meths_destroy, self.meths_repair)}
        self.temperature = sol.obj() * self.own_settings.mh_alns_init_temp_factor + 0.000000001
        self.next_segment = 0

    @staticmethod
    def select_method(meths: List[Method], weights=None) -> Method:
        """Randomly select a method from the given list with probabilities proportional to the given weights.

        :param meths: list of methods from which to select one
        :param weights: list of probabilities for the methods; if None, uniform probability is used
        """
        if weights is None:
            return np.random.choice(meths)
        else:
            return np.random.choice(meths, p=weights/sum(weights))

    def select_method_pair(self) -> Tuple[Method, Method]:
        """Select a destroy and repair method pair according to current weights."""
        destroy = self.select_method(self.meths_destroy,
                                     np.fromiter((self.score_data[m.name].weight for m in self.meths_destroy),
                                                 dtype=float, count=len(self.meths_destroy)))
        repair = self.select_method(self.meths_repair,
                                    np.fromiter((self.score_data[m.name].weight for m in self.meths_repair),
                                                dtype=float, count=len(self.meths_repair)))
        return destroy, repair

    def metropolis_criterion(self, sol_new: Solution, sol_current: Solution) -> bool:
        """Apply Metropolis criterion as acceptance decision, return True when sol_new should be accepted."""
        if sol_new.is_better(sol_current):
            return True
        return np.random.random_sample() <= exp(-abs(sol_new.obj() - sol_current.obj()) / self.temperature)

    @staticmethod
    def get_number_to_destroy(num_elements: int, own_settings=settings, dest_min_abs=None, dest_min_ratio=None,
                              dest_max_abs=None, dest_max_ratio=None) -> int:
        """Randomly sample the number of elements to destroy in the destroy operator based on the parameter settings.

        :param num_elements: number of elements to destroy from (e.g., the size of the solution)
        :param own_settings: a settings object to be used which overrides the global settings
        :param dest_min_abs: absolute minimum number of elements to destroy overriding settings
        :param dest_min_ratio: relative minimum ratio of elements to destroy overriding settings
        :param dest_max_abs: absolute maximum number of elements to destroy overriding settings
        :param dest_max_ratio: relative maximum ratio of elements to destroy overriding settings
        """
        if dest_min_abs is None:
            dest_min_abs = own_settings.mh_alns_dest_min_abs
        if dest_min_ratio is None:
            dest_min_ratio = own_settings.mh_alns_dest_min_ratio
        if dest_max_abs is None:
            dest_max_abs = own_settings.mh_alns_dest_max_abs
        if dest_max_ratio is None:
            dest_max_ratio = own_settings.mh_alns_dest_max_ratio
        a = max(dest_min_abs, int(dest_min_ratio * num_elements))
        b = min(dest_max_abs, int(dest_max_ratio * num_elements))
        return np.random.randint(a, b+1) if b >= a else b+1

    def update_operator_weights(self):
        """Update operator weights at segment ends and re-initialize scores"""
        if self.iteration == self.next_segment:
            if self.own_settings.mh_alns_logscores:
                self.log_scores()
            self.next_segment = self.iteration + self.own_settings.mh_alns_segment_size
            gamma = self.own_settings.mh_alns_gamma
            for m in chain(self.meths_destroy, self.meths_repair):
                data = self.score_data[m.name]
                if data.applied:
                    data.weight = data.weight * (1 - gamma) + gamma * data.score / data.applied
                    data.score = 0
                    data.applied = 0

    def update_after_destroy_and_repair_performed(self, destroy: Method, repair: Method, sol_new: Solution,
                                                  sol_incumbent: Solution, sol: Solution):
        """Update current solution, incumbent, and all operator score data according to performed destroy+repair.

        :param destroy: applied destroy method
        :param repair: applied repair method
        :param sol_new: obtained new solution
        :param sol_incumbent: current incumbent solution
        :param sol: current (last accepted) solution
        """
        destroy_data = self.score_data[destroy.name]
        repair_data = self.score_data[repair.name]
        destroy_data.applied += 1
        repair_data.applied += 1
        score = 0
        if sol_new.is_better(sol_incumbent):
            score = self.own_settings.mh_alns_sigma1
            # print('better than incumbent')
            sol_incumbent.copy_from(sol_new)
            sol.copy_from(sol_new)
        elif sol_new.is_better(sol):
            score = self.own_settings.mh_alns_sigma2
            # print('better than current')
            sol.copy_from(sol_new)
        elif sol.is_better(sol_new) and self.metropolis_criterion(sol_new, sol):
            score = self.own_settings.mh_alns_sigma3
            # print('accepted although worse')
            sol.copy_from(sol_new)
        elif sol_new != sol:
            sol_new.copy_from(sol)
        destroy_data.score += score
        repair_data.score += score

    def cool_down(self):
        """Apply geometric cooling."""
        self.temperature *= self.own_settings.mh_alns_temp_dec_factor

    def log_scores(self):
        """Write information on received scores and weight update to log."""
        indent = ' ' * 32
        s = f"{indent}scores at end of iteration {self.iteration}:\n"
        s += f"{indent} method    applied   score    weight"
        for m in chain(self.meths_destroy, self.meths_repair):
            data = self.score_data[m.name]
            s += f"\n{indent}{m.name:>7} {data.applied:10d} {data.score:7d} {data.weight:10.3f}"
        self.iter_logger.info(LogLevel.indent(s))

    def alns(self, sol: Solution):
        """Perform adaptive large neighborhood search (ALNS) on given solution."""
        self.next_segment = self.iteration + self.own_settings.mh_alns_segment_size
        sol_incumbent = sol.copy()
        sol_new = sol.copy()
        while True:
            destroy, repair = self.select_method_pair()
            res = self.perform_method_pair(destroy, repair, sol_new)
            self.update_after_destroy_and_repair_performed(destroy, repair, sol_new, sol_incumbent, sol)
            if res.terminate:
                sol.copy_from(sol_incumbent)
                return
            self.update_operator_weights()
            self.cool_down()

    def run(self) -> None:
        """Actually performs the construction heuristics followed by the ALNS."""
        sol = self.incumbent.copy()
        assert self.incumbent_valid or self.meths_ch
        self.perform_sequentially(sol, self.meths_ch)
        self.alns(sol)
