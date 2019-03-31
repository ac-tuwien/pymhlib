"""Adaptive Large Neighborhood Search (ALNS).

The module realizes a classical ALNS based on the scheduler module.
"""

from typing import List
import numpy as np
from math import exp
from itertools import chain
from dataclasses import dataclass

from mhlib.settings import get_settings_parser, settings
from mhlib.solution import Solution
from mhlib.scheduler import Scheduler, Method


parser = get_settings_parser()
parser.add("--mh_alns_segment_size", type=int, default=200, help='ALNS segment size')
parser.add("--mh_alns_sigma1", type=int, default=33, help='ALNS score for new global best solution')
parser.add("--mh_alns_sigma2", type=int, default=9, help='ALNS score for better than current solution')
parser.add("--mh_alns_sigma3", type=int, default=13, help='ALNS score for worse accepted solution')
parser.add("--mh_alns_gamma", type=int, default=0.1, help='ALNS ')
parser.add("--mh_alns_init_temp_factor", type=int, default=1.05, help='ALNS factor for determining initial temperature')
parser.add("--mh_alns_temp_dec_factor", type=int, default=0.99975, help='ALNS factor for decreasing the temperature')
parser.add("--mh_alns_dest_min_abs", type=int, default=4, help='ALNS minimum number of elements to destroy')
parser.add("--mh_alns_dest_max_abs", type=int, default=60, help='ALNS maximum number of elements to destroy')
parser.add("--mh_alns_dest_min_ratio", type=int, default=0.05, help='ALNS minimum ratio of elements to destroy')
parser.add("--mh_alns_dest_max_ratio", type=int, default=0.35, help='ALNS maximum ratio of elements to destroy')
parser.add("--mh_alns_logscores", default=False, action='store_true',
           help='ALNS write out log information on scores')
parser.add("--no_mh_alns_logscores", dest='mh_alns_logscores', action='store_false')


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
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_destroy: List[Method], meths_repair: List[Method],
                 own_settings: dict = None):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meths_destroy: list of destroy methods
        :param meths_repair: list of repair methods
        :param own_settings: optional dictionary with specific settings
        """
        super().__init__(sol, meths_ch + meths_destroy + meths_repair, own_settings)
        self.meths_ch = meths_ch
        self.meths_destroy = meths_destroy
        self.meths_repair = meths_repair
        self.score_data = {m.name: ScoreData() for m in chain(self.meths_destroy, self.meths_repair)}
        self.temperature = sol.obj() * self.own_settings.mh_alns_init_temp_factor + 0.000001

    @staticmethod
    def select_method(meths: List[Method], weights=None):
        """Randomly select a method from the given list with probabilities proportional to the given weights.

        :param meths: list of methods from which to select one
        :param weights: list of probabilities for the methods; if None, uniform probability is used
        """
        if weights is None:
            return np.random.choice(meths)
        else:
            return np.random.choice(meths, p=weights/sum(weights))

    def metropolis_criterion(self, sol_new: Solution, sol_incumbent: Solution) -> bool:
        """Apply Metropolis criterion as acceptance decision, return True when sol_new should be accepted."""
        if sol_new.is_better(sol_incumbent):
            return True
        return np.random.random_sample() <= exp(-abs(sol_new.obj() - sol_incumbent.obj())/self.temperature)

    @staticmethod
    def get_number_to_destroy(num_elements: int, own_settings=settings):
        """Randomly sample the number of elements to destroy in the destroy operator based on the parameter settings."""
        a = max(own_settings.mh_alns_dest_min_abs, int(own_settings.mh_alns_dest_min_ratio * num_elements))
        b = min(own_settings.mh_alns_dest_max_abs, int(own_settings.mh_alns_dest_max_ratio * num_elements))
        return np.random.randint(a, b+1)

    def log_scores(self):
        """Write information on received scores and weight update to log."""
        indent = ' '*32
        s = f"{indent}scores at end of iteration {self.iteration}:\n"
        s += f"{indent} method    applied   score    weight"
        for m in chain(self.meths_destroy, self.meths_repair):
            data = self.score_data[m.name]
            s += f"\n{indent}{m.name:>7} {data.applied:10d} {data.score:7d} {data.weight:10.3f}"
        self.iter_logger.info(s)

    def alns(self, sol: Solution):
        """Perform adaptive large neighborhood search (ALNS) to given solution."""
        next_segment = self.iteration + self.own_settings.mh_alns_segment_size
        sol_incumbent = sol.copy()
        sol_new = sol.copy()
        while True:
            destroy = self.select_method(self.meths_destroy,
                                         np.fromiter((self.score_data[m.name].weight for m in self.meths_destroy),
                                                     dtype=float, count=len(self.meths_destroy)))
            repair = self.select_method(self.meths_repair,
                                        np.fromiter((self.score_data[m.name].weight for m in self.meths_repair),
                                                    dtype=float, count=len(self.meths_repair)))
            res = self.perform_method_pair(destroy, repair, sol_new)
            destroy_data = self.score_data[destroy.name]
            repair_data = self.score_data[repair.name]
            destroy_data.applied += 1
            repair_data.applied += 1
            score = 0
            if sol_new.is_better(sol_incumbent):
                score = self.own_settings.mh_alns_sigma1
                sol_incumbent = sol_new
            elif sol_new.is_better(sol):
                score = self.own_settings.mh_alns_sigma2
                sol.copy_from(sol_new)
            elif self.metropolis_criterion(sol_new, sol):
                score = self.own_settings.mh_alns_sigma3
                sol.copy_from(sol_new)
            else:
                sol_new.copy_from(sol)
            destroy_data.score += score
            repair_data.score += score
            if res.terminate:
                sol.copy_from(sol_incumbent)
                return
            if self.iteration == next_segment:
                # end of segment: update weights and re-initialize scores
                if self.own_settings.mh_alns_logscores:
                    self.log_scores()
                next_segment = self.iteration + self.own_settings.mh_alns_segment_size
                gamma = self.own_settings.mh_alns_gamma
                for m in chain(self.meths_destroy, self.meths_repair):
                    data = self.score_data[m.name]
                    if data.applied:
                        data.weight = data.weight * (1 - gamma) + gamma * data.score / data.applied
                        data.score = 0
                        data.applied = 0

    def run(self):
        """Actually performs the construction heuristics followed by the ALNS."""
        sol = self.incumbent.copy()

        # perform all construction heuristics, take best solution
        for m in self.next_method(self.meths_ch):
            res = self.perform_method(m, sol)
            if res.terminate:
                break
        if self.incumbent.is_better(sol):
            sol.copy_from(self.incumbent)

        self.alns(sol)
