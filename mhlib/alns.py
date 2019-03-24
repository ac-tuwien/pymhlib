"""Adaptive Large Neighborhood Search (ALNS).

The module realizes a classical ALNS based on the scheduler module.
"""

from typing import List
import numpy as np
from math import exp

from mhlib.settings import get_settings_parser
from mhlib.solution import Solution
from mhlib.scheduler import Scheduler, Method


parser = get_settings_parser()
parser.add("--mh_alns_segment", type=int, default=200, help='ALNS segment size')
parser.add("--mh_alns_sigma1", type=int, default=33, help='ALNS score for new global best solution')
parser.add("--mh_alns_sigma2", type=int, default=9, help='ALNS score for better than current solution')
parser.add("--mh_alns_sigma3", type=int, default=13, help='ALNS score for worse accepted solution')
parser.add("--mh_alns_init_temp_factor", type=int, default=1.05, help='ALNS factor for determining initial temperature')
parser.add("--mh_alns_temp_dec_factor", type=int, default=0.99975, help='ALNS factor for decreasing the temperature')
parser.add("--mh_alns_dest_min_abs", type=int, default=4, help='ALNS minimum number of elements to destroy')
parser.add("--mh_alns_dest_max_abs", type=int, default=60, help='ALNS maximum number of elements to destroy')
parser.add("--mh_alns_dest_min_percent", type=int, default=5, help='ALNS minimum percentage of elements to destroy')
parser.add("--mh_alns_dest_max_percent", type=int, default=35, help='ALNS maximum percentage of elements to destroy')


class ALNS(Scheduler):
    """An adaptive large neighborhood search (ALNS).

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meths_de: list of destroy methods
        - meths_repair: list of repair methods
        - weights_destroy: array of weights for destroy methods
        - weights_repair: array of weights for repair methods
        - scores_destroy: array of scores for destroy methods for current segment
        - scores_repair: array of scores for repair methods for current segment
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
        self.weights_destroy = np.full((len(self.meths_destroy)), 1.0)
        self.weights_repair = np.full((len(self.meths_repair)), 1.0)
        self.scores_destroy = np.full((len(self.meths_destroy)), 1.0)
        self.scores_repair = np.full((len(self.meths_repair)), 1.0)
        self.temperature = sol.obj() * self.own_settings.mh_alns_init_temp_factor

    @staticmethod
    def select_method(meths: List[Method], weights=None):
        """Randomly select a method from the given list with probabilities proportional to the given weights.

        :param meths: list of methods from which to select one
        :param weights: list of probabilities for the methods; if None, uniform probability is used
        """
        if weights is None:
            return np.random.choice(meths)
        else:
            assert len(meths) == len(weights)
            return np.random.choice(meths, p=weights/sum(weights))

    def metropolis_criterion(self, sol_new: Solution, sol_incumbent: Solution) -> bool:
        """Apply Metropolis criterion as acceptance decision, return True when sol_new should be accepted."""
        if sol_new.is_better(sol_incumbent):
            return True
        return np.random.random_sample() <= exp(-abs(sol_new.obj() - sol_incumbent.obj())/self.temperature)

    def alns(self, sol: Solution):
        """Perform adaptive large neighborhood search (ALNS) to given solution."""
        sol_incumbent = sol.copy()
        sol_new = sol.copy()
        while True:
            destroy = self.select_method(self.meths_destroy, self.weights_destroy)
            repair = self.select_method(self.meths_repair, self.weights_repair)
            res = self.perform_method_pair(destroy, repair, sol_new)
            # TODO update scores
            # TODO also print final scores in statistics(?)
            if sol_new.is_better(sol_incumbent):
                sol_incumbent = sol_new
            elif self.metropolis_criterion(sol_new, sol):
                sol.copy_from(sol_new)
            else:
                sol_new.copy_from(sol)
            if res.terminate:
                sol.copy_from(sol_incumbent)
                return

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
