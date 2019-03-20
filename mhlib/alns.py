"""Adaptive Large Neighborhood Search (ALNS).

The module realizes a classical ALNS based on the scheduler module.
"""

from typing import List
import time
import numpy as np

from mhlib.settings import get_settings_parser
from mhlib.solution import Solution
from mhlib.scheduler import Scheduler, Method


parser = get_settings_parser()
parser.add("--mh_alns_segment", type=int, default=200, help='ALNS segment size')


class ALNS(Scheduler):
    """An adaptive large neighborhood search (ALNS).

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meths_de: list of destroy methods
        - meths_repair: list of repair methods
        - scores_destroy: array of scores for destroy methods
        - scores_repair: array of scores for repair methods
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
        self.scores_destroy = np.full((len(self.meths_destroy)), 1.0)
        self.scores_repair = np.full((len(self.meths_repair)), 1.0)

    @staticmethod
    def select_method(meths: List[Method], scores=None):
        """Randomly select a method from the given list with probabilities proportional to the given scores.

        :param meths: list of methods from which to select one
        :param scores: list of probabilities for the methods; if None, uniform probability is used
        """
        if scores is None:
            return np.random.choice(meths)
        else:
            assert len(meths) == len(scores)
            return np.random.choice(meths, p=scores/sum(scores))

    def alns(self, sol: Solution):
        """Perform adaptive large neighborhood search (ALNS) to given solution."""
        sol2 = sol.copy()
        while True:
            destroy = self.select_method(self.meths_destroy, self.scores_destroy)
            repair = self.select_method(self.meths_repair, self.scores_repair)
            t_destroy_start = time.process_time()
            res = self.perform_method(destroy, sol2, delayed_success=True)
            t_destroy = time.process_time() - t_destroy_start
            terminate = res.terminate
            if not terminate:
                res = self.perform_method(repair, sol2)
            self.delayed_success_update(destroy, sol.obj(), time.process_time()-t_destroy, sol2)
            # TODO one destroy+repair should count as one iteration in the statistics
            # TODO update scores
            # TODO also print final scores in statistics(?)
            if sol2.is_better(sol):
                sol.copy_from(sol2)
                if terminate or res.terminate:
                    return
            else:
                if terminate or res.terminate:
                    return
                sol2.copy_from(sol)

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
