
from typing import List
from itertools import cycle
import random

from mhlib.scheduler import Method, Result
from mhlib.settings import get_settings_parser
from mhlib.solution import Solution


parser = get_settings_parser()
parser.add("--mh_pop_size", type=int, default=100, help='Population size')
parser.add("--mh_tournament_size", type=int, default=10, help='Tournament size')


class Population(List[Solution]):
    def __init__(self, sol: Solution, meths_ch: List[Method], own_settings: dict = None):
        super().__init__(self)
        self.meths_ch = meths_ch
        self.own_settings = own_settings

        meths_cycle = cycle(self.meths_ch)

        # cycle through construction heuristics to generate population
        # perform all construction heuristics, take best solution
        while len(self) < self.own_settings.mh_pop_size:
            m = next(meths_cycle)
            sol = sol.copy()
            res = Result()
            m.func(sol, m.par, res)
            self.append(sol)

            if res.terminate:
                break

    def best(self):
        """Get index of best individual
        """
        best = 0
        for i in range(len(self)):
            if self[i].is_better(self[best]):
                best = i

        return best

    def worst(self):
        """Get index of worst individual
        """
        worst = 0
        for i in range(len(self)):
            if self[i].is_worse(self[worst]):
                worst = i

        return worst

    def selection(self):
        """Tournament selection.
        """
        k = self.own_settings.mh_tournament_size

        best = random.randint(1, len(self) - 1)

        for i in range(k - 1):
            individual = random.randint(1, len(self) - 1)
            if self[individual].is_better(self[best]):
                best = individual

        return best
