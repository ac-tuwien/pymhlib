
from typing import List
from itertools import cycle
import random
from statistics import stdev

import numpy as np

from pymhlib.scheduler import Method, Result
from pymhlib.settings import get_settings_parser, settings, OwnSettings, boolArg
from pymhlib.solution import Solution


parser = get_settings_parser()
parser.add_argument("--mh_pop_size", type=int, default=100, help='Population size')
parser.add_argument("--mh_pop_tournament_size", type=int, default=10, help='Group size in tournament selection')
parser.add_argument("--mh_pop_dupelim", type=boolArg, default=True, help='Prevent duplicates in population')


class Population(np.ndarray):
    """ Maintains a set of solutions, called a population and provides elementary methods.

    Attributes
        - own_settings: own settings object with possibly individualized parameter values
    """

    def __new__(cls, sol: Solution, meths_ch: List[Method], own_settings: dict = None):
        """Create population of mh_pop_size solutions using the list of construction heuristics if given.

        If sol is None or no constructors are given, the population is initialized empty.
        sol itself is just used as template for obtaining further solutions.
        """
        own_settings = OwnSettings(own_settings) if own_settings else settings
        size = own_settings.mh_pop_size
        obj = super(Population, cls).__new__(cls, size, Solution)
        obj.own_settings = own_settings
        if sol is not None and meths_ch:
            # cycle through construction heuristics to generate population
            # perform all construction heuristics, take best solution
            meths_cycle = cycle(meths_ch)
            idx = 0
            while idx < size:
                m = next(meths_cycle)
                sol = sol.copy()
                res = Result()
                m.func(sol, m.par, res)
                if own_settings.mh_pop_dupelim and obj.duplicates_of(sol) != []:
                    continue  # do not add this duplicate
                obj[idx] = sol
                if res.terminate:
                    break
                idx += 1
        return obj

    def best(self) -> int:
        """Get index of best solution."""
        best = 0
        for i in range(len(self)):
            if self[i].is_better(self[best]):
                best = i
        return best

    def worst(self) -> int:
        """Get index of worst solution."""
        worst = 0
        for i in range(len(self)):
            if self[i].is_worse(self[worst]):
                worst = i
        return worst

    def tournament_selection(self) -> int:
        """Select one solution with tournament selection with replacement and return its index."""
        k = self.own_settings.mh_pop_tournament_size
        best = random.randrange(len(self))
        for i in range(k - 1):
            idx = random.randrange(len(self))
            if self[idx].is_better(self[best]):
                best = idx
        return best
    
    def select(self) -> int:
        """Select one solution and return its index.
        
        So far calls tournament_selection. May be extended in the future.
        """
        return self.tournament_selection()

    def duplicates_of(self, solution: Solution) -> List[int]:
        """ Get a list of duplicates of the provided solution."""
        return [i for i, sol in enumerate(self) if sol is not None and sol == solution]

    def obj_avg(self) -> float:
        """ Returns the average of all solutions' objective values."""
        if len(self) < 1:
            raise ValueError("average requires at least one element")

        return sum([float(sol.obj()) for sol in self]) / len(self)

    def obj_std(self) -> float:
        """ Returns the standard deviation of all solutions' objective values."""
        return stdev([float(sol.obj()) for sol in self])
