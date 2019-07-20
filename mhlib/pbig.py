"""A population-based iterated greedy (PBIG) algorithm.
.
The parameter is a method that randomly creates a solution, which will be used to generate an initial population.
Until a termination criterion is met, the list of d&r (destroy and recreate) methods
is applied to each individual of the population, resulting in a temporary population.
The best of the current and the temporary iteration form a new generation for which the process will be repeated.
"""

from typing import List
from itertools import cycle
import functools

from mhlib.population import Population
from mhlib.scheduler import Method, Scheduler
from mhlib.settings import get_settings_parser
from mhlib.solution import Solution


parser = get_settings_parser()
parser.add("--mh_pbig_pop_size", type=int, default=20, help='PBIG population size')


class PBIG(Scheduler):
    """A population-based iterated greedy (PBIG) algorithm.

    Attributes
        - sol: solution object, in which final result will be returned
        - population: population of solutions
        - meths_ch: list of construction heuristic methods
        - meths_dr: list of destruct and recreate methods
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_dr: List[Method],
                 own_settings: dict = None):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meths_dr: list of destruct and recreate methods
        :param own_settings: optional dictionary with specific settings
        """
        super().__init__(sol, meths_ch+meths_dr, own_settings)
        self.population = Population(sol, meths_ch, self.own_settings)
        self.meths_ch = meths_ch
        self.meths_dr = meths_dr

    def run(self):
        """Actually performs the construction heuristics followed by the PBIG."""

        population = self.population

        meths_dr_cycle = cycle(self.meths_dr)

        while True:
            changed: List[Solution] = []

            for individual in self.population:
                modified = individual.copy()
                method = next(meths_dr_cycle)
                res = self.perform_method(method, modified)

                if res.terminate:
                    return

                if res.changed:
                    changed.append(modified)

                    # Update population best
                    if modified.is_better(self.incumbent):
                        self.incumbent = modified  # Update best solution

            # Add new individuals to population and take the best
            def compare(lhs: Solution, rhs: Solution):
                return lhs.is_better(rhs)

            population.extend(changed)
            sorted(population, key=functools.cmp_to_key(compare), reverse=True)
            population = population[0:self.own_settings.mh_pbig_pop_size]
