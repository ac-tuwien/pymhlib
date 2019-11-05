"""A population-based iterated greedy (PBIG) algorithm.
.
The parameter is a method that randomly creates a solution, which will be used to generate an initial population.
Until a termination criterion is met, the list of d&r (destroy and recreate) methods
is applied to each individual of the population, resulting in a temporary population.
The best of the current and the temporary iteration form a new generation for which the process will be repeated.
"""

from typing import List
from itertools import cycle

from pymhlib.population import Population
from pymhlib.scheduler import Method, Scheduler
from pymhlib.solution import Solution


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
        population = Population(sol, meths_ch, own_settings)
        super().__init__(sol, meths_ch+meths_dr, own_settings, population=population)
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
                        self.incumbent = modified  # update best solution

            # Add new individuals to population and take the best
            for individual in changed:
                worst = population.worst()
                if individual.is_better(population[worst]):
                    population[worst] = individual
