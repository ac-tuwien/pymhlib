"""A population-based iterated greedy search

"""

from typing import List
from itertools import cycle

from mhlib.scheduler import Method, Scheduler
from mhlib.settings import get_settings_parser
from mhlib.solution import Solution


parser = get_settings_parser()
parser.add("--mh_pbig_pop_size", type=int, default=20, help='PBIG population size')


class PBIG(Scheduler):
    """A population-based iterated greedy algorithm.

    Attributes
        - sol: solution object, in which final result will be returned
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
        self.meths_ch = meths_ch
        self.meths_dr = meths_dr

    def run(self):
        """Actually performs the construction heuristics followed by the PBIG."""

        population: List[Solution] = []

        meths_cycle = cycle(self.meths_ch)

        # Cycle through construction heuristics to generate population
        # perform all construction heuristics, take best solution
        while len(population) < self.own_settings.mh_pbig_pop_size:
            m = next(meths_cycle)
            individual = self.incumbent.copy()
            res = self.perform_method(m, individual)
            population.append(individual)
            if res.terminate:
                return

        meths_dr_cycle = cycle(self.meths_dr)

        terminate = False
        while not terminate:
            best: Solution = self.incumbent

            nextgen: List[Solution] = []
            for individual in population:
                modified = individual.copy()
                meth = next(meths_dr_cycle)
                res = self.perform_method(meth, modified)

                if res.terminate:
                    terminate = True
                    break

                if res.changed and modified.is_better(individual):
                    nextgen.append(modified)

                    # Update population best
                    if modified.is_better(best):
                        best = modified
                else:
                    # Individual was not changed
                    nextgen.append(individual)

            population = nextgen   # Replace old population with new
            self.incumbent = best  # Update best solution
