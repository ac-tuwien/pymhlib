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
        - meths_li: list of local improvement methods
        - meths_sh: list of shaking methods
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_li: List[Method], meths_sh: List[Method],
                 own_settings: dict = None):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meths_li: list of local improvement methods
        :param meths_sh: list of shaking methods
        :param own_settings: optional dictionary with specific settings
        """
        super().__init__(sol, meths_ch+meths_li+meths_sh, own_settings)
        self.meths_ch = meths_ch
        self.meths_li = meths_li
        self.meths_sh = meths_sh

    def run(self):
        """Actually performs the construction heuristics followed by the PBIG."""

        #        population = List[Solution]
        population: List[Solution] = []

        meths_cycle = cycle(self.meths_ch)

        # cycle through construction heuristics to generate population
        # perform all construction heuristics, take best solution
        while len(population) < self.own_settings.mh_pbig_pop_size:
            m = next(meths_cycle)
            indiv = self.incumbent.copy()
            res = self.perform_method(m, indiv)
            population.append(indiv)
            if res.terminate:
                print(f"Break detected but ignored for now")
                # break

        # TODO already pass in delete and reconstruct methods
        # TODO append all individual methods
        meths_dr: List[Method] = [self.meths_li[0], self.meths_sh[0]]
        meths_dr_cycle = cycle(meths_dr)

        terminate = False
        while not terminate:
            best: Solution = self.incumbent

            nextgen: List[Solution] = []
            for indiv in population:
                mod = indiv.copy()
                m = next(meths_dr_cycle)
                res = self.perform_method(m, mod)

                if res.terminate:
                    print(f"Break detected")  # TODO remove debug output
                    terminate = True
                    break

                if res.changed:
                    if mod.is_better(indiv):
                        nextgen.append(mod)

                        # update population best
                        if mod.is_better(best):
                            best = mod
                    else:
                        nextgen.append(indiv)
                else:
                    nextgen.append(indiv)  # indiv eq mod

            population = nextgen   # replace old population with new
            self.incumbent = best  # update best solution
            print(f"Best {best.obj()}")
