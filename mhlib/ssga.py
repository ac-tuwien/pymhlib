"""A steady-state genetic algorithm

"""

from typing import List
from itertools import cycle

from mhlib.scheduler import Method, Scheduler
from mhlib.settings import get_settings_parser
from mhlib.permutation_solution import PermutationSolution, cycle_crossover, partial_matched_crossover, \
    edge_recombination

import random

parser = get_settings_parser()
parser.add("--mh_ssga_pop_size", type=int, default=100, help='SSGA population size')
parser.add("--mh_ssga_tournament_size", type=int, default=10, help='SSGA tournament size')
parser.add("--mh_ssga_cross_prob", type=int, default=0.01, help='SSGA crossover probability')

class SSGA(Scheduler):
    """A steady state genetic algorithm.

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meth_sel: a selection method
        - meth_re: a recombination method
        - meth_mu: a mutation method
    """

    def __init__(self, sol: PermutationSolution, meths_ch: List[Method],  # meth_sel: Method, meth_re: Method,
                 meth_mu: Method,
                 own_settings: dict = None):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meth_sel: a selection method
        :param meth_re: a recombination method
        :param meth_mu: a mutation method
        :param own_settings: optional dictionary with specific settings
        """
        #        super().__init__(sol, meths_ch+meth_sel+meth_re+meth_mu, own_settings)
        super().__init__(sol, meths_ch + [meth_mu], own_settings)
        self.meths_ch = meths_ch

        #        self.meth_sel = meth_sel
        #        self.meth_re = meth_re
        self.meth_mu = meth_mu

    def selection(self):
        """Tournament selection.

        :return: a solution
        """
        pop = self.population
        k = self.own_settings.mh_ssga_tournament_size

        best = pop[random.randint(1, len(pop)) - 1]

        for i in range(k - 1):
            indiv = pop[random.randint(1, len(pop)) - 1]
            if indiv.is_better(best):
                best = indiv

        return best

    def run(self):
        """Actually performs the construction heuristics followed by the PBIG."""

        #        population = List[Solution]
        self.population: List[PermutationSolution] = []
        population = self.population

        meths_cycle = cycle(self.meths_ch)

        # TODO check what self.incumbent is initalized to

        # cycle through construction heuristics to generate population
        # perform all construction heuristics, take best solution
        while len(population) < self.own_settings.mh_ssga_pop_size:
            m = next(meths_cycle)
            indiv = self.incumbent.copy()
            res = self.perform_method(m, indiv)
            population.append(indiv)
            population[-1].invalidate()
            if population[-1].is_better(self.incumbent):
                self.incumbent = population[-1].copy()

            if res.terminate:
                break

        while True:
            self.iteration += 1

            # Selection
            parent1 = self.selection()
            parent2 = self.selection()

            # Crossover
            if random.random() < self.own_settings.mh_ssga_cross_prob:
                # TODO find way to make 'Method' receiving two solutions as parameters
                #  and returning a new one (or two)
                # TODO create list of crossover operations for PMX, cycle crossover, ...
                # a = random.randint(0,len(parent1.x) -2)
                # b = random.randint(a+1,len(parent1.x) -1)
                # child1 = partial_matched_crossover(parent1, parent2, range(a,b))
                # child2 = partial_matched_crossover(parent2, parent1, range(a,b))
                # child1, child2 = cycle_crossover(parent1.copy(), parent2.copy())
                child1 = edge_recombination(parent1, parent2)
                child2 = edge_recombination(parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            res1 = self.perform_method(self.meth_mu, child1)
            res2 = self.perform_method(self.meth_mu, child2)

            if res1.terminate or res2.terminate:
                break

            # Replacement
            # Set parent 1 and 2 to the best two
            # of: parent1, parent2, child1, child2
            if child1.is_better(parent1):
                parent1 = child1
            elif child1.is_better(parent2):
                parent2 = child1

            if child2.is_better(parent1):
                parent1 = child2
            elif child2.is_better(parent2):
                parent2 = child2

            if parent1.is_better(self.incumbent):
                self.incumbent = parent1.copy()

            if parent2.is_better(self.incumbent):
                self.incumbent = parent2.copy()

            # Replace best
            # TODO this should not necessary, find out why it is and fix
            best = population[0]
            for indiv in population:
                if indiv.is_better(best):
                    best = indiv

            self.incumbent = best  # update best solution
