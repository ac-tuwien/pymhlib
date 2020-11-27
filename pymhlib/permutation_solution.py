"""A generic solution class for solutions that are represented by permutations of integers."""

import numpy as np
from abc import ABC
from typing import List, Tuple

from pymhlib.solution import VectorSolution, TObj
import random


class PermutationSolution(VectorSolution, ABC):
    """Solution that is represented by a permutation of 0,...length-1."""

    def __init__(self, length: int, init=True, **kwargs):
        """Initializes the solution with 0,...,length-1 if init is set."""
        super().__init__(length, init=False, **kwargs)
        if init:
            self.x[:] = np.arange(length)

    def copy_from(self, other: 'PermutationSolution'):
        super().copy_from(other)

    def initialize(self, k):
        """Random initialization."""
        np.random.shuffle(self.x)
        self.invalidate()

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
        """
        super().check()
        if set(self.x) != set(range(len(self.x))):
            raise ValueError("Solution is no permutation of 0,...,length-1")

    def apply_two_exchange_move(self, p1: int, p2: int):
        """The values at positions p1 and p2 are exchanged in self.x.

        Note that the obj_val is not changed here nor is invalidate() called yet, as it is
        assumed that obj_val is updated by a corresponding delta evaluation.
        """
        x = self.x
        x[p1], x[p2] = x[p2], x[p1]

    def two_exchange_neighborhood_search(self, best_improvement) -> bool:
        """Perform the systematic search of the 2-exchange neighborhood, in which two elements are exchanged.

        The neighborhood is searched in a randomized ordering.
        A problem-specific delta-evaluation can be performed by overloading two_exchange_move_delta_eval.

        :param best_improvement:  if set, the neighborhood is completely searched and a best neighbor is kept;
            otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
            better solution.

        :return: True if an improved solution has been found
        """
        n = self.inst.n
        best_delta = 0
        best_p1 = None
        best_p2 = None
        order = np.arange(n)
        np.random.shuffle(order)
        for idx, p1 in enumerate(order[:n - 1]):
            for p2 in order[idx + 1:]:
                # consider exchange of positions p1 and p2
                delta = self.two_exchange_move_delta_eval(p1, p2)
                if self.is_better_obj(delta, best_delta):
                    if not best_improvement:
                        self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
                        self.obj_val += delta
                        return True
                    best_delta = delta
                    best_p1 = p1
                    best_p2 = p2
        if best_p1:
            self.x[best_p1], self.x[best_p2] = self.x[best_p2], self.x[best_p1]
            self.obj_val += best_delta
            return True
        return False

    def two_exchange_move_delta_eval(self, p1: int, p2: int) -> TObj:
        """Return delta value in objective when exchanging positions p1 and p2 in self.x.

        The solution is not changed.
        This is a helper function for delta-evaluating solutions when searching a neighborhood that should
        be overloaded with a more efficient implementation for a concrete problem.
        Here we perform the move, calculate the objective value from scratch and revert the move.

        :param p1: first position
        :param p2: second position
        """
        obj = self.obj()
        x = self.x
        x[p1], x[p2] = x[p2], x[p1]
        self.invalidate()
        delta = self.obj() - obj
        x[p1], x[p2] = x[p2], x[p1]
        self.obj_val = obj
        return delta

    def random_two_exchange_move_delta_eval(self) -> Tuple[Tuple[int, int], TObj]:
        """Choose random move in the two-exchange neighborhood and perform delta eval., returning (p1, p2, delta_obj).

        The solution is not changed here yet.
        Primarily used in simulated annealing.
        """
        p1 = random.randint(0, len(self.x) - 2)
        p2 = random.randint(p1 + 1, len(self.x) - 1)
        delta_obj = self.two_exchange_move_delta_eval(p1, p2)
        return (p1, p2), delta_obj

    def two_opt_neighborhood_search(self, best_improvement) -> bool:
        """Systematic search of the 2-opt neighborhood, i.e., consider all inversions of subsequences.

        The neighborhood is searched in a randomized ordering.

        :param best_improvement:  if set, the neighborhood is completely searched and a best neighbor is kept;
            otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
            better solution.

        :return: True if an improved solution has been found
        """
        n = self.inst.n
        best_delta = 0
        best_p1 = None
        best_p2 = None
        order = np.arange(n)
        np.random.shuffle(order)
        for idx, p1 in enumerate(order[:n - 1]):
            for p2 in order[idx + 1:]:
                pa, pb = (p1, p2) if p1 < p2 else (p2, p1)
                # consider the move that self.x is inverted from position p1 to position p2
                delta = self.two_opt_move_delta_eval(pa, pb)
                if self.is_better_obj(delta, best_delta):
                    if not best_improvement:
                        self.apply_two_opt_move(pa, pb)
                        self.obj_val += delta
                        return True
                    best_delta = delta
                    best_p1 = pa
                    best_p2 = pb
        if best_p1:
            self.apply_two_opt_move(best_p1, best_p2)
            self.obj_val += best_delta
            return True
        return False

    def apply_two_opt_move(self, p1: int, p2: int):
        """The subsequence from p1 to p2 is inverted in self.x.

        Note that the obj_val is not changed here nor is invalidate() called yet, as it is
        assumed that obj_val is updated by a corresponding delta evaluation.
        """
        self.x[p1:(p2 + 1)] = self.x[p1:(p2 + 1)][::-1]

    def two_opt_move_delta_eval(self, p1: int, p2: int) -> int:
        """ Return the delta in the objective value when inverting self.x from position p1 to position p2.

        The function returns the difference in the objective function if the move would be performed,
        the solution, however, is not changed.
        This function should be overwritten in a concrete class.
        Here we actually perform a less efficient complete evaluation of the modified solution.
        """
        orig_obj = self.obj_val
        self.apply_two_opt_move(p1, p2)
        self.invalidate()
        delta = self.obj() - orig_obj
        self.apply_two_opt_move(p1, p2)
        self.obj_val = orig_obj
        return delta

    def random_two_opt_move_delta_eval(self) -> Tuple[Tuple[int, int], TObj]:
        """Choose random move in 2-opt neighborhood and perform delta evaluation, returning (move, delta_obj).

        The solution is not changed here yet.
        Primarily used in simulated annealing.
        """
        p1 = random.randrange(len(self.x)-1)
        p2 = random.randint(p1+1, len(self.x)-1)
        delta_obj = self.two_opt_move_delta_eval(p1, p2)
        return (p1, p2), delta_obj

    def partially_mapped_crossover(self, other: 'PermutationSolution') -> 'PermutationSolution':
        """Partially mapped crossover (PMX).

        Copies the current solution, selects a random subsequence from the other parent and realizes this subsequence
        in the child by corresponding pairwise exchanges.

        :param other: second parent
        :return: new offspring solution
        """

        size = len(self.x)

        # determine random subsequence
        begin = random.randrange(size)
        end = random.randrange(size - 1)
        if begin == end:
            end = end + 1
        if begin > end:
            begin, end = end, begin

        child = self.copy()

        # adopt subsequence from parent b by corresponding pairwise exchanges
        pos = np.empty(size, int)
        for i, elem in enumerate(child.x):
            pos[elem] = i
        for i in range(begin, end):
            elem = other.x[i]
            j = pos[elem]
            if i != j:
                elem_2 = child.x[i]
                child.x[i], child.x[j] = elem, elem_2
                pos[elem], pos[elem_2] = i, j
        child.invalidate()
        return child

    def cycle_crossover(self, other: 'PermutationSolution') -> 'PermutationSolution':
        """ Cycle crossover.

        A randomized crossover method that adopts absolute positions of the elements from the parents.

        :param other: second parent
        :return: new offspring solution
        """
        size = len(self.x)
        pos = np.empty(size, int)
        for i, elem in enumerate(self.x):
            pos[elem] = i

        # detect all cycles
        group = np.full(size, 0)
        group_id = 1
        for i in range(size):
            if group[i]:
                continue
            j = i
            while not group[j]:
                group[j] = group_id
                elem = other.x[j]
                j = pos[elem]
            group_id += 1

        # perform exchange
        child = self.copy()
        for i in range(size):
            if child.x[i] % 2 == 1:
                child.x[pos] = other.x[pos]
        child.invalidate()
        return child

    def edge_recombination(self, other: 'PermutationSolution') -> 'PermutationSolution':
        """ Edge recombination.

        This is a classical recombination operator for the traveling salesman problem, for example.
        It creates an adjacency list, i.e., a list of neighbors in the cyclically viewed parent permutations,
        for each element.
        A start element is randomly chosen.
        From this current element the next is iteratively determined by either choosing a neighbor with the smallest
        adjacency list (ties are broken randomly), or, if the list is of remaining neighbors is empty,
        by choosing some other not yet visited element at random.

        :param other: second parent
        :return new offspring solution
        """
        def append_if_not_contained(nbs, nb):
            if nb not in nbs:
                nbs.append(nb)
        size = len(self.x)
        adj_lists: List[List[int]] = [list() for _ in range(size)]
        for i, elem in enumerate(self.x):
            append_if_not_contained(adj_lists[elem], self.x[(i-1) % size])
            append_if_not_contained(adj_lists[elem], self.x[(i+1) % size])
        for i, elem in enumerate(other.x):
            append_if_not_contained(adj_lists[elem], other.x[(i-1) % size])
            append_if_not_contained(adj_lists[elem], other.x[(i+1) % size])
        unvisited = set(range(size))
        child = self.copy()
        elem = random.randrange(size)
        for i in range(size-1):
            # accept elem and remove it from unvisited and adjacency list
            child.x[i] = elem
            unvisited.remove(elem)
            for j in adj_lists[elem]:
                adj_lists[j].remove(elem)
            # select next elem
            if not adj_lists[elem]:
                sel = random.choice(list(unvisited))
            else:
                candidates = [adj_lists[elem][0]]
                degree = len(adj_lists[candidates[0]])
                for e2 in adj_lists[elem][1:]:
                    degree_e2 = len(adj_lists[e2])
                    if degree_e2 < degree:
                        candidates = [e2]
                    elif degree_e2 == degree:
                        candidates.append(e2)
                sel = random.choice(candidates)
                adj_lists[elem].clear()
            elem = sel
        child.x[-1] = elem
        child.invalidate()
        return child
