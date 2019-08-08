"""A generic solution class for solutions that are represented by permutations of integers."""

import numpy as np
from abc import ABC

from mhlib.solution import VectorSolution
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

    def two_exchange_neighborhood_search(self, best_improvement) -> bool:
        """Perform the systematic search of the 2-exchange neighborhood, in which two elements are exchanged.

        The neighborhood is searched in a randomized ordering.
        Note that frequently, a more problem-specific neighborhood search with delta-evaluation is
        much more efficient!

        :param best_improvement:  if set, the neighborhood is completely searched and a best neighbor is kept;
            otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
            better solution.

        :return: True if an improved solution has been found
        """
        n = self.inst.n
        best_obj = orig_obj = self.obj()
        best_p1 = None
        best_p2 = None
        order = np.arange(n)
        np.random.shuffle(order)
        for idx, p1 in enumerate(order[:n - 1]):
            for p2 in order[idx + 1:]:
                self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
                if self.two_exchange_delta_eval(p1, p2):
                    if self.is_better_obj(self.obj(), best_obj):
                        if not best_improvement:
                            return True
                        best_obj = self.obj()
                        best_p1 = p1
                        best_p2 = p2
                    self.x[p1], self.x[p2] = self.x[p2], self.x[p1]
                    self.obj_val = orig_obj
                    assert self.two_exchange_delta_eval(p1, p2, False)
        if best_p1:
            self.x[best_p1], self.x[best_p2] = self.x[best_p2], self.x[best_p1]
            self.obj_val = best_obj
            return True
        self.obj_val = orig_obj
        return False

    def two_exchange_delta_eval(self, p1: int, p2: int, update_obj_val=True, allow_infeasible=False) -> bool:
        """A 2-exchange move was performed, if feasible update other solution data accordingly, else revert.

        It can be assumed that the solution was in a correct state with a valid objective value in obj_val
        *before* the already applied move, obj_val_valid therefore is True.
        The default implementation just calls invalidate() and returns True.

        :param p1: first position
        :param p2: second position
        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
            the update of other data done
        """
        if update_obj_val:
            self.invalidate()
        return True

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

        TODO: revise

        :param other: second parent
        :return: new offspring solution
        """
        pos_a = {}
        for i in range(0, len(self.x)):
            pos_a[self.x[i]] = i

        # Detect cycles
        group = np.full(len(self.x), -1)

        group_id = 0
        for i in range(0, len(self.x)):
            if group[i] != -1:
                # Position already in a cycle
                continue

            # Create a new cycle
            pos = i
            while group[pos] == -1:
                # Element at pos i is not yet assigned to a group
                group[pos] = group_id
                sym = other.x[pos]
                pos = pos_a[sym]

            # sanity check
            assert pos == i
            group_id += 1

        # Perform exchange
        child = self.copy()

        for pos in range(0, len(child.x)):
            if child[pos] % 2 == 0:
                continue

            child.x[pos] = other.x[pos]
        child.invalidate()
        return child

    def edge_recombination(self, other: 'PermutationSolution') -> 'PermutationSolution':
        """ Edge recombination.

        This is a classical recombination operator for the traveling salesman problem, for example.
        It creates an adjacency list, i.e., a list of neighbors in the cyclically viewed parent permutations,
        for each element.
        A start element is randomly chosen and removed from the adjacency lists.
        From this current element the next is iteratively determined by either choosing an element randomly
        from the element's adjacency list, or, if the list is empty, by choosing some other not yet visited element.

        :param other: second parent
        :return new offspring solution

        TODO: revise
        """
        nbh = {}

        for i in range(0, len(self.x)):
            elem_a = self.x[i]

            if elem_a not in nbh:
                nbh[elem_a] = []

            nbh[elem_a].append(self.x[i - 1])
            pos_next = i + 1 if i + 1 < len(self.x) else 0
            nbh[elem_a].append(self.x[pos_next])

            elem_b = other.x[i]

            if elem_b not in nbh:
                nbh[elem_b] = []

            nbh[elem_b].append(other.x[i - 1])
            pos_next = i + 1 if i + 1 < len(other.x) else 0
            nbh[elem_b].append(other.x[pos_next])

        x = []
        start = random.randrange(len(self.x))
        x.append(start)

        while len(x) < len(self.x):
            for i in nbh:
                while start in nbh[i]:
                    nbh[i].remove(start)

            if len(nbh[start]) > 0:
                # determine bh of x that has fewest neighbors
                choices = [nbh[start][0]]
                for i in nbh[start]:
                    elem = choices[0]
                    if len(nbh[i]) < len(nbh[elem]):
                        choices = [i]  # new minimum found
                    elif len(nbh[i]) == len(nbh[elem]):
                        choices.append(elem)  # add equal choice

                start = random.choice(choices)
            else:
                # find random not in child
                tries = 0
                while start in x:
                    tries += 1
                    start = random.randrange(len(self.x))

            x.append(start)

        child = self.copy()
        assert len(child.x) == len(x)
        child.x = x
        child.invalidate()
        return child

