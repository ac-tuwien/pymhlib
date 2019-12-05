"""A generic class for solutions that are represented by fixed-length binary vectors."""

import numpy as np
from abc import ABC
import random
from typing import Tuple

from pymhlib.solution import VectorSolution, TObj


class BinaryVectorSolution(VectorSolution, ABC):
    """Abstract solution class with fixed-length 0/1 vector as solution representation.

    Attributes
        - x: 0/1 vector representing a solution
    """

    def __init__(self, length, **kwargs):
        """Initializes the solution vector with zeros."""
        super().__init__(length, dtype=bool, **kwargs)

    def dist(self, other: 'BinaryVectorSolution'):
        """Return Hamming distance of current solution to other solution."""
        return sum(np.logical_xor(self.x, other.x))

    def initialize(self, k):
        """Random initialization."""
        self.x = np.random.randint(0, 2, len(self.x), dtype=bool)
        self.invalidate()

    def k_random_flips(self, k):
        """Perform k random flips and call invalidate()."""
        for i in range(k):
            p = random.randrange(self.inst.n)
            self.x[p] = not self.x[p]
        self.invalidate()

    def check(self):
        """Check if valid solution.

        Raises ValueError if problem detected.
        """
        super().check()
        for v in self.x:
            if not 0 <= v <= 1:
                raise ValueError("Invalid value in BinaryVectorSolution: {self.x}")

    def k_flip_neighborhood_search(self, k: int, best_improvement: bool) -> bool:
        """Perform one major iteration of a k-flip local search, i.e., search one neighborhood.

        If best_improvement is set, the neighborhood is completely searched and a best neighbor is kept;
        otherwise the search terminates in a first-improvement manner, i.e., keeping a first encountered
        better solution.

        :returns: True if an improved solution has been found.
        """
        x = self.x
        assert 0 < k <= len(x)
        better_found = False
        best_sol = self.copy()
        perm = np.random.permutation(len(x))  # permutation for randomization of enumeration order
        p = np.full(k, -1)  # flipped positions
        # initialize
        i = 0  # current index in p to consider
        while i >= 0:
            # evaluate solution
            if i == k:
                if self.is_better(best_sol):
                    if not best_improvement:
                        return True
                    best_sol.copy_from(self)
                    better_found = True
                i -= 1  # backtrack
            else:
                if p[i] == -1:
                    # this index has not yet been placed
                    p[i] = (p[i-1] if i > 0 else -1) + 1
                    self.flip_variable(perm[p[i]])
                    i += 1  # continue with next position (if any)
                elif p[i] < len(x) - (k - i):
                    # further positions to explore with this index
                    self.flip_variable(perm[p[i]])
                    p[i] += 1
                    self.flip_variable(perm[p[i]])
                    i += 1
                else:
                    # we are at the last position with the i-th index, backtrack
                    self.flip_variable(perm[p[i]])
                    p[i] = -1  # unset position
                    i -= 1
        if better_found:
            self.copy_from(best_sol)
            self.invalidate()
        return better_found

    def flip_variable(self, pos: int):
        """Flip the variable at position pos and possibly incrementally update objective value or invalidate.

        This generic implementation just calls invalidate() after flipping the variable.
        """
        self.x[pos] = not self.x[pos]
        self.invalidate()

    def flip_move_delta_eval(self, pos: int) -> TObj:
        """Determine delta in objective value when flipping position p.

        Here the solution is evaluated from scratch. If possible, it should be overloaded by a more
        efficient delta evaluation.
        """
        obj = self.obj()
        self.x[pos] = not self.x[pos]
        self.invalidate()
        delta = self.obj() - obj
        self.x[pos] = not self.x[pos]
        self.obj_val = obj
        return delta

    def random_flip_move_delta_eval(self) -> Tuple[int, TObj]:
        """Choose random move in the flip neighborhood and perform delta evaluation, returning (move, delta_obj).

        The solution is not changed here yet.
        Primarily used in simulated annealing.
        """
        p = random.randrange(len(self.x))
        delta_obj = self.flip_move_delta_eval(p)
        return p, delta_obj
