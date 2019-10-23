"""A generic class for solutions that are represented by fixed-length binary vectors."""

import numpy as np
from abc import ABC

from mhlib.solution import VectorSolution


class BinaryVectorSolution(VectorSolution, ABC):
    """Abstract solution class with fixed-length 0/1 vector as solution representation.

    Attributes
        - x: 0/1 vector representing a solution
    """

    def __init__(self, length, **kwargs):
        """Initializes the solution vector with zeros."""
        super().__init__(length, dtype=bool, **kwargs)

    def initialize(self, k):
        """Random initialization."""
        self.x = np.random.randint(0, 2, len(self.x))
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

        TODO allow incremental evaluation
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
                self.invalidate()
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
                    x[perm[p[i]]] = not x[perm[p[i]]]
                    i += 1  # continue with next position (if any)
                elif p[i] < len(x) - (k - i):
                    # further positions to explore with this index
                    x[perm[p[i]]] = not x[perm[p[i]]]
                    p[i] += 1
                    x[perm[p[i]]] = not x[perm[p[i]]]
                    i += 1
                else:
                    # we are at the last position with the i-th index, backtrack
                    x[perm[p[i]]] = not x[perm[p[i]]]
                    p[i] = -1  # unset position
                    i -= 1
        if better_found:
            self.copy_from(best_sol)
            self.invalidate()
        return better_found
