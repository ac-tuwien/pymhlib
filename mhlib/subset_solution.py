"""A generic solution class for solutions that are arbitrary cardinality subsets of a given set."""

import numpy as np
import random
from abc import ABC

from mhlib.solution import VectorSolution, Solution


class SubsetSolution(VectorSolution, ABC):
    """Generic class for solutions that are arbitrary cardinality subsets of a given set.

    Attributes
        - all_elements: complete set of which a subset shall be selected
        - sel: number of selected elements
        - x: array with elements, where x[:sel] are the sorted selected ones;
            if unselected_elems_in_x returns True, all not selected elements are maintained in x[sel:]
    """
    def __init__(self, all_elements, inst=None, alg=None, init=True):
        """Initialize empty solution.

        :param all_elements: complete set of elements, may be a generator
        :param inst: associated problem instance to store
        :param alg: associated algorithm object to store
        :param init: if False the solution is not initialized
        """
        dtype = type(next(iter(all_elements)))
        super().__init__(len(all_elements), dtype=dtype, inst=inst, alg=alg, init=False)
        self.all_elements = all_elements
        self.sel = 0
        if init:
            if self.unselected_elems_in_x():
                self.x = np.array(all_elements)
            else:
                self.x = np.empty(len(all_elements), dtype=dtype)

    @classmethod
    def unselected_elems_in_x(cls) -> bool:
        """Return True if the unselected elements are maintained in x[sel:], i.e., behind the selected ones."""
        return True

    def copy_from(self, other: 'SubsetSolution'):
        self.sel = other.sel
        if self.unselected_elems_in_x():
            super().copy_from(other)
        else:
            self.x[:self.sel] = other.x[:self.sel]
            Solution.copy_from(self, other)

    def __repr__(self):
        return str(self.x[:self.sel])

    def clear(self):
        self.sel = 0
        self.invalidate()

    def __eq__(self, other: 'SubsetSolution') -> bool:
        return self.obj() == other.obj() and self.x[:self.sel] == other.x[:other.sel]

    def initialize(self, k):
        """Random construction of a new solution by applying random_fill to an initially empty solution."""
        self.clear()
        self.random_fill(self.x if self.unselected_elems_in_x() else list(self.all_elements))
        self.check()  # TODO finally remove

    def check(self, unsorted=False):
        """Check correctness of solution; throw an exception if error detected.

        :param unsorted: if set, it is not checked if s is sorted
        """
        all_elements_set = set(self.all_elements)
        length = len(all_elements_set)
        if not 0 <= self.sel <= length:
            raise ValueError(f"Invalid attribute sel in solution: {self.sel}")
        if len(self.x) != length:
            raise ValueError(f"Invalid length of solution array x: {self.x}")
        if unsorted:
            if self.unselected_elems_in_x():
                if set(self.x) != all_elements_set:
                    raise ValueError(f"Invalid solution - x is not a permutation of V: {self.x}"
                                     " (sorted: {sorted(self.x)})")
            else:
                sol_set = set(self.x[:self.sel])
                if not sol_set.issubset(set(self.all_elements)) or len(sol_set) != self.sel:
                    raise ValueError(f"Solution not simple subset of V: {self.x[:self.sel]}, {self.all_elements}")
        if not unsorted:
            old_v = self.x[0]
            for v in self.x[1:self.sel]:
                if v <= old_v:
                    raise ValueError(f"Solution not sorted: value {v} in {self.x}")
                old_v = v
        super().check()

    def sort_sel(self):
        """Sort selected elements in x."""
        self.x[:self.sel].sort()

    def random_fill(self, pool: list) -> int:
        """Scans elements from pool in random order and selects those whose inclusion is feasible.

        The pool may be x[sel:].
        Elements in pool must not yet be selected.
        Uses element_added_delta_eval() which should be properly overloaded.
        Reorders elements in pool so that the selected ones appear in pool[: return-value].
        """
        if not self.may_be_extendible():
            return 0
        x = self.x
        selected = 0
        for i in range(len(pool)):
            ir = random.randrange(i, len(pool))
            if selected != ir:
                pool[selected], pool[ir] = pool[ir], pool[selected]
            x[self.sel] = pool[selected]
            self.sel += 1
            if self.element_added_delta_eval():
                selected += 1
                if not self.may_be_extendible():
                    break
        if selected:
            self.sort_sel()
        return selected

    def remove_some(self, k):
        """Removes min(k,sel) randomly selected elements from the solution.

        Uses element_removed_delta_eval, which should be overloaded and adapted to the problem.
        The elements are removed even when the solution becomes infeasible.
        """
        x = self.x
        k = min(k, self.sel)
        if k > 0:
            for i in range(k):
                j = random.randrange(self.sel)
                self.sel -= 1
                if j != self.sel:
                    x[j], x[self.sel] = x[self.sel], x[j]
                self.element_removed_delta_eval(allow_infeasible=True)
            self.sort_sel()

    def two_exchange_random_fill_neighborhood_search(self, best_improvement) -> bool:
        """Search 2-exchange neighborhood followed by random_fill.

        Each selected location is tried to be exchanged with each unselected one followed by a random_fill().

        The neighborhood is searched in a randomized fashion.
        Overload two_exchange_delta_eval for problem-specific efficient delta evaluation.
        Returns True if the solution could be improved, otherwise the solution remains unchanged.
        """
        sel = self.sel
        x = self.x
        num_neighbors = 0
        random.shuffle(x[:sel])
        random.shuffle(x[sel:])
        orig = self.copy()
        best = self.copy()
        for i, v in enumerate(x[:sel]):
            for j, vu in enumerate(x[sel:]):
                x[i], x[sel+j] = vu, v
                num_neighbors += 1
                if self.two_exchange_delta_eval(i, sel+j):
                    # neighbor is feasible
                    random_fill_applied = False
                    if self.may_be_extendible():
                        assert self.unselected_elems_in_x()  # TODO update to work with external pool
                        self.random_fill(self.x[sel:])
                        random_fill_applied = True
                    if self.is_better(best):
                        # new best solution found
                        if not best_improvement:
                            self.sort_sel()
                            return True
                        best.copy_from(self)
                    if random_fill_applied:
                        self.copy_from(orig)
                    else:
                        x[i], x[sel+j] = v, vu
                        assert self.two_exchange_delta_eval(i, sel+j, False)
                        self.obj_val = orig.obj()
        if best.is_better(orig):
            # return new best solution
            self.copy_from(best)
            self.sort_sel()
            return True
        return False

    # Methods to be specialized for efficient move calculations

    def may_be_extendible(self) -> bool:
        """Quick check if the solution has chances to be extended by adding further elements."""
        return self.sel < len(self.x)

    def element_removed_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        """Element x[sel] has been removed in the solution, if feasible update other solution data, else revert.

        It can be assumed that the solution was in a correct state with a valid objective value before the move.
        The default implementation just calls invalidate() and returns True.
        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
            the update of other data done
        :return: True if feasible, False if infeasible
        """
        if update_obj_val:
            self.invalidate()
        return True

    def element_added_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        """Element x[sel-1] was added to a solution, if feasible update further solution data, else revert.

        It can be assumed that the solution was in a correct state with a valid objective value before the move.
        The default implementation just calls invalidate() and returns True.
        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
            the update of other data done
        :return: True if feasible, False if infeasible
        """
        if update_obj_val:
            self.invalidate()
        return True

    def two_exchange_delta_eval(self, p1, p2, update_obj_val=True, allow_infeasible=False) -> bool:
        """A 2-exchange move has been performed, if feasible update other solution data accordingly, else revert.

        It can be assumed that the solution was in a correct state with a valid objective value before the move.
        The default implementation just calls invalidate() and returns True.
        :param p1: first exchanged position; x[p1] was added
        :param p2: second exchanged position; x[p2] was removed
        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
            the update of other data done
        :return: True if feasible, False if infeasible
        """
        if update_obj_val:
            self.invalidate()
        return True
