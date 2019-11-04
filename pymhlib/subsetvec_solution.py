"""A generic class for solutions that are arbitrary cardinality subsets of a given set represented in vector form."""

import numpy as np
import random
from abc import ABC
from typing import Union
from itertools import chain

from pymhlib.solution import VectorSolution, Solution


class SubsetVectorSolution(VectorSolution, ABC):
    """Generic class for solutions that are arbitrary cardinality subsets of a given set represented in vector form.

    Attributes
        - all_elements: complete set of which a subset shall be selected
        - sel: number of selected elements
        - x: array with elements, where x[:sel] are the *sorted* selected ones;
            if unselected_elems_in_x returns True, all not selected elements are maintained in x[sel:]
    """
    def __init__(self, all_elements, inst=None, alg=None, init=True):
        """Initialize empty solution.

        :param all_elements: complete set of elements
        :param inst: associated problem instance to store
        :param alg: associated algorithm object to store
        :param init: if False the solution is not initialized
        """
        self.all_elements = set(all_elements)
        super().__init__(len(self.all_elements), dtype=self.dtype(), inst=inst, alg=alg, init=False)
        self.sel = 0
        if init:
            if self.unselected_elems_in_x():
                self.x = np.fromiter(self.all_elements, self.dtype(), len(self.all_elements))
            else:
                self.x = np.empty(len(self.all_elements), dtype=self.dtype())

    @classmethod
    def unselected_elems_in_x(cls) -> bool:
        """Return True if the unselected elements are maintained in x[sel:], i.e., behind the selected ones."""
        return True

    def dtype(self):
        """Return type of the elements."""
        elem = None
        for elem in self.all_elements:
            break
        return type(elem)

    def copy_from(self, other: 'SubsetVectorSolution'):
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

    def __eq__(self, other: 'SubsetVectorSolution') -> bool:
        return self.obj() == other.obj() and np.array_equal(self.x[:self.sel], other.x[:other.sel])

    def initialize(self, k):
        """Random construction of a new solution by applying fill to an initially empty solution."""
        self.clear()
        self.fill(None if self.unselected_elems_in_x() else list(self.all_elements))
        self.invalidate()

    def check(self, unsorted=False):
        """Check correctness of solution; throw an exception if error detected.

        :param unsorted: if set, it is not checked if s is sorted
        """
        length = len(self.all_elements)
        if not 0 <= self.sel <= length:
            raise ValueError(f"Invalid attribute sel in solution: {self.sel}")
        if len(self.x) != length:
            raise ValueError(f"Invalid length of solution array x: {self.x}")
        if self.unselected_elems_in_x():
            if set(self.x) != self.all_elements:
                raise ValueError(f"Invalid solution - x is not a permutation of V: {self.x}"
                                 " (sorted: {sorted(self.x)})")
        else:
            sol_set = set(self.x[:self.sel])
            if not sol_set.issubset(self.all_elements) or len(sol_set) != self.sel:
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

    def fill(self, pool: Union[list, np.ndarray] = None, random_order: bool = True) -> int:
        """Scans elements from pool (by default in random order) and selects those whose inclusion is feasible.

        Elements in pool must not yet be selected.
        if unselected_elems_in_x() is true, parameter pool must either be None, in which case x[sel:] is used as pool,
        or x[sel:_] for some _ > sel.
        If random_order is set, the elements in the pool are processed in random order.
        Uses element_added_delta_eval() which should be properly overloaded.
        Reorders elements in pool so that the selected ones appear in pool[:return-value].
        """
        if not self.may_be_extendible():
            return 0
        x = self.x
        if pool is None:
            pool = x[self.sel:]
        assert not self.unselected_elems_in_x() or np.may_share_memory(x[self.sel:self.sel+1], pool)
        selected = 0
        for i in range(len(pool)):
            if random_order:
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
        """Search 2-exchange neighborhood followed by fill() with random ordering.

        Each selected location is tried to be exchanged with each unselected one followed by a fill().

        The neighborhood is searched in a randomized fashion.
        Overload the methods element_removed_delta_eval and element_added_delta_eval for an efficient problem-specific
        delta evaluation.
        Returns True if the solution could be improved, otherwise the solution remains unchanged.
        """
        sel = self.sel
        x = self.x
        orig_obj = self.obj()
        self_backup = None
        x_sel_orig = x[:sel].copy()
        random.shuffle(x[:sel])
        best = self.copy()
        num_neighbors = 0
        for i, v in enumerate(x[:sel]):
            if i != sel-1:
                x[i], x[sel-1] = x[sel-1], x[i]
            self.sel -= 1
            self.element_removed_delta_eval(allow_infeasible=True)
            obj1 = self.obj()
            pool = self.get_extension_pool()
            random.shuffle(pool)
            v_pos = np.nonzero(pool == v)[0][0]
            if v_pos:
                pool[0], pool[v_pos] = pool[v_pos], pool[0]
            for j, vu in enumerate(pool[1:]):
                x[sel-1], pool[j+1] = vu, x[sel-1]
                self.sel += 1
                num_neighbors += 1
                if self.element_added_delta_eval():
                    # neighbor is feasible
                    random_fill_applied = False
                    if self.may_be_extendible():
                        self_backup = self.copy()
                        self.fill(self.get_extension_pool())
                        random_fill_applied = True
                    if self.is_better(best):
                        # new best solution found
                        if not best_improvement:
                            self.sort_sel()
                            return True
                        best.copy_from(self)
                    if random_fill_applied:
                        if i != self.sel:
                            x[i], x[sel-1] = x[sel-1], x[i]
                        self.copy_from(self_backup)
                    self.sel -= 1
                    self.element_removed_delta_eval(update_obj_val=False, allow_infeasible=True)
                    self.obj_val = obj1
                x[sel-1], pool[j+1] = pool[j+1], vu
            self.sel += 1
            self.element_added_delta_eval(update_obj_val=False, allow_infeasible=True)
            self.obj_val = orig_obj
            if i != sel-1:
                x[i], x[sel-1] = x[sel-1], x[i]
        if self.is_better_obj(best.obj(), orig_obj):
            # return new best solution
            self.copy_from(best)
            self.sort_sel()
            return True
        x[:sel] = x_sel_orig
        return False

    def subset_crossover(self, other: 'SubsetVectorSolution') -> 'SubsetVectorSolution':
        """Performs a general crossover operation on two subset solutions.

        A new child solution is constructed by considering all elements in the parent solutions in random order.
        If feasible an element gets added, otherwise it will not be present in the child solution.
        Finally, also all elements that do not appear in the parents are also considered for inclusion in random order.

        :param other: second parent for crossover
        :return: a new child solution
        """
        parent_elems = set(self.x[:self.sel]).union(other.x[:other.sel])
        child = self.copy()
        child.clear()
        for i, elem in enumerate(chain(parent_elems, self.all_elements - parent_elems)):
            child.x[i] = elem
        np.random.shuffle(child.x[:len(parent_elems)])
        np.random.shuffle(child.x[len(parent_elems):])
        child.fill(random_order=False)
        child.invalidate()
        return child

    # Methods to be specialized for efficient move calculations

    def get_extension_pool(self):
        """Return a list of yet unselected elements that may possibly be added."""
        if self.unselected_elems_in_x():
            return self.x[self.sel:]
        else:
            return list(set(self.all_elements) - set(self.x[:self.sel]))

    def may_be_extendible(self) -> bool:
        """Quick check if the solution has chances to be extended by adding further elements."""
        return self.sel < len(self.x)

    def element_removed_delta_eval(self, update_obj_val=True, allow_infeasible=False) -> bool:
        """Element x[sel] has been removed in the solution, if feasible update other solution data,
        else revert.

        This is a helper function for delta-evaluating solutions when searching a neighborhood that needs
        to be overloaded for a concrete problem.
        It can be assumed that the solution was in a correct state with a valid objective value in obj_val
        *before* the already applied move, obj_val_valid therefore is True.
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

        This is a helper function for delta-evaluating solutions when searching a neighborhood that needs
        to be overloaded for a concrete problem.
        It can be assumed that the solution was in a correct state with a valid objective value in obj_val
        *before* the already applied move, obj_val_valid therefore is True.
        The default implementation just calls invalidate() and returns True.

        :param update_obj_val: if set, the objective value should also be updated or invalidate needs to be called
        :param allow_infeasible: if set and the solution is infeasible, the move is nevertheless accepted and
            the update of other data done
        :return: True if feasible, False if infeasible
        """
        if update_obj_val:
            self.invalidate()
        return True
