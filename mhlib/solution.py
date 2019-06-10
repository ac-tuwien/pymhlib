"""
Abstract class representing a candidate solution to an optimization problem.

For an optimization problem to solve you have to derive from this class.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar

from mhlib.settings import settings, get_settings_parser

parser = get_settings_parser()
parser.add("--mh_maxi", default=True, action='store_true',
           help='maximize the objective function, else minimize')
parser.add("--no_mh_maxi", dest='mh_maxi', action='store_false')

TObj = TypeVar('TObj', int, float)  # Type of objective value


class Solution(ABC):
    """Abstract base class for a candidate solution.

    Attributes
        - obj_val: objective value; valid if obj_val_valid is set
        - obj_val_valid: indicates if obj_val has been calculated and is valid
        - inst: optional reference to an problem instance object
        - alg: optional reference to an algorithm object using this solution
    """

    def __init__(self, inst=None, alg=None):
        self.obj_val: TObj = -1
        self.obj_val_valid: bool = False
        self.inst = inst
        self.alg = alg

    @abstractmethod
    def copy(self):
        """Return a (deep) clone of the current solution."""

    @abstractmethod
    def copy_from(self, other: 'Solution'):
        """Make the current solution a (deep) copy of the other."""
        # self.inst = other.inst
        # self.alg = other.alg
        self.obj_val = other.obj_val
        self.obj_val_valid = other.obj_val_valid

    @abstractmethod
    def __repr__(self):
        return str(self.obj())

    @abstractmethod
    def calc_objective(self) -> TObj:
        """Determine the objective value and return it."""
        raise NotImplementedError

    def obj(self) -> TObj:
        """Return objective value.

        Returns stored value if already known or calls calc_objective() otherwise.
        """
        if not self.obj_val_valid:
            self.obj_val = self.calc_objective()
            self.obj_val_valid = True
        return self.obj_val

    def invalidate(self):
        """Mark the stored objective value obj_val as not valid anymore.

        Needs to be called whenever the solution is changed and obj_val not updated accordingly.
        """
        self.obj_val_valid = False

    @abstractmethod
    def initialize(self, k):
        """Construct an initial solution in a fast non-sophisticated way.

        :param k: is increased from 0 onwards for each call of this method
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: "Solution") -> bool:
        """Return true if the other solution is equal to the current one."""
        raise NotImplementedError

    def is_better(self, other: "Solution") -> bool:
        """Return True if the current solution is better in terms of the objective function than the other.

        Considers parameter settings.mh_maxi.
        """
        if settings.mh_maxi:
            return self.obj() > other.obj()
        else:
            return self.obj() < other.obj()

    def is_worse(self, other: "Solution") -> bool:
        """Return True if the current solution is worse in terms of the objective function than the other.

        Considers parameter settings.mh_maxi.
        """
        if settings.mh_maxi:
            return self.obj() < other.obj()
        else:
            return self.obj() > other.obj()

    @classmethod
    def is_better_obj(cls, obj1: TObj, obj2: TObj) -> bool:
        """Return True if the obj1 is a better objective value than obj2.

        Considers parameter settings.mh_maxi.
        """
        if settings.mh_maxi:
            return obj1 > obj2
        else:
            return obj1 < obj2

    @classmethod
    def is_worse_obj(cls, obj1: TObj, obj2: TObj) -> bool:
        """Return True if obj1 is a worse objective value than obj2.

        Considers parameter settings.mh_maxi.
        """
        if settings.mh_maxi:
            return obj1 < obj2
        else:
            return obj1 > obj2

    def dist(self, other):
        """Return distance of current solution to other solution.

        The default implementation just returns 0 if the solutions are the same and 1 otherwise.
        """
        return self.obj() == other.obj()

    def __hash__(self):
        """Return hash value for solution.

        The default implementation returns the hash value of the objective value.
        """
        return hash(self.obj())

    @abstractmethod
    def check(self):
        """Check validity of solution.

        If a problem is encountered, raise an exception.
        The default implementation just re-calculates the objective value.
        """
        if self.obj_val_valid:
            old_obj = self.obj_val
            self.invalidate()
            if old_obj != self.obj():
                raise ValueError(f'Solution has wrong objective value: {old_obj}, should be {self.obj()}')

    @abstractmethod
    def crossover(self, other: "Solution"):
        """ Performs a crossover operation of two solutions.

        A crossover operation of two solutions is performed and the result is returned.
        """
        raise NotImplementedError


class VectorSolution(Solution, ABC):
    """Abstract solution class with integer vector as solution representation.

    Attributes
        - x: vector representing a solution, realized ba a numpy.ndarray
    """

    def __init__(self, length, init=True, dtype=int, init_value=0, **kwargs):
        """Initializes the solution vector with zeros."""
        super().__init__(**kwargs)
        self.x = np.full([length], init_value, dtype=dtype) if init else np.empty([length], dtype=dtype)

    def copy_from(self, other: 'VectorSolution'):
        super().copy_from(other)
        self.x[:] = other.x

    def __repr__(self):
        return str(self.x)

    def __eq__(self, other: 'VectorSolution') -> bool:
        return self.obj() == other.obj() and np.array_equal(self.x, other.x)


class BoolVectorSolution(VectorSolution, ABC):
    """Abstract solution class with 0/1 vector as solution representation.

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
                raise ValueError("Invalid value in BoolVectorSolution: {self.x}")
