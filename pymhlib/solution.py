"""
Abstract base class representing a candidate solution to an optimization problem and some derived still generic classes.

The abstract base class Solution represents a candidate solution to an optimization problem.
Derived classes VectorSolution, BinaryVectorSolution, and SetSolution are for solutions which are
represented bei general fixed-length vectors, boolean fixed-length vectors and sets of arbitrary elements.

For a concrete optimization problem to solve you have to derive from one of these classes.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar
import random

from pymhlib.settings import settings, get_settings_parser

parser = get_settings_parser()
parser.add_argument("--mh_xover_pts", type=int, default=1, help='number of crossover points in multi-point crossover')

TObj = TypeVar('TObj', int, float)  # Type of objective value


class Solution(ABC):
    """Abstract base class for a candidate solution.

    Class variables
        - to maximize: default is True, i.e., to maximize; override with False if the goal is to minimize

    Attributes
        - obj_val: objective value; valid if obj_val_valid is set
        - obj_val_valid: indicates if obj_val has been calculated and is valid
        - inst: optional reference to a problem instance object
        - alg: optional reference to an algorithm object using this solution
    """

    to_maximize = True

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

    def __eq__(self, other: "Solution") -> bool:
        """Return true if the other solution is equal to the current one.

        The default implementation returns True if the objective values are the same.
        """
        return self.obj() == other.obj()

    def is_better(self, other: "Solution") -> bool:
        """Return True if the current solution is better in terms of the objective function than the other."""
        return self.obj() > other.obj() if self.to_maximize else self.obj() < other.obj()

    def is_worse(self, other: "Solution") -> bool:
        """Return True if the current solution is worse in terms of the objective function than the other."""
        return self.obj() < other.obj() if self.to_maximize else self.obj() > other.obj()

    @classmethod
    def is_better_obj(cls, obj1: TObj, obj2: TObj) -> bool:
        """Return True if the obj1 is a better objective value than obj2."""
        return obj1 > obj2 if cls.to_maximize else obj1 < obj2

    @classmethod
    def is_worse_obj(cls, obj1: TObj, obj2: TObj) -> bool:
        """Return True if obj1 is a worse objective value than obj2."""
        return obj1 < obj2 if cls.to_maximize else obj1 > obj2

    def dist(self, other):
        """Return distance of current solution to other solution.

        The default implementation just returns 0 if the solutions have the same objective value.
        """
        return self.obj() != other.obj()

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


class VectorSolution(Solution, ABC):
    """Abstract solution class with fixed-length integer vector as solution representation.

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

    def uniform_crossover(self, other: 'VectorSolution') -> 'VectorSolution':
        """Uniform crossover of the current solution with the given other solution."""
        child = self.copy()
        #  randomly replace elements with those from other solution
        for i in range(len(self.x)):
            if random.getrandbits(1):
                child.x[i] = other.x[i]
        child.invalidate()
        return child

    def multi_point_crossover(self, other: 'VectorSolution') -> 'VectorSolution':
        """Multi-point crossover of current and other given solution.

        The number of crossover points is passed in settings.mh_xover_pts.
        """
        child = self.copy()
        size = len(self.x)
        points = np.random.choice(size, settings.mh_xover_pts, replace=False)
        points.sort()
        if len(points) % 2:
            points.append(size)
        points = points.reshape(len(points)/2, 2)
        for a, b in points:
            child.x[a:b] = other.x[a:b]
        child.invalidate()
        return child


class SetSolution(Solution, ABC):
    """Abstract solution class with a set as solution representation.

    Attributes
        - s: set representing a solution
    """

    def __init__(self, **kwargs):
        """Initializes the solution with the empty set."""
        super().__init__(**kwargs)
        self.s = set()

    def copy_from(self, other: 'SetSolution'):
        super().copy_from(other)
        self.s = other.s.copy()

    def __repr__(self):
        return str(self.s)

    def __eq__(self, other: 'SetSolution') -> bool:
        return self.obj() == other.obj() and self.s == other.s

    def initialize(self, k):
        """Set the solution to the empty set."""
        self.s.clear()
