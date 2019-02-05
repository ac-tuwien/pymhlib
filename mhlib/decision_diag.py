"""Generic classes for decision diagrams (DDs)."""

from typing import Any, Dict, List, TypeVar, Optional
from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict

TNum = TypeVar('TNum', int, float)
TKey = Any


Arc = namedtuple("Arc", ("s:Node", "t:Node", "key:TKey", "length:TNum"))


class Node(ABC):
    """An abstract class for a node of a DD."""
    id_max = 0

    def __init__(self, id: Any):
        """Create a node.

        Attributes
            - id: a DD-unique ID for printing the node
            - next: dictionary for outgoing arcs, the key is a value assigned to a variable
            - prev: list of ingoing arcs
            - z_bp: length of best path from root to the node
        """
        self.id = id
        self.next: Dict[TKey, Arc] = defaultdict(None)
        self.prev: List[Arc] = list()
        self.z_bp: Optional[TNum] = None

    def __repr__(self):
        return f"Node({id}, prev={self.prev!s}, next={self.next!s})\n"

    @abstractmethod
    def expand(self, depth: int) -> List['Node']:
        """Expand node, returning all successor nodes.

        The successor nodes will be correctly connected to the current node by corresponding arcs.
        :param depth: depth of the current node
        """
        pass


class DecisionDiag:
    """An abstract class for a DD."""

    def __init__(self, root: Node):
        """Initialize DD with root node."""
        self.r = root
        self.node_class = root.__class__
