"""Generic classes for decision diagrams (DDs)."""

from typing import List, TypeVar, Optional
from abc import ABC, abstractmethod
import networkx as nx


TNum = TypeVar('TNum', int, float)


class Node(ABC):
    """An abstract class for a node of a DD."""

    def __init__(self, id_):
        """Create a node.

        Attributes
            - id: a DD-unique ID for printing the node
            - z_bp: length of a best path from r to the node
        """
        self.id_ = id_
        self.z_bp: Optional[TNum] = None

    def __repr__(self, detailed=False):
        if detailed:
            return f"Node: {self.id_}"
        else:
            return f"{self.id_}"

    @abstractmethod
    def __hash__(self):
        """A hash function used for the graph and to determine identical states."""
        pass

    @abstractmethod
    def __eq__(self, other: 'Node'):
        """Return True if the nodes represent the same states."""
        return self is other

    @abstractmethod
    def expand(self, depth) -> List['Node']:
        """Expand node, returning all successor nodes.

        The successor nodes and the corresponding arcs are added to the graph.
        :param depth: optional depth of the current node
        """
        pass


class DecisionDiag(nx.MultiDiGraph):
    """An abstract class for a DD."""

    def __init__(self, inst, r: Node):
        """Initialize DD with root node.

        Attributes
            - inst: problem instance
            - r: root node
            - NodeType: specific node type to be used determined from r
        """
        super().__init__()
        self.inst = inst
        self.r = r
        self.NodeType = r.__class__
        self.add_node(r)
