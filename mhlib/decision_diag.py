"""Generic classes for decision diagrams (DDs)."""

from typing import DefaultDict, List, TypeVar, Optional
from abc import ABC, abstractmethod
from itertools import count
from dataclasses import dataclass
from collections import defaultdict


TNum = TypeVar('TNum', int, float)


@dataclass
class Arc:
    """An arc in the DD

    Attributes:
        - u, v: source and target nodes
        - val: value of the arc, i.e., the value assigned to a corresponding variable etc.
        - length: arc length
    """
    u: 'Node'
    v: 'Node'
    value: int
    length: TNum


class Node(ABC):
    """An abstract class for a node of a DD.

    Attributes
        - id: a DD-unique ID for printing the node
        - z_bp: length of a best path from r to the node
        - pred: list of ingoing arcs
        - succ: dict with outgoing arcs, with values as keys
    """

    def __init__(self, id_):
        """Create a node.

        """
        self.id_ = id_
        self.z_bp: Optional[TNum] = None
        self.pred: List[Arc] = list()
        self.succ: DefaultDict[(int, Arc)] = defaultdict(lambda: None)

    def __repr__(self, detailed=False):
        if detailed:
            return f"Node({self.id_}"
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


class DecisionDiag:
    """An abstract class for a DD."""

    def __init__(self, inst, r: Node):
        """Initialize DD with root node.

        Attributes
            - inst: problem instance
            - id_generator: yields successive IDs for the nodes
            - r: root node
            - NodeType: specific node type to be used determined from r
            - layers list of list of nodes at each layer
        """
        super().__init__()
        self.inst = inst
        self.id_generator = count()
        self.r = r
        self.NodeType = r.__class__
        self.layers = [[r]]

    def __repr__(self):
        s = f"DD, {len(self.layers)} layers:\n"
        for i, layer in enumerate(self.layers):
            s += f"Layer {i}, {len(layer)} node(s):\n"
            for node in layer:
                s += repr(node) + " "
            s += "\n"
        return s + "\n"

    def create_successor_node(self, node: Node, value: int, length: TNum, *args) -> Node:
        """Creates a successor node for node, connects it with an arc and returns it.

        :param node: source node
        :param value: value of the arc, i.e., value assigned to a corresponding variable
        :param length: arc length
        :param args: parameters passed to the initialization of the node (without id_)
        """
        succ_node = self.NodeType(next(self.id_generator), *args)
        assert not node.succ[value]
        arc = Arc(node, succ_node, value, length)
        node.succ[value] = arc
        succ_node.pred.append(arc)
        return succ_node

    @abstractmethod
    def expand_node(self, node: Node, depth) -> List[Node]:
        """Expand node, creating all successor nodes, and returns them as list.

        The successor nodes and the corresponding arcs are added to the graph.
        :param node: the node to be expanded; must not yet have any successors
        :param depth: optional depth of the current node
        """
        return []
