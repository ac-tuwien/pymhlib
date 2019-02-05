"""Generic classes for decision diagrams (DDs)."""

from typing import DefaultDict, List, TypeVar
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

    def __init__(self, id_, z_bp: TNum):
        """Create a node.

        """
        self.id_ = id_
        self.z_bp = z_bp
        self.pred: List[Arc] = list()
        self.succ: DefaultDict[(int, Arc)] = defaultdict(lambda: None)

    def __repr__(self):
            return f"Node({self.id_}"

    @abstractmethod
    def __hash__(self):
        """A hash function used for the graph and to determine identical states."""
        pass

    @abstractmethod
    def __eq__(self, other: 'Node'):
        """Return True if the nodes represent the same states."""
        return self is other


class DecisionDiag(ABC):
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
        """Creates a successor node for node, connects it with an arc, sets z_bp, and returns the successor node.

        :param node: source node
        :param value: value of the arc, i.e., value assigned to a corresponding variable
        :param length: arc length
        :param args: parameters passed to the initialization of the node (without id_)
        """
        succ_node = self.NodeType(next(self.id_generator), node.z_bp + length, *args)
        assert not node.succ[value]
        arc = Arc(node, succ_node, value, length)
        node.succ[value] = arc
        succ_node.pred.append(arc)
        return succ_node

    def merge_nodes(self, nodes: List[Node]):
        """Merge given list of nodes into the first node.

        All input nodes are not yet expanded.
        """
        n1 = nodes[0]
        for n2 in nodes:
            n1.pred += n2.pred
            for arc in n2.pred:
                arc.v = n1
                z_bp_new = arc.u.z_bp + arc.length
                if z_bp_new > n1.z_bp:
                    n1.z_bp = z_bp_new
            self.merge_state(n1, n2)

    @abstractmethod
    def expand_node(self, node: Node, depth: int) -> List[Node]:
        """Expand node, creating all successor nodes, and returns them as list.

        The successor nodes and the corresponding arcs are added to the graph.
        z_bp is also set in the successor nodes..
        :param node: the node to be expanded; must not yet have any successors
        :param depth: optional depth of the current node
        """
        return []

    @abstractmethod
    def merge_state(self, node: Node, node2: Node):
        """Merge state of second node into state of first node."""
        pass
