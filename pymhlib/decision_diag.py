"""Generic classes for decision diagrams (DDs)."""

from typing import Dict, DefaultDict, List, Optional
from abc import ABC, abstractmethod
from itertools import count
from dataclasses import dataclass
from collections import defaultdict

from pymhlib.solution import VectorSolution, TObj
from pymhlib.settings import settings


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
    length: TObj

    def __repr__(self):
        return f"Arc({self.u.id_}-{self.v.id_}, value={self.value}, length={self.length})"


class State(ABC):
    """Problem-specific state information in a node."""

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other: 'State'):
        return self is other

    @abstractmethod
    def __repr__(self):
        pass


class Node(State, ABC):
    """An abstract class for a node of a DD.

    Attributes
        - id: a DD-unique ID for printing the node
        - state: the problem-specific state data
        - z_bp: length of a best path from r to the node
        - pred: list of ingoing arcs
        - succ: dict with outgoing arcs, with values as keys
    """

    def __init__(self, id_, state: State, z_bp: TObj):
        """Create a node."""
        self.id_ = id_
        self.state = state
        self.z_bp = z_bp
        self.pred: List[Arc] = list()
        self.succ: Dict[int, Arc] = dict()

    def __repr__(self):
        return f"Node {self.id_}: z_bp={self.z_bp}, state={self.state}"

    def __hash__(self):
        """A hash function used for the graph and to determine identical states."""
        return hash(self.state)

    def __eq__(self, other: 'Node'):
        """Return True if the nodes represent the same states."""
        return self.state == other.state


NodePool = Dict[State, Node]


class DecisionDiag(ABC):
    """An abstract class for a DD.

    Attributes
        - inst: problem instance
        - id_generator: yields successive IDs for the nodes
        - r: root node
        - t_state: state a target node
        - t: target node; only set when actually reached
        - sol: solution object in which final solution will be stored
        - NodeType: specific node type to be used determined from r
        - layers dict of dict of nodes at each layer
    """

    def __init__(self, inst, r: Node, t_state: State, sol: VectorSolution):
        super().__init__()
        self.inst = inst
        self.id_generator = count()
        self.r = r
        self.t_state = t_state
        self.t: Optional[Node, None] = None
        self.sol = sol
        self.NodeType = r.__class__
        self.layers: DefaultDict[int, NodePool] = defaultdict(dict)
        self.layers[0][r.state] = r

    def __repr__(self):
        s = f"DD, {len(self.layers)} layers:\n"
        for i, layer in self.layers.items():
            s += f"Layer {i}, {len(layer)} node(s):\n"
            for node in layer.values():
                s += f" {node!s}\n"
        return s

    def get_successor_node(self, node_pool: NodePool, node: Node, value: int, length: TObj, state: State) -> Node:
        """Look up or create a successor for a node, connect them with an arc, set z_bp, and return the successor.

        :param node_pool: node pool in which to either find already existing node or add new node
        :param node: source node
        :param value: value of the arc, i.e., value assigned to a corresponding variable
        :param length: arc length
        :param state: state of the successor node
        """
        z_bp_new = node.z_bp + length
        if state in node_pool:
            succ_node = node_pool[state]
            if self.sol.is_better_obj(z_bp_new, succ_node.z_bp):
                succ_node.z_bp = z_bp_new
        else:
            if state == self.t_state:
                self.t = succ_node = self.NodeType('t', state, z_bp_new)
            else:
                succ_node = self.NodeType(next(self.id_generator), state, z_bp_new)
            node_pool[state] = succ_node
        assert value not in node.succ
        arc = Arc(node, succ_node, value, length)
        node.succ[value] = arc
        succ_node.pred.append(arc)
        return succ_node

    def expand_layer(self, depth: int) -> bool:
        """Expand all nodes in layers[depth], creating layers[depth+1.

        :return: True if all nodes have been expanded, else False
        """
        layers = self.layers
        if depth not in layers or not layers[depth]:
            return False
        depth_succ = depth + 1
        further_nodes_remaining = False
        for s, n in layers[depth].items():
            if self.expand_node(n, depth, layers[depth_succ]):
                further_nodes_remaining = True
        return further_nodes_remaining

    def expand_all(self, dd_type: str, max_width: int = 1):
        """Expand nodes layer-wise until no unexpanded nodes remain, i.e., create complete DD.

        :param dd_type: of DD to create: 'exact', 'relaxed', or 'restricted'
        :param max_width: maximum with in case of type 'relaxed' or 'restricted'
        """
        for depth in count(0):
            if not self.expand_layer(depth):
                break
            if dd_type == 'relaxed':
                self.relax_layer(self.layers[depth+1], max_width)
            elif dd_type == 'restricted':
                self.restrict_layer(self.layers[depth+1], max_width)
            elif dd_type != 'exact':
                raise ValueError(f"Invalid dd_type: {dd_type}")

    @classmethod
    def get_sorted_nodes(cls, node_pool):
        """Return a sorted list of the nodes in the given node_pool, with the most promising node first."""
        return sorted(node_pool.values(), key=lambda n: n.z_bp, reverse=settings.mh_maxi)

    def relax_layer(self, node_pool: NodePool, max_width: int = 1):
        """Relax the last created layer at the given depth to the given maximum width."""
        if len(node_pool) > max_width:
            nodes_sorted = self.get_sorted_nodes(node_pool)
            self.merge_nodes(nodes_sorted[max_width-1:], node_pool)

    def restrict_layer(self, node_pool: NodePool, max_width: int = 1):
        """Restrict the last created layer at the given depth to the given maximum width."""
        if len(node_pool) > max_width:
            nodes_sorted = self.get_sorted_nodes(node_pool)
            for node in nodes_sorted[-1:max_width-1:-1]:
                self.delete_node(node, node_pool)

    @staticmethod
    def delete_node(node: Node, node_pool: NodePool):
        """Deletes the specified node from the DD and node_pool, together with all its arcs.

        The nodes must not have any successors yet and must not be the r or t.
        """
        assert not node.succ  # node must not have successors
        del node_pool[node.state]
        for arc in node.pred:
            del arc.u.succ[arc.value]

    def derive_best_path(self) -> List[int]:
        """Derives from a completely constructed DD a best path and returns it as list of arc values."""
        node = self.t
        path = []
        assert node
        for idx in range(len(self.sol.x)-1, -1, -1):
            for pred in node.pred:
                # print(pred, node.z_bp, pred.u.z_bp + pred.length)
                if node.z_bp == pred.u.z_bp + pred.length:
                    path.append(pred.value)
                    node = pred.u
                    break
            else:
                raise ValueError(f"Invalid z_bp value at node {node!s}")
        return path[::-1]

    def merge_nodes(self, nodes: List[Node], node_pool: NodePool):
        """Merge given list of nodes into the first node.

        All input nodes are not yet expanded and are assumed to be in the given node_pool.
        """
        assert len(nodes) >= 2
        merged_node = nodes[0]
        merged_state = merged_node.state
        for n in nodes[1:]:
            merged_state = self.merge_states(merged_state, n.state)
            if merged_state is n.state:
                merged_node = n
        if merged_node.state != merged_state:
            n = node_pool[merged_node.state]
            if n:
                merged_node = n
            else:
                del node_pool[merged_node.state]
                merged_node.state = merged_state
                node_pool[merged_node.state] = merged_node
        for n in nodes:
            if n is merged_node:
                continue
            merged_node.pred += n.pred
            for arc in n.pred:
                arc.v = merged_node
                z_bp_new = arc.u.z_bp + arc.length
                if self.sol.is_better_obj(z_bp_new, merged_node.z_bp):
                    merged_node.z_bp = z_bp_new
            del node_pool[n.state]

    # Problem-specific abstract methods

    @abstractmethod
    def expand_node(self, node: Node, depth: int, node_pool: NodePool) -> bool:
        """Expand node, creating all successor nodes in node_pool.

        The successor nodes and the corresponding arcs are added to the graph.
        z_bp is also set in the successor nodes..
        :param node: the node to be expanded; must not yet have any successors
        :param depth: optional depth of the current node
        :param node_pool: pool of nodes in which to look for already existing node or create new nodes
        :return: True if nodes that need further expansion have been created
        """
        return False

    @abstractmethod
    def merge_states(self, state1: State, state2: State) -> State:
        """Return merged state of the two given states.

        May return directly state1 or state2 if one of this state dominates the other.
        """
        pass
