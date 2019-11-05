"""Demo application solving the minimum vertex cover problem.

Given an undirected simple graph, find a minimum subset of the vertices so that from each edge in the graph at
least one of its end points is in this subset.
"""

import networkx as nx
import random
from typing import Any
import heapq
from itertools import combinations
from typing import Tuple

from pymhlib.solution import SetSolution, TObj
from pymhlib.settings import get_settings_parser
from pymhlib.scheduler import Result
from pymhlib.demos.graphs import create_or_read_simple_graph

parser = get_settings_parser()


class VertexCoverInstance:
    """Minimum vertex cover problem instance.

    Given an undirected simple graph, find a minimum subset of the vertices so that from each edge in the graph at
    least one of its end points is in this subset.

    Attributes
        - graph: the graph for which we want to find a minimum vertex cover
        - n: number of nodes
        - m: number of edges
    """

    def __init__(self, name: str):
        """Create or read graph with given name."""
        self.graph = create_or_read_simple_graph(name)
        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()

    def __repr__(self):
        """Write out the instance data."""
        return f"n={self.n} m={self.m}\n"


class VertexCoverSolution(SetSolution):
    """Solution to a minimum vertex cover instance.

    Attributes
        - s: set of selected elements
    """

    to_maximize = False

    def __init__(self, inst: VertexCoverInstance):
        super().__init__(inst=inst)

    def copy(self):
        sol = VertexCoverSolution(self.inst)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'VertexCoverSolution'):
        super().copy_from(other)

    def calc_objective(self):
        return len(self.s)

    def check(self):
        """Check if valid solution.

        :raises ValueError: if problem detected.
        """
        super().check()
        if not self.s.issubset(set(range(self.inst.n))):
            raise ValueError(f'Invalid value in solution set: {self.s}')
        for u, v in self.inst.graph.edges:
            if u not in self.s and v not in self.s:
                raise ValueError(f'Edge ({u},{v}) not covered')

    def remove_redundant(self) -> bool:
        """Scheduler method that checks for each node in the current vertex cover if it can be removed.

        The nodes are processed in random order.

        :return: True if solution could be improved
        """
        s = self.s
        x = list(s)
        random.shuffle(x)
        for u in x:
            for v in self.inst.graph.neighbors(u):
                if v not in s:
                    break
            else:
                s.remove(u)
        if len(s) < len(x):
            self.invalidate()
            return True
        return False

    def two_approximation_construction(self):
        """Perform a randomized 2-approximation construction algorithm.

        Randomly select an uncovered edge and include both end nodes in vertex cover until all edges are covered.
        """
        g: nx.Graph = self.inst.graph.copy()
        s = self.s
        s.clear()
        edge_list = list(g.edges)
        random.shuffle(edge_list)
        for u, v in edge_list:
            if not g.has_edge(u, v):
                continue
            s.add(u)
            s.add(v)
            g.remove_nodes_from((u, v))
        self.invalidate()

    # noinspection PyCallingNonCallable
    def greedy_construction(self, use_degree=True):
        """Degree-based greedy or pure random construction heuristic."""
        g: nx.Graph = self.inst.graph.copy()
        s = self.s
        s.clear()
        nodes = list(g.nodes)
        random.shuffle(nodes)

        heap = None
        if use_degree:
            heap = [(-g.degree(u), u) for u in g.nodes]
            heapq.heapify(heap)

        # noinspection PyCallingNonCallable
        def node_yielder():
            if use_degree:
                while heap:
                    d, node = heapq.heappop(heap)
                    if d == 0:
                        return
                    if not g.has_node(node) or g.degree(node) != -d:
                        continue
                    yield node
            else:
                for node in nodes:
                    if g.degree(node):
                        yield node

        for u in node_yielder():
            for v in g.neighbors(u):
                if v not in s:
                    s.add(u)
                    if use_degree:
                        for v2 in g.neighbors(u):
                            v2_d = g.degree(v2) - 1
                            if v2_d:
                                heapq.heappush(heap, (-v2_d, v2))
                    g.remove_node(u)
                    break
        self.invalidate()

    def construct(self, par, _result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)

    def local_improve(self, _par, result):
        """Search add-one-remove-at-least-two-nodes neighborhood in first-improvement manner."""
        g: nx.Graph = self.inst.graph
        s = self.s
        x = list(set(range(self.inst.n)).difference(s))
        random.shuffle(x)
        for u in x:
            # when adding u, can we remove >= 2 neighboring nodes?
            s.add(u)
            removable = []
            for v in g.neighbors(u):
                if v not in s:
                    continue
                for v2 in g.neighbors(v):
                    if v2 not in s:
                        break
                else:
                    removable.append(v)
            if len(removable) >= 2:
                # find two non-adjacent
                for va, vb in combinations(removable, 2):
                    if not g.has_edge(va, vb):
                        s.remove(va)
                        s.remove(vb)
                        removable.remove(va)
                        removable.remove(vb)
                        removed = {va, vb}
                        for vc in removable:
                            for vr in removed:
                                if g.has_edge(vc, vr):
                                    break
                            else:
                                s.remove(vc)
                                removed.add(vc)
                        self.invalidate()
                        return
            s.remove(u)
        result.changed = False

    def shaking(self, par: Any, _result: Result):
        """Add par so far unselected nodes and apply remove_redundant."""
        s = self.s
        x = set(range(self.inst.n)).difference(s)
        to_add = random.sample(x, max(len(x), par))
        for u in to_add:
            s.add(u)
        self.remove_redundant()

    def initialize(self, k):
        """Initialize solution by taking all nodes and applying local_improve."""
        super().initialize(0)
        self.greedy_construction(k == 0)
        # self.two_approximation_construction()
        # self.remove_redundant()
        self.check()

    def random_move_delta_eval(self) -> Tuple[int, TObj]:
        """Choose a random move and perform delta evaluation for it, return (move, delta_obj)."""
        raise NotImplementedError

    def apply_neighborhood_move(self, pos: int):
        """This method applies a given neighborhood move accepted by SA,
            without updating the obj_val or invalidating, since obj_val is updated incrementally by the SA scheduler."""
        raise NotImplementedError

    def crossover(self, other: 'VertexCoverSolution') -> 'VertexCoverSolution':
        raise NotImplementedError


if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    run_optimization('Minimum Vertex Cover', VertexCoverInstance, VertexCoverSolution, "frb40-19-1.mis")
