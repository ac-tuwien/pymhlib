"""Demo program addressing the MKP with decision diagrams."""

import numpy as np
from typing import List

from ..settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from ..log import init_logger
from .mkp import MKPInstance, MKPSolution
from ..decision_diag import Node, DecisionDiag


class MKPNode(Node):
    """A DD node for the MKP.

    Additional attributes
        - y: so far used amount of each resource
    """

    def __init__(self, id_, y):
        """Create new node with given ID and state."""
        super().__init__(id_)
        self.y = y

    def __repr__(self):
            return super().__repr__(True) + f", y={self.y!s})"

    def __hash__(self):
        return hash(tuple(self.y))

    def __eq__(self, other: 'MKPNode'):
        return self.y == other.y


class MKPDecisionDiag(DecisionDiag):
    """A DD for the MKP."""

    def __init__(self, inst: MKPInstance, r: MKPNode):
        super().__init__(inst, r)

    def expand_node(self, node: MKPNode, depth)-> List[MKPNode]:
        assert not node.succ
        successors = [self.create_successor_node(node, 0, 0, node.y.copy())]
        y_new = node.y + self.inst.r[:, depth]
        if np.all(y_new <= self.inst.b):
            successors.append(self.create_successor_node(node, 1, self.inst.p[depth], y_new))
        return successors


if __name__ == '__main__':
    """Test for the DD classes."""
    import os
    import logging

    parser = get_settings_parser()
    parser.add("--inst_file", type=str, default=os.path.join('mhlib', 'demos', 'mknap-small.txt'),
               help='problem instance file')
    # parser.set_defaults(seed=3)

    parse_settings()
    init_logger()
    logger = logging.getLogger("mhlib")
    logger.info(f"mhlib demo for using decision diagrams for the MKP")
    logger.info(get_settings_as_str())
    instance = MKPInstance(settings.inst_file)
    logger.info(f"MKP instance read:\n" + str(instance))

    solution = MKPSolution(instance)
    # solution.initialize(0)
    logger.info(f"Solution: {solution}, obj={solution.obj()}\n")

    root = MKPNode('0', np.zeros(instance.m, dtype=int))
    decision_diag = MKPDecisionDiag(instance, root)
    new_nodes = decision_diag.expand_node(root, 0)
    decision_diag.layers.append([])
    decision_diag.layers[1] = new_nodes
    print(decision_diag)
