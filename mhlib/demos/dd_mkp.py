"""Demo program addressing the MKP with decision diagrams."""
from typing import List

import numpy as np

from ..settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from ..log import init_logger
from .mkp import MKPInstance, MKPSolution
from ..decision_diag import Node, DecisionDiag


class MKPNode(Node):
    """A DD node for the MKP.

    Attributes
        - dd: associated DecisionDiag
        - y: so far used amount of each resource
    """

    def __init__(self, id_, inst):
        super().__init__(id_)
        self.y = np.zeros([inst.m], dtype=int)

    def __repr__(self, detailed=False):
        if detailed:
            return super().__repr__() + f", y={self.y!s})\n"
        else:
            return f"{self.id_}"

    def __hash__(self):
        return hash(tuple(self.y))

    def __eq__(self, other: 'MKPNode'):
        return self.y == other.y

    def expand(self, depth) -> List['Node']:
        pass  # TODO


if __name__ == '__main__':
    import os
    import logging

    """Test for the DD classes."""
    parser = get_settings_parser()
    parser.add("--inst_file", type=str, default=os.path.join('mhlib', 'demos', 'mknapcb5-01.txt'),
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

    r = MKPNode('0', instance)
    decision_diag = DecisionDiag(instance, r)
    logger.info("DD=" + str(decision_diag))
