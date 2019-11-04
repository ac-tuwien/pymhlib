"""Demo program addressing the MKP with decision diagrams."""

import numpy as np
from typing import DefaultDict
from itertools import count

from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from pymhlib.log import init_logger
from pymhlib.demos.mkp import MKPInstance, MKPSolution
from pymhlib.decision_diag import State, Node, DecisionDiag


class MKPState(State, tuple):
    """The state corresponds to an int-tuple indicating for each resource the amount already used."""

    @staticmethod
    def __new__(cls, y):
        return tuple.__new__(tuple, y)

    def __hash__(self):
        return tuple.__hash__(self)

    def __eq__(self, other: 'MKPState'):
        return tuple.__eq__(self, other)

    def __repr__(self):
        return "y=" + tuple.__str__(self)


class MKPNode(Node):
    """A DD node for the MKP."""
    pass


class MKPDecisionDiag(DecisionDiag):
    """A DD for the MKP."""

    def __init__(self, inst: MKPInstance):
        super().__init__(inst, MKPNode('r', MKPState((0,)*inst.m), 0), MKPState(inst.b),
                         MKPSolution(inst))

    def expand_node(self, node: MKPNode, depth: int, node_pool: DefaultDict[(State, Node)])-> bool:
        assert not node.succ
        no_target = depth < self.inst.n-1
        self.get_successor_node(node_pool, node, 0, 0, node.state if no_target else self.t_state)
        y_new = node.state + self.inst.r[:, depth]
        if np.all(y_new <= self.inst.b):
            self.get_successor_node(node_pool, node, 1, self.inst.p[depth],
                                    MKPState(y_new) if no_target else self.t_state)
        return no_target

    def merge_states(self, state1: MKPState, state2: MKPState) -> MKPState:
        y = np.amin([state1, state2], axis=0)
        if y == state1:
            return state1
        if y == state2:
            return state2
        return MKPState(y)

    def derive_solution(self) -> MKPSolution:
        path = self.derive_best_path()
        # print(path)
        sel_pos = count()
        not_sel_pos = count(len(self.sol.x)-1, -1)
        for i, v in enumerate(path):
            idx = next(sel_pos) if v else next(not_sel_pos)
            self.sol.x[idx] = i
        self.sol.sel = next(sel_pos)
        self.sol.calc_y()
        assert self.sol.sel - 1 == next(not_sel_pos)
        # print(self.sol.x)
        return self.sol


def main():
    """Test for the DD classes."""
    import logging
    from pymhlib.demos.common import data_dir
    parser = get_settings_parser()
    parser.add("--inst_file", type=str, default=data_dir+'mknap-small.txt', help='problem instance file')
    # parser.set_defaults(seed=3)

    parse_settings()
    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info(f"pymhlib demo for using decision diagrams for the MKP")
    logger.info(get_settings_as_str())
    instance = MKPInstance(settings.inst_file)
    logger.info(f"MKP instance read:\n" + str(instance))

    # solution = MKPSolution(instance)
    # solution.initialize(0)
    dd = MKPDecisionDiag(instance)
    dd.expand_all('relaxed')
    logger.info(dd)
    sol = dd.derive_solution()
    # sol.check()
    logger.info(f"Solution: {sol}, obj={sol.obj()}")


if __name__ == '__main__':
    main()
