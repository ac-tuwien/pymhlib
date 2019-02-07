"""Some functions common for all demo applications."""

import logging

from mhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from mhlib.log import init_logger
from mhlib.scheduler import GVNS, Method
from pkg_resources import resource_filename


def run_gvns_demo(problem_name: str, Instance, Solution, default_inst_file: str):
    """Run GVNS for any of the demo problems specified by the parameters."""
    parser = get_settings_parser()
    parser.add("--inst_file", type=str, default=default_inst_file,
               help='problem instance file')
    parser.add("--meths_ch", type=int, default=1,
               help='number of construction heuristics to be used in OC')
    parser.add("--meths_li", type=int, default=1,
               help='number of local improvement methods to be used in OC')
    parser.add("--meths_sh", type=int, default=5,
               help='number of shaking methods to be used in OC')
    # parser.set_defaults(seed=3)

    parse_settings()
    init_logger()
    logger = logging.getLogger("mhlib")
    logger.info(f"mhlib demo for solving {problem_name}")
    logger.info(get_settings_as_str())
    instance = Instance(settings.inst_file)
    logger.info(f"{problem_name} instance read:\n" + str(instance))
    solution = Solution(instance)
    # solution.initialize(0)

    logger.info(f"Solution: {solution}, obj={solution.obj()}\n")

    gvns = GVNS(solution,
                [Method(f"ch{i}", Solution.construct, i) for i in range(settings.meths_ch)],
                [Method(f"li{i}", Solution.local_improve, i) for i in range(1, settings.meths_li + 1)],
                [Method(f"sh{i}", Solution.shaking, i) for i in range(1, settings.meths_sh + 1)])
    gvns.run()
    logger.info("")
    gvns.method_statistics()
    gvns.main_results()

data_dir = resource_filename("mhlib", "demos/data/")
