"""Some functions common for all demo applications."""

import logging
from pkg_resources import resource_filename

from mhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from mhlib.log import init_logger
from mhlib.scheduler import Method
from mhlib.gvns import GVNS
from mhlib.alns import ALNS
from mhlib.pbig import PBIG


data_dir = resource_filename("mhlib", "demos/data/")


def run_optimization(problem_name: str, instance_class, solution_class, default_inst_file: str, own_settings=None):
    """Run optimization algorithm given by parameter alg on given problem instance."""
    parser = get_settings_parser()
    parser.add("--alg", type=str, default='gvns', help='optimization algorithm to be used (gvns, alns)')
    parser.add("--inst_file", type=str, default=default_inst_file,
               help='problem instance file')
    parser.add("--meths_ch", type=int, default=1,
               help='number of construction heuristics to be used')
    parser.add("--meths_li", type=int, default=1,
               help='number of local improvement methods to be used')
    parser.add("--meths_sh", type=int, default=5,
               help='number of shaking methods to be used')
    parser.add("--meths_de", type=int, default=3,
               help='number of destroy methods to be used')
    parser.add("--meths_re", type=int, default=3,
               help='number of repair methods to be used')
    # parser.set_defaults(seed=3)

    parse_settings()
    init_logger()
    logger = logging.getLogger("mhlib")
    logger.info(f"mhlib demo for solving {problem_name}")
    logger.info(get_settings_as_str())
    instance = instance_class(settings.inst_file)
    logger.info(f"{problem_name} instance read:\n" + str(instance))
    solution = solution_class(instance)
    # solution.initialize(0)

    logger.info(f"solution_class: {solution}, obj={solution.obj()}\n")

    if settings.alg == 'gvns':
        alg = GVNS(solution,
                   [Method(f"ch{i}", solution_class.construct, i) for i in range(settings.meths_ch)],
                   [Method(f"li{i}", solution_class.local_improve, i) for i in range(1, settings.meths_li + 1)],
                   [Method(f"sh{i}", solution_class.shaking, i) for i in range(1, settings.meths_sh + 1)],
                   own_settings)
    elif settings.alg == 'alns':
        alg = ALNS(solution,
                   [Method(f"ch{i}", solution_class.construct, i) for i in range(settings.meths_ch)],
                   [Method(f"de{i}", solution_class.destroy, i) for i in range(1, settings.meths_de + 1)],
                   [Method(f"re{i}", solution_class.repair, i) for i in range(1, settings.meths_re + 1)],
                   own_settings)
    elif settings.alg == 'pbig':
        alg = PBIG(solution,
                   [Method(f"ch{i}", solution_class.construct, i) for i in range(settings.meths_ch)],
                   [Method(f"li{i}", solution_class.local_improve, i) for i in range(1, settings.meths_li + 1)] +
                   [Method(f"sh{i}", solution_class.shaking, i) for i in range(1, settings.meths_sh + 1)],
                   own_settings)
    else:
        raise ValueError('Invalid optimization algorithm selected (settings.alg): ', settings.alg)

    alg.run()
    logger.info("")
    alg.method_statistics()
    alg.main_results()
