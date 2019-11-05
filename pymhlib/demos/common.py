"""Some functions common for all demo applications."""

import logging
from typing import Callable

from pkg_resources import resource_filename

from pymhlib.alns import ALNS
from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.par_alns import ParallelALNS
from pymhlib.pbig import PBIG
from pymhlib.sa import SA
from pymhlib.scheduler import Method
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from pymhlib.solution import Solution
from pymhlib.ssga import SteadyStateGeneticAlgorithm

data_dir = resource_filename("pymhlib", "demos/data/")


def add_general_arguments_and_parse_settings(default_inst_file: str = 'inst.dat', seed: int = 0):
    """Some general parameters are registered and the settings are parsed.

    :param seed: optional seed value for the random number generators; 0: random initialization
    :param default_inst_file: default instance file to be loaded and solved
    """
    parser = get_settings_parser()
    parser.add_argument("--alg", type=str, default='gvns', help='optimization algorithm to be used '
                                                                '(gvns, alns, pbig, par_alns, ssga, sa)')
    parser.add_argument("--inst_file", type=str, default=default_inst_file,
                        help='problem instance file')
    parser.add_argument("--meths_ch", type=int, default=1,
                        help='number of construction heuristics to be used')
    parser.add_argument("--meths_li", type=int, default=1,
                        help='number of local improvement methods to be used')
    parser.add_argument("--meths_sh", type=int, default=5,
                        help='number of shaking methods to be used')
    parser.add_argument("--meths_de", type=int, default=3,
                        help='number of destroy methods to be used')
    parser.add_argument("--meths_re", type=int, default=3,
                        help='number of repair methods to be used')
    parse_settings(seed=seed)


def run_optimization(problem_name: str, instance_class, solution_class, default_inst_file: str = "inst.dat",
                     own_settings: dict = None, embedded: bool = False, iter_cb: Callable = None,
                     seed: int = 0) -> Solution:
    """Initialize and run optimization algorithm given by parameter alg on given problem instance.

    First, some general parameters for the algorithm to be applied, the instance file, and the methods to
    be applied are registered and the settings are parsed.
    Then the loggers are initialized, instance and solution objects are created and the chosen algorithm is
    performed. The resulting solution is finally returned.

    :param problem_name: name of the problem to be printed
    :param instance_class: class of the instance to be solved
    :param solution_class: concrete solution class to be used
    :param default_inst_file: default instance file to be loaded and solved
    :param own_settings: optional run-specific settings dictionary
    :param embedded: if set it is assumed that the call is embedded in a Notebook or other larger framework,
        and therefore, the parameters are assumed to be already registered and parsed
    :param iter_cb: optional callback function that is called each iteration by some of the algorithms
    :param seed: optional seed value for the random number generators; 0: random initialization
    """
    if not embedded:
        add_general_arguments_and_parse_settings(default_inst_file, seed)

    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info(f"pymhlib demo for solving {problem_name}")
    logger.info(get_settings_as_str())
    instance = instance_class(settings.inst_file)
    logger.info(f"{problem_name} instance read:\n" + str(instance))
    solution = solution_class(instance)
    # solution.initialize(0)

    logger.info(f"Solution: {solution}, obj={solution.obj()}\n")

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
    elif settings.alg == 'par_alns':
        alg = ParallelALNS(solution,
                           [Method(f"ch{i}", solution_class.construct, i) for i in range(settings.meths_ch)],
                           [Method(f"de{i}", solution_class.destroy, i) for i in range(1, settings.meths_de + 1)],
                           [Method(f"re{i}", solution_class.repair, i) for i in range(1, settings.meths_re + 1)],
                           own_settings)
    elif settings.alg == 'ssga':
        alg = SteadyStateGeneticAlgorithm(solution,
                                          [Method(f"ch{i}", solution_class.construct, i) for i in
                                           range(settings.meths_ch)],
                                          solution_class.crossover,
                                          Method(f"mu", solution_class.shaking, 1),
                                          Method(f"ls", solution_class.local_improve, 1),
                                          own_settings)
    elif settings.alg == 'sa':
        alg = SA(solution,
                 [Method(f"ch{i}", solution_class.construct, i) for i in range(settings.meths_ch)],
                 solution_class.random_move_delta_eval, solution_class.apply_neighborhood_move, iter_cb, own_settings)
    else:
        raise ValueError('Invalid optimization algorithm selected (settings.alg): ', settings.alg)

    alg.run()
    logger.info("")
    alg.method_statistics()
    alg.main_results()
    return solution
