"""
Provides configuration file and command line argument parsing functionality to all modules.

Parameters can be decentrally defined in any module by getting the global parser via get_settings_parser
and registering them by add_argument(). parse_settings() needs to be called one in the main program, then
all parameters are available under the global Namespace settings.

For the usage of config files see the documentation of configargparse or call the program with -h.
"""

import pickle
import numpy as np
import random
import configargparse


settings = configargparse.Namespace()  # global Namespace with all settings
unknown_args = []  # global list with all unknown parameters
_parser = None  # single global settings parser


def get_settings_parser():
    """Returns the single global argument parser for adding parameters.

    Parameters can be added in all modules by add_argument.
    After calling parse() once in the main program, all settings
    are available in the global settings dictionary.
    """
    global _parser
    if not _parser:
        _parser = configargparse.ArgParser(  # default_config_files=["default.cfg"],
                                           formatter_class=configargparse.ArgumentDefaultsRawHelpFormatter)
        _parser.set_defaults(seed=0)
    return _parser


def parse_settings(return_unknown=False, default_config_files=None):
    """Parses the config files and command line arguments and initializes settings and unknown_parameters.

    Needs to be called once in the main program (or more generally after all arguments have been added to the parser.
    Also seeds the random number generators based on parameter seed.

    Parameters
        - return_unknown: return unknown parameters as list in global variable unknown_args; otherwise raise exception
        - default_config_files: list of default config files to read
    """
    global settings, unknown_args
    p = get_settings_parser()
    p.add_argument('--seed', type=int, help='seed for the random number generators (0: random init)')
    p.add_argument('-c', '--config', is_config_file=True, help='config file to be read')
    p._default_config_files = default_config_files if default_config_files else []
    if return_unknown:
        _, unknown_args[:] = p.parse_known_args(namespace=settings)
    else:
        p.parse_args(namespace=settings)

    # random seed; per default a random seed is generated
    if settings.seed == 0:
        settings.seed = np.random.randint(np.iinfo(np.int32).max)
    np.random.seed(settings.seed)
    random.seed(settings.seed)


def save_settings(filename):
    """Save settings to given binary file."""
    with open(filename, 'wb') as f:
        pickle.dump(settings, f, pickle.HIGHEST_PROTOCOL)


def load_settings(filename):
    """Load settings from given binary file."""
    with open(filename, 'rb') as f:
        global settings
        settings.__dict__ = vars(pickle.load(f))


def get_settings_as_str():
    """Get all parameters and their values as descriptive multi-line string."""
    s = "\nsettings:\n"
    for key, value in sorted(vars(settings).items()):
        s += f"{key}={value}\n"
    return s


class OwnSettings:
    """An individualized settings storage, which falls back to the default setting for not provided parameters."""

    def __init__(self, own_settings: dict = None):
        self.__dict__ = own_settings if own_settings else dict()

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            val = settings.__getattribute__(item)
            self.__setattr__(item, val)
            return val

