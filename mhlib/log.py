"""mhlib specific logging objects.

Two logging objects are maintained specifically for the mhlib:
- logger with name "mhlib" for the general information and
- iter_logger with name "mhlib_iter" for the iteration-wise logging.

init() must be called to initialize this module, i.e., create these objects.
"""

import logging
import logging.handlers
import sys

from mhlib.settings import settings, get_settings_parser, parse_settings

parser = get_settings_parser()
parser.add("--mh_out", type=str, default="None",
           help='file to write general output into (None: stdout)')
parser.add("--mh_log", type=str, default="None",
           help='file to write iteration-wise logging into (None: stdout)')


def init_logger():
    """Initialize logger objects."""

    # logger for general output
    logger = logging.getLogger("mhlib")
    formatter = logging.Formatter("%(message)s")
    if settings.mh_out == 'None':
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(settings.mh_out, "w")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # logger for iteration-wise output
    iter_logger = logging.getLogger("mhlib_iter")
    if settings.mh_log == 'None':
        iter_handler = handler
    else:
        iter_file_handler = logging.FileHandler(settings.mh_log, "w")
        iter_file_handler.setFormatter(formatter)
        iter_handler = logging.handlers.MemoryHandler(
            capacity=1024 * 100,
            flushLevel=logging.ERROR,
            target=iter_file_handler
        )
        iter_handler.setFormatter(formatter)
    iter_logger.addHandler(iter_handler)
    iter_logger.propagate = False
    iter_logger.setLevel(logging.INFO)


class IndentLevel:
    """Manage indentation of log messages according to specified levels."""
    level = 0
    indent_str = "  > "
    format_str = "%(message)s"

    @classmethod
    def reset(cls, value=0):
        """Reset indentation level to the given value."""
        cls.level = value
        cls.set_format()

    @classmethod
    def increase(cls):
        """Increase indentation level by one."""
        cls.level += 1
        cls.set_format()

    @classmethod
    def decrease(cls):
        """Decrease indentation level by one."""
        cls.level -= 1
        assert(cls.level >= 0)
        cls.set_format()

    @classmethod
    def set_format(cls):
        format_str = cls.indent_str * cls.level + cls.format_str
        formatter = logging.Formatter(format_str)
        for name in ['mhlib', 'mhlib_iter']:
            logger = logging.getLogger(name)
            for h in logger.handlers:
                h.setFormatter(formatter)


def test():
    """Some basic module tests."""
    init_logger()
    logger = logging.getLogger("mhlib")
    logger.info('This is an info to logger')
    logger.error('This is an error to logger')
    iter_logger = logging.getLogger("mhlib_iter")
    iter_logger.info('This is an info to iter_logger')
    iter_logger.error('This is an error to iter_logger')
    IndentLevel.increase()
    logger.info('This is an info to logger at level 1')
    IndentLevel.increase()
    logger.info('This is an info to iter_logger at level 2')
    IndentLevel.decrease()
    IndentLevel.decrease()
    logger.info('This is an info to logger at level 0')


if __name__ == "__main__":
    parse_settings()
    test()
