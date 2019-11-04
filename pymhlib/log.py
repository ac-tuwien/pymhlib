"""pymhlib specific logging objects.

Two logging objects are maintained specifically for the pymhlib:
    - logger with name "pymhlib" for the general information and
    - iter_logger with name "pymhlib_iter" for the iteration-wise logging.

init() must be called to initialize this module, i.e., create these objects.
"""

import logging
import logging.handlers
import sys

from pymhlib.settings import settings, get_settings_parser, parse_settings

parser = get_settings_parser()
parser.add_argument("--mh_out", type=str, default="None",
                    help='file to write general output into (None: stdout)')
parser.add_argument("--mh_log", type=str, default="None",
                    help='file to write iteration-wise logging into (None: stdout)')


def init_logger():
    """Initialize logger objects."""

    # logger for general output
    logger = logging.getLogger("pymhlib")
    formatter = logging.Formatter("%(message)s")
    if settings.mh_out == 'None':
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(settings.mh_out, "w")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # logger for iteration-wise output
    iter_logger = logging.getLogger("pymhlib_iter")
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
    iter_logger.handlers = []
    iter_logger.addHandler(iter_handler)
    iter_logger.propagate = False
    iter_logger.setLevel(logging.INFO)


class LogLevel:
    """Manage indentation of log messages according to specified levels.

    Indentation is most meaningful when embedding optimization algorithms within others.

    This class can also be used as context manager in a with statement.

    If indentation is used and some multi-line log message is written, write Loglevel.s after each "\n"
    in order to do the indentation for all lines.

    Class attributes
        - level: level of indentation
        - s: actual string used for current indentation
        - indent_str: prefix used for each indentation level
        - format_str: unindented format string for logging
    """
    level = 0
    s = ""
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
        """Activate the format for the currently set level."""
        cls.s = cls.indent_str * cls.level
        format_str = cls.s + cls.format_str
        formatter = logging.Formatter(format_str)
        for name in ['pymhlib', 'pymhlib_iter']:
            logger = logging.getLogger(name)
            for h in logger.handlers:
                h.setFormatter(formatter)

    def __enter__(self):
        """When used as context manager in with statement and entering context, increase level."""
        self.increase()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When used as context manager in with statement and leaving context, decrease level."""
        self.decrease()

    @classmethod
    def indent(cls, s: s) -> s:
        """Correctly indent the given string, which may be a multi-line message."""
        return cls.s + s.replace('\n', f'\n{cls.s}')


def test():
    """Some basic module tests."""
    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info('This is an info to logger')
    logger.error('This is an error to logger')
    iter_logger = logging.getLogger("pymhlib_iter")
    iter_logger.info('This is an info to iter_logger')
    iter_logger.error('This is an error to iter_logger')
    LogLevel.increase()
    logger.info('This is an info to logger at level 1')
    LogLevel.increase()
    logger.info('This is an info to iter_logger at level 2')
    LogLevel.decrease()
    LogLevel.decrease()
    logger.info('This is an info to logger at level 0')


if __name__ == "__main__":
    parse_settings()
    test()
