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


def test():
    """Some basic module tests."""
    init_logger()
    logger = logging.getLogger("mhlib")
    logger.info('This is an info to logger')
    logger.error('This is an error to logger')
    iter_logger = logging.getLogger("mhlib_iter")
    iter_logger.info('This is an info to iter_logger')
    iter_logger.error('This is an error to iter_logger')


if __name__ == "__main__":
    parse_settings()
    test()
