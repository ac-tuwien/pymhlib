#!/usr/bin/python3

import multiprocessing as mp
import time
import logging
from configargparse import Namespace
from mhlib.settings import parse_settings, settings, get_settings_parser, set_settings
from mhlib.log import init_logger
from mhlib.scheduler import Method
from mhlib.gvns import GVNS
from mhlib.alns import ALNS


parser = get_settings_parser()
parser.add("--mh_workers", type=int, default=2,
           help='number of used worker processes in multiprocessing')


def sub(param):
    print("In", param, settings)
    a = 0
    time.sleep(1)
    while False:
        a += 1
    return param


par = [1]


def it():
    a = 0
    for b in range(10):
        a += 1
        yield (a, time.asctime(), par)
        par[0] += 1


def process_init(s: Namespace, new_seed: int):
    s.seed = new_seed
    set_settings(s)
    print("process init", settings)


if __name__ == '__main__':
    parse_settings()
    print(type(settings))
    init_logger()
    logger = logging.getLogger("mhlib")

    seed = 0 if settings.mh_workers > 1 else settings.seed
    g = it()
    with mp.Pool(processes=settings.mh_workers, initializer=process_init, initargs=(settings, seed)) as pool:
        pit = pool.imap_unordered(sub, g)
        print("Started")
        for r in pit:
            print("Result:", r)
        # time.sleep(2)
    print("Finish")
