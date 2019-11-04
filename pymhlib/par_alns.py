"""Parallel implementation of Adaptive Large Neighborhood Search (ALNS).

The module realizes a parallel version of the ALNS metaheuristic utilizing multiple cores via multiple processes.
The destroy plus repair operations are delegated as distributed tasks to a process pool in an asynchronous way.
Note that a reasonable speedup will only be gained if the repair/destroy operations are time-expensive.
Further note that the implementation will not be deterministic anymore for more than one worker even
if a seed > 0 is specified.
"""

import multiprocessing as mp
import time
from typing import Iterable, Tuple
from configargparse import Namespace
from pymhlib.settings import settings, set_settings
from pymhlib.solution import Solution, TObj
from pymhlib.scheduler import Method, Result
from pymhlib.alns import ALNS


class ParallelALNS(ALNS):
    """A parallel version of the ALNS utilizing multiple cores via multiple processes.

    The destroy plus repair operations are delegated as distributed tasks to a process pool in an asynchronous way.
    """

    def operators_generator(self, sol: Solution) -> Iterable[Tuple[Method, Method, Solution]]:
        """Generator yielding a selected repair and destroy operators and the solution to apply them to.

        :param sol: Solution to be modified.
        :return: yields infinitely often a tuple of a selected destroy and repair methods and the solution to which
            they should be applied
        """
        while True:
            destroy, repair = self.select_method_pair()
            yield destroy, repair, sol

    @staticmethod
    def process_init(s: Namespace, new_seed: int):
        """Initialization of new worker process."""
        s.seed = new_seed
        set_settings(s)

    @staticmethod
    def perform_method_pair_in_worker(params: Tuple[Method, Method, Solution])\
            -> Tuple[Method, Method, Solution, Result, TObj, float, float]:
        """Performs the given destroy and repair operator pair on the given solution in a worker process."""
        destroy, repair, sol = params
        res = Result()
        obj_old = sol.obj()
        t_start = time.process_time()
        destroy.func(sol, destroy.par, res)
        t_destroyed = time.process_time()
        repair.func(sol, repair.par, res)
        t_end = time.process_time()
        return destroy, repair, sol, res, obj_old, t_destroyed-t_start, t_end-t_destroyed

    def alns(self, sol: Solution):
        """Perform adaptive large neighborhood search (ALNS) on given solution."""
        self.next_segment = self.iteration + self.own_settings.mh_alns_segment_size
        sol_incumbent = sol.copy()
        sol_new = sol.copy()
        operators = self.operators_generator(sol_new)
        worker_seed = 0 if settings.mh_workers > 1 else settings.seed
        with mp.Pool(processes=settings.mh_workers,
                     initializer=self.process_init, initargs=(settings, worker_seed)) as worker_pool:
            result_iter = worker_pool.imap_unordered(self.perform_method_pair_in_worker, operators)
            for result in result_iter:
                # print("Result:", result)
                destroy, repair, sol_result, res, obj_old, t_destroy, t_repair = result
                sol_new.copy_from(sol_result)
                self.update_stats_for_method_pair(destroy, repair, sol, res, obj_old, t_destroy, t_repair)
                self.update_after_destroy_and_repair_performed(destroy, repair, sol_new, sol_incumbent, sol)
                if res.terminate:
                    sol.copy_from(sol_incumbent)
                    return
                self.update_operator_weights()
