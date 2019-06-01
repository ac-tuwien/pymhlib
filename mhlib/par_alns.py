"""Parallel implementation of Adaptive Large Neighborhood Search (ALNS).

The module realizes a parallel version of the ALNS metaheuristic utilizing multiple cores via multiple processes.
The destroy plus repair operations are delegated as distributed tasks to a process pool in an asynchronous way.
Note that a reasonable speedup will only be gained if the repair/destroy operations are time-expensive.
Further note that the implementation will not be deterministic anymore for more than one worker even
if a seed > 0 is specified.
"""

import multiprocessing as mp
import time
import numpy as np
from itertools import chain
from typing import List, Iterable, Tuple
from configargparse import Namespace
from mhlib.settings import settings, set_settings
from mhlib.solution import Solution, TObj
from mhlib.scheduler import Method, Result
from mhlib.alns import ALNS


class ParallelALNS(ALNS):
    """A parallel version of the ALNS utilizing multiple cores via multiple processes.

    The destroy plus repair operations are delegated as distributed tasks to a process pool in an asynchronous way.
    """

    def operators_generator(self, list_sol: List[Solution]) -> Iterable[Tuple[Method, Method, Solution]]:
        """Generator yielding a selected repair and destroy operators and the solution to apply them to.

        :param list_sol: List with a single argument which is the solution to be modified.
        """
        while True:
            destroy = self.select_method(self.meths_destroy,
                                         np.fromiter((self.score_data[m.name].weight for m in self.meths_destroy),
                                                     dtype=float, count=len(self.meths_destroy)))
            repair = self.select_method(self.meths_repair,
                                        np.fromiter((self.score_data[m.name].weight for m in self.meths_repair),
                                                    dtype=float, count=len(self.meths_repair)))
            yield destroy, repair, list_sol[0]

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
        next_segment = self.iteration + self.own_settings.mh_alns_segment_size
        sol_incumbent = sol.copy()
        list_sol = [sol.copy()]
        operators = self.operators_generator(list_sol)
        worker_seed = 0 if settings.mh_workers > 1 else settings.seed
        with mp.Pool(processes=settings.mh_workers,
                     initializer=self.process_init, initargs=(settings, worker_seed)) as worker_pool:
            result_iter = worker_pool.imap_unordered(self.perform_method_pair_in_worker, operators)
            for result in result_iter:
                # print("Result:", result)
                destroy, repair, sol_new, res, obj_old, t_destroy, t_repair = result
                self.update_stats_for_method_pair(destroy, repair, sol, res, obj_old, t_destroy, t_repair)
                destroy_data = self.score_data[destroy.name]
                repair_data = self.score_data[repair.name]
                destroy_data.applied += 1
                repair_data.applied += 1
                score = 0
                if sol_new.is_better(sol_incumbent):
                    score = self.own_settings.mh_alns_sigma1
                    # print('better than incumbent')
                    sol_incumbent.copy_from(sol_new)
                    sol.copy_from(sol_new)
                    list_sol[0] = sol_new
                elif sol_new.is_better(sol):
                    score = self.own_settings.mh_alns_sigma2
                    # print('better than current')
                    sol.copy_from(sol_new)
                    list_sol[0] = sol_new
                elif sol.is_better(sol_new) and self.metropolis_criterion(sol_new, sol):
                    score = self.own_settings.mh_alns_sigma3
                    # print('accepted although worse')
                    sol.copy_from(sol_new)
                    list_sol[0] = sol_new
                elif sol_new != sol:
                    sol_new.copy_from(sol)
                destroy_data.score += score
                repair_data.score += score
                if res.terminate:
                    sol.copy_from(sol_incumbent)
                    return
                if self.iteration == next_segment:
                    # end of segment: update weights and re-initialize scores
                    if self.own_settings.mh_alns_logscores:
                        self.log_scores()
                    next_segment = self.iteration + self.own_settings.mh_alns_segment_size
                    gamma = self.own_settings.mh_alns_gamma
                    for m in chain(self.meths_destroy, self.meths_repair):
                        data = self.score_data[m.name]
                        if data.applied:
                            data.weight = data.weight * (1 - gamma) + gamma * data.score / data.applied
                            data.score = 0
                            data.applied = 0
        print("Finish")
