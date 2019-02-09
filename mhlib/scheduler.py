"""General scheduler for realizing (G)VNS, GRASP, IG and similar metaheuristics.

The module is intended for metaheuristics in which a set of methods (or several of them) are
in some way repeatedly applied to candidate solutions.
"""

from typing import Callable, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import time
import logging
from math import log10

from mhlib.settings import settings, get_settings_parser
from mhlib.solution import Solution, TObj


parser = get_settings_parser()
parser.add("--mh_titer", type=int, default=100,
           help='maximum number of iterations (<0: turned off)')
parser.add("--mh_tciter", type=int, default=-1,
           help='maximum number of iterations without improvement (<0: turned off)')
parser.add("--mh_ttime", type=int, default=-1,
           help='time limit [s] (<0: turned off)')
parser.add("--mh_tctime", type=int, default=-1,
           help='maximum time [s] without improvement (<0: turned off)')
parser.add("--mh_tobj", type=float, default=-1,
           help='objective value at which should be terminated when reached (<0: turned off)')
parser.add("--mh_lnewinc", default=True, action='store_true',
           help='write iteration log if new incumbent solution')
parser.add("--no_mh_lnewinc", dest='mh_lnewinc', action='store_false')
parser.add("--mh_lfreq", type=int, default=0,
           help='frequency of writing iteration logs (0: none, >0: number of iterations, -1: iteration 1,2,5,10,20,...')


@dataclass
class Result:
    """Data in conjunction with a method application's result.

    Attributes
        - changed: if false, the solution has not been changed by the method application
        - terminate: if true, a termination condition has been fulfilled
    """
    changed: bool = True
    terminate: bool = False


@dataclass
class Method:
    """A method to be applied by the scheduler.

    Attributes
        - name: name of the method; must be unique over all used methods
        - method: a function called for a Solution object
        - par: a parameter provided when calling the method
    """
    name: str
    func: Callable[[Solution, Any, Result], None]
    par: Any


@dataclass
class MethodStatistics:
    """Class that collects data on the applications of a Method.

    Attributes
        - applications: number of applications of this method
        - netto_time: accumulated time of all applications of this method without further costs (e.g., VND)
        - successes: number of applications in which an improved solution was found
        - obj_gain: sum of gains in the objective values over all successful applications
        - brutto_time: accumulated time of all applications of this method including further costs (e.g., VND)
    """
    applications: int = 0
    netto_time: float = 0.0
    successes: int = 0
    obj_gain: float = 0.0
    brutto_time: float = 0.0


class Scheduler(ABC):
    """Abstract class for metaheuristics that work by iteratively applying certain operators.

    Attributes
        - incumbent: incumbent solution, i.e., initial solution and finally solution so far encountered
        - incumbent_iteration: iteration in which incumbent was found
        - incumbent_time: time at which incumbent was found
        - methods: list of all Methods
        - method_stats: dict of MethodStatistics for each Method
        - iteration: overall number of method applications
        - time_start: starting time of algorithm
        - run_time: overall runtime (set when terminating)
        - logger: mhlib's logger for logging general info
        - iter_logger: mhlib's logger for logging iteration info
    """
    eps = 1e-12  # epsilon value for is_logarithmic_number()
    log10_2 = log10(2)  # log10(2)
    log10_5 = log10(5)  # log10(5)

    def __init__(self, sol: Solution, methods: List[Method]):
        self.incumbent = sol
        self.incumbent_iteration = 0
        self.incumbent_time = 0.0
        self.methods = methods
        self.method_stats = {method.name: MethodStatistics() for method in methods}
        self.iteration = 0
        self.time_start = time.process_time()
        self.run_time = None
        self.logger = logging.getLogger("mhlib")
        self.iter_logger = logging.getLogger("mhlib_iter")
        self.log_iteration_header()
        self.log_iteration(None, sol, True, True)

    def update_incumbent(self, sol, current_time):
        """If the given solution is better than incumbent (or we do not have an incumbent yet) update it."""
        if not self.incumbent or sol.is_better(self.incumbent):
            self.incumbent.copy_from(sol)
            self.incumbent_iteration = self.iteration
            self.incumbent_time = current_time
            return True

    @staticmethod
    def next_method(meths: List, randomize: bool, repeat: bool):
        """Generator for obtaining a next method from a given list of methods.

        Parameters
            - meths: List of methods
            - randomize: random order, otherwise consider given order
            - repeat: repeat infinitely, otherwise just do one pass
        """
        if randomize:
            meths = meths.copy()
        while True:
            if randomize:
                random.shuffle(meths)
            for method in meths:
                yield method
            if not repeat:
                break

    def perform_method(self, method: Method, sol: Solution, delayed_success=False) -> Result:
        """Performs method on given solution and returns Results object.

        Also updates incumbent, iteration and the method's statistics in method_stats.
        Furthermore checks the termination condition and eventually sets terminate in the returned Results object.

        Parameters
            - method: method to be performed
            - sol: solution to which the method is applied
            - delayed_success: if set the success is not immediately determined and updated but at some later
                call of delayed_success_update()

        Returns
            - Results object
        """
        res = Result()
        obj_old = sol.obj()
        t_start = time.process_time()
        method.func(sol, method.par, res)
        t_end = time.process_time()
        sol.check()
        ms = self.method_stats[method.name]
        ms.applications += 1
        ms.netto_time += t_end - t_start
        obj_new = sol.obj()
        if not delayed_success:
            ms.brutto_time += t_end - t_start
            if sol.is_better_obj(sol.obj(), obj_old):
                ms.successes += 1
                ms.obj_gain += obj_new - obj_old
        self.iteration += 1
        new_incumbent = self.update_incumbent(sol, t_end - self.time_start)
        terminate = self.check_termination()
        self.log_iteration(method, sol, new_incumbent, terminate)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = True
        return res

    def delayed_success_update(self, method: Method, obj_old: TObj, t_start: TObj, sol: Solution):
        """Update an earlier performed method's success information in method_stats.

        Parameters
            - method: earlier performed method
            - old_obj: objective value of solution to which method had been applied
            - t_start: time when the application of method dad started
            - sol: current solution considered the final result of the method
        """
        t_end = time.process_time()
        ms = self.method_stats[method.name]
        ms.brutto_time += t_end - t_start
        obj_new = sol.obj()
        if sol.is_better_obj(sol.obj(), obj_old):
            ms.successes += 1
            ms.obj_gain += obj_new - obj_old

    def check_termination(self):
        """Check termination conditions and return True when to terminate."""
        t = time.process_time()
        if 0 <= settings.mh_titer <= self.iteration or \
                0 <= settings.mh_tciter <= self.iteration - self.incumbent_iteration or \
                0 <= settings.mh_ttime <= t - self.time_start or \
                0 <= settings.mh_tctime <= t - self.incumbent_time or \
                0 <= settings.mh_tobj and not self.incumbent.is_worse_obj(self.incumbent.obj(), settings.mh_tobj):
            return True

    def log_iteration_header(self):
        """Writes iteration log header."""
        s = f"{'iteration':>10} {'best':>17} {'current':>12} {'time':>12} {'method':>7}"
        self.iter_logger.info(s)

    @staticmethod
    def is_logarithmic_number(x: int):
        lr = log10(x) % 1
        return abs(lr) < __class__.eps or abs(lr-__class__.log10_2) < __class__.eps or \
            abs(lr-__class__.log10_5) < __class__.eps

    def log_iteration(self, method: Optional[Method], sol: Solution, new_incumbent: bool, in_any_case: bool):
        """Writes iteration log info.

        A line is written if in_any_case is set or in dependence of settings.mh_lfreq and settings.mh_lnewinc.

        Parameters
            - method: applied method or None (if initially given solution)
            - sol: current solution
            - new_incumbent: true if the method yielded a new incumbent solution
            - in_any_case: turns filtering of iteration logs off
        """
        log = in_any_case or new_incumbent and settings.mh_lnewinc
        if not log:
            lfreq = settings.mh_lfreq
            if lfreq > 0 and self.iteration % lfreq == 0:
                log = True
            elif lfreq < 0 and self.is_logarithmic_number(self.iteration):
                log = True
        if log:
            name = method.name if method else '-'
            s = f"{self.iteration:>10d} {self.incumbent.obj():16.5f} {sol.obj():16.5f} " \
                f"{time.process_time()-self.time_start:9.4f} {name:>7}"
            self.iter_logger.info(s)

    @abstractmethod
    def run(self):
        """Actually performs the optimization."""
        pass

    @staticmethod
    def sdiv(x, y):
        """Safe division: return x/y if y!=0 and nan otherwise."""
        if y == 0:
            return float('nan')
        else:
            return x/y

    def method_statistics(self):
        """Write overall statistics."""
        if not self.run_time:
            self.run_time = time.process_time() - self.time_start
        s = "Method statistics:\n"
        s += " method    iter   succ succ-rate%  tot-obj-gain  avg-obj-gain rel-succ%  net-time  " \
             "net-time%  brut-time  brut-time%\n"
        total_applications = 0
        total_netto_time = 0.0
        total_successes = 0
        total_brutto_time = 0.0
        total_obj_gain = 0.0
        for ms in self.method_stats.values():
            total_applications += ms.applications
            total_netto_time += ms.netto_time
            total_successes += ms.successes
            total_brutto_time += ms.brutto_time
            total_obj_gain += ms.obj_gain

        for name, ms in self.method_stats.items():
            s += f"{name:>7} {ms.applications:7d} {ms.successes:6d} " \
                 f"{self.sdiv(ms.successes, ms.applications)*100:10.4f} " \
                 f"{ms.obj_gain:13.5f} {self.sdiv(ms.obj_gain, ms.applications):13.5f} " \
                 f"{self.sdiv(ms.successes, total_successes)*100:9.4f} " \
                 f"{ms.netto_time:9.4f} {self.sdiv(ms.netto_time, self.run_time)*100:10.4f} " \
                 f"{ms.brutto_time:10.4f} {self.sdiv(ms.brutto_time, self.run_time)*100:11.4f}\n"
        s += f"{'SUM/AVG':>7} {total_applications:7d} {total_successes:6d} " \
             f"{self.sdiv(total_successes, total_applications)*100:10.4f} " \
             f"{total_obj_gain:13.5f} {self.sdiv(total_obj_gain, total_applications):13.5f} " \
             f"{self.sdiv(self.sdiv(total_successes, len(self.method_stats)), total_successes)*100:9.4f} " \
             f"{total_netto_time:9.4f} {self.sdiv(total_netto_time, self.run_time)*100:10.4f} " \
             f"{total_brutto_time:10.4f} {self.sdiv(total_brutto_time, self.run_time)*100:11.4f}\n"
        self.logger.info(s)

    def main_results(self):
        """Write main results to logger."""
        s = f"best solution: {self.incumbent}\nbest obj: {self.incumbent.obj()}\n" \
            f"best iteration: {self.incumbent_iteration}\n" \
            f"total iterations: {self.iteration}\n" \
            f"best time [s]: {self.incumbent_time:.3f}\n" \
            f"total time [s]: {self.run_time:.4f}\n"
        self.logger.info(s)
        self.incumbent.check()


class GVNS(Scheduler):
    """A general variable neighborhood search (GVNS).

    Attributes
        - sol: solution object, in which final result will be returned
        - meths_ch: list of construction heuristic methods
        - meths_li: list of local improvement methods
        - meths_sh: list of shaking methods
    """

    def __init__(self, sol: Solution, meths_ch: List[Method], meths_li: List[Method], meths_sh: List[Method]):
        """Initialization.

        Parameters
            - meths_ch: list of construction heuristic methods
            - meths_li: list of local improvement methods
            - meths_sh: list of shaking methods
            - incumbent: incumbent solution, i.e., best solution so far
        """
        super().__init__(sol, meths_ch+meths_li+meths_sh)
        self.meths_ch = meths_ch
        self.meths_li = meths_li
        self.meths_sh = meths_sh

    def vnd(self, sol: Solution) -> bool:
        """Perform variable neighborhood descent (VND) on given solution.

        Returns true if a global  termination condition is fulfilled, else False.
        """
        sol2 = sol.copy()
        while True:
            for m in self.next_method(self.meths_li, False, False):
                res = self.perform_method(m, sol2)
                if sol2.is_better(sol):
                    sol.copy_from(sol2)
                    if res.terminate:
                        return True
                    break
                else:
                    if res.terminate:
                        return True
                    if res.changed:
                        sol2.copy_from(sol)
            else:  # local optimum reached
                return False

    def gvns(self, sol: Solution):
        """Perform general variable neighborhood search (GVNS) to given solution."""
        sol2 = sol.copy()
        self.vnd(sol2)
        use_vnd = bool(self.meths_li)
        while True:
            for m in self.next_method(self.meths_sh, False, True):
                t_start = time.process_time()
                res = self.perform_method(m, sol2, delayed_success=use_vnd)
                terminate = res.terminate
                if not terminate and use_vnd:
                    terminate = self.vnd(sol2)
                self.delayed_success_update(m, sol.obj(), t_start, sol2)
                if sol2.is_better(sol):
                    sol.copy_from(sol2)
                    if terminate or res.terminate:
                        return
                    break
                else:
                    if terminate or res.terminate:
                        return
                    sol2.copy_from(sol)
            else:
                break

    def run(self):
        """Actually performs the construction heuristics followed by the GVNS."""
        sol = self.incumbent.copy()

        # perform all construction heuristics, take best solution
        for m in self.next_method(self.meths_ch, False, False):
            res = self.perform_method(m, sol)
            if res.terminate:
                break
        if self.incumbent.is_better(sol):
            sol.copy_from(self.incumbent)

        self.gvns(sol)
