"""Demo application for showing the integration of a Julia module, solving the MAXSAT problem.

The Julia module julia-maxsat.jl is used via Python's julia interface package.
It provides a concrete Solution class for solving the MAXSAT problem in essentially the same way as maxsat.py.
The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.
"""

from julia import Julia
# jl = Julia(sysimage="/home1/guenther/sys.so")  # only use when compiled Julia system image available
from julia import Base, Main
import os

from mhlib.demos.maxsat import MAXSATInstance, MAXSATSolution

Main.eval(r'include("'+os.path.dirname(__file__)+r'/julia-maxsat.jl")')

if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir
    # run_optimization('Julia-MAXSAT', MAXSATInstance, MAXSATSolution, data_dir+"advanced.cnf")
