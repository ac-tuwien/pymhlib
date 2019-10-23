"""Demo application for showing the integration with the Julia language, solving the MAXSAT problem.

Julia and Python's julia package must be installed properly.
The Julia module julia-maxsat.jl is used via Python's julia interface package.
It provides a concrete Solution class for solving the MAXSAT problem in essentially the same way as maxsat.py.
The goal is to maximize the number of clauses satisfied in a boolean function given in conjunctive normal form.
"""

from julia import Julia
#jl = Julia(sysimage="/home/guenther/s.so")  # only use when compiled Julia system image available
from julia import Base, Main
import os

from mhlib.demos.maxsat import MAXSATInstance, MAXSATSolution

Main.eval(r'include("'+os.path.dirname(__file__)+r'/julia-maxsat.jl")')

if __name__ == '__main__':
    from mhlib.demos.common import run_optimization, data_dir
    print(Main, Base)
    run_optimization('Julia-MAXSAT', MAXSATInstance, Main.JuliaMAXSAT.JuliaMAXSATSolution, data_dir+"advanced.cnf")
