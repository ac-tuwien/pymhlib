"""Demo application for showing the integration with the Julia language, solving the MAXSAT problem.

Julia and Python's julia package must be installed properly.
The Julia module julia-maxsat.jl is used via Python's julia interface package.
It provides a concrete solution class for solving the MAXSAT problem in essentially the same way as maxsat.py.

Note that there is the alternative Julia main module julia-maxsat.py.

The goal in the MAXSAT problem is to maximize the number of clauses satisfied in a boolean function given in
conjunctive normal form.
"""

# from julia import Julia
# jl = Julia(sysimage="/home/guenther/s.so")  # only use when compiled Julia system image available
from julia import Base, Main
import os

from pymhlib.demos.maxsat import MAXSATInstance

Main.eval(r'include("'+os.path.dirname(__file__)+r'/julia-maxsat.jl")')

if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)
    run_optimization('Julia-MAXSAT',Main.JuliaMAXSAT.JuliaMAXSATInstance, Main.JuliaMAXSAT.JuliaMAXSATSolution,
                     data_dir+"maxsat-adv1.cnf")
