#!/usr/local/bin/julia
"""Demo application for showing the integration with the Julia language, solving the MAXSAT problem.

Julia must be installed properly with package PyCall.
This here is a Julia main program that uses pymhlib for the MAXSAT problem instance and
metaheuristics but julia-maxsat.jl for the concrete solution class
for performance reasons.

Note that there is the alternative Python main module julia-maxsat.py.

The goal in the MAXSAT problem is to maximize the number of clauses satisfied in a boolean
function given in conjunctive normal form.
"""

ENV["PYTHONPATH"] = "."
using PyCall

maxsat = pyimport("pymhlib.demos.maxsat")

include("julia-maxsat.jl")
# using  JuliaMAXSAT

common = pyimport("pymhlib.demos.common")
settings = pyimport("pymhlib.settings")
parser = settings.get_settings_parser()
parser.set_defaults(mh_titer=1000)
common.run_optimization("Julia-MAXSAT", JuliaMAXSAT.JuliaMAXSATInstance,
    JuliaMAXSAT.JuliaMAXSATSolution, common.data_dir*"maxsat-adv1.cnf")
