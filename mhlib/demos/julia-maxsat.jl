"""Demo application for showing the integration of a Julia module, solving the MAXSAT problem.

The Julia module julia-maxsat.jl is used via Python's julia interface package.
It provides a concrete Solution class for solving the MAXSAT problem in essentially
the same way as maxsat.py. The goal is to maximize the number of clauses satisfied in a
boolean function given in conjunctive normal form.
"""

module JuliaMAXSAT

using PyCall

math = pyimport("math")
println("JuliaMAXSAT initialized", math.sin(3))

py_solution = pyimport("mhlib.solution")
show(py_solution)

@pydef mutable struct JuliaMAXSATSolution <: (py_solution.BoolVectorSolution)
    function test(self, x=10)
        println("JuliaMAXSAT.test")
    end
end

sol = JuliaMAXSATSolution()

end