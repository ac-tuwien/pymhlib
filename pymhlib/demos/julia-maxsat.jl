"""Demo application for showing the integration with the Julia language, solving the MAXSAT problem.

The Julia module julia-maxsat.jl is used via Python's julia interface package.
Julia and Python's julia package must be installed properly.
This Julia module provides a concrete Solution class for solving the MAXSAT problem in essentially
the same way as maxsat.py. The goal is to maximize the number of clauses satisfied in a
boolean function given in conjunctive normal form.
"""

module JuliaMAXSAT

using PyCall
using Random
# using StatsBase

# math = pyimport("math")
# println("JuliaMAXSAT initialized", math.sin(3))

py_maxsat = pyimport("pymhlib.demos.maxsat")
py_solution = pyimport("pymhlib.binvec_solution")


"""
    JuliaMAXSATInstance

Python instance class that augments MAXSATInstance by a reference julia_inst to Julia-specific
instance data for a better performance.
"""
@pydef mutable struct JuliaMAXSATInstance <: py_maxsat.MAXSATInstance
    function __init__(self, file_name)
        pybuiltin(:super)(JuliaMAXSATInstance,self).__init__(file_name)
        self.julia_inst = JMAXSATInstance(self)
    end
end


"""
    JMAXSATInstance

Juliy-specific instance data, which are derived from Python's MAXSATInstance.
This separate structure enables a better performance of the Julia code since
no data conversion has to take place at each lookup of some instance data.
"""
struct JMAXSATInstance
    clauses::Array{Int,2}
    variable_usage::Array{Int,2}

    function JMAXSATInstance(inst::PyObject)
        cl::Vector{Vector{Int}} = inst."clauses"
        vu::Vector{Vector{Int}} = inst."variable_usage"
        for c in vu c .+= 1 end
        return new(make_2d_array(cl), make_2d_array(vu))
    end
end

"""
    make_2d_array(a, fill)

Turns an Array of Arrays of Ints into a 2D array, filling up with zeros.
"""
function make_2d_array(a::Vector{Vector{Int}})::Array{Int,2}
    size1 = length(a)
    size2 = maximum(length(a[i]) for i in 1:size1)
    a2 = zeros(Int, size2, size1)
    for i in 1:size1
        a2[1:length(a[i]),i] = a[i]
    end
    return a2
end


"""Solution to a MAXSAT instance.

Attributes
    - inst: associated MAXSATInstance
    - x: binary incidence vector
    - destroyed: list of indices of variables that have been destroyed by the ALNS's destroy op.
"""
@pydef mutable struct JuliaMAXSATSolution <: py_solution.BinaryVectorSolution

    to_maximize = true

    function __init__(self, inst)
        pybuiltin(:super)(JuliaMAXSATSolution,self).__init__(inst.n, inst=inst)
        self.destroyed = nothing
    end

    function calc_objective(self)
        """Count the number of satisfied clauses."""
        return obj(self.x, self.inst.julia_inst)
    end

    function copy(self)
        sol = JuliaMAXSATSolution(self.inst)
        sol.copy_from(self)
        return sol
    end

    function check(self)
        x = PyArray(self."x")
        if length(x) != self.inst.n
            throw(DomainError("Invalid length of solution"))
        end
        pybuiltin(:super)(JuliaMAXSATSolution,self).check()
    end

    function construct(self, par, _result)
        """
        Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        self.initialize(par)
    end

    function local_improve(self, par, _result)
        """Perform one k_flip_neighborhood_search."""
        x = self.x
        obj_val = self.obj()
        new_obj_val = k_flip_neighborhood_search!(x, obj_val, self.inst.julia_inst, par, false)
        if new_obj_val > obj_val
            PyArray(self."x")[:] = x
            self.obj_val = new_obj_val
            return true
        end
        return false
    end

    function shaking(self, par, _result)
        """Scheduler method that performs shaking by flipping par random positions."""
        x = PyArray(self."x")
        for i in 1:par
            p = rand(1:length(x))
            x[p] = !x[p]
        end
        self.invalidate()
    end

    function destroy(self, par, _result)
        """Destroy operator for ALNS selects par*ALNS.get_number_to_destroy positions
        uniformly at random for removal.

        Selected positions are stored with the solution in list self.destroyed.
        """
        x = PyArray(self."x")
        num = min(ALNS.get_number_to_destroy(length(x)) * par, length(x))
        self.destroyed = sample(1:length(x), num, replace=false)
        self.invalidate()
    end

    function repair(self, _par, _result)
        """Repair operator for ALNS assigns new random values to all positions in self.destroyed."""
        @assert !(self.destroyed === nothing)
        x = PyArray(self."x")
        for p in self.destroyed
            x[p] = rand(0:1)
        end
        self.destroyed = nothing
        self.invalidate()
    end


    function crossover(self, other)
        """ Perform uniform crossover as crossover."""
        return self.uniform_crossover(other)
    end

end


function obj(x::Vector{Bool}, julia_inst::JMAXSATInstance)
    """Count the number of satisfied clauses."""
    fulfilled_clauses = 0
    for clause in eachcol(julia_inst.clauses)
        for v in clause
            if v == 0
                break
            end
            if x[abs(v)] == (v > 0)
                fulfilled_clauses += 1
                break
            end
        end
    end
    return fulfilled_clauses
end


function locimp(self)::Bool
    """Perform one k_flip_neighborhood_search."""
    x = self.x
    obj_val = self.obj()
    new_obj_val = k_flip_neighborhood_search!(x, obj_val, self.inst.julia_inst, 1, false)
    if new_obj_val > obj_val
        PyArray(self."x")[:] = x
        self.obj_val = new_obj_val
        return true
    end
    return false
end

function k_flip_neighborhood_search!(x::Vector{Bool}, obj_val::Int, julia_inst::JMAXSATInstance,
                                     k::Int, best_improvement::Bool)::Int
    """Perform one major iteration of a k-flip local search, i.e., search one neighborhood.

    If best_improvement is set, the neighborhood is completely searched and a best neighbor is
    kept; otherwise the search terminates in a first-improvement manner, i.e., keeping a first
    encountered better solution.

    :returns: Objective value.
    """
    len_x = length(x)
    @assert 0 < k <= len_x
    better_found = false
    best_sol = copy(x)
    best_obj = obj_val
    perm = randperm(len_x)  # random permutation for randomizing enumeration order
    p = fill(-1, k)  # flipped positions
    # initialize
    i = 1  # current index in p to consider
    while i >= 1
        # evaluate solution
        if i == k + 1
            if obj_val > best_obj
                if !best_improvement
                    return true
                end
                best_sol[:] = x
                best_obj = obj_val
                better_found = true
            end
            i -= 1  # backtrack
        else
            if p[i] == -1
                # this index has not yet been placed
                p[i] = (i>1 ? p[i-1] : 0) + 1
                obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
                i += 1  # continue with next position (if any)
            elseif p[i] < len_x - (k - i)
                # further positions to explore with this index
                obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
                p[i] += 1
                obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
                i += 1
            else
                # we are at the last position with the i-th index, backtrack
                obj_val = flip_variable!(x, perm[p[i]], julia_inst, obj_val)
                p[i] = -1  # unset position
                i -= 1
            end
        end
    end
    if better_found
        x[:] = best_sol
        obj_val = best_obj
    end
    return obj_val
end


function flip_variable!(x::Vector{Bool}, pos::Int, julia_inst::JMAXSATInstance, obj_val::Int)::Int
    val = !x[pos]
    x[pos] = val
    for clause in view(julia_inst.variable_usage,:,pos)
        if clause == 0 break end
        fulfilled_by_other = false
        val_fulfills_now = false
        for v in view(julia_inst.clauses,:,clause)
            if v == 0 break end
            if abs(v) == pos
                val_fulfills_now = (v>0 ? val : !val)
            elseif x[abs(v)] == (v>0 ? 1 : 0)
                fulfilled_by_other = true
                break  # clause fulfilled by other variable, no change
            end
        end
        if !fulfilled_by_other
            obj_val += (val_fulfills_now ? 1 : -1)
        end
    end
    return obj_val
end


# using Random
# Random.seed!(3)

end  # module JuliaMAXSAT
