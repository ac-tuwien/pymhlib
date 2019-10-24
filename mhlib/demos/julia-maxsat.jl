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

py_instance = pyimport("mhlib.demos.maxsat")
py_solution = pyimport("mhlib.binvec_solution")

"""Solution to a MAXSAT instance.

Attributes
    - inst: associated MAXSATInstance
    - x: binary incidence vector
    - destroyed: list of indices of variables that have been destroyed by the ALNS's destroy op.
"""

"""
    JuliaMAXSATInstance

    Additional instance data specifically for Julia for a better performance.
    Clauses are stored in a 2D array, where values 0 indicate an earlier end of the clause.
"""
struct JuliaMAXSATInstance
    clauses::Array{Int,2}
    variable_usage::Array{Int,2}
end


"""
    make_2d_array(a, fill)

Turns an Array of Arrays of Ints into a 2D array, filling up with zeros.
"""
function make_2d_array(a::Array{Array{Int,1}})::Array{Int,2}
    size1 = length(a)
    size2 = maximum(length(a[i]) for i in 1:size1)
    a2 = zeros(Int, size1, size2)
    for i in 1:size1
        a2[i,1:length(a[i])] = a[i]
    end
    return a2
end


@pydef mutable struct JuliaMAXSATSolution <: (py_solution.BinaryVectorSolution)

    function __init__(self, inst)
        pybuiltin(:super)(JuliaMAXSATSolution,self).__init__(inst.n, inst=inst)
        self.destroyed = nothing
        clauses::Array{Array{Int,1},1} = inst."clauses"
        clauses_2d = make_2d_array(clauses)
        variable_usage::Array{Array{Int,1},1} = inst."variable_usage"
        for clauses in variable_usage clauses .+= 1 end
        variable_usage_2d = make_2d_array(variable_usage)
        self.julia_inst = JuliaMAXSATInstance(clauses_2d, variable_usage_2d)
    end

    function calc_objective(self)
        """Count the number of satisfied clauses."""
        x::Array{Int,1} = self.x
        clauses::Array{Int,2} = self.julia_inst.clauses
        fulfilled_clauses = 0
        for clause in eachrow(clauses)
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
        self.k_flip_neighborhood_search(par, false)
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
        """Destroy operator for ALNS selects par*ALNS.get_number_to_destroy positions uniformly at random for removal.

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

    function k_flip_neighborhood_search(self, k, best_improvement)
        """Perform one major iteration of a k-flip local search, i.e., search one neighborhood.

        If best_improvement is set, the neighborhood is completely searched and a best neighbor is
        kept; otherwise the search terminates in a first-improvement manner, i.e., keeping a first
        encountered better solution.

        :returns: True if an improved solution has been found.
        """
        x = PyArray(self."x")
        len_x = length(x)
        julia_inst = self.julia_inst
        @assert 0 < k <= len_x
        better_found = false
        best_sol = self.copy()
        perm = randperm(len_x)  # random permutation for randomizing enumeration order
        p = fill(-1, k)  # flipped positions
        # initialize
        i = 1  # current index in p to consider
        while i >= 1
            # evaluate solution
            if i == k + 1
                if self.is_better(best_sol)
                    if !best_improvement
                        return true
                    end
                    best_sol.copy_from(self)
                    better_found = true
                end
                i -= 1  # backtrack
            else
                if p[i] == -1
                    # this index has not yet been placed
                    p[i] = (i>1 ? p[i-1] : 0) + 1
                    self.obj_val = flip_variable(x, perm[p[i]], julia_inst, self.obj_val)
                    i += 1  # continue with next position (if any)
                elseif p[i] < len_x - (k - i)
                    # further positions to explore with this index
                    self.obj_val = flip_variable(x, perm[p[i]], julia_inst, self.obj_val)
                    p[i] += 1
                    self.obj_val = flip_variable(x, perm[p[i]], julia_inst, self.obj_val)
                    i += 1
                else
                    # we are at the last position with the i-th index, backtrack
                    self.obj_val = flip_variable(x, perm[p[i]], julia_inst, self.obj_val)
                    p[i] = -1  # unset position
                    i -= 1
                end
            end
        end
        if better_found
            self.copy_from(best_sol)
            self.invalidate()
        end
        return better_found
    end

    function crossover(self, other)
        """ Perform uniform crossover as crossover."""
        return self.uniform_crossover(other)
    end

end


function flip_variable(x::PyArray, pos::Int, julia_inst::JuliaMAXSATInstance, obj_val::Int)::Int
    val = !x[pos]
    x[pos] = val
    for clause in julia_inst.variable_usage[pos,:]
        if clause == 0 break end
        fulfilled_by_other = false
        val_fulfills_now = false
        for v in julia_inst.clauses[clause,:]
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
