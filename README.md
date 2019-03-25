## `mhlib` - A Toolbox for Metaheuristics and Hybrid Optimization Methods

_This project is still in its early phase of development, please come back later!_

`mhlib` is a collection of modules supporting the efficient implementation of metaheuristics 
and certain hybrid optimization approaches for solving primarily combinatorial optimization 
problems in Python.

![ ](mh.png =8x)

This Python `mhlib` version emerged from the 
[C++ `mhlib`](https://bitbucket.org/ads-tuwien/mhlib) to which it has certain similarities 
but also many differences.

The main purpose of the library is to support rapid prototyping and teaching. 
While ultimately efficient implementations of such algorithms in compiled 
languages like C++ will likely be faster, the expected advantage of the Python
implementation lies in the expected faster implementation.

`mhlib` is developed primarily by the 
[Algorithms and Complexity Group of TU Wien](https://www.ac.tuwien.ac.at), 
Vienna, Austria, since 2019.

#### Contributors:
- [Günther Raidl](https://www.ac.tuwien.ac.at/raidl) (mainly responsible)
- Nikolaus Frohner
- Thomas Jatschka
- Daniel Obszelka
- Andreas Windbichler

### Major Components

- **log.py**:
    Provides two logger objects, one for writing out general log information, one for 
    iteration-wise.
- **settings.py**:
    Allows for defining module-specific parameters directly in each module in a distributed
    way, while values for these parameters can be provided as program arguments or in
    configuration files.
- **solution.py**:
    An abstract base class that represents a candidate solution to the optimization problem.
- **subset_solution.py**:
    A more specific solution class for problems in which solutions are subsets of a 
    larger set.
- **permutation_solution.py**:
    A more specific solution class for problems in which solutions are permutations of a
    set of elements.
- **scheduler.py**:
    A general framework for local search, iterated local search, variable neighborhood 
    search, GRASP, etc.
- **decision_diag.py**:
    A generic class for (relaxed) decision diagrams for optimization.
    
#### Demos

For demonstration purposes, simple metaheuristics are provided for the following
well-known combinatorial optimization problems.

- Maximum satisfiability problem
- Maximum independent set problem
- Multidimensional 0-1 knapsack problem
- Quadratic assignment problem    
         

### Changelog: major changes over major releases ##

#### Version 0.0.1 - Initial version