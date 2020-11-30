## `pymhlib` - A Toolbox for Metaheuristics and Hybrid Optimization Methods

Build status: 
[![Build Status](https://travis-ci.org/ac-tuwien/pymhlib.svg?branch=master)](https://travis-ci.org/ac-tuwien/pymhlib)

_This project is still in early development, any feedback is much appreciated!_

`pymhlib` is a collection of modules supporting the efficient implementation of metaheuristics 
and certain hybrid optimization approaches for solving primarily combinatorial optimization 
problems in Python 3.7+.

![ ](mh.png)

This Python `mhlib` version emerged from the 
[C++ `mhlib`](https://bitbucket.org/ads-tuwien/mhlib) to which it has certain similarities 
but also many differences.

Note that there also exists a more recent efficient **Julia-implementation** of this libraries, 
following a similar design concept: [Julia MHLib.jl](https://github.com/ac-tuwien/MHLib.jl)

The main purpose of the library is to support rapid prototyping and teaching. 
While ultimately efficient implementations of such algorithms in compiled 
languages like Julia or C++ will in general be much faster, an advantage of 
the Python implementation lies in the possibly quicker development cycle.


`pymhlib` is developed primarily by the 
[Algorithms and Complexity Group of TU Wien](https://www.ac.tuwien.ac.at), 
Vienna, Austria, since 2019.

#### Contributors:
- [Günther Raidl](https://www.ac.tuwien.ac.at/raidl) (primarily responsible)
- [Nikolaus Frohner](https://www.ac.tuwien.ac.at/nfrohner)
- Thomas Jatschka
- Daniel Obszelka
- Andreas Windbichler

### Installation

Major versions of `pymhlib` can be installed from `PyPI` via

    python3 -m pip install -U pymhlib
    
and development versions are available at https://github.com/ac-tuwien/pymhlib.

### Major Components

- **solution.py**:
    An abstract base class `Solution`that represents a candidate solution to an optimization problem and
    derived classes `VectorSolution`, `BinaryVectorSolution`, and `SetSolution` for solutions which are
    represented bei general fixed-length vectors, boolean vectors or sets of arbitrary elements.
- **binvec_solution.py**:
    A more specific solution class `BinaryVectorSolution` for problems in which solutions are represented by
    fixed-length binary vectors.
- **subsetvec_solution.py**:
    A more specific solution class `SubsetVectorSolution` for problems in which solutions are subsets of a 
    larger set. The set is realized by an efficient numpy array which is split into two parts, 
    the one with the included elements in sorted order and the one with the remaining elements.
- **permutation_solution.py**:
    A more specific solution class `PermutationSolution` for problems in which solutions are permutations of a
    set of elements.
- **scheduler.py**:
    A an abstract framework for single metaheuristics that rely on iteratively applying certain 
    methods to a current solution. Modules like gvns.py and alns.py extend this abstract class towards
    more specific metaheuristics.
- **gvns.py**:
    A framework for local search, iterated local search, (general) variable neighborhood 
    search, GRASP, etc.
- **alns.py**:
    A framework for adaptive large neighborhood search (ALNS).
- **par_alns.py**:
    A multi-process implementation of the ALNS where destroy+repair operations are parallelized.
- **population.py**
    A population class for population-based metaheuristics.
- **pbig.py**:
    A population based iterated greedy (PBIG) algorithm.
- **ssga.py**:
    A steady-state genetic algorithm (SSGA).
- **sa.py**:
    A simulated annealing (SA) algorithm with geometric cooling.
- **decision_diag.py**:
    A generic class for (relaxed) decision diagrams for combinatorial optimization.
- **log.py**:
    Provides two logger objects, one for writing out general log information, which is typically
    written into a `*.out`  file, and one for iteration-wise information, which is typically
    written into a `*.log` file. The latter is buffered in order to work also efficiently, e.g., 
    on network drives and massive detailed log information. 
    A class `LogLevel` is provided for indented writing of log information according to a current level, 
    which might be used for hierarchically embedded components of a larger optimization framework,
    such as a local search that is embedded in a population-based approach.   
- **settings.py**:
    Allows for defining module-specific parameters directly in each module in an independent  distributed
    way, while values for these parameters can be provided as program arguments or in
    configuration files. Most `pyhmlib` modules rely on this mechanism for their external parameters.

Modules/scripts for analyzing results of many runs:

- **multi_run_summary.py**:
    Collects essential information from multiple `pymhlib` algorithm runs found in the respective out and log files
    and returns a corresponding pandas dataframe if used as a module or as a plain ASCII table when used as
    independent script. The module can be easily configured to extract also arbitrary application-specific data.
    
- **aggregate_results.py**:
    Calculate grouped basic statistics for one or two dataframes/TSV files obtained e.g. from `multi-run-summary.py`.
    In particular, two test series with different algorithms or different settings can be statistically
    compared, including Wilcoxon signed rank tests. The module can be used as standalone script as well 
    as module called, e.g., from a jupyter notebook.


#### Demos

For demonstration purposes, simple metaheuristic approaches are provided in the `demo` subdirectory for the following
well-known combinatorial optimization problems. They can be startet by

    python3 -m pymhlib.demos.<problem> ...

where `<problem>` is one of the following and `...` represents further parameters that can be seen by providing 
the option `-h`.
It is recommended to take such a demo as template 
for solving your own problem.

- **`maxsat`**: maximum satisfiability problem based on `BinaryVectorSolution`
- **`tsp`**: traveling salesperson problem based on `PermutationSolution`
- **`qap`**: quadratic assignment problem based on `PermutationSolution`
- **`vertex_cover`**: minimum vertex cover problem based on `SetSolution`
- **`graph_coloring`**: graph coloring problem based on `VectorSolution`
- **`misp`**: maximum (weighted) independent set problem based on `SubsetVectorSolution`
- **`mkp`**: multidimensional 0-1 knapsack problem based on `SubsetVectorSolution`

Shared code of these demos is found in the submodules `pymhlib.demos.common` and `pymhlib.demos.graphs`,
test instance data in `pymhlib.demos.data`.

Moreover, `julia-maxsat.py` and `julia-maxsat.jl` demonstrate the integration with the Julia programming language.
Implementing time-critical parts of an application in Julia may accelerate the code substantially.
To run this demo, Julia must be set up correctly and Python's `julia` package must be installed.
While this demo derives a whole solution class in Julia, `julia-maxsat2.py` is a variant where only two functions 
are realized in Julia.
 

### Changelog

Major changes in releases:

#### Version 0.1.3
- bug fix in 2-opt neighborhood search of permutation representation

#### Version 0.1.2
- directory renamed to pymhlib to correspond to module name
- bug fix in Metropolis criterion of ALNS
- boolean arguments must now be specified in the command line as any other parameter

#### Version 0.1.1
- basic functionality test `tests/test_all.py` for all problems and
  algorithms added
- polishing, minor fixes

#### Version 0.1.0 
- ALNS and parallel ALNS added
- graph coloring, TSP, and minimum vertex cover demos added
- population based iterated greedy and steady state genetic algorithms added
- SA with geometric cooling added
- demos.graphs introduced
- mhlib renamed to pymhlib
- demo for interfacing with Julia added
- many smaller improvements, bug fixes, improvements in documentation 

#### Version 0.0.1 
- Initial version
