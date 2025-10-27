"""
# HPRLP.jl

Julia interface for the HPR-LP (Halpern-Peaceman-Rachford Linear Programming) CUDA solver.

This module provides Julia bindings to the HPRLP C/C++ library for solving 
large-scale linear programming problems using GPU acceleration.

## Quick Start

```julia
using HPRLP
using SparseArrays

# Create model from arrays: minimize c'x subject to AL <= Ax <= AU, l <= x <= u
A = sparse([1.0 2.0; 3.0 1.0])
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l = [0.0, 0.0]
u = [Inf, Inf]
c = [-3.0, -5.0]

model = Model(A, AL, AU, l, u, c)
result = solve(model)

# Check result
if result.status == "OPTIMAL"
    println("Optimal value: ", result.primal_obj)
    println("Solution: ", result.x)
end
```

## Solving from MPS File

```julia
model = Model("problem.mps")
result = solve(model)
```

## Model Reuse with Custom Parameters

```julia
# Create model once
model = Model(A, AL, AU, l, u, c)

# Solve with default parameters
result1 = solve(model)

# Solve again with different parameters
params = Parameters(
    max_iter = 10000,
    stop_tol = 1e-9,
    time_limit = 3600.0,
    device_number = 0
)
result2 = solve(model, params)

# Model is automatically freed when garbage collected
```

For more details, see the documentation for `Model`, `solve`, `Parameters`, and `Results`.
"""
module HPRLP

using SparseArrays
using LinearAlgebra
using JuMP

export Model, Parameters, Results, solve, free, is_optimal

# Include submodules
include("wrapper.jl")
include("interface.jl")
include("utils.jl")

end # module
