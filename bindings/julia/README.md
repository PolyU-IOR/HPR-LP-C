# HPRLP-C Julia Interface

Julia interface for HPRLP-C (Halpern-Peaceman-Rachford Linear Programming) solver - a GPU-accelerated linear programming solver using C/CUDA.

## Installation
```bash
cd bindings/julia
bash install.sh
```
This script compiles the backend and installs the Julia package of HPR-LP-C locally.

## Examples
### Example 1: Build model directly from matrices
```bash
cd examples
julia --project=../package example_direct_lp.jl
```

**Quick overview:**
The following snippet demonstrates how to define and solve a small LP problem directly from matrices.
For a complete version with additional options, see example_direct_lp.jl.
```julia
using HPRLP
using SparseArrays

# Define LP: minimize c'x subject to AL ≤ A*x ≤ AU, l ≤ x ≤ u
A  = sparse([1.0 2.0; 3.0 1.0])
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l  = [0.0, 0.0]
u  = [Inf, Inf]
c  = [-3.0, -5.0]

# Create model
model = Model(A, AL, AU, l, u, c)

# Configure solver parameters
params = Parameters(
    stop_tol = 1e-9,
    device_number = 0
)

# Solve
result = solve(model, params)

if is_optimal(result)
    println("Optimal objective: ", result.primal_obj)
    println("Optimal solution:  ", result.x)
end
```
---
### `Model(A, AL, AU, l, u, c)`
Create an LP model directly from matrices and vectors.

| Argument | Description |
|-----------|--------------|
| `A` | Sparse constraint matrix (m×n) |
| `AL`, `AU` | Constraint bounds |
| `l`, `u` | Variable bounds |
| `c` | Objective coefficients |

**Returns:** `Model` object

## Example 2: Read model from an MPS file

```
cd examples
julia --project=../package example_mps_file.jl
```

**Quick overview:**
The following snippet demonstrates how to define and solve an LP problem directly from an MPS file. For a complete version with additional options, see example_mps_file.jl.
```julia
using HPRLP

# Create model from MPS file
model = Model("problem.mps")

# Solve
result = solve(model)
println(result)
```

---

### `Model(filename::String)`
Create an LP model by reading an MPS file.

| Argument | Description |
|-----------|--------------|
| `filename` | Path to the MPS file |

**Returns:** `Model` object

---

### `solve(model::Model, params=nothing)`
Solve the given LP model.

| Argument | Description |
|-----------|--------------|
| `model` | LP model object |
| `params` | Optional solver parameters |
| **Returns** | `Results` object |

---


## Example 3: Use JuMP to construct and solve LPs

```
cd examples
julia --project=../package example_jump.jl
```


## `Parameters`
Solver configuration options (specified as keyword arguments):

| Parameter | Description | Default |
|------------|-------------|----------|
| `max_iter` | Maximum iterations | Unlimited |
| `stop_tol` | Stopping tolerance | `1e-4` |
| `time_limit` | Time limit in seconds | `3600` |
| `device_number` | CUDA device ID | `0` |
| `check_iter` | Convergence check interval | `150` |
| `use_Ruiz_scaling` | Apply Ruiz scaling | `true` |
| `use_Pock_Chambolle_scaling` | Apply Pock–Chambolle scaling | `true` |
| `use_bc_scaling` | Apply bounds/cost scaling | `true` |

---

## `Results`
The `Results` object contains solution and performance information after solving the LP:

| Field | Description |
|--------|-------------|
| `status` | Solver status: `"OPTIMAL"`, `"TIME_LIMIT"`, `"ITER_LIMIT"`, `"ERROR"` |
| `x` | Primal solution |
| `y` | Dual solution |
| `primal_obj` | Primal objective value |
| `gap` | Duality gap |
| `residuals` | Final KKT residual |
| `iter` | Total iterations |
| `time` | Solve time (seconds) |
| `iter4`, `iter6`, `iter8` | Iterations to reach 1e-4/1e-6/1e-8 tolerance |
| `time4`, `time6`, `time8` | Time to reach corresponding tolerance |
