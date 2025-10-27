# HPR-LP-C MATLAB Interface

MATLAB interface for HPR-LP-C (Halpern–Peaceman–Rachford Linear Programming) solver — a GPU-accelerated linear programming solver using C/CUDA.

---

## Prerequisites

- MATLAB **R2018a** or later  
- **CUDA Toolkit** (for compilation)  
- **GCC-11** or **GCC-12** (recommended)  
- **Linux** operating system 

**Note:** The MEX interface requires access to CUDA headers during compilation. Make sure CUDA is installed and either:
- Set `CUDA_HOME` environment variable, or
- Set `CUDA_PATH` environment variable, or
- Have CUDA installed in `/usr/local/cuda` or `/opt/cuda`

## Installation

```bash
cd bindings/matlab
bash install.sh
```

This will:
- Detect CUDA installation
- Build the C++/CUDA library
- Compile the MEX interface with CUDA headers
- Set up the HPRLP package

## Examples

### Example 1: Build model directly from matrices

```bash
cd examples
matlab -batch "example_direct_lp"              # Solve from arrays
```

Or run interactively in MATLAB:
```matlab
cd examples
example_direct_lp
```

**Quick overview**:
The following snippet demonstrates how to define and solve an LP problem directly from matrices.
For a complete version with additional options, see example_direct_lp.m.


```matlab
% Define LP: minimize c'x subject to AL <= Ax <= AU, l <= x <= u
A = sparse([1.0, 2.0; 3.0, 1.0]);
AL = [-inf; -inf];
AU = [10.0; 12.0];
l = [0.0; 0.0];
u = [inf; inf];
c = [-3.0; -5.0];

% Create model
model = hprlp.Model.from_arrays(A, AL, AU, l, u, c);

% Configure solver
param = hprlp.Parameters();
param.stop_tol = 1e-9;
param.device_number = 0;

% Solve
result = model.solve(param);

if strcmp(result.status, 'OPTIMAL')
    fprintf('Optimal: %.6f\n', result.primal_obj);
    fprintf('Solution: x = [%.6f, %.6f]\n', result.x(1), result.x(2));
end
```

### `hprlp.Model.from_arrays(A, AL, AU, l, u, c)`
Create an LP model from matrices and vectors.

**Arguments:**
- `A` - Constraint matrix (m×n sparse or dense)
- `AL`, `AU` - Constraint bounds (column vectors)
- `l`, `u` - Variable bounds (column vectors)
- `c` - Objective coefficients (column vector)

**Returns:** Model object

---
### Example 2: Solve from MPS File

```bash
cd examples
matlab -batch "example_mps_file"              # Solve from MPS file
```

Or run interactively in MATLAB:
```matlab
cd examples
example_mps_file
```

**Quick overview**:
The following snippet demonstrates how to define and solve an LP problem from an MPS file.
For a complete version with additional options, see example_mps_file.m.

```matlab
% Create model from MPS file
model = hprlp.Model.from_mps('problem.mps');

% Solve
result = model.solve();
disp(result);
```
### `hprlp.Model.from_mps(filename)`
Create an LP model from the MPS file.

**Arguments:**
- `filename` - Path to MPS file

**Returns:** Model object

### `model.solve(param)`
Solve the LP model.

**Arguments:**
- `param` - (Optional) Parameters object. If omitted, default parameters are used.

**Returns:** Result object

## `hprlp.Parameters`
Solver configuration:
- `max_iter` - Maximum iterations (default: 2147483647)
- `stop_tol` - Stopping tolerance (default: 1e-4)
- `time_limit` - Time limit in seconds (default: 3600)
- `device_number` - CUDA device ID (default: 0)
- `check_iter` - Convergence check interval (default: 150)
- `use_Ruiz_scaling` - Ruiz scaling (default: true)
- `use_Pock_Chambolle_scaling` - Pock-Chambolle scaling (default: true)
- `use_bc_scaling` - Bounds/cost scaling (default: true)

**Example:**
```matlab
param = hprlp.Parameters();
param.stop_tol = 1e-9;
param.device_number = 0;
```
---

## `hprlp.Result`
Solution information:
- `status` - "OPTIMAL", "TIME_LIMIT", "ITER_LIMIT", "ERROR"
- `x` - Primal solution (n×1 vector)
- `y` - Dual solution (m×1 vector)
- `primal_obj` - Primal objective value
- `gap` - Duality gap
- `residuals` - Final KKT residual
- `iter` - Total iterations
- `time` - Solve time (seconds)
- `iter4/6/8` - Iterations to reach 1e-4/6/8 tolerance
- `time4/6/8` - Time to reach tolerance

**Checking optimality:**
```matlab
if strcmp(result.status, 'OPTIMAL')
    fprintf('Solution is optimal!\n');
end
```

## Setting Up MATLAB Path

To use HPRLP in your MATLAB scripts, add the bindings directory to your path:

```matlab
addpath('/path/to/HPR-LP-C/bindings/matlab');
```

Or add this line to your `startup.m` file for automatic loading.

## Requirements

- MATLAB R2018a or later
- NVIDIA GPU with CUDA support
- CUDA Toolkit (compatible with your MATLAB version)
- C++ compiler compatible with MATLAB MEX
