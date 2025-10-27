# HPR-LP-C Python Interface

Python interface for HPR-LP-C (Halpern Peaceman--Rachford Linear Programming) solver — a GPU-accelerated linear programming solver using C/CUDA.


## Installation

```bash
cd HPR-LP-C/bindings/python
python -m pip install .         # or: python -m pip install -e .
```

## Examples


### Example 1: Build model directly from matrices

```bash
cd examples
python example_direct_lp.py              # Solve from arrays
```

**Quick overview**:
The following snippet demonstrates how to define and solve an LP problem directly from matrices.
For a complete version with additional options, see example_direct_lp.py.

```python
import numpy as np
from scipy import sparse
import hprlp

# Define LP: minimize c'x subject to AL <= Ax <= AU, l <= x <= u
A = sparse.csr_matrix([[1.0, 2.0], [3.0, 1.0]])
AL = np.array([-np.inf, -np.inf])
AU = np.array([10.0, 12.0])
l = np.array([0.0, 0.0])
u = np.array([np.inf, np.inf])
c = np.array([-3.0, -5.0])

# Create model
model = hprlp.Model.from_arrays(A, AL, AU, l, u, c)

# Configure solver
param = hprlp.Parameters()
param.stop_tol = 1e-9
param.device_number = 0

# Solve
result = model.solve(param)

if result.is_optimal():
    print(f"Optimal: {result.primal_obj}")
    print(f"Solution: {result.x}")
```




### `Model.from_arrays(A, AL, AU, l, u, c)`
Create an LP model from matrices and vectors.

**Arguments:**
- `A` - Constraint matrix (scipy sparse or dense)
- `AL`, `AU` - Constraint bounds (numpy arrays)
- `l`, `u` - Variable bounds (numpy arrays)
- `c` - Objective coefficients (numpy array)

**Returns:** Model object

---

### Example 2: Solve from MPS File

```bash
cd examples
python example_mps_file     # Solve from MPS file
```

**Quick overview:**
The following snippet demonstrates how to define and solve an LP problem directly from an MPS file. For a complete version with additional options, see example_mps_file.py.

```python
import hprlp

# Create model from MPS file
model = hprlp.Model.from_mps("problem.mps")

# Solve
result = model.solve()
print(result)
```

### `Model.from_mps(filename)`
Create an LP model from the MPS file.

**Arguments:**
- `filename` - Path to MPS file

**Returns:** Model object

### `Model.solve(param=None)`
Solve the LP model.

**Arguments:**
- `param` - (Optional) Parameters object

**Returns:** Results object

---

## `Parameters`
Solver configuration:
- `max_iter` - Maximum iterations (default: unlimited)
- `stop_tol` - Stopping tolerance (default: 1e-4)
- `time_limit` - Time limit in seconds (default: 3600)
- `device_number` - CUDA device ID (default: 0)
- `check_iter` - Convergence check interval (default: 150)
- `use_Ruiz_scaling` - Ruiz scaling (default: True)
- `use_Pock_Chambolle_scaling` - Pock-Chambolle scaling (default: True)
- `use_bc_scaling` - Bounds/cost scaling (default: True)

### `Results`
Solution information:
- `status` - "OPTIMAL", "TIME_LIMIT", "ITER_LIMIT", "ERROR"
- `x` - Primal solution
- `y` - Dual solution
- `primal_obj` - Primal objective value
- `gap` - Duality gap
- `residuals` - Final KKT residual
- `iter` - Total iterations
- `time` - Solve time (seconds)
- `iter4/6/8` - Iterations to reach 1e-4/6/8 tolerance
- `time4/6/8` - Time to reach tolerance
