# C Examples

Pure C examples demonstrating how to use HPR-LP from C programs.

## Examples

**example_direct_lp.c** - Solve LP from arrays in CSR format
```bash
make example_direct_lp
./example_direct_lp
```

**example_mps_file.c** - Solve LP from MPS file
```bash
make example_mps_file
./example_mps_file
```

## Building

Build the HPRLP library first:
```bash
cd ../..
make
make shared   # Build shared library for dynamic linking
```

Then build and run examples:
```bash
cd examples/c
make          # Build all examples
make run      # Build and run all examples
```

Or build and run individual examples:
```bash
make example_direct_lp
./example_direct_lp

make example_mps_file
./example_mps_file
```

**Note:** The Makefile will automatically build `libhprlp.so` if it doesn't exist, but if you get a "cannot open shared object file" error, run `make shared` in the project root first.

## Basic Usage

```c
#include "HPRLP.h"

// Create model from arrays (CSR format)
LP_info_cpu* model = create_model_from_arrays(
    m, n, nnz,
    rowPtr, colIndex, values,
    AL, AU, l, u, c,
    false  // is_csc=false for CSR format
);

// Or create a model from the MPS file
LP_info_cpu* model = create_model_from_mps("model.mps");

// Solve the model with custom parameters
HPRLP_parameters param;
param.stop_tol = 1e-9;
param.device_number = 0;
HPRLP_results result = solve(model, &param);

// Access solution
if (result.x != NULL) {
    // Use result.x (primal solution)
    // Use result.y (dual solution)
    
    // Free solution arrays
    free(result.x);
    free(result.y);
}

// Free the model
free_model(model);
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

See the example source code for complete working implementations.
