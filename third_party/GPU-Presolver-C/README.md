# GPU Presolver + Folding

This package provides a C++17/CUDA pipeline for reducing linear programs before
they are sent to an LP solver. It combines two complementary stages:

1. **Folding** detects compatible repeated structure and aggregates rows and
   columns while retaining a map to the original model.
2. **GPU presolve** applies LP presolve reductions to the folded model, or
   directly to the original model when folding is disabled or finds no valid
   aggregation.

After the reduced LP is solved, the package can replay presolve operations and
unfold the solution back to the original model space.

```text
Reduction: original LP -> optional folding -> GPU presolve -> reduced LP
Recovery:  reduced solution -> GPU postsolve -> unfold -> original-space solution
```

## Scope

The supported model form is:

```text
minimize    c' x + c0
subject to  AL <= A x <= AU
            l  <= x   <= u
```

The package currently supports LP models. QP presolve and FME projection are
not part of the active C++ pipeline.

## Requirements

- CMake 3.24 or newer
- C++17 compiler
- CUDA toolkit with `nvcc`, CUDA Runtime, and cuSPARSE
- NVIDIA GPU compatible with the configured CUDA architecture

The default architecture is `90-real`. Override it when targeting another GPU.

## Build and Test

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90-real

cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

## CMake Integration

Place the package under `third_party`, then link its CMake target:

```cmake
add_subdirectory(
    third_party/GPU-Presolver-C-folding/cpp
    ${CMAKE_BINARY_DIR}/gpu-presolver
    EXCLUDE_FROM_ALL)

target_link_libraries(your_target PRIVATE gpu_presolver_core)
```

## Command-Line Use

Run the presolve + folding pipeline on an MPS file from the package root:

```bash
./cpp/build/gpu_presolver_mps /path/to/model.mps
```

The output distinguishes all three model sizes:

- `original_rows`, `original_cols`: input model
- `folded_rows`, `folded_cols`: model after folding
- `reduced_rows`, `reduced_cols`, `reduced_nnz`: final output model

## Presolve and Postsolve

The combined API is declared in `gpu_presolver/folding/folding.hpp`. Typical
use is:

```cpp
#include "gpu_presolver/folding/folding.hpp"

gpu_presolver::presolve::PresolveParams params;
params.enable_folding = true;
params.record_postsolve_tape = true;

gpu_presolver::folding::FoldingPipelineSummary summary =
    gpu_presolver::folding::run_gpu_presolve_with_folding(
        device_lp,
        params,
        /*keep_reduced_lp=*/true,
        /*keep_folded_lp=*/true);

// Handle summary.presolve.has_infeasible or .has_unbounded before solving.
// Solve summary.presolve.reduced_lp, producing x_red, y_red, and z_red.
gpu_presolver::folding::UnfoldedSolutionHost original_solution =
    gpu_presolver::folding::postsolve_and_unfold_to_host(
        x_red, y_red, z_red, summary);

// original_solution.x, .y, and .z are in the original model space.

gpu_presolver::presolve::free_gpu_presolve_reduced_lp(summary.presolve);
gpu_presolver::folding::free_folded_lp(summary.folding);
```

The workflow is:

1. Run folding and presolve, then check the returned status.
2. Solve `summary.presolve.reduced_lp` to obtain `x_red`, `y_red`, and `z_red`.
3. Call `postsolve_and_unfold_to_host` to recover the original-space solution.

`summary` contains the reduced LP, presolve status and recovery record, folding
result and map, and the original/folded/reduced model sizes. Keep it alive until
recovery finishes. Postsolve also requires `record_postsolve_tape = true`.
Release its reduced LP and folding data with the cleanup functions shown above.

For presolve without folding, include
`gpu_presolver/presolve/gpu_presolve.hpp` and call
`run_gpu_presolve_with_reduced_lp` directly.

## Command-Line Options

The main options are:

- `--presolve <0|1>`: enable or disable GPU presolve; default `1`
- `--folding <0|1>`: enable or disable folding; default `1`
- `--device <id>`: select a visible CUDA device
- `--config <path>`: load optional settings from a TOML file
- `-h`, `--help`: print usage

Command-line values override the optional config. Presolve internals remain
available through `PresolveParams` and `config/default.toml`.

## Main Components

- `cpp/include/gpu_presolver/folding`: folding and unfolding API
- `cpp/include/gpu_presolver/presolve`: presolve and postsolve API
- `cpp/src/folding`: CUDA folding implementation
- `cpp/src/presolve`: CUDA presolve and postsolve implementation
- `cpp/tools`: MPS command-line tools
- `cpp/tests`: folding, presolve, postsolve, and rule tests

## License

See `LICENSE`.
