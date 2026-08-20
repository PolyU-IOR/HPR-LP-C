# Source layout

The implementation is grouped by responsibility:

- `api/`: public model-construction and model-lifetime API.
- `batch/`: batched shared-matrix solver.
- `cli/`: command-line programs.
- `cuda_kernels/`: compiled CUDA kernels, separated into shared solver
  primitives and the same simple/generic/unit/dictionary/structured families
  used by the iteration backend subsystem.
- `gpu/memory/`: GPU allocation strategies.
- `gpu/preprocessing/`: device-model transfer, workspace setup, sparse-pattern
  analysis, specialized-operator preparation, and cleanup.
- `io/`: MPS parsing and model construction.
- `presolve/`: PSLP and GPU-Presolver-C adapters.
- `solver/`: solve orchestration, reporting, CUDA graph management, scaling,
  power iteration, iteration logic, and the backend selection/dispatch
  subsystem.
- `support/`: shared CUDA/C++ utility functions.

Files named `*.inc.cu` are implementation fragments owned by the nearest
`.cu` file. They keep tightly coupled CUDA helpers in one translation unit
while separating distinct responsibilities for navigation and maintenance.
They must not be compiled independently.

Headers under `include/` mirror the source ownership hierarchy. `HPRLP.h` remains
the stable public umbrella; internal headers use their canonical categorized
paths.
