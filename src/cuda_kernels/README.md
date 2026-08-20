# CUDA kernel layout

This directory contains compiled CUDA kernels only. Runtime backend
classification and dispatch remain under `src/solver/backends`.

```text
cuda_kernels/
├── shared/
│   ├── vector_kernels.cu
│   ├── scaling_kernels.cu
│   ├── residual_kernels.cu
│   └── halpern_kernels.cu
└── backends/
    ├── detail/
    ├── simple/
    ├── generic/
    ├── unit/
    ├── dictionary/
    └── structured/
```

The backend directories match the implementation families selected by the
solver:

- `simple/`: pointwise X/Z and Y kernels paired with separate cuSPARSE SpMVOp.
- `generic/`: general fused CSR kernels and segmented-row fallbacks.
- `unit/`: unit-factorized, signed-unit, and column-tile kernels.
- `dictionary/`: dictionary-coded and packed-state kernels.
- `structured/`: detected structured, row-template, affine-block, stencil,
  and grid/slack/Laplacian kernels.
- `detail/`: device-only helpers shared by multiple backend kernel files.

Headers use the same hierarchy under `include/cuda_kernels`. The legacy
`HPR_cuda_kernels.cuh` is now a compatibility umbrella; new code should include
the narrow header for the category it launches.

Keep backend selection out of this directory. Preprocessing certifies
eligibility, autotuning selects an eligible pair, solver classifiers route the
iteration, and these files provide only the concrete GPU kernels.
