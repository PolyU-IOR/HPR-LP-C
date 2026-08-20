# Solver backend subsystem

This directory owns the X/Z and Y iteration-update sides.

## Runtime flow

The solve loop chooses check versus normal mode. Each side classifier then owns
all backend conditions and calls one concrete implementation function.

```text
solve loop
├── check
│   ├── update_zx_check_gpu -> update_zx_check_cusparse_gpu
│   └── update_y_check_gpu  -> update_y_check_cusparse_gpu
└── normal
    ├── update_zx_normal_gpu -> classify x_backend -> concrete X/Z function
    └── update_y_normal_gpu  -> classify y_backend -> concrete Y function
```

Check mode currently uses the canonical cuSPARSE SpMVOp path because it materializes
additional residual/restart state:

- X/Z: `x_temp`, `x_bar`, and `z_bar`.
- Y: `y_temp`, `y_bar`, and `y_obj`.

Specialized normal kernels omit those outputs. A future specialized check
implementation should be added as another concrete implementation and selected
by the corresponding classifier.

## Layout

```text
backends/
├── xz/
│   ├── update_xz_classifier.inc.cu
│   ├── simple_cusparse_backend.inc.cu
│   └── specialized/
│       ├── generic_fused_backends.inc.cu
│       ├── unit_backends.inc.cu
│       ├── dictionary_backends.inc.cu
│       └── structured_backends.inc.cu
├── y/
│   ├── update_y_classifier.inc.cu
│   ├── simple_cusparse_backend.inc.cu
│   └── specialized/
│       ├── generic_fused_backends.inc.cu
│       ├── unit_backends.inc.cu
│       ├── dictionary_backends.inc.cu
│       └── structured_backends.inc.cu
├── autotune/
└── detail/
```

The responsibilities are deliberately strict:

- `update_*_classifier.inc.cu` contains backend conditions and routing only.
- `simple_cusparse_backend.inc.cu` contains the concrete simple check and normal
  implementations.
- Files under `specialized/` contain concrete specialized normal
  implementations only; they contain no backend selection logic.

The implementation families are:

- `generic_fused_backends.inc.cu`: general fused sparse and segmented paths.
- `unit_backends.inc.cu`: unit-factorized, signed-unit, column-tile, and
  active-scatter paths.
- `dictionary_backends.inc.cu`: packed and fixed-degree dictionary paths.
- `structured_backends.inc.cu`: detected structured, stencil, grid/slack,
  row-template, and affine-block paths.

These files are sibling implementation families under the same specialized
category; the split does not add another selection layer.

The corresponding compiled kernels use the same family names under
`../../cuda_kernels/backends/`. This directory owns conditions, routing, and
host launch orchestration; `cuda_kernels/` owns concrete `__global__`
implementations.

## Iteration completion

X/Z and Y update functions only update solver variables; none advances the
Halpern state. Completion is owned by each iteration driver:

- `solve.cu` advances after a check X/Z-Y pair.
- Single-update CUDA graph capture records one advance after its normal pair.
- Autotune advances after every normal probe pair and check-evaluation pair.
- Batched graph preparation computes all per-iteration factors and advances the
  canonical state by the whole batch, so the captured updates add no advances.

This keeps update functions independent of iteration scheduling and guarantees
that each logical iteration advances exactly once.

## Analysis and selection

Preprocessing analyzes the matrix and prepares readiness flags, row buckets,
compact operators, dictionaries, and detected structural metadata. Candidate
discovery converts those readiness results into compatible X/Z and Y pairs.
Autotuning benchmarks eligible pairs and installs `x_backend` and `y_backend`.
The runtime classifiers validate the selected backend/readiness combination and
fall back to the simple or generic implementation where appropriate.

```text
preprocess analysis -> eligible metadata -> candidate pairs
                    -> autotune selection -> runtime classifiers
                    -> concrete backend functions
```

These `*.inc.cu` files are owned by `../iteration/main_iterate.cu` and remain in
one CUDA translation unit to preserve existing internal linkage. They must not
be compiled independently.
