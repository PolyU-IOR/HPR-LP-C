# Changelog

All notable public changes to HPR-LP-C are recorded here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Removed the CUDA Toolkit 13.3 build-time requirement. CUDA 13.3 and newer
  automatically use experimental `cusparseSpMVOp` ALG1; older toolkits compile
  the regular `cusparseSpMV` CSR ALG2 fallback.
- Made Make, CMake, examples, and Python packaging use a detected or
  compiler-supported CUDA architecture instead of assuming `sm_100` when no
  GPU is visible.
- Centralized architecture detection for plain Make and CMake builds. B200,
  H100, A100, and GeForce RTX 30/40/50 series GPUs select `sm_100`, `sm_90`,
  `sm_80`, `sm_86`, `sm_89`, and `sm_120`, respectively, while `GPU_SM`
  remains available as an explicit override.
- Renamed generic CUDA launch macros that collided with CUDA 13.0
  `cooperative_groups` internals.

## [0.1.3] - 2026-08-21

### Added

- Added runtime-certified CUDA backends for repeated-row, affine-block,
  fixed-degree, signed-unit, packed-dictionary, segmented-row,
  windowed-stencil, and grid/slack/Laplacian matrix structures. Ineligible
  matrices continue to use the canonical cuSPARSE path.
- Added a bounded 24-update CUDA graph for eligible normal iterations. The
  graph cannot cross convergence-check, output, progress, or time-limit
  boundaries.
- Added GPU-Presolver-C as a selectable, vendored presolve/postsolve backend,
  including optional folding and an original-model fallback. The embedded PSLP
  backend remains available.
- Added adaptive row and column reduction, reduced-state backend selection,
  and optional CUDA virtual-memory allocation with a `cudaMalloc` fallback.
- Added detailed phase timing, numerical-range diagnostics, and a compact
  solution summary.
- Added resumable Python and Julia dataset runners with multi-GPU scheduling,
  per-instance logs, CSV summaries, solved counts, and shifted geometric mean
  timing.
- Added optional HDF5 input to the command-line and Julia interfaces.

### Changed

- Reorganized the implementation into focused API, I/O, presolve, solver,
  preprocessing, backend, and CUDA-kernel components while keeping
  `include/HPRLP.h` as the stable public entry point.
- Made GPU-Presolver-C the default presolver. Use `--presolver pslp` for the
  embedded PSLP path or `--presolver none` to disable presolve.
- Moved iteration scalars and Halpern factors to asynchronously updated device
  state where safe and kept restart-controlling reductions on the canonical
  solver stream.
- Stabilized Curtis--Reid scaling in the logarithmic domain and added
  deterministic handling for exceptionally long rows.
- Streamed `.mps.gz` input through zlib instead of expanding the complete model
  into a temporary file.
- Set the publication build defaults to `GPU_SM=100` and
  `CMAKE_CUDA_ARCHITECTURES=100` for NVIDIA B200.
- Require CUDA 13.3 or newer and use experimental `cusparseSpMVOp` ALG1.
- Added prefix-aware Make installation and CMake package export support.

### Fixed

- Value-initialized model storage so failed MPS reads can be cleaned up safely.
- Kept the Julia `C_HPRLP_results` mirror synchronized with the native result
  structure.
- Synchronized ultra-wide primal movement-norm results before restart logic
  consumes them.
- Corrected reduced-state mask maintenance, full/reduced state synchronization,
  progress-monitor counters, guarded sigma trials, and rollback behavior.
- Hardened presolve cleanup, postsolve validation, and fallback paths.

### Compatibility

- The C/C++ solver API and command-line interface remain compatible with the
  0.1.x public release, apart from the documented expanded presolver selector.
- HPR-LP-C remains a Linux/NVIDIA CUDA solver and has no CPU solver backend.
- This publication package requires CUDA Toolkit 13.3 or newer and targets
  NVIDIA B200 compute capability 10.0 (`sm_100`).
- Reference validation uses tolerance `1e-6`, a 1,000-second time limit, and a
  convergence-check interval of 150 iterations.

## [0.1.2]

The 0.1.2 public baseline is official repository commit
`358295ca9af3a9f1413174f2f63e5bdf3032c548`.
