# Changelog

## Unreleased

- Use the CUDA 13.3 experimental `cusparseSpMVOp` ALG1 API for single-vector
  sparse operations when it is available. CUDA 12.x and CUDA 13.0--13.2 now
  compile the same wrapper against `cusparseSpMV` CSR ALG2, including the GPU
  presolver folding path, so the public build no longer requires CUDA 13.3.
- Stream `.mps.gz` input directly through zlib instead of expanding the full
  model into an anonymous temporary file. This removes the tens-of-gigabytes
  temporary-disk requirement exposed by the large Oliver-Hinder MCF models.
- Value-initialize model storage before MPS parsing so failed reads can be
  cleaned up safely instead of freeing uninitialized fields.
- Simplified the README quick start to the default `make` command and restored
  the public package and banner version to 0.1.3.
- Ported the validated ultra-wide movement-norm synchronization fix from
  `HPR-LP-C-private` commit `4922b51`: primal restart movement vectors with at
  least `2^27` elements use the blocking host-result form of the ordinary
  32-bit `cublasDnrm2`, while smaller primal vectors and all dual vectors keep
  the queued device-result path. Movement and residual kernels now share the
  workspace stream. This is a synchronization workaround, not an index-width
  change.
- Fixed Julia dataset-worker stack corruption after a solve by keeping the
  `C_HPRLP_results` ABI mirror synchronized with the five reduced-matrix ratio
  statistics at the end of the native `HPRLP_results` struct.

All notable public changes to HPR-LP-C are recorded here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Added a complete solve-time breakdown, including SpMV/backend analysis
  time and total wall time from the model solve boundary to solution collection.
- Added Python and Julia dataset runners with CSV summaries, `SGM10` and
  solved-count rows, and saved solver iteration logs. The Python runner also
  supports resuming completed batches.
- Added finite absolute-value ranges for `A`, `AL`, `AU`, `l`, `u`, and `c`
  before and after presolve.

### Changed

- Batched eligible reduced-matrix iterations into the existing 24-update CUDA
  Graph cadence while preserving every convergence, output, progress, and
  time-limit boundary used by the single-update path.
- Decoupled full-state backend autotuning from reduced-matrix enablement, so
  C-only specialized backends remain available before reduced activation and
  during full-state checks while compact reduced x/y implementations retain
  their independent fused-versus-cuSPARSE choices.
- Reorganized implementation sources by responsibility and split the largest
  CUDA translation units into focused internal implementation fragments while
  preserving the public header layout.
- Reorganized iteration updates into X/Z and Y classifiers that own all
  backend conditions, paired simple check/normal cuSPARSE implementations,
  family-specific specialized implementation files, and separate autotuning.
- Split the monolithic CUDA kernel source and declaration header into shared
  vector/scaling/residual/Halpern primitives and matching
  simple/generic/unit/dictionary/structured backend-family directories.
- Reorganized non-kernel headers to mirror API, batch, I/O, GPU preprocessing,
  presolve, solver, and support ownership. Removed historical flat internal
  include paths; the root `HPRLP.h` public entry point remains unchanged.
- Moved Halpern-factor advancement out of the update layer into the solver,
  single-update CUDA graph, autotune probes, and batched-graph preparation.
- Simplified the command-line presolver interface to
  `--presolver <pslp|gpu|none>` and removed command-line CSV writing.
- Centralized the solution summary and standardized phase-time formatting.

## Recent commit history

### Recover one-sided feasibility stalls safely (2026-08-01)

- Added a late-feasibility stall detector for trajectories where exactly one
  feasibility residual has crossed tolerance while the other remains within
  `100 * tolerance`. It requires a stable residual side and 150 checks without
  a 10% target improvement before applying a bounded residual-balance restart.
- Limited the recovery to one correction-direction transition and made every
  correction a monitored trial. If the target residual rebounds by more than
  0.5% over the next observation window, the solver restores the prior sigma
  and blocks further one-sided corrections for that solve, preventing the
  repeated factor-four sigma ping-pong observed on `bdry2`.
- Added regression coverage for activation, side changes, the late-feasibility
  band, bounded sigma updates, accepted MCF-scale numerical variation, and the
  `bdry2` rollback case.
- The latest server validation at tolerance `1e-6` solved 49/49 Hans
  instances within 2,000 seconds, with solve-time SGM10 7.1387598445 seconds
  (`Aug01_HPRLPC_one_sided_trial_guard_Hans_devices03_2000s`). It solved
  12/12 Oliver Hinder instances within 9,600 seconds, with solve-time SGM10
  759.2789461223 seconds
  (`Jul31_HPRLPC_one_sided_trial_guard_Oliver_Hinder_devices023_9600s_retry1`).
  Focused runs solved `bdry2` in 29,376,600 iterations / 559.12 seconds,
  `mcf_5000_50_500` in 545,850 iterations / 1,303.47 seconds total, and
  `mcf_5000_100_400` in 1,229,250 iterations / 2,754.88 seconds total.

### Preserve movement-based sigma candidates at ordinary restarts (2026-07-31)

- Kept a finite movement-based sigma candidate for ordinary restart flags
  1--3 when only one feasibility residual has crossed the stopping tolerance,
  instead of reversing the candidate with an immediate factor-four residual
  correction.
- Retained residual balancing in the explicit phase-2 flag-4 trial path,
  where acceptance and rollback protect the incumbent trajectory, and kept
  the separate ultra-low residual-floor safeguard.
- Added a regression assertion for the preserved-candidate decision. On the
  Hans dataset at tolerance `1e-6` and a 2,000-second limit, the change solved
  49/49 instances with an SGM10 solve time of 7.2907 seconds, improving on the
  Jul26 reference value of 7.4098 seconds.

### Reduced-matrix synchronization and lifecycle fixes (2026-07-31)

- Aligned reduced-state mask maintenance, fixed-column shifts, incremental
  extensions, adaptive rebases, and full/reduced state synchronization with
  the Julia reference lifecycle.
- Restored the required full-state scatter before check/residual observations
  and synchronized compact state after canonical full-state check updates.
- Aligned reduced-mode bound metadata and independent X/Y backend autotuning,
  including Halpern-factor advancement during Y probes.

### Progress-aware restart and sigma control (2026-07-31)

- Added progress-aware phase identification, guarded long-restart decisions,
  residual-balanced sigma trials, trial acceptance/rollback, and terminal
  recovery controls while retaining the legacy restart criteria as the base.
- Added public controls for the progress monitor, sigma safeguards, rebalance
  restarts, restart cooldown/guard behavior, debug logging, and fixed sigma.
- Suppressed repeated feasibility-converged rebalance messages and recorded
  the C/Julia restart, sigma, and CUDA-alignment decisions in a review note.
- Corrected progress-monitor counter indexing and replaced per-element global
  atomics with block reductions, preserving the metrics while reducing
  monitoring overhead on large models.

### Input, memory-policy, scaling, and dataset-runner updates (2026-07-31)

- Added validated HDF5 model input to the Julia binding and optional native
  HDF5 support to the standalone CLI, preserving the objective constant and
  all bound types through the C API.
- Added multi-device Julia dataset scheduling with shared CSV output and one
  promptly flushed log per instance; expanded the Python runner and language
  bindings for the new runtime controls.
- Added a post-presolve automatic policy: models with more than two million
  columns and more than twice as many columns as rows use reduced mode with
  compression disabled; all other models use full state with compression.
- Made compressible-memory selection solve-local and restored the prior mode
  after each solve, with ordinary-allocation fallback and allocation
  diagnostics.
- Stabilized Curtis-Reid scaling in the logarithmic domain and added GPU-side
  finite numerical-range reporting after scaling.
- Hardened GPU-presolve cleanup and fallback paths, standardized CSR SpMV on
  cuSPARSE ALG2, and improved solver error handling for large-model failures.
- Added opt-in reduced-phase timing diagnostics and documented the complete
  server-tested implementation and experimental findings.

### `455ffeb` - Default to the GPU presolver (2026-07-21)

- Changed the library and command-line default presolver from PSLP to
  GPU-Presolver-C.

### `0387db1` - Improve timing summaries and dataset runs (2026-07-21)

- Added total solve time and a centralized solution summary.
- Added the Julia dataset runner and expanded Python/Julia CSV timing output.
- Removed deprecated command-line CSV options and presolver aliases.
- Improved Python batch-run error reporting and log collection.

### `1cd34ec` - Add detailed phase timing (2026-07-21)

- Split timing into presolve, setup, scaling, power iteration, and main-loop
  solve phases.
- Moved MPS read timing to the public model-creation API.
- Exposed read and phase timing through C/C++, Python, Julia, and MATLAB.

### `7205d79` - Import HPR-LP-C 0.1.3 (2026-07-21)

- Imported the CUDA solver, CPU and GPU presolvers, language bindings,
  examples, build systems, tests, and release documentation.
- Performance improvement: runtime-certified structured-matrix kernels avoid
  general sparse operations when a model is eligible, and the bounded
  24-update CUDA graph reduces normal-iteration kernel-launch overhead.
  Reduced synchronization and asynchronous scalar updates further reduce CPU
  and GPU waiting. The commit contains no reproducible numeric speedup, so no
  percentage or multiplier is claimed here.

## [0.1.3] - 2026-07-17

### Added

- General runtime-certified backends for repeated-row, affine-block,
  fixed-degree, signed-unit, packed-dictionary, segmented-row,
  windowed-stencil, and grid/slack/Laplacian matrix structures.
- Device-side backend probes and structural eligibility checks.
- Compact structural metadata and optional CUDA virtual-memory allocation
  with a `cudaMalloc` fallback.
- Make and CMake support for all optimized CUDA sources.

### Changed

- Reduced unnecessary host/device synchronization in the solver iteration.
- Moved iteration scalars and Halpern factors to asynchronously updated device
  state where safe.
- Retained the validated bounded 24-update normal-iteration CUDA graph. Its
  structural policy prevents a batch from crossing convergence-check or
  output boundaries, and it falls back to the single-update graph elsewhere.
- Updated the public package version to 0.1.3.
- Simplified the public build so it produces only the solver libraries and
  command-line executable.
- Added prefix-aware Make installation and complete CMake dependency export.
- Made the installed static CMake target self-contained for CUDA device
  linking and CUDA runtime use in downstream C++ consumers.

### Compatibility

- The command-line interface and C/C++ solver API remain compatible with the
  0.1.x public release.
- Matrices that do not satisfy a structured-backend certificate use the
  canonical cuSPARSE path.
- The bounded graph changes launch organization only; iteration counting,
  checkpoint placement, tolerance, restart behavior, and public APIs are
  unchanged.

### Release correction

- Restored the bounded graph path that was present in the benchmarked v0.1.3
  candidate but was inadvertently removed while deleting rejected graph
  experiments from the first public staging archive.
- Kept residual and adaptive-restart reductions on the canonical solver
  stream. The removed auxiliary-stream experiment could occasionally perturb
  a restart decision and change the numerical iteration path.
- Verification commands use a 1,000-second time limit, tolerance `1e-6`, and
  check interval `150`.

## [0.1.2]

The upstream 0.1.2 release is represented by official repository commit
`358295ca9af3a9f1413174f2f63e5bdf3032c548`.
