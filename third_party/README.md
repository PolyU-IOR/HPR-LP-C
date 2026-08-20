# Third-party dependencies

This directory contains source dependencies distributed with HPR-LP-C.

- `PSLP`: embedded CPU presolve and postsolve implementation used by the
  HPR-LP-C v0.1.3 solver.
- `GPU-Presolver-C`: C++17/CUDA presolve and folding library used by
  `--presolver gpu`. CMake builds it by default and can disable it with
  `-DBUILD_GPU_PRESOLVER=OFF`.

Each dependency retains its own license and upstream documentation.
