# HPR-LP-C

A C implementation of the Halpern Peaceman--Rachford (HPR) method for solving linear programming (LP) problems on GPUs.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 12.4 or later recommended)
- cuBLAS, cuSOLVER, cuSPARSE libraries (included with CUDA Toolkit)

## Building

The build system creates libraries and executables:

```bash
make clean    # Remove build artifacts
make          # Build everything (recommended)
```

This creates:
- `lib/libhprlp.a` - Static library for C/C++ linking
- `lib/libhprlp.so` - Shared library for language bindings (Python, Julia, MATLAB)
- `build/solve_mps_file` - MPS file solver executable

## Usage

### Solving from MPS Files

```bash
# Show help
./build/solve_mps_file -h

# Run with default settings
./build/solve_mps_file -i data/model.mps

# Run with custom settings
./build/solve_mps_file -i data/model.mps --tol 1e-4 --time-limit 3600
```

### Using as a Library

See the `examples/` directory for standalone demos showing how to link against `libhprlp.a` and solve LP problems programmatically.

See `examples/c(cpp)/README.md` for details on using HPR-LP-C in external projects.

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input <path>` | Path to input MPS file | *required* |
| `--device <id>` | CUDA device ID | 0 |
| `--max-iter <N>` | Maximum iterations | unlimited |
| `--tol <eps>` | Stopping tolerance | 1e-4 |
| `--time-limit <sec>` | Time limit in seconds | 3600 |
| `--check-iter <N>` | Convergence check interval | 150 |
| `--ruiz <true/false>` | Ruiz scaling | true |
| `--pock <true/false>` | Pock-Chambolle scaling | true |
| `--bc <true/false>` | Bounds/cost scaling | true |
| `-h, --help` | Show help message | - |

### Language Interface Installation

**Python (pip only):**
```bash
cd HPR-LP-C/bindings/python
python -m pip install .         # or: python -m pip install -e .
```

**Julia Interface:**
```bash
cd bindings/julia
bash install.sh
```

**MATLAB Interface:**
```bash
cd bindings/matlab
bash install.sh
```

See the respective README files in `bindings/` for detailed usage instructions.

### Language Interfaces

HPR-LP-C provides native interfaces for multiple languages:

- **Python**: Native Python bindings via pybind11 (see `bindings/python/README.md`)
- **MATLAB**: Native MEX interface with OOP wrapper (see `bindings/matlab/README.md`)
- **Julia**: Native Julia interface (see `bindings/julia/README.md`)
- **C/C++**: Direct API usage (see `examples/c/` and `examples/cpp/`)


## License

MIT License

Copyright (c) 2025 HPR-LP Contributors

See the [LICENSE](LICENSE) file for full details.

## Reference

Kaihuang Chen, [Defeng Sun](https://www.polyu.edu.hk/ama/profile/dfsun//), [Yancheng Yuan](https://www.polyu.edu.hk/ama/people/academic-staff/dr-yuan-yancheng/?sc_lang=en), Guojun Zhang, and [Xinyuan Zhao](https://scholar.google.com/citations?user=nFG8lEYAAAAJ&hl=en), “[HPR-LP: An implementation of an HPR method for solving linear programming](https://www.polyu.edu.hk/ama/profile/dfsun//files/HPR-LP_Published2025.pdf)”, arXiv:2408.12179 (August 2024), [Mathematical Programming Computation](https://link.springer.com/journal/12532) 17:4 (2025), doi.org/10.1007/s12532-025-00292-0.

## Contributors

### Core Team
- **Kaihuang Chen** - Developer
- **ZheXuan Gu** - Developer
- **Defeng Sun** - Principal Investigator
- **Yancheng Yuan** - Contributor
- **Guojun Zhang** - Contributor
- **Xinyuan Zhao** - Contributor

### Acknowledgments
- Community contributors and testers

## Contact

For questions, issues, or collaboration inquiries:
- Open an issue on GitHub: https://github.com/PolyU-IOR/HPR-LP-C/issues

## Version

Current version: **0.1.0**

## Available HPR-LP implementations

For the C implementation on GPUs, please visit the following repository:
[https://github.com/PolyU-IOR/HPR-LP-C](https://github.com/PolyU-IOR/HPR-LP-C)

For the Julia implementation on CPUs and GPUs, please visit the following repository:
[https://github.com/PolyU-IOR/HPR-LP](https://github.com/PolyU-IOR/HPR-LP)

For the Python implementation on GPUs, please visit the following repository:
[https://github.com/PolyU-IOR/HPR-LP-Python](https://github.com/PolyU-IOR/HPR-LP-Python)

For the MATLAB implementation on CPUs, please visit the following repository:
[https://github.com/PolyU-IOR/HPR-LP-MATLAB](https://github.com/PolyU-IOR/HPR-LP-MATLAB)
