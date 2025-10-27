# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Dropped `bindings/python/install.sh` and `install_new.sh`; Python installs now rely solely on `pip install`

### Planned
- Add more comprehensive test suite
- Improve error handling and reporting
- Add support for additional problem formats
- Performance benchmarks against other solvers

## [0.1.0] - 2025-10-26

### Added
- Initial public release
- CUDA-accelerated HPR-LP solver implementation
- Support for MPS file format
- C/C++ API with static and shared libraries
- Python bindings via pybind11
- Julia bindings via CCall
- MATLAB bindings via MEX
- Command-line solver executable
- Ruiz scaling, Pock-Chambolle scaling, and bounds/cost scaling
- CMake and Makefile build systems
- Comprehensive examples for all language interfaces
- Documentation and README files
- MIT License

### Features
- GPU-accelerated linear programming solver
- Halpern Peaceman-Rachford splitting algorithm
- Multi-language support (C/C++, Python, Julia, MATLAB)

### Build System
- Makefile with auto-detection of CUDA and GCC versions
- CMake configuration for cross-platform builds
- Separate static and shared library builds
- Language binding installers

---

## Version History

- **0.1.0** (2025-10-26) - Initial release

[Unreleased]: https://github.com/PolyU-IOR/HPR-LP-C/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.0
