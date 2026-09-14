# Contributing to HPR-LP-C

Thank you for your interest in contributing to HPR-LP-C. This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-account>/HPR-LP-C.git
   cd HPR-LP-C
   ```
3. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- NVIDIA B200 GPU (Compute Capability 10.0)
- CUDA Toolkit 13.3 or newer
- GCC 9-12 with C++17 support
- GNU Make or CMake 3.18 or newer
- zlib development headers
- Python 3.8+ (for Python bindings)
- Julia 1.6+ (for Julia bindings)
- MATLAB R2020a+ (for MATLAB bindings)

### Building
```bash
make clean
make GPU_SM=100 -j
```

### Quick verification
```bash
# Run the command-line solver with explicit release settings
./build/solve_mps_file -i data/model.mps \
  --tol 1e-6 --time-limit 1000 --check-iter 150

# Run a C++ example
make -C examples/cpp GPU_SM=100 run

# Build and run the Python example
python -m pip install ./bindings/python
python bindings/python/examples/example_direct_lp.py
```

## How to Contribute

### Reporting Bugs
- Check if the issue already exists in [Issues](https://github.com/PolyU-IOR/HPR-LP-C/issues)
- If not, create a new issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs. actual behavior
  - System information (OS, CUDA version, GPU model)
  - Error messages and logs

### Suggesting Enhancements
- Open an issue with the `enhancement` label
- Describe the feature and its use case
- Explain why it would be useful to users

### Pull Requests
1. Ensure your code follows the existing style
2. Include a minimal reproduction command or example when applicable
3. Update documentation as needed
4. Commit with clear, descriptive messages:
   ```
   feat: Add support for new constraint types
   fix: Resolve memory leak in CUDA kernels
   docs: Update installation instructions
   ```
5. Push to your fork and create a pull request

### Code Style Guidelines
- **C/C++/CUDA**: Follow the existing style in the codebase
  - Use descriptive variable names
  - Add comments for complex algorithms
  - Keep functions focused and modular
- **Python**: Follow PEP 8
- **Julia**: Follow Julia style guidelines
- **MATLAB**: Follow MATLAB best practices

### Commit Message Format
We follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, no logic change)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive criticism
- Respect differing opinions and experiences

### Unacceptable Behavior
- Harassment, discriminatory language, or personal attacks
- Trolling or inflammatory comments
- Public or private harassment
- Publishing others' private information

### Enforcement
Violations can be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Questions?

Feel free to:
- Open a [Discussion](https://github.com/PolyU-IOR/HPR-LP-C/discussions)
- Ask in the issue tracker
- Contact the maintainers directly

## License

By contributing to HPR-LP-C, you agree that your contributions will be licensed under the MIT License.

---

Thank you for improving HPR-LP-C.
