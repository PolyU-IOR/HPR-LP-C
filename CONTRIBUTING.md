# Contributing to HPR-LP

Thank you for your interest in contributing to HPR-LP! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/HPR-LP-C.git
   cd HPR-LP-C
   ```
3. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- NVIDIA GPU with CUDA support (Compute Capability 5.2+)
- CUDA Toolkit 12.4+
- GCC 9-12 (for compatibility)
- Python 3.8+ (for Python bindings)
- Julia 1.6+ (for Julia bindings)
- MATLAB R2020a+ (for MATLAB bindings)

### Building
```bash
make clean
make -j
```

### Running Tests
```bash
# Test the solver
./build/solve_mps_file -i data/model.mps

# Test C/C++ examples
cd examples/cpp && make && ./example_direct_lp

# Test Python bindings
cd bindings/python && python -m pip install .
python examples/example_direct_lp.py
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
2. Add tests if applicable
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

By contributing to HPR-LP, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making HPR-LP better! 🚀
