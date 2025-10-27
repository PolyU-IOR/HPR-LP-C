# Makefile for HPRLP Julia interface
# Builds Julia package and runs tests

.PHONY: all install test examples clean help

# Default target
all: install

# Install Julia package and dependencies
install:
	@echo "Installing HPRLP.jl and dependencies..."
	@julia --project=. -e 'using Pkg; Pkg.instantiate()'
	@echo "Installation complete!"
	@echo "Run 'make test' to verify installation"

# Run tests
test: install
	@echo "Testing HPRLP.jl installation..."
	@./test.sh

# Run examples
examples: install
	@echo "Running HPRLP.jl examples..."
	@julia --project=. examples/run_all_examples.jl

# Run specific example
example_direct: install
	@julia --project=. examples/example_direct_lp.jl

example_mps: install
	@julia --project=. examples/example_mps_file.jl

# Clean build artifacts
clean:
	@echo "Cleaning Julia artifacts..."
	@rm -rf Manifest.toml
	@echo "Clean complete!"

# Development: Start Julia REPL with package loaded
repl: install
	@julia --project=. -i -e 'using HPRLP'

# Show help
help:
	@echo "HPRLP.jl Makefile"
	@echo "================="
	@echo ""
	@echo "Targets:"
	@echo "  all (default)   - Install package and dependencies"
	@echo "  install         - Install Julia package"
	@echo "  test            - Run tests"
	@echo "  examples        - Run all examples"
	@echo "  example_direct  - Run direct LP example"
	@echo "  example_mps     - Run MPS file example"
	@echo "  repl            - Start Julia REPL with HPRLP loaded"
	@echo "  clean           - Clean build artifacts"
	@echo "  help            - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - Julia 1.6 or later"
	@echo "  - HPRLP C/CUDA library built (run 'make' in parent directory)"
	@echo "  - CUDA 11.0+ and NVIDIA GPU"
