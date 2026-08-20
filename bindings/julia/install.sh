#!/bin/bash
# Installation script for HPRLP.jl

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PACKAGE_DIR="$SCRIPT_DIR/package"

echo "======================================"
echo "HPRLP.jl Installation"
echo "======================================"
echo ""

# Check Julia
if ! command -v julia &> /dev/null; then
    echo "ERROR: Julia not found. Please install Julia 1.6+"
    exit 1
fi

echo "✓ Julia found: $(julia --version)"
echo ""

# Check if shared library exists
if [ ! -f "$SCRIPT_DIR/../../lib/libhprlp.so" ]; then
    echo "ERROR: Shared library not found at lib/libhprlp.so"
    echo "Please build the library first:"
    echo "  cd ../.."
    echo "  make"
    exit 1
fi

echo "✓ HPRLP library found"
echo ""

# Install Julia dependencies
echo "Installing Julia dependencies..."
cd "$PACKAGE_DIR"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
cd "$SCRIPT_DIR"

echo ""
echo "======================================"
echo "✓ Installation complete!"
echo "======================================"
echo ""
echo "Usage from Julia:"
echo "  cd bindings/julia/package"
echo "  julia --project=."
echo "  using HPRLP"
echo "  result = solve(A, AL, AU, l, u, c)"
echo ""
echo "See examples/ folder for usage examples"
echo ""
