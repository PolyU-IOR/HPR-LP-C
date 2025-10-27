#!/bin/bash
# Complete installation script for HPRLP MATLAB interface

set -e  # Exit on error

echo "======================================"
echo "HPRLP MATLAB - Installation"
echo "======================================"
echo

# Check if we're in the matlab directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to find compatible GCC
find_compatible_gcc() {
    # Try to find GCC-12 or GCC-11 (both avoid GLIBCXX_3.4.32)
    for gcc_ver in g++-12 g++-11 g++-10; do
        if command -v $gcc_ver &> /dev/null; then
            echo "$gcc_ver"
            return 0
        fi
    done
    return 1
}

# Check for compatible GCC
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 0: Checking GCC compatibility"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

COMPATIBLE_GCC=$(find_compatible_gcc)
if [ -n "$COMPATIBLE_GCC" ]; then
    GCC_PATH=$(which $COMPATIBLE_GCC)
    GCC_VERSION=$($COMPATIBLE_GCC --version | head -n1)
    echo -e "${GREEN}✓ Found compatible compiler: $COMPATIBLE_GCC${NC}"
    echo "  Path: $GCC_PATH"
    echo "  Version: $GCC_VERSION"
    USE_COMPATIBLE_GCC=true
else
    echo -e "${YELLOW}⚠ No GCC-12/11/10 found. Will use system default compiler.${NC}"
    echo -e "${YELLOW}  This may cause GLIBCXX compatibility issues with MATLAB.${NC}"
    echo ""
    echo "To avoid potential issues, install a compatible GCC:"
    echo "  Ubuntu/Debian: sudo apt install g++-12"
    echo "  Fedora/RHEL:   sudo dnf install gcc-c++-12"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    USE_COMPATIBLE_GCC=false
fi
echo

# Check MATLAB installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking MATLAB installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ! command -v matlab &> /dev/null; then
    echo -e "${RED}✗ MATLAB not found in PATH${NC}"
    echo "Please ensure MATLAB is installed and accessible."
    echo "You may need to add MATLAB to your PATH:"
    echo "  export PATH=/usr/local/MATLAB/R2023b/bin:\$PATH"
    exit 1
fi

MATLAB_VERSION=$(matlab -batch "disp(version)" 2>/dev/null | head -n 1)
echo -e "${GREEN}✓ MATLAB found: $MATLAB_VERSION${NC}"
echo

# Build main C++/CUDA library
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Building C++/CUDA library"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd ../..
make clean > /dev/null 2>&1
if make -j$(nproc) > /tmp/hprlp_build.log 2>&1; then
    echo -e "${GREEN}✓ C++/CUDA library built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build C++/CUDA library${NC}"
    echo "See /tmp/hprlp_build.log for details"
    exit 1
fi

# Check that shared library exists
if [ ! -f "lib/libhprlp.so" ]; then
    echo -e "${RED}✗ Shared library lib/libhprlp.so not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Shared library lib/libhprlp.so found${NC}"
cd bindings/matlab
echo

# Build MEX interface
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Building MEX interface"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Detect CUDA installation path
echo "Detecting CUDA installation..."
CUDA_PATH=""
if [ -n "$CUDA_HOME" ]; then
    CUDA_PATH="$CUDA_HOME"
elif [ -n "$CUDA_PATH" ]; then
    CUDA_PATH="$CUDA_PATH"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -d "/opt/cuda" ]; then
    CUDA_PATH="/opt/cuda"
elif command -v nvcc &> /dev/null; then
    CUDA_PATH="$(dirname $(dirname $(which nvcc)))"
fi

if [ -z "$CUDA_PATH" ] || [ ! -d "$CUDA_PATH" ]; then
    echo -e "${RED}✗ CUDA installation not found${NC}"
    echo "Please set CUDA_HOME or CUDA_PATH environment variable"
    exit 1
fi

echo -e "${GREEN}✓ CUDA found at: $CUDA_PATH${NC}"

# Create the MATLAB build script
cat > build_mex.m << EOFMATLAB
function build_mex()
    % Build MEX interface for HPRLP with GLIBCXX compatibility
    
    fprintf('Building HPRLP MEX interface...\n');
    
    % Get paths
    root_dir = fullfile(pwd, '..', '..');
    include_dir = fullfile(root_dir, 'include');
    lib_dir = fullfile(root_dir, 'lib');
    cuda_include_dir = '${CUDA_PATH}/include';
    src_file = fullfile('src', 'hprlp_mex.cpp');
    
    % Get absolute path for RPATH
    lib_dir_abs = char(java.io.File(lib_dir).getCanonicalPath());
    
    % Check if source file exists
    if ~exist(src_file, 'file')
        error('MEX source file not found: %s', src_file);
    end
    
    % Determine which GCC to use (passed from shell script)
    use_compatible_gcc = ${USE_COMPATIBLE_GCC};
    gcc_path = '${GCC_PATH}';
    
    % MEX compilation command with:
    % - RPATH for runtime library loading
    % - Old ABI for compatibility
    % - Optional GCC version specification
    % - CUDA include directory for headers
    try
        mex_args = {'-v', ...
            ['-I', include_dir], ...
            ['-I', cuda_include_dir], ...
            ['-L', lib_dir], ...
            '-lhprlp', ...
            '-R2018a', ...
            ['LDFLAGS="\$LDFLAGS -Wl,-rpath,', lib_dir_abs, '"'], ...
            'CXXFLAGS="\$CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0"'};
        
        if use_compatible_gcc && ~isempty(gcc_path)
            fprintf('✓ Using compatible GCC: %s\n', gcc_path);
            mex_args{end+1} = ['GCC="', gcc_path, '"'];
        else
            fprintf('⚠ Using system default compiler (may cause GLIBCXX issues)\n');
        end
        
        mex_args{end+1} = src_file;
        mex_args{end+1} = '-outdir';
        mex_args{end+1} = '+hprlp/private';
        
        mex(mex_args{:});
        
        fprintf('✓ MEX interface compiled successfully\n');
        fprintf('  Library path embedded: %s\n', lib_dir_abs);
        
        % Test if MEX file was created
        mex_file = fullfile('+hprlp', 'private', ['hprlp_mex.', mexext]);
        if exist(mex_file, 'file')
            fprintf('✓ MEX file created: %s\n', mex_file);
            
            % Check GLIBCXX dependencies
            if isunix && ~ismac
                [status, result] = system(sprintf('objdump -T "%s" 2>/dev/null | grep GLIBCXX | awk ''{print \$5}'' | sort -u', mex_file));
                if status == 0 && ~isempty(result)
                    fprintf('\nGLIBCXX versions required:\n');
                    fprintf('%s', result);
                    
                    % Check if GLIBCXX_3.4.32 is present (problematic)
                    if contains(result, 'GLIBCXX_3.4.32') || contains(result, 'GLIBCXX_3.4.31')
                        fprintf('\n⚠ WARNING: MEX file requires GLIBCXX >= 3.4.31\n');
                        fprintf('  This may cause compatibility issues with MATLAB.\n');
                        fprintf('  Consider installing GCC-12 and rebuilding.\n');
                    else
                        fprintf('✓ GLIBCXX versions are compatible with MATLAB\n');
                    end
                end
            end
        else
            error('MEX file was not created');
        end
        
    catch ME
        fprintf('✗ MEX compilation failed:\n');
        fprintf('%s\n', ME.message);
        exit(1);
    end
end
EOFMATLAB

# Run MATLAB to compile MEX
if matlab -batch "build_mex" > /tmp/hprlp_mex_build.log 2>&1; then
    echo -e "${GREEN}✓ MEX interface compiled successfully${NC}"
else
    echo -e "${RED}✗ Failed to compile MEX interface${NC}"
    echo "See /tmp/hprlp_mex_build.log for details"
    cat /tmp/hprlp_mex_build.log
    exit 1
fi
echo

# Test installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Testing installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create a simple test script
cat > test_install.m << 'EOFTEST'
function test_install()
    % Quick test of HPRLP installation
    fprintf('Testing HPRLP installation...\n');
    
    try
        % Test Parameters class
        param = hprlp.Parameters();
        fprintf('✓ Parameters class loaded\n');
        
        % Test simple problem
        A = sparse([1.0, 2.0; 3.0, 1.0]);
        AL = [-inf; -inf];
        AU = [10.0; 12.0];
        l = [0.0; 0.0];
        u = [inf; inf];
        c = [-3.0; -5.0];
        
        fprintf('✓ Problem data created\n');
        
        % Try to solve
        result = hprlp.solve(A, AL, AU, l, u, c, param);
        
        fprintf('✓ Solver executed successfully\n');
        fprintf('  Status: %s\n', result.status);
        fprintf('  Objective: %.6f\n', result.primal_obj);
        fprintf('  Iterations: %d\n', result.iter);
        
        fprintf('\n✓ All tests passed!\n');
        
    catch ME
        fprintf('✗ Test failed:\n');
        fprintf('%s\n', ME.message);
        exit(1);
    end
end
EOFTEST

if matlab -batch "test_install" > /tmp/hprlp_test.log 2>&1; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
    cat /tmp/hprlp_test.log | grep "✓"
else
    echo -e "${YELLOW}⚠ Test had issues${NC}"
    echo "See /tmp/hprlp_test.log for details"
    
    # Check for GLIBCXX error
    if grep -q "GLIBCXX" /tmp/hprlp_test.log; then
        echo
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}GLIBCXX compatibility issue detected!${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo "This happens when the MEX file requires a newer C++ library"
        echo "version than MATLAB provides."
        echo
        echo -e "${BLUE}Quick fix:${NC} Start MATLAB with:"
        echo "  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        echo "  matlab"
        echo
        echo -e "${BLUE}Better fix:${NC} Install GCC-12 and rebuild:"
        echo "  sudo apt install g++-12"
        echo "  cd $SCRIPT_DIR"
        echo "  bash install.sh"
        echo
        echo "See GLIBCXX_COMPATIBILITY.md for more solutions."
    fi
    
    # Check for library path error
    if grep -q "libhprlp.so" /tmp/hprlp_test.log; then
        echo
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}Library path issue detected!${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo "In MATLAB, run: hprlp.setup_library_path()"
        echo "Or use the launcher: ./matlab_hprlp.sh"
    fi
fi
echo

# Clean up temporary files
rm -f build_mex.m test_install.m

echo "======================================"
echo "✓ Installation Complete!"
echo "======================================"
echo
echo "Usage from MATLAB:"
echo "  1. Add the matlab directory to your MATLAB path:"
echo "     cd bindings/matlab"
echo "     addpath(pwd)"
echo
echo "  2. Run examples:"
echo "     cd examples"
echo "     example_direct_lp"
echo "     example_mps_file"
echo
echo "  3. Use in your code:"
echo "     param = hprlp.Parameters();"
echo "     result = hprlp.solve(A, AL, AU, l, u, c, param);"
echo
echo "  For MPS files:"
echo "     result = hprlp.solve_mps('problem.mps', param);"
echo
