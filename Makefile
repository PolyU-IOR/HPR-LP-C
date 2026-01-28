## Makefile to build HPRLP as a library and executables

# Compiler and CUDA architecture
# Try to auto-detect CUDA installation
CUDA_PATH ?= $(shell if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; \
                      elif [ -d /opt/cuda ]; then echo /opt/cuda; \
                      elif [ -n "$${CUDA_HOME}" ]; then echo $${CUDA_HOME}; \
                      elif command -v nvcc >/dev/null 2>&1; then dirname $$(dirname $$(command -v nvcc)); \
                      else echo /usr/local/cuda; fi)

NVCC := $(CUDA_PATH)/bin/nvcc
AR := ar
RANLIB := ranlib

# Check if nvcc exists
ifeq ($(shell test -x $(NVCC) && echo yes),)
    $(error CUDA compiler not found at $(NVCC). Please install CUDA or set CUDA_PATH variable)
endif

# Auto-detect compute capability via nvidia-smi (override with `make GPU_SM=86`)
NVIDIA_SMI := $(shell command -v nvidia-smi 2>/dev/null)
DETECTED_CC := $(shell test -n "$(NVIDIA_SMI)" && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
# Convert x.y -> xy (e.g., 8.6 -> 86); ignore if N/A
ifneq ($(DETECTED_CC),)
	ifeq ($(DETECTED_CC),N/A)
		DETECTED_SM :=
	else
		DETECTED_SM := $(subst .,,$(DETECTED_CC))
	endif
endif

ifeq ($(strip $(GPU_SM)),)
	ifeq ($(strip $(DETECTED_SM)),)
		CUDA_ARCH := -arch=sm_52
	else
		CUDA_ARCH := -arch=sm_$(DETECTED_SM)
	endif
else
	CUDA_ARCH := -arch=sm_$(GPU_SM)
endif

# Using detected GPU architecture: $(CUDA_ARCH)

# Auto-detect suitable GCC version (prefer older versions for compatibility)
# Try to find GCC-12, GCC-11, GCC-10, or fall back to system default
# GCC-13+ requires GLIBCXX_3.4.32 which is not widely available
HOST_COMPILER := $(shell \
	if command -v g++-12 >/dev/null 2>&1; then echo g++-12; \
	elif command -v g++-11 >/dev/null 2>&1; then echo g++-11; \
	elif command -v g++-10 >/dev/null 2>&1; then echo g++-10; \
	elif command -v g++-9 >/dev/null 2>&1; then echo g++-9; \
	else command -v g++ >/dev/null 2>&1 && echo g++ || echo ""; fi)

# Check if we found a compiler
ifeq ($(HOST_COMPILER),)
    $(error No suitable C++ compiler found. Please install g++)
endif

# Warn if using GCC-13 or newer (may cause compatibility issues)
GCC_VERSION := $(shell $(HOST_COMPILER) -dumpversion 2>/dev/null | cut -d. -f1)
ifneq ($(GCC_VERSION),)
    GCC_MAJOR := $(shell echo $(GCC_VERSION) | cut -d. -f1)
    ifeq ($(shell test $(GCC_MAJOR) -ge 13 && echo yes),yes)
        $(warning Using GCC $(GCC_VERSION) which may require GLIBCXX_3.4.32+ on target systems)
        $(warning Consider installing GCC-12 or earlier for better compatibility: sudo apt-get install g++-12)
    endif
endif

# Directory structure
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
LIB_DIR := lib

# Flags and includes
# Use C++11 for better compatibility with MATLAB and older systems
# Add flags to avoid GLIBCXX_3.4.32 dependency when possible
NVCC_FLAGS := -w -O2 --std=c++11 $(CUDA_ARCH) -Xcompiler -fPIC -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin $(HOST_COMPILER)
INCLUDES := -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/cuda_kernels -I$(CUDA_PATH)/include

# Libraries - auto-detect lib vs lib64
CUDA_LIB_DIR := $(shell if [ -d $(CUDA_PATH)/lib64 ]; then echo $(CUDA_PATH)/lib64; \
                         else echo $(CUDA_PATH)/lib; fi)
LIB_DIRS := -L$(CUDA_LIB_DIR) -L$(LIB_DIR)
<<<<<<< HEAD
LIBS := -lcublas -lcusolver -lcusparse -lcurand
=======
LIBS := -lcublas -lcusolver -lcusparse
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e

# Library sources (solver core without main files)
LIB_SOURCES := \
	$(SRC_DIR)/mps_reader.cpp \
	$(SRC_DIR)/utils.cu \
	$(SRC_DIR)/scaling.cu \
	$(SRC_DIR)/preprocess.cu \
	$(SRC_DIR)/power_iteration.cu \
	$(SRC_DIR)/main_iterate.cu \
	$(SRC_DIR)/HPRLP.cu \
	$(SRC_DIR)/cuda_kernels/HPR_cuda_kernels.cu

# Object files for library
LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(LIB_SOURCES))) \
               $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(LIB_SOURCES)))

# Static library
STATIC_LIB := $(LIB_DIR)/libhprlp.a
SHARED_LIB := $(LIB_DIR)/libhprlp.so

# Executables
RUN_MPS := $(BUILD_DIR)/solve_mps_file

# Default target: build both static and shared libraries, plus executables
all: $(STATIC_LIB) $(SHARED_LIB) $(RUN_MPS)

# Build shared library (for language bindings: Python, Julia, MATLAB, etc.)
shared: $(SHARED_LIB)

# Build static library
$(STATIC_LIB): $(LIB_OBJECTS) | $(LIB_DIR)
	@echo "Creating static library libhprlp.a..."
	@$(AR) rcs $@ $(LIB_OBJECTS)
	@$(RANLIB) $@

# Build shared library (for Python ctypes)
# Use -Xcompiler to pass static linking flags to the host compiler
$(SHARED_LIB): $(LIB_OBJECTS) | $(LIB_DIR)
	@echo "Creating shared library libhprlp.so..."
	@$(NVCC) -shared $(NVCC_FLAGS) -o $@ $(LIB_OBJECTS) $(LIB_DIRS) $(LIBS) \
		-Xlinker --exclude-libs,ALL \
		-Xcompiler -static-libstdc++ \
		-Xcompiler -static-libgcc

# Compile library object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) $(BUILD_DIR)/cuda_kernels
	@echo "Compiling $(notdir $<)..."
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR) $(BUILD_DIR)/cuda_kernels
	@echo "Compiling $(notdir $<)..."
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cuda_kernels/%.o: $(SRC_DIR)/cuda_kernels/%.cu | $(BUILD_DIR)/cuda_kernels
	@echo "Compiling $(notdir $<)..."
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# MPS file reader executable (statically linked for standalone use)
$(RUN_MPS): $(SRC_DIR)/solve_mps_file.cpp $(STATIC_LIB) | $(BUILD_DIR)
	@echo "Linking executable solve_mps_file..."
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(STATIC_LIB) $(LIB_DIRS) $(LIBS)
	@echo "Build complete: ./build/solve_mps_file"

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cuda_kernels:
	@mkdir -p $(BUILD_DIR)/cuda_kernels

$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(LIB_DIR)

install: $(STATIC_LIB)
	@echo "Installing library and headers..."
	@mkdir -p /usr/local/lib
	@mkdir -p /usr/local/include/hprlp
	@cp $(STATIC_LIB) /usr/local/lib/
	@cp -r $(INCLUDE_DIR)/* /usr/local/include/hprlp/
	@echo "Installed to /usr/local/lib and /usr/local/include/hprlp"

help:
	@echo "HPRLP Makefile - Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build static and shared libraries, plus executables"
	@echo "  make shared       - Build only shared library (for language bindings)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install      - Install library to /usr/local (requires sudo)"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Targets:"
	@echo "  $(STATIC_LIB)     - Static library (.a, for C/C++ linking)"
	@echo "  $(SHARED_LIB)     - Shared library (.so, for Python/Julia/MATLAB bindings)"
	@echo "  $(RUN_MPS)        - MPS file solver executable"
	@echo ""
	@echo "Options:"
	@echo "  GPU_SM=<arch>     - Override GPU architecture (e.g., make GPU_SM=86)"
	@echo "  CUDA_PATH=<path>  - Override CUDA installation path"
	@echo ""
	@echo "Current configuration:"
	@echo "  CUDA_PATH:  $(CUDA_PATH)"
	@echo "  NVCC:       $(NVCC)"
	@echo "  CUDA_ARCH:  $(CUDA_ARCH)"
	@echo "  LIB_DIR:    $(CUDA_LIB_DIR)"

.PHONY: all shared clean install help
