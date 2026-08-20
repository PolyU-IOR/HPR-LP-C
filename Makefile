## Makefile to build HPRLP as a library and executables

# Compiler and CUDA architecture
# Try to auto-detect CUDA installation
CUDA_PATH ?= $(shell if [ -n "$${CUDA_HOME}" ] && [ -x "$${CUDA_HOME}/bin/nvcc" ]; then echo "$${CUDA_HOME}"; \
                      elif [ -x "$${HOME}/cuda-13.3/bin/nvcc" ]; then echo "$${HOME}/cuda-13.3"; \
                      elif [ -d /usr/local/cuda ]; then echo /usr/local/cuda; \
                      elif [ -d /opt/cuda ]; then echo /opt/cuda; \
                      elif command -v nvcc >/dev/null 2>&1; then dirname $$(dirname $$(command -v nvcc)); \
                      else echo /usr/local/cuda; fi)

NVCC := $(CUDA_PATH)/bin/nvcc
CC := $(shell command -v gcc >/dev/null 2>&1 && echo gcc || echo cc)
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
		CUDA_ARCH := -arch=sm_75
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
PSLP_DIR := third_party/PSLP
PREFIX ?= /usr/local
DESTDIR ?=

# Flags and includes
# CUDA 13's CCCL (Thrust, CUB, and libcu++) requires C++17.
# Add flags to avoid GLIBCXX_3.4.32 dependency when possible.
NVCC_FLAGS := -w -O2 --std=c++17 -DCUSPARSE_ENABLE_EXPERIMENTAL_API $(CUDA_ARCH) -Xcompiler -fPIC -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin $(HOST_COMPILER)
NVCC_CXX17_FLAGS := -w -O2 --std=c++17 -DCUSPARSE_ENABLE_EXPERIMENTAL_API $(CUDA_ARCH) -Xcompiler -fPIC -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin $(HOST_COMPILER)
DEPFLAGS := -MMD -MP
INCLUDES := -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/cuda_kernels -I$(CUDA_PATH)/include
PSLP_INCLUDES := -I$(PSLP_DIR)/include/PSLP -I$(PSLP_DIR)/include/core -I$(PSLP_DIR)/include/data_structures -I$(PSLP_DIR)/include/explorers
PSLP_DEFINES := -DPSLP_VERSION=\"0.0.8\" -D_POSIX_C_SOURCE=200809L -DNDEBUG
PSLP_CFLAGS := -O3 -fPIC
GPU_PRESOLVER_DIR := third_party/GPU-Presolver-C/cpp
GPU_PRESOLVER_INCLUDES := -I$(GPU_PRESOLVER_DIR)/include

# Optional native HDF5 support for solve_mps_file. Reuse the HDF5_jll
# installation already managed by the Julia binding so the CLI does not
# require a separate system HDF5 development package.
JULIA ?= $(shell command -v julia 2>/dev/null)
HDF5_JULIA_PROJECT ?= bindings/julia/package
HDF5_ROOT ?= $(shell if [ -n "$(JULIA)" ]; then \
	$(JULIA) --project=$(HDF5_JULIA_PROJECT) -e \
	'using HDF5; print(dirname(dirname(HDF5.API.libhdf5)))' 2>/dev/null; \
	fi)
HDF5_RUNTIME_DIRS ?= $(shell if [ -n "$(HDF5_ROOT)" ]; then \
	$(JULIA) --project=$(HDF5_JULIA_PROJECT) -e \
	'using HDF5, Libdl; Libdl.dlopen(HDF5.API.libhdf5); paths = filter(path -> occursin("/.julia/artifacts/", path), Libdl.dllist()); print(join(unique(dirname.(paths)), ":"))' 2>/dev/null; \
	fi)
HDF5_MPI_ROOT ?= $(shell if [ -n "$(HDF5_ROOT)" ]; then \
	$(JULIA) --project=$(HDF5_JULIA_PROJECT) -e \
	'using HDF5, Libdl; Libdl.dlopen(HDF5.API.libhdf5); paths = filter(path -> startswith(basename(path), "libmpi."), Libdl.dllist()); isempty(paths) || print(dirname(dirname(first(paths))))' 2>/dev/null; \
	fi)
HDF5_TRANSITIVE_LIBS ?= $(shell if [ -n "$(HDF5_ROOT)" ]; then \
	$(JULIA) --project=$(HDF5_JULIA_PROJECT) -e \
	'using HDF5, Libdl; Libdl.dlopen(HDF5.API.libhdf5); paths = filter(path -> occursin("/.julia/artifacts/", path) && endswith(basename(path), ".so") && !startswith(basename(path), "libhdf5") && !(basename(path) in ("libmpicxx.so", "libmpifort.so")), Libdl.dllist()); print(join(paths, " "))' 2>/dev/null; \
	fi)
ifneq ($(strip $(HDF5_ROOT)),)
	HDF5_CLI_FLAGS := -DHPRLP_HAS_HDF5=1 -I$(HDF5_ROOT)/include
	ifneq ($(strip $(HDF5_MPI_ROOT)),)
		HDF5_CLI_FLAGS += -I$(HDF5_MPI_ROOT)/include
	endif
	HDF5_CLI_LIBS := -Xcompiler=-Wl\\,--no-as-needed \
		$(HDF5_ROOT)/lib/libhdf5.so $(HDF5_TRANSITIVE_LIBS)
	ifneq ($(strip $(HDF5_RUNTIME_DIRS)),)
		HDF5_CLI_LIBS += \
			-Xcompiler=-Wl\\,--disable-new-dtags\\,-rpath\\,$(HDF5_RUNTIME_DIRS)
	endif
endif

# Libraries - auto-detect lib vs lib64
CUDA_LIB_DIR := $(shell if [ -d $(CUDA_PATH)/lib64 ]; then echo $(CUDA_PATH)/lib64; \
                         else echo $(CUDA_PATH)/lib; fi)
CUDA_COMPAT_DIR ?= $(shell for candidate in \
	$(CUDA_PATH)/compat \
	$${HOME}/cuda-compat-13.3/usr/local/cuda-13.3/compat; do \
	if [ -d "$${candidate}" ]; then echo "$${candidate}"; break; fi; done)
# Use legacy DT_RPATH so an unrelated CUDA entry in LD_LIBRARY_PATH cannot
# override the toolkit selected at build time. CUDA 13.3+ builds use the newer
# experimental SpMVOp symbols; older builds use the cusparseSpMV ALG2 fallback.
CUDA_RPATH_FLAGS := -Xlinker --disable-new-dtags \
	-Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)
ifneq ($(strip $(CUDA_COMPAT_DIR)),)
	CUDA_RPATH_FLAGS += -Xlinker -rpath -Xlinker $(CUDA_COMPAT_DIR)
endif
LIB_DIRS := -L$(CUDA_LIB_DIR) -L$(LIB_DIR)
LIBS := -lcublas -lcusolver -lcusparse -lcurand -lcuda -lz

# Library sources (solver core without main files)
LIB_SOURCES := \
	$(SRC_DIR)/presolve/pslp_integration.cpp \
	$(SRC_DIR)/presolve/gpu_presolver_integration.cpp \
	$(SRC_DIR)/io/mps_reader.cpp \
	$(SRC_DIR)/gpu/memory/compressible_memory.cu \
	$(SRC_DIR)/support/cuda_utils.cu \
	$(SRC_DIR)/solver/scaling.cu \
	$(SRC_DIR)/gpu/preprocessing/preprocess.cu \
	$(SRC_DIR)/solver/power_iteration.cu \
	$(SRC_DIR)/solver/restart_control.cpp \
	$(SRC_DIR)/solver/progress_monitor.cu \
	$(SRC_DIR)/solver/reduced_matrix.cu \
	$(SRC_DIR)/solver/iteration/main_iterate.cu \
	$(SRC_DIR)/solver/solver_output.cpp \
	$(SRC_DIR)/solver/cuda_graph.cu \
	$(SRC_DIR)/solver/solve.cu \
	$(SRC_DIR)/api/model.cpp \
	$(SRC_DIR)/batch/batched_solver.cu \
	$(SRC_DIR)/cuda_kernels/shared/vector_kernels.cu \
	$(SRC_DIR)/cuda_kernels/shared/scaling_kernels.cu \
	$(SRC_DIR)/cuda_kernels/shared/residual_kernels.cu \
	$(SRC_DIR)/cuda_kernels/shared/halpern_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/simple/simple_update_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/generic/generic_fused_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/unit/unit_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/unit/signed_unit_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/dictionary/dictionary_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/structured/structured_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/structured/factorized_stencil_kernels.cu \
	$(SRC_DIR)/cuda_kernels/backends/structured/grid_slack_laplacian_kernels.cu

PSLP_SOURCES := $(filter-out $(PSLP_DIR)/src/core/Debugger.c,$(wildcard $(PSLP_DIR)/src/core/*.c)) $(wildcard $(PSLP_DIR)/src/explorers/*.c)
GPU_PRESOLVER_SOURCES := \
	$(GPU_PRESOLVER_DIR)/src/folding/folding.cu \
	$(GPU_PRESOLVER_DIR)/src/folding/folding_kernels.cu \
	$(GPU_PRESOLVER_DIR)/src/folding/folding_map.cu \
	$(GPU_PRESOLVER_DIR)/src/folding/folding_reduce.cu \
	$(GPU_PRESOLVER_DIR)/src/folding/folding_refinement.cu \
	$(GPU_PRESOLVER_DIR)/src/model/lp_problem.cpp \
	$(GPU_PRESOLVER_DIR)/src/presolve/presolve_config.cpp \
	$(GPU_PRESOLVER_DIR)/src/presolve/gpu_postsolve.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/gpu_presolve.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/gpu_presolve_kernels.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/presolve_structs.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_close_bounds.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_activity_checks.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_doubleton_eq.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_dual_fix.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_empty_cols.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_empty_rows.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_parallel_cols.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_parallel_rows.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_primal_propagation.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_redundant_bounds.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_singleton_cols.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_singleton_rows.cu \
	$(GPU_PRESOLVER_DIR)/src/presolve/rules/rule_structural_l1_substitution.cu

# Object files for library
LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(LIB_SOURCES))) \
               $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(LIB_SOURCES)))

PSLP_OBJECTS := $(patsubst $(PSLP_DIR)/%.c,$(BUILD_DIR)/pslp/%.o,$(PSLP_SOURCES))
GPU_PRESOLVER_OBJECTS := \
	$(patsubst $(GPU_PRESOLVER_DIR)/%.cpp,$(BUILD_DIR)/gpu_presolver/%.o,$(filter %.cpp,$(GPU_PRESOLVER_SOURCES))) \
	$(patsubst $(GPU_PRESOLVER_DIR)/%.cu,$(BUILD_DIR)/gpu_presolver/%.o,$(filter %.cu,$(GPU_PRESOLVER_SOURCES)))
DEPENDENCY_FILES := $(LIB_OBJECTS:.o=.d) $(PSLP_OBJECTS:.o=.d) $(GPU_PRESOLVER_OBJECTS:.o=.d)

# Static library
STATIC_LIB := $(LIB_DIR)/libhprlp.a
SHARED_LIB := $(LIB_DIR)/libhprlp.so

# Executables
RUN_MPS := $(BUILD_DIR)/solve_mps_file

# Default target: build both libraries and the command-line solver.
all: $(STATIC_LIB) $(SHARED_LIB) $(RUN_MPS)

# Build shared library (for language bindings: Python, Julia, MATLAB, etc.)
shared: $(SHARED_LIB)

# Build static library
$(STATIC_LIB): $(LIB_OBJECTS) $(PSLP_OBJECTS) $(GPU_PRESOLVER_OBJECTS) | $(LIB_DIR)
	@echo "Creating static library libhprlp.a..."
	@rm -f $@
	@$(AR) rcs $@ $(LIB_OBJECTS) $(PSLP_OBJECTS) $(GPU_PRESOLVER_OBJECTS)
	@$(RANLIB) $@

# Build shared library (for Python ctypes)
# Use -Xcompiler to pass static linking flags to the host compiler
$(SHARED_LIB): $(LIB_OBJECTS) $(PSLP_OBJECTS) $(GPU_PRESOLVER_OBJECTS) | $(LIB_DIR)
	@echo "Creating shared library libhprlp.so..."
	@$(NVCC) -shared $(NVCC_FLAGS) $(CUDA_RPATH_FLAGS) -o $@ $(LIB_OBJECTS) $(PSLP_OBJECTS) $(GPU_PRESOLVER_OBJECTS) $(LIB_DIRS) $(LIBS) \
		-Xlinker --exclude-libs,ALL \
		-Xcompiler -static-libstdc++ \
		-Xcompiler -static-libgcc

$(BUILD_DIR)/presolve/gpu_presolver_integration.o: $(SRC_DIR)/presolve/gpu_presolver_integration.cpp | $(BUILD_DIR)
	@echo "Compiling presolve/gpu_presolver_integration.cpp..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_CXX17_FLAGS) $(DEPFLAGS) $(INCLUDES) $(PSLP_INCLUDES) $(GPU_PRESOLVER_INCLUDES) -DHPRLP_HAS_GPU_PRESOLVER=1 -c $< -o $@

# Compile library object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) $(BUILD_DIR)/cuda_kernels
	@echo "Compiling $(notdir $<)..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_FLAGS) $(DEPFLAGS) $(INCLUDES) $(PSLP_INCLUDES) -DHPRLP_HAS_GPU_PRESOLVER=1 -c $< -o $@

$(BUILD_DIR)/solver/reduced_matrix.o: $(SRC_DIR)/solver/reduced_matrix.cu | $(BUILD_DIR) $(BUILD_DIR)/cuda_kernels
	@echo "Compiling reduced_matrix.cu (C++17 for CUB)..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_CXX17_FLAGS) $(DEPFLAGS) $(INCLUDES) $(PSLP_INCLUDES) -DHPRLP_HAS_GPU_PRESOLVER=1 -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR) $(BUILD_DIR)/cuda_kernels
	@echo "Compiling $(notdir $<)..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_FLAGS) $(DEPFLAGS) $(INCLUDES) $(PSLP_INCLUDES) -DHPRLP_HAS_GPU_PRESOLVER=1 -c $< -o $@

$(BUILD_DIR)/cuda_kernels/%.o: $(SRC_DIR)/cuda_kernels/%.cu | $(BUILD_DIR)/cuda_kernels
	@echo "Compiling $(notdir $<)..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_FLAGS) $(DEPFLAGS) $(INCLUDES) $(PSLP_INCLUDES) -c $< -o $@

$(BUILD_DIR)/pslp/%.o: $(PSLP_DIR)/%.c | $(BUILD_DIR)/pslp $(BUILD_DIR)/pslp/src $(BUILD_DIR)/pslp/src/core $(BUILD_DIR)/pslp/src/explorers
	@echo "Compiling PSLP $(notdir $<)..."
	@$(CC) $(PSLP_CFLAGS) $(DEPFLAGS) $(PSLP_INCLUDES) $(PSLP_DEFINES) -c $< -o $@

$(BUILD_DIR)/gpu_presolver/%.o: $(GPU_PRESOLVER_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling GPU-Presolver-C $(notdir $<)..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_CXX17_FLAGS) $(DEPFLAGS) $(INCLUDES) $(GPU_PRESOLVER_INCLUDES) -c $< -o $@

$(BUILD_DIR)/gpu_presolver/%.o: $(GPU_PRESOLVER_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling GPU-Presolver-C $(notdir $<)..."
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCC_CXX17_FLAGS) $(DEPFLAGS) $(INCLUDES) $(GPU_PRESOLVER_INCLUDES) -c $< -o $@

# MPS/HDF5 file reader executable (statically linked for standalone use)
$(RUN_MPS): $(SRC_DIR)/cli/solve_mps_file.cpp $(STATIC_LIB) | $(BUILD_DIR)
	@echo "Linking executable solve_mps_file..."
	@$(NVCC) $(NVCC_FLAGS) $(CUDA_RPATH_FLAGS) $(INCLUDES) $(HDF5_CLI_FLAGS) -o $@ $< \
		$(STATIC_LIB) $(LIB_DIRS) $(LIBS) $(HDF5_CLI_LIBS)
	@echo "Build complete: ./build/solve_mps_file"

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cuda_kernels:
	@mkdir -p $(BUILD_DIR)/cuda_kernels

$(BUILD_DIR)/pslp:
	@mkdir -p $(BUILD_DIR)/pslp

$(BUILD_DIR)/pslp/src:
	@mkdir -p $(BUILD_DIR)/pslp/src

$(BUILD_DIR)/pslp/src/core:
	@mkdir -p $(BUILD_DIR)/pslp/src/core

$(BUILD_DIR)/pslp/src/explorers:
	@mkdir -p $(BUILD_DIR)/pslp/src/explorers

$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(LIB_DIR)

install: all
	@echo "Installing HPR-LP-C under $(DESTDIR)$(PREFIX)..."
	@install -d $(DESTDIR)$(PREFIX)/lib
	@install -d $(DESTDIR)$(PREFIX)/bin
	@install -d $(DESTDIR)$(PREFIX)/include/hprlp
	@install -m 0644 $(STATIC_LIB) $(DESTDIR)$(PREFIX)/lib/
	@install -m 0755 $(SHARED_LIB) $(DESTDIR)$(PREFIX)/lib/
	@install -m 0755 $(RUN_MPS) $(DESTDIR)$(PREFIX)/bin/
	@cp -R $(INCLUDE_DIR)/HPRLP.h \
		$(INCLUDE_DIR)/api \
		$(INCLUDE_DIR)/batch \
		$(INCLUDE_DIR)/cuda_kernels \
		$(INCLUDE_DIR)/gpu \
		$(INCLUDE_DIR)/io \
		$(INCLUDE_DIR)/presolve \
		$(INCLUDE_DIR)/solver \
		$(INCLUDE_DIR)/support \
		$(DESTDIR)$(PREFIX)/include/hprlp/
	@echo "Installed libraries, headers, and solve_mps_file under $(DESTDIR)$(PREFIX)"

help:
	@echo "HPRLP Makefile - Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build static and shared libraries, plus the command-line solver"
	@echo "  make shared       - Build only shared library (for language bindings)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make install      - Install under PREFIX (default: /usr/local)"
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
	@echo "  PREFIX=<path>     - Installation prefix (default: /usr/local)"
	@echo "  DESTDIR=<path>    - Optional staged-install root"
	@echo ""
	@echo "Current configuration:"
	@echo "  CUDA_PATH:  $(CUDA_PATH)"
	@echo "  NVCC:       $(NVCC)"
	@echo "  CUDA_ARCH:  $(CUDA_ARCH)"
	@echo "  LIB_DIR:    $(CUDA_LIB_DIR)"

.PHONY: all shared clean install help

-include $(DEPENDENCY_FILES)
