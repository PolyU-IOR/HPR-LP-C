# Makefile for HPRLP C++ Examples

# Path to HPRLP installation
HPRLP_ROOT := ../..
HPRLP_LIB_DIR := $(HPRLP_ROOT)/lib
HPRLP_INCLUDE_DIR := $(HPRLP_ROOT)/include

# Auto-detect CUDA
CUDA_PATH ?= $(shell if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; \
                      elif [ -d /opt/cuda ]; then echo /opt/cuda; \
                      elif [ -n "$${CUDA_HOME}" ]; then echo $${CUDA_HOME}; \
                      elif command -v nvcc >/dev/null 2>&1; then dirname $$(dirname $$(command -v nvcc)); \
                      else echo /usr/local/cuda; fi)

NVCC := $(CUDA_PATH)/bin/nvcc

# Check if nvcc exists
ifeq ($(shell test -x $(NVCC) && echo yes),)
    $(error CUDA compiler not found at $(NVCC))
endif

# Auto-detect GPU architecture
NVIDIA_SMI := $(shell command -v nvidia-smi 2>/dev/null)
DETECTED_CC := $(shell test -n "$(NVIDIA_SMI)" && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
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

# Compiler flags
NVCC_FLAGS := -w -O2 --std=c++17 $(CUDA_ARCH)
INCLUDES := -I$(HPRLP_INCLUDE_DIR) -I$(CUDA_PATH)/include

# CUDA library path
CUDA_LIB_DIR := $(shell if [ -d $(CUDA_PATH)/lib64 ]; then echo $(CUDA_PATH)/lib64; \
                         else echo $(CUDA_PATH)/lib; fi)

# Linker flags
LIB_DIRS := -L$(HPRLP_LIB_DIR) -L$(CUDA_LIB_DIR)
LIBS := -lhprlp -lcublas -lcusolver -lcusparse
LDFLAGS := -Xlinker -rpath -Xlinker $(HPRLP_LIB_DIR)

# Targets
EXAMPLE_DIRECT := example_direct_lp
EXAMPLE_MPS := example_mps_file

all: $(EXAMPLE_DIRECT) $(EXAMPLE_MPS)

$(EXAMPLE_DIRECT): example_direct_lp.cpp
	@echo "Building $(EXAMPLE_DIRECT)..."
	@if [ ! -f $(HPRLP_LIB_DIR)/libhprlp.a ]; then \
		echo "Error: libhprlp.a not found. Please run 'make' in $(HPRLP_ROOT) first."; \
		exit 1; \
	fi
	@if [ ! -f $(HPRLP_LIB_DIR)/libhprlp.so ]; then \
		echo "Building shared library..."; \
		cd $(HPRLP_ROOT) && make shared; \
	fi
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIB_DIRS) $(LIBS) $(LDFLAGS)
	@echo "Built: $@"

$(EXAMPLE_MPS): example_mps_file.cpp
	@echo "Building $(EXAMPLE_MPS)..."
	@if [ ! -f $(HPRLP_LIB_DIR)/libhprlp.a ]; then \
		echo "Error: libhprlp.a not found. Please run 'make' in $(HPRLP_ROOT) first."; \
		exit 1; \
	fi
	@if [ ! -f $(HPRLP_LIB_DIR)/libhprlp.so ]; then \
		echo "Building shared library..."; \
		cd $(HPRLP_ROOT) && make shared; \
	fi
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIB_DIRS) $(LIBS) $(LDFLAGS)
	@echo "Built: $@"

clean:
	@rm -f $(EXAMPLE_DIRECT) $(EXAMPLE_MPS)

run-direct: $(EXAMPLE_DIRECT)
	@./$(EXAMPLE_DIRECT)

run-mps: $(EXAMPLE_MPS)
	@./$(EXAMPLE_MPS)

run: $(EXAMPLE_DIRECT) $(EXAMPLE_MPS)
	@echo "Running example_direct_lp..."
	@./$(EXAMPLE_DIRECT)
	@echo ""
	@echo "Running example_mps_file..."
	@./$(EXAMPLE_MPS)

help:
	@echo "HPRLP C++ Examples"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all examples"
	@echo "  make run          - Build and run all examples"
	@echo "  make run-direct   - Build and run direct LP example"
	@echo "  make run-mps      - Build and run MPS file example"
	@echo "  make clean        - Remove all executables"
	@echo ""
	@echo "Targets:"
	@echo "  example_direct_lp   - Demonstrates creating and solving LP from raw arrays"
	@echo "  example_mps_file    - Demonstrates creating and solving LP from MPS file"
	@echo ""
	@echo "Note: libhprlp.a must be built first (run 'make' in $(HPRLP_ROOT))"

.PHONY: all clean run run-direct run-mps help
