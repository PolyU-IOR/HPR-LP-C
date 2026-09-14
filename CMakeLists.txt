cmake_minimum_required(VERSION 3.18)

# Select the CUDA architecture before enabling the CUDA language so compiler
# identification uses the architecture of the first visible GPU. An explicit
# -DCMAKE_CUDA_ARCHITECTURES=<arch> always wins. If no GPU is visible, retain
# the B200 publication package's sm_100 fallback default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE DETECTED_CC
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(DETECTED_CC AND NOT DETECTED_CC STREQUAL "N/A")
        # nvidia-smi emits one line per GPU. Build for the first visible one.
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" DETECTED_CC "${DETECTED_CC}")
        string(REPLACE "." "" DETECTED_SM "${DETECTED_CC}")
        string(STRIP "${DETECTED_SM}" DETECTED_SM)
    endif()

    if(DETECTED_SM)
        set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_SM}")
        message(STATUS "Detected CUDA architecture: sm_${DETECTED_SM}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "100")
        message(STATUS "No visible GPU detected; using B200 fallback: sm_100")
    endif()
endif()

project(HPRLP VERSION 0.1.3 LANGUAGES C CXX CUDA)

# CUDA 13's CCCL (Thrust, CUB, and libcu++) requires C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(BUILD_STATIC_LIB "Build static library" ON)
option(BUILD_SHARED_LIB "Build shared library" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_GPU_PRESOLVER "Build GPU-Presolver-C and enable --presolver gpu" ON)

if(NOT BUILD_STATIC_LIB AND NOT BUILD_SHARED_LIB)
    message(FATAL_ERROR "At least one of BUILD_STATIC_LIB or BUILD_SHARED_LIB must be ON")
endif()

# Find the CUDA toolkit required by the NVIDIA B200 release.
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 13.3)
    message(FATAL_ERROR
        "The HPR-LP-C 0.1.3 B200 release requires CUDA Toolkit 13.3 or newer; found ${CUDAToolkit_VERSION}")
endif()
message(STATUS "CUDA ${CUDAToolkit_VERSION}: using cusparseSpMVOp ALG1")
find_package(ZLIB REQUIRED)
find_package(HDF5 QUIET COMPONENTS C)

# Auto-detect suitable GCC version (prefer older versions for compatibility)
# Try to find GCC-12, GCC-11, GCC-10, or fall back to system default
find_program(GCC_12 g++-12)
find_program(GCC_11 g++-11)
find_program(GCC_10 g++-10)
find_program(GCC_9 g++-9)

if(GCC_12)
    set(CMAKE_CUDA_HOST_COMPILER ${GCC_12})
    message(STATUS "Using GCC-12 as CUDA host compiler")
elseif(GCC_11)
    set(CMAKE_CUDA_HOST_COMPILER ${GCC_11})
    message(STATUS "Using GCC-11 as CUDA host compiler")
elseif(GCC_10)
    set(CMAKE_CUDA_HOST_COMPILER ${GCC_10})
    message(STATUS "Using GCC-10 as CUDA host compiler")
elseif(GCC_9)
    set(CMAKE_CUDA_HOST_COMPILER ${GCC_9})
    message(STATUS "Using GCC-9 as CUDA host compiler")
else()
    message(STATUS "Using system default GCC as CUDA host compiler")
endif()

# Warn if using GCC-13 or newer (may cause compatibility issues)
if(CMAKE_CUDA_HOST_COMPILER)
    execute_process(
        COMMAND ${CMAKE_CUDA_HOST_COMPILER} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "^[0-9]+" GCC_MAJOR "${GCC_VERSION}")
    if(GCC_MAJOR GREATER_EQUAL 13)
        message(WARNING "Using GCC ${GCC_VERSION} which may require GLIBCXX_3.4.32+ on target systems")
        message(WARNING "Consider installing GCC-12 or earlier for better compatibility")
    endif()
endif()

# Compiler flags (matching Makefile configuration)
# Use -O2 and -w to suppress warnings, add GLIBCXX compatibility flag
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -O2 -Xcompiler -fPIC -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
add_compile_definitions(CUSPARSE_ENABLE_EXPERIMENTAL_API)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cuda_kernels
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(PSLP_DIR ${PROJECT_SOURCE_DIR}/third_party/PSLP)
file(GLOB PSLP_SOURCES CONFIGURE_DEPENDS
    ${PSLP_DIR}/src/core/*.c
    ${PSLP_DIR}/src/explorers/*.c
)
list(REMOVE_ITEM PSLP_SOURCES ${PSLP_DIR}/src/core/Debugger.c)

add_library(pslp_embedded OBJECT ${PSLP_SOURCES})
target_include_directories(pslp_embedded BEFORE
    PRIVATE
        ${PSLP_DIR}/include/PSLP
        ${PSLP_DIR}/include/core
        ${PSLP_DIR}/include/data_structures
        ${PSLP_DIR}/include/explorers
)
target_compile_definitions(pslp_embedded PRIVATE PSLP_VERSION="0.0.8" NDEBUG)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(pslp_embedded PRIVATE _POSIX_C_SOURCE=200809L)
endif()
target_compile_options(pslp_embedded PRIVATE -w -O3)
set_target_properties(pslp_embedded PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_GPU_PRESOLVER)
    set(GPU_PRESOLVER_DIR
        ${PROJECT_SOURCE_DIR}/third_party/GPU-Presolver-C/cpp)
    if(NOT EXISTS ${GPU_PRESOLVER_DIR}/CMakeLists.txt)
        message(FATAL_ERROR
            "Vendored GPU-Presolver-C source is missing from ${GPU_PRESOLVER_DIR}")
    endif()

    set(GPU_PRESOLVER_SOURCES
        ${GPU_PRESOLVER_DIR}/src/folding/folding.cu
        ${GPU_PRESOLVER_DIR}/src/folding/folding_kernels.cu
        ${GPU_PRESOLVER_DIR}/src/folding/folding_map.cu
        ${GPU_PRESOLVER_DIR}/src/folding/folding_reduce.cu
        ${GPU_PRESOLVER_DIR}/src/folding/folding_refinement.cu
        ${GPU_PRESOLVER_DIR}/src/model/lp_problem.cpp
        ${GPU_PRESOLVER_DIR}/src/presolve/presolve_config.cpp
        ${GPU_PRESOLVER_DIR}/src/presolve/gpu_postsolve.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/gpu_presolve.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/gpu_presolve_kernels.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/presolve_structs.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_close_bounds.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_activity_checks.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_doubleton_eq.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_dual_fix.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_empty_cols.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_empty_rows.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_parallel_cols.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_parallel_rows.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_primal_propagation.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_redundant_bounds.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_singleton_cols.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_singleton_rows.cu
        ${GPU_PRESOLVER_DIR}/src/presolve/rules/rule_structural_l1_substitution.cu
    )

    add_library(gpu_presolver_embedded OBJECT ${GPU_PRESOLVER_SOURCES})
    target_include_directories(gpu_presolver_embedded PUBLIC
        ${GPU_PRESOLVER_DIR}/include
    )
    target_compile_features(gpu_presolver_embedded PUBLIC
        cxx_std_17
        cuda_std_17
    )
    set_target_properties(gpu_presolver_embedded PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Source files for the library
set(HPRLP_LIB_SOURCES
    src/presolve/pslp_integration.cpp
    src/io/mps_reader.cpp
    src/gpu/memory/compressible_memory.cu
    src/support/cuda_utils.cu
    src/solver/scaling.cu
    src/gpu/preprocessing/preprocess.cu
    src/solver/power_iteration.cu
    src/solver/restart_control.cpp
    src/solver/progress_monitor.cu
    src/solver/reduced_matrix.cu
    src/solver/iteration/main_iterate.cu
    src/solver/solver_output.cpp
    src/solver/cuda_graph.cu
    src/solver/solve.cu
    src/api/model.cpp
    src/batch/batched_solver.cu
    src/cuda_kernels/shared/vector_kernels.cu
    src/cuda_kernels/shared/scaling_kernels.cu
    src/cuda_kernels/shared/residual_kernels.cu
    src/cuda_kernels/shared/halpern_kernels.cu
    src/cuda_kernels/backends/simple/simple_update_kernels.cu
    src/cuda_kernels/backends/generic/generic_fused_kernels.cu
    src/cuda_kernels/backends/unit/unit_kernels.cu
    src/cuda_kernels/backends/unit/signed_unit_kernels.cu
    src/cuda_kernels/backends/dictionary/dictionary_kernels.cu
    src/cuda_kernels/backends/structured/structured_kernels.cu
    src/cuda_kernels/backends/structured/factorized_stencil_kernels.cu
    src/cuda_kernels/backends/structured/grid_slack_laplacian_kernels.cu
)
if(BUILD_GPU_PRESOLVER)
    list(APPEND HPRLP_LIB_SOURCES src/presolve/gpu_presolver_integration.cpp)
endif()

# CUDA 13 CUB device primitives used by the reduced-matrix path require C++17.
set_source_files_properties(src/solver/reduced_matrix.cu PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>")

# Create static library (default)
if(BUILD_STATIC_LIB)
    add_library(hprlp_static STATIC ${HPRLP_LIB_SOURCES})
    target_sources(hprlp_static PRIVATE $<TARGET_OBJECTS:pslp_embedded>)
    if(BUILD_GPU_PRESOLVER)
        target_sources(hprlp_static PRIVATE
            $<TARGET_OBJECTS:gpu_presolver_embedded>)
        target_compile_definitions(hprlp_static PRIVATE
            HPRLP_HAS_GPU_PRESOLVER=1)
        target_include_directories(hprlp_static PRIVATE
            ${GPU_PRESOLVER_DIR}/include)
        set_source_files_properties(src/presolve/gpu_presolver_integration.cpp
            PROPERTIES CXX_STANDARD 17)
    endif()
    
    # Link CUDA libraries
    target_link_libraries(hprlp_static
        PUBLIC
            CUDA::cublas
            CUDA::cusolver
            CUDA::cusparse
            CUDA::curand
            CUDA::cuda_driver
            CUDA::cudart
            ZLIB::ZLIB
    )
    
    # Set library properties
    set_target_properties(hprlp_static PROPERTIES
        OUTPUT_NAME hprlp
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        # Resolve device symbols in the installed static library.  Without
        # this, a downstream CMake C++ executable can inherit unresolved
        # registration symbols from the CUDA object files.
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Set include directories for the target
    target_include_directories(hprlp_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include/hprlp>
        PRIVATE
            ${PSLP_DIR}/include/PSLP
            ${PSLP_DIR}/include/core
            ${PSLP_DIR}/include/data_structures
            ${PSLP_DIR}/include/explorers
    )
endif()

# Create shared library (for language bindings: Python, Julia, MATLAB, etc.)
if(BUILD_SHARED_LIB)
    add_library(hprlp_shared SHARED ${HPRLP_LIB_SOURCES})
    target_sources(hprlp_shared PRIVATE $<TARGET_OBJECTS:pslp_embedded>)
    if(BUILD_GPU_PRESOLVER)
        target_sources(hprlp_shared PRIVATE
            $<TARGET_OBJECTS:gpu_presolver_embedded>)
        target_compile_definitions(hprlp_shared PRIVATE
            HPRLP_HAS_GPU_PRESOLVER=1)
        target_include_directories(hprlp_shared PRIVATE
            ${GPU_PRESOLVER_DIR}/include)
        set_source_files_properties(src/presolve/gpu_presolver_integration.cpp
            PROPERTIES CXX_STANDARD 17)
    endif()
    
    # Link CUDA libraries
    target_link_libraries(hprlp_shared
        PUBLIC
            CUDA::cublas
            CUDA::cusolver
            CUDA::cusparse
            CUDA::curand
            CUDA::cuda_driver
            CUDA::cudart
            ZLIB::ZLIB
    )
    
    # Set library properties
    set_target_properties(hprlp_shared PROPERTIES
        OUTPUT_NAME hprlp
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Add linker flags to exclude static libraries (matching Makefile)
    if(UNIX AND NOT APPLE)
        set_target_properties(hprlp_shared PROPERTIES
            LINK_FLAGS "-Wl,--exclude-libs,ALL"
        )
    endif()
    
    # Set include directories for the target
    target_include_directories(hprlp_shared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include/hprlp>
        PRIVATE
            ${PSLP_DIR}/include/PSLP
            ${PSLP_DIR}/include/core
            ${PSLP_DIR}/include/data_structures
            ${PSLP_DIR}/include/explorers
    )
endif()

# Alias to use the preferred library (static by default, or shared if static is disabled)
if(BUILD_STATIC_LIB)
    add_library(hprlp ALIAS hprlp_static)
elseif(BUILD_SHARED_LIB)
    add_library(hprlp ALIAS hprlp_shared)
endif()


# MPS/HDF5 file solver executable (statically linked for standalone use)
if(BUILD_STATIC_LIB)
    add_executable(solve_mps_file src/cli/solve_mps_file.cpp)
    target_link_libraries(solve_mps_file PRIVATE hprlp_static)
elseif(BUILD_SHARED_LIB)
    add_executable(solve_mps_file src/cli/solve_mps_file.cpp)
    target_link_libraries(solve_mps_file PRIVATE hprlp_shared)
endif()
if(HDF5_FOUND)
    target_compile_definitions(solve_mps_file PRIVATE HPRLP_HAS_HDF5=1)
    target_include_directories(solve_mps_file PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(solve_mps_file PRIVATE ${HDF5_LIBRARIES})
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(restart_control_test
        tests/restart_control_test.cpp
        src/solver/restart_control.cpp)
    add_test(NAME restart_control_test COMMAND restart_control_test)
endif()

# Build Python bindings
if(BUILD_PYTHON_BINDINGS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/CMakeLists.txt)
        add_subdirectory(bindings/python)
    else()
        message(WARNING "Python bindings CMakeLists.txt not found")
    endif()
endif()

# Installation
include(GNUInstallDirs)

# Install libraries
if(BUILD_STATIC_LIB)
    install(TARGETS hprlp_static
        EXPORT HPRLPTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(BUILD_SHARED_LIB)
    install(TARGETS hprlp_shared
        EXPORT HPRLPTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install executable
install(TARGETS solve_mps_file
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Preserve the organized header hierarchy under <prefix>/include/hprlp.
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hprlp
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Export targets
install(EXPORT HPRLPTargets
    FILE HPRLPTargets.cmake
    NAMESPACE HPRLP::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HPRLP
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HPRLPConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HPRLPConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HPRLPConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HPRLP
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HPRLPConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HPRLPConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HPRLP
)

# Print configuration summary
message(STATUS "")
message(STATUS "HPRLP 0.1.3 B200 Configuration Summary")
message(STATUS "============================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD} (required by CUDA 13 CCCL)")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build static library: ${BUILD_STATIC_LIB}")
message(STATUS "Build shared library: ${BUILD_SHARED_LIB}")
message(STATUS "Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Build GPU-Presolver-C dependency: ${BUILD_GPU_PRESOLVER}")
message(STATUS "Native solve_mps_file HDF5 support: ${HDF5_FOUND}")
message(STATUS "Compiler flags:")
message(STATUS "  CUDA: ${CMAKE_CUDA_FLAGS}")
message(STATUS "  CXX:  ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "")
