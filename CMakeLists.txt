cmake_minimum_required(VERSION 3.18)
project(HPRLP VERSION 0.1.1 LANGUAGES CXX CUDA)

# Set C++ and CUDA standards (C++11 for better compatibility with MATLAB and older systems)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(BUILD_STATIC_LIB "Build static library" ON)
option(BUILD_SHARED_LIB "Build shared library" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_DOXYGEN "Build documentation" OFF)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Auto-detect CUDA architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Try to detect GPU compute capability using nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE DETECTED_CC
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(DETECTED_CC AND NOT DETECTED_CC STREQUAL "N/A")
        string(REPLACE "." "" DETECTED_SM "${DETECTED_CC}")
        string(STRIP "${DETECTED_SM}" DETECTED_SM)
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_SM})
        message(STATUS "Detected CUDA architecture: sm_${DETECTED_SM}")
    else()
        # Fallback to sm_52 if detection fails (same as Makefile)
        set(CMAKE_CUDA_ARCHITECTURES "52")
        message(STATUS "Using default CUDA architecture: sm_52")
    endif()
endif()

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

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cuda_kernels
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files for the library
set(HPRLP_LIB_SOURCES
    src/mps_reader.cpp
    src/utils.cu
    src/scaling.cu
    src/preprocess.cu
    src/power_iteration.cu
    src/main_iterate.cu
    src/HPRLP.cu
    src/cuda_kernels/HPR_cuda_kernels.cu
)

# Create static library (default)
if(BUILD_STATIC_LIB)
    add_library(hprlp_static STATIC ${HPRLP_LIB_SOURCES})
    
    # Link CUDA libraries
    target_link_libraries(hprlp_static
        PUBLIC
            CUDA::cublas
            CUDA::cusolver
            CUDA::cusparse
            CUDA::curand
    )
    
    # Set library properties
    set_target_properties(hprlp_static PROPERTIES
        OUTPUT_NAME hprlp
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Set include directories for the target
    target_include_directories(hprlp_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
endif()

# Create shared library (for language bindings: Python, Julia, MATLAB, etc.)
if(BUILD_SHARED_LIB)
    add_library(hprlp_shared SHARED ${HPRLP_LIB_SOURCES})
    
    # Link CUDA libraries
    target_link_libraries(hprlp_shared
        PUBLIC
            CUDA::cublas
            CUDA::cusolver
            CUDA::cusparse
            CUDA::curand
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
            $<INSTALL_INTERFACE:include>
    )
endif()

# Alias to use the preferred library (static by default, or shared if static is disabled)
if(BUILD_STATIC_LIB)
    add_library(hprlp ALIAS hprlp_static)
elseif(BUILD_SHARED_LIB)
    add_library(hprlp ALIAS hprlp_shared)
endif()

# Public headers (for installation)
set(HPRLP_PUBLIC_HEADERS
    include/HPRLP.h
    include/structs.h
    include/constants.h
    include/version.h
    include/mps_reader.h
    include/utils.h
    include/preprocess.h
    include/scaling.h
    include/power_iteration.h
    include/main_iterate.h
)

# MPS file solver executable (statically linked for standalone use)
if(BUILD_STATIC_LIB)
    add_executable(solve_mps_file src/solve_mps_file.cpp)
    target_link_libraries(solve_mps_file PRIVATE hprlp_static)
elseif(BUILD_SHARED_LIB)
    add_executable(solve_mps_file src/solve_mps_file.cpp)
    target_link_libraries(solve_mps_file PRIVATE hprlp_shared)
endif()

# Build examples
if(BUILD_EXAMPLES)
    # Note: examples/CMakeLists.txt needs to be created
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
        add_subdirectory(examples)
    endif()
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

install(FILES ${HPRLP_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hprlp
)

install(DIRECTORY include/cuda_kernels
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

# Doxygen documentation
if(ENABLE_DOXYGEN)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    else()
        message(WARNING "Doxygen not found, documentation will not be generated")
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "HPRLP Configuration Summary")
message(STATUS "============================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD} (for compatibility)")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build static library: ${BUILD_STATIC_LIB}")
message(STATUS "Build shared library: ${BUILD_SHARED_LIB}")
message(STATUS "Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Enable documentation: ${ENABLE_DOXYGEN}")
message(STATUS "Compiler flags:")
message(STATUS "  CUDA: ${CMAKE_CUDA_FLAGS}")
message(STATUS "  CXX:  ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "")
