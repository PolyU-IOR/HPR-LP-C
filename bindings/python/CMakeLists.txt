cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Respect externally provided interpreter hints
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
if(DEFINED Python3_ROOT_DIR)
    set(Python3_FIND_STRATEGY LOCATION)
endif()
set(Python3_FIND_VIRTUALENV FIRST)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

# If pybind11 not found via CMake, try to find it via Python
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake, trying Python import...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS ${PYBIND11_CMAKE_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Python module source
set(PYBIND_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hprlp_pybind.cpp
)

# Create Python module
pybind11_add_module(_hprlp_core ${PYBIND_SOURCE})

# Link against hprlp library (prefer shared for Python bindings)
if(TARGET hprlp_shared)
    target_link_libraries(_hprlp_core PRIVATE hprlp_shared)
elseif(TARGET hprlp_static)
    target_link_libraries(_hprlp_core PRIVATE hprlp_static)
else()
    message(FATAL_ERROR "No hprlp library target found")
endif()

# Link CUDA runtime
target_link_libraries(_hprlp_core PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cudart
)

# Include directories
target_include_directories(_hprlp_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cuda_kernels
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Ensure the extension is placed in CMAKE_LIBRARY_OUTPUT_DIRECTORY
# (setup.py sets this to the wheel staging directory)
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_hprlp_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
endif()

# Install the Python module when invoking `cmake --install`
install(TARGETS _hprlp_core
    LIBRARY DESTINATION hprlp
)

# Install Python package files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hprlp/
    DESTINATION hprlp
    FILES_MATCHING PATTERN "*.py"
)

message(STATUS "Python bindings will be built and emitted to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
