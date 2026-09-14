cmake_minimum_required(VERSION 3.24)

project(GPUPresolverCpp LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.3)
  message(STATUS
    "GPU-Presolver-C: using cusparseSpMVOp ALG1 with CUDA ${CUDAToolkit_VERSION}")
else()
  message(STATUS
    "GPU-Presolver-C: using cusparseSpMV CSR ALG2 fallback with CUDA ${CUDAToolkit_VERSION}")
endif()

if(NOT DEFINED GPUPRESOLVER_CUSPARSE_LINK_TARGET OR
   "${GPUPRESOLVER_CUSPARSE_LINK_TARGET}" STREQUAL "")
  set(GPUPRESOLVER_CUSPARSE_LINK_TARGET CUDA::cusparse)
  if(NOT TARGET CUDA::cusparse)
    set(GPUPRESOLVER_CUSPARSE_LINK_TARGET "")
    find_library(GPUPRESOLVER_CUSPARSE_LIBRARY
      NAMES cusparse
      HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDAToolkit_TARGET_DIR}/lib
        ${CUDAToolkit_TARGET_DIR}/lib64
        ${CUDAToolkit_BIN_DIR}/../lib
        ${CUDAToolkit_BIN_DIR}/../lib64
        ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/lib64
        ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/x86_64-linux/lib
    )
    if(NOT GPUPRESOLVER_CUSPARSE_LIBRARY)
      message(FATAL_ERROR "Could not locate cuSPARSE library for gpu_presolver_core")
    endif()
    set(GPUPRESOLVER_CUSPARSE_LINK_TARGET ${GPUPRESOLVER_CUSPARSE_LIBRARY})
  endif()
endif()

if(NOT EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cusparse.h")
  message(FATAL_ERROR "Could not locate cusparse.h in the selected CUDA toolkit")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
enable_testing()

add_library(gpu_presolver_core
  src/folding/folding.cu
  src/folding/folding_kernels.cu
  src/folding/folding_map.cu
  src/folding/folding_reduce.cu
  src/folding/folding_refinement.cu
  src/model/lp_problem.cpp
  src/presolve/presolve_config.cpp
  src/presolve/gpu_postsolve.cu
  src/presolve/gpu_presolve.cu
  src/presolve/gpu_presolve_kernels.cu
  src/presolve/presolve_structs.cu
  src/presolve/rules/rule_close_bounds.cu
  src/presolve/rules/rule_activity_checks.cu
  src/presolve/rules/rule_doubleton_eq.cu
  src/presolve/rules/rule_dual_fix.cu
  src/presolve/rules/rule_empty_cols.cu
  src/presolve/rules/rule_empty_rows.cu
  src/presolve/rules/rule_parallel_cols.cu
  src/presolve/rules/rule_parallel_rows.cu
  src/presolve/rules/rule_primal_propagation.cu
  src/presolve/rules/rule_redundant_bounds.cu
  src/presolve/rules/rule_singleton_cols.cu
  src/presolve/rules/rule_singleton_rows.cu
  src/presolve/rules/rule_structural_l1_substitution.cu
)

target_include_directories(gpu_presolver_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.3)
  target_compile_definitions(gpu_presolver_core PUBLIC
    CUSPARSE_ENABLE_EXPERIMENTAL_API
  )
endif()

target_link_libraries(gpu_presolver_core PUBLIC
  ${GPUPRESOLVER_CUSPARSE_LINK_TARGET}
)

set_target_properties(gpu_presolver_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(test_lp_problem
  tests/test_lp_problem.cpp
)

target_link_libraries(test_lp_problem PRIVATE
  gpu_presolver_core
)

add_test(NAME test_lp_problem COMMAND test_lp_problem)

add_executable(test_presolve_config
  tests/test_presolve_config.cpp
)

target_link_libraries(test_presolve_config PRIVATE
  gpu_presolver_core
)

target_compile_definitions(test_presolve_config PRIVATE
  GPU_PRESOLVER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME test_presolve_config COMMAND test_presolve_config)

add_executable(test_gpu_direct_port
  tests/test_gpu_direct_port.cpp
)

target_link_libraries(test_gpu_direct_port PRIVATE
  gpu_presolver_core
)

add_test(NAME test_gpu_direct_port COMMAND test_gpu_direct_port)

add_executable(test_rule_close_bounds
  tests/test_rule_close_bounds.cpp
)

target_link_libraries(test_rule_close_bounds PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_close_bounds COMMAND test_rule_close_bounds)

add_executable(test_gpu_presolve_kernels
  tests/test_gpu_presolve_kernels.cpp
)

target_link_libraries(test_gpu_presolve_kernels PRIVATE
  gpu_presolver_core
)

add_test(NAME test_gpu_presolve_kernels COMMAND test_gpu_presolve_kernels)

add_executable(test_gpu_postsolve
  tests/test_gpu_postsolve.cpp
)

target_link_libraries(test_gpu_postsolve PRIVATE
  gpu_presolver_core
)

add_test(NAME test_gpu_postsolve COMMAND test_gpu_postsolve)

add_executable(test_folding
  tests/test_folding.cpp
)

target_link_libraries(test_folding PRIVATE
  gpu_presolver_core
)

add_test(NAME test_folding COMMAND test_folding)

add_executable(test_rule_empty_rows
  tests/test_rule_empty_rows.cpp
)

target_link_libraries(test_rule_empty_rows PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_empty_rows COMMAND test_rule_empty_rows)

add_executable(test_rule_singleton_rows
  tests/test_rule_singleton_rows.cpp
)

target_link_libraries(test_rule_singleton_rows PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_singleton_rows COMMAND test_rule_singleton_rows)

add_executable(test_rule_singleton_cols
  tests/test_rule_singleton_cols.cpp
)

target_link_libraries(test_rule_singleton_cols PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_singleton_cols COMMAND test_rule_singleton_cols)

add_executable(test_rule_activity_checks
  tests/test_rule_activity_checks.cpp
)

target_link_libraries(test_rule_activity_checks PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_activity_checks COMMAND test_rule_activity_checks)

add_executable(test_rule_primal_propagation
  tests/test_rule_primal_propagation.cpp
)

target_link_libraries(test_rule_primal_propagation PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_primal_propagation COMMAND test_rule_primal_propagation)

add_executable(test_rule_parallel_rows
  tests/test_rule_parallel_rows.cpp
)

target_link_libraries(test_rule_parallel_rows PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_parallel_rows COMMAND test_rule_parallel_rows)

add_executable(test_rule_parallel_cols
  tests/test_rule_parallel_cols.cpp
)

target_link_libraries(test_rule_parallel_cols PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_parallel_cols COMMAND test_rule_parallel_cols)

add_executable(test_rule_doubleton_eq
  tests/test_rule_doubleton_eq.cpp
)

target_link_libraries(test_rule_doubleton_eq PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_doubleton_eq COMMAND test_rule_doubleton_eq)

add_executable(test_rule_dual_fix
  tests/test_rule_dual_fix.cpp
)

target_link_libraries(test_rule_dual_fix PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_dual_fix COMMAND test_rule_dual_fix)

add_executable(test_rule_redundant_bounds
  tests/test_rule_redundant_bounds.cpp
)

target_link_libraries(test_rule_redundant_bounds PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_redundant_bounds COMMAND test_rule_redundant_bounds)

add_executable(test_rule_structural_l1_substitution
  tests/test_rule_structural_l1_substitution.cpp
)

target_link_libraries(test_rule_structural_l1_substitution PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_structural_l1_substitution COMMAND test_rule_structural_l1_substitution)

add_executable(test_rule_empty_cols
  tests/test_rule_empty_cols.cpp
)

target_link_libraries(test_rule_empty_cols PRIVATE
  gpu_presolver_core
)

add_test(NAME test_rule_empty_cols COMMAND test_rule_empty_cols)

add_executable(test_mpsreader
  third_party/cpu_presolve_reference/tests/test_main.cpp
  third_party/cpu_presolve_reference/tests/test_mpsreader.cpp
  third_party/cpu_presolve_reference/src/core/lp_model.cpp
  third_party/cpu_presolve_reference/src/core/sparse_matrix.cpp
  third_party/cpu_presolve_reference/src/io/mpsreader.cpp
)

target_include_directories(test_mpsreader PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpu_presolve_reference/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpu_presolve_reference/tests
)

add_test(NAME test_mpsreader COMMAND test_mpsreader)

# The test suite uses assert() for its checks. Keep those checks active for
# Release and RelWithDebInfo builds, where CMake otherwise defines NDEBUG.
set(GPU_PRESOLVER_TEST_TARGETS
  test_lp_problem
  test_presolve_config
  test_gpu_direct_port
  test_rule_close_bounds
  test_gpu_presolve_kernels
  test_gpu_postsolve
  test_folding
  test_rule_empty_rows
  test_rule_singleton_rows
  test_rule_singleton_cols
  test_rule_activity_checks
  test_rule_primal_propagation
  test_rule_parallel_rows
  test_rule_parallel_cols
  test_rule_doubleton_eq
  test_rule_dual_fix
  test_rule_redundant_bounds
  test_rule_structural_l1_substitution
  test_rule_empty_cols
  test_mpsreader
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(test_target IN LISTS GPU_PRESOLVER_TEST_TARGETS)
    target_compile_options(${test_target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>
    )
  endforeach()
endif()

add_executable(gpu_presolver_mps
  tools/gpu_presolver_mps.cpp
  third_party/cpu_presolve_reference/src/core/lp_model.cpp
  third_party/cpu_presolve_reference/src/core/sparse_matrix.cpp
  third_party/cpu_presolve_reference/src/io/mpsreader.cpp
)

target_link_libraries(gpu_presolver_mps PRIVATE
  gpu_presolver_core
)

target_include_directories(gpu_presolver_mps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpu_presolve_reference/include
)

add_executable(gpu_presolver_export_reduced
  tools/gpu_presolver_export_reduced.cpp
  third_party/cpu_presolve_reference/src/core/lp_model.cpp
  third_party/cpu_presolve_reference/src/core/sparse_matrix.cpp
  third_party/cpu_presolve_reference/src/io/mpsreader.cpp
)

target_link_libraries(gpu_presolver_export_reduced PRIVATE
  gpu_presolver_core
)

target_include_directories(gpu_presolver_export_reduced PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpu_presolve_reference/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

add_executable(gpu_presolver_check_postsolve_solution
  tools/gpu_presolver_check_postsolve_solution.cpp
  third_party/cpu_presolve_reference/src/core/lp_model.cpp
  third_party/cpu_presolve_reference/src/core/sparse_matrix.cpp
  third_party/cpu_presolve_reference/src/io/mpsreader.cpp
)

target_link_libraries(gpu_presolver_check_postsolve_solution PRIVATE
  gpu_presolver_core
)

target_include_directories(gpu_presolver_check_postsolve_solution PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpu_presolve_reference/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

option(GPU_PRESOLVER_BUILD_CPU_REFERENCE "Build historical CPU reference scratch target" OFF)

if(GPU_PRESOLVER_BUILD_CPU_REFERENCE)
add_library(cpu_presolve_reference
  third_party/cpu_presolve_reference/src/core/lp_model.cpp
  third_party/cpu_presolve_reference/src/core/sparse_matrix.cpp
  third_party/cpu_presolve_reference/src/io/mpsreader.cpp
  third_party/cpu_presolve_reference/src/presolve/presolver.cpp
  third_party/cpu_presolve_reference/src/c_api/cpu_presolve_c.cpp
)

target_include_directories(cpu_presolve_reference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpu_presolve_reference/include
)

add_executable(cpu_presolve_reference_tests
  third_party/cpu_presolve_reference/tests/test_main.cpp
  third_party/cpu_presolve_reference/tests/test_lp_model.cpp
  third_party/cpu_presolve_reference/tests/test_mpsreader.cpp
  third_party/cpu_presolve_reference/tests/test_presolver.cpp
  third_party/cpu_presolve_reference/tests/test_sparse_matrix.cpp
  third_party/cpu_presolve_reference/tests/test_c_api.cpp
)

target_link_libraries(cpu_presolve_reference_tests PRIVATE
  cpu_presolve_reference
)

add_test(NAME cpu_presolve_reference_tests COMMAND cpu_presolve_reference_tests)

add_executable(cpu_presolve_reference_mps
  third_party/cpu_presolve_reference/tools/cpu_presolve_mps.cpp
)

target_link_libraries(cpu_presolve_reference_mps PRIVATE
  cpu_presolve_reference
)
endif()
