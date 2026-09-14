#include "folding_internal.cuh"

#include <cuda_runtime.h>
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13030
#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#endif
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gpu_presolver::folding {
namespace {

constexpr double kMaxFoldingReduceSize = 0.8;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

void throw_if_cusparse_error(cusparseStatus_t status, const char* context) {
  if (status == CUSPARSE_STATUS_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": cusparse status " +
                           std::to_string(static_cast<int>(status)));
}

void synchronize_for_timing(const char* context) {
  throw_if_cuda_error(cudaDeviceSynchronize(), context);
}

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

bool reduction_too_small(std::int32_t num_row_color,
                         std::int32_t num_col_color,
                         std::int32_t num_row,
                         std::int32_t num_col) {
  return static_cast<double>(num_row_color) > kMaxFoldingReduceSize * static_cast<double>(num_row) &&
         static_cast<double>(num_col_color) > kMaxFoldingReduceSize * static_cast<double>(num_col);
}

void spmv_csr(const presolve::DeviceCsrMatrix& A, const double* x, double* y) {
  if (A.rows == 0) {
    return;
  }
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t mat = nullptr;
  cusparseDnVecDescr_t x_desc = nullptr;
  cusparseDnVecDescr_t y_desc = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13030
  cusparseSpMVOpDescr_t operation_desc = nullptr;
  cusparseSpMVOpPlan_t operation_plan = nullptr;
#endif
  void* buffer = nullptr;
  std::size_t buffer_size = 0;
  const double alpha = 1.0;
  const double beta = 0.0;

  throw_if_cusparse_error(cusparseCreate(&handle), "cusparseCreate spmv");
  throw_if_cusparse_error(
      cusparseCreateCsr(&mat,
                        A.rows,
                        A.cols,
                        A.nnz,
                        A.rowPtr,
                        A.colVal,
                        A.nzVal,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F),
      "cusparseCreateCsr spmv");
  throw_if_cusparse_error(cusparseCreateDnVec(&x_desc, A.cols, const_cast<double*>(x), CUDA_R_64F),
                          "cusparseCreateDnVec x");
  throw_if_cusparse_error(cusparseCreateDnVec(&y_desc, A.rows, y, CUDA_R_64F),
                          "cusparseCreateDnVec y");
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13030
  throw_if_cusparse_error(
      cusparseSpMVOp_bufferSize(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                mat,
                                x_desc,
                                y_desc,
                                y_desc,
                                CUDA_R_64F,
                                CUSPARSE_SPMVOP_ALG1,
                                &buffer_size),
      "cusparseSpMVOp_bufferSize");
  if (buffer_size > 0) {
    throw_if_cuda_error(cudaMalloc(&buffer, buffer_size), "cudaMalloc spmv buffer");
  }
  throw_if_cusparse_error(
      cusparseSpMVOp_createDescr(
          handle, &operation_desc, CUSPARSE_OPERATION_NON_TRANSPOSE,
          mat, x_desc, y_desc, y_desc, CUDA_R_64F,
          CUSPARSE_SPMVOP_ALG1, buffer),
      "cusparseSpMVOp_createDescr");
  throw_if_cusparse_error(
      cusparseSpMVOp_createPlan(
          handle, operation_desc, &operation_plan, nullptr, 0),
      "cusparseSpMVOp_createPlan");
  throw_if_cusparse_error(
      cusparseSpMVOp(
          handle, operation_plan, &alpha, &beta, x_desc, y_desc, y_desc),
      "cusparseSpMVOp");
  cusparseSpMVOp_destroyPlan(operation_plan);
  cusparseSpMVOp_destroyDescr(operation_desc);
#else
  constexpr cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_CSR_ALG2;
  throw_if_cusparse_error(
      cusparseSpMV_bufferSize(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              mat,
                              x_desc,
                              &beta,
                              y_desc,
                              CUDA_R_64F,
                              algorithm,
                              &buffer_size),
      "cusparseSpMV_bufferSize");
  if (buffer_size > 0) {
    throw_if_cuda_error(cudaMalloc(&buffer, buffer_size), "cudaMalloc spmv buffer");
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
  throw_if_cusparse_error(
      cusparseSpMV_preprocess(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              mat,
                              x_desc,
                              &beta,
                              y_desc,
                              CUDA_R_64F,
                              algorithm,
                              buffer),
      "cusparseSpMV_preprocess");
#endif
  throw_if_cusparse_error(
      cusparseSpMV(handle,
                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                   &alpha,
                   mat,
                   x_desc,
                   &beta,
                   y_desc,
                   CUDA_R_64F,
                   algorithm,
                   buffer),
      "cusparseSpMV");
#endif
  cudaFree(buffer);
  cusparseDestroyDnVec(x_desc);
  cusparseDestroyDnVec(y_desc);
  cusparseDestroySpMat(mat);
  cusparseDestroy(handle);
}

void init_color(const presolve::LPInfoGpu& device_model,
                FoldingWorkspace& workspace,
                double tolerance) {
  workspace.num_row_color = 1;
  workspace.num_col_color = 1;
  if (workspace.num_row > 0) {
    throw_if_cuda_error(cudaMemset(workspace.row_color_id, 0, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_row)),
                        "cudaMemset init row_color_id");
  }
  if (workspace.num_col > 0) {
    throw_if_cuda_error(cudaMemset(workspace.col_color_id, 0, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_col)),
                        "cudaMemset init col_color_id");
  }
  workspace.num_row_color = folding_update_row_color_id(workspace, device_model.AL, tolerance);
  workspace.num_row_color = folding_update_row_color_id(workspace, device_model.AU, tolerance);
  workspace.num_col_color = folding_update_col_color_id(workspace, device_model.c, tolerance);
  workspace.num_col_color = folding_update_col_color_id(workspace, device_model.l, tolerance);
  workspace.num_col_color = folding_update_col_color_id(workspace, device_model.u, tolerance);
  folding_write_row_color_sig(workspace, 0, 2ULL);
  folding_write_col_color_sig(workspace, 0, 5ULL);
}

}  // namespace

bool refine_color(const presolve::LPInfoGpu& device_model,
                  FoldingWorkspace& workspace,
                  double tolerance,
                  bool verbose,
                  FoldingRefinementProfile* profile) {
  const auto total_start = std::chrono::steady_clock::now();

  const auto init_start = std::chrono::steady_clock::now();
  init_color(device_model, workspace, tolerance);
  synchronize_for_timing("cudaDeviceSynchronize init_color");
  if (profile != nullptr) {
    profile->init_color_seconds +=
        seconds_since(init_start, std::chrono::steady_clock::now());
  }
  if (reduction_too_small(workspace.num_row_color,
                          workspace.num_col_color,
                          workspace.num_row,
                          workspace.num_col)) {
    if (verbose) {
      std::cout << "Not enough folding reduction, reduced size "
                << workspace.num_row_color << " row, "
                << workspace.num_col_color << " col.\n";
    }
    if (profile != nullptr) {
      profile->refinement_ok = false;
      profile->final_row_colors = workspace.num_row_color;
      profile->final_col_colors = workspace.num_col_color;
      profile->total_seconds =
          seconds_since(total_start, std::chrono::steady_clock::now());
    }
    return false;
  }

  const std::int32_t max_round = std::max<std::int32_t>(1, workspace.num_row + workspace.num_col);
  for (std::int32_t round = 1; round <= max_round; ++round) {
    const auto spmv_at_start = std::chrono::steady_clock::now();
    spmv_csr(device_model.AT, workspace.row_color_sig, workspace.col_color_sig);
    synchronize_for_timing("cudaDeviceSynchronize spmv AT");
    if (profile != nullptr) {
      profile->spmv_at_seconds +=
          seconds_since(spmv_at_start, std::chrono::steady_clock::now());
    }

    const std::int32_t prev_col_color_num = workspace.num_col_color;
    const auto refine_col_start = std::chrono::steady_clock::now();
    workspace.num_col_color =
        folding_update_col_color_id(workspace, workspace.col_color_sig, tolerance);
    synchronize_for_timing("cudaDeviceSynchronize refine col");
    const bool col_changed = workspace.num_col_color > prev_col_color_num;
    folding_write_col_color_sig(workspace, round, 1ULL);
    synchronize_for_timing("cudaDeviceSynchronize write col sig");
    if (profile != nullptr) {
      profile->refine_col_seconds +=
          seconds_since(refine_col_start, std::chrono::steady_clock::now());
    }

    const auto spmv_a_start = std::chrono::steady_clock::now();
    spmv_csr(device_model.A, workspace.col_color_sig, workspace.row_color_sig);
    synchronize_for_timing("cudaDeviceSynchronize spmv A");
    if (profile != nullptr) {
      profile->spmv_a_seconds +=
          seconds_since(spmv_a_start, std::chrono::steady_clock::now());
    }

    const std::int32_t prev_row_color_num = workspace.num_row_color;
    const auto refine_row_start = std::chrono::steady_clock::now();
    workspace.num_row_color =
        folding_update_row_color_id(workspace, workspace.row_color_sig, tolerance);
    synchronize_for_timing("cudaDeviceSynchronize refine row");
    const bool row_changed = workspace.num_row_color > prev_row_color_num;
    folding_write_row_color_sig(workspace, round, 2ULL);
    synchronize_for_timing("cudaDeviceSynchronize write row sig");
    if (profile != nullptr) {
      profile->refine_row_seconds +=
          seconds_since(refine_row_start, std::chrono::steady_clock::now());
      profile->rounds = round;
    }

    if (reduction_too_small(workspace.num_row_color,
                            workspace.num_col_color,
                            workspace.num_row,
                            workspace.num_col)) {
      if (verbose) {
        std::cout << "Not enough folding reduction, reduced size "
                  << workspace.num_row_color << " row, "
                  << workspace.num_col_color << " col.\n";
      }
      if (profile != nullptr) {
        profile->refinement_ok = false;
        profile->final_row_colors = workspace.num_row_color;
        profile->final_col_colors = workspace.num_col_color;
        profile->total_seconds =
            seconds_since(total_start, std::chrono::steady_clock::now());
      }
      return false;
    }
    if (!(row_changed || col_changed)) {
      if (profile != nullptr) {
        profile->refinement_ok = true;
        profile->final_row_colors = workspace.num_row_color;
        profile->final_col_colors = workspace.num_col_color;
        profile->total_seconds =
            seconds_since(total_start, std::chrono::steady_clock::now());
      }
      return true;
    }
  }

  if (verbose) {
    std::cout << "Hybrid signature color refinement reached round limit ("
              << max_round << "); stopping.\n";
  }
  if (profile != nullptr) {
    profile->refinement_ok = true;
    profile->final_row_colors = workspace.num_row_color;
    profile->final_col_colors = workspace.num_col_color;
    profile->total_seconds =
        seconds_since(total_start, std::chrono::steady_clock::now());
  }
  return true;
}

}  // namespace gpu_presolver::folding
