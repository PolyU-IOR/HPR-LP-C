#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

__global__ void _kernel_compute_row_nnz(std::int32_t* row_nnz,
                                        const std::int32_t* rowPtr,
                                        std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    row_nnz[i] = rowPtr[i + 1] - rowPtr[i];
  }
}

__global__ void _kernel_singleton_row_support(std::int32_t* singleton_col,
                                              double* singleton_val,
                                              const std::int32_t* row_nnz,
                                              const std::int32_t* rowPtr,
                                              const std::int32_t* colVal,
                                              const double* nzVal,
                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    const std::int32_t nnz_i = row_nnz[i];
    if (nnz_i == 1) {
      const std::int32_t p = rowPtr[i];
      singleton_col[i] = colVal[p];
      singleton_val[i] = nzVal[p];
    } else {
      singleton_col[i] = -1;
      singleton_val[i] = 0.0;
    }
  }
}

__global__ void _kernel_compute_row_activity_bounds(double* row_min,
                                                    double* row_max,
                                                    const std::int32_t* rowPtr,
                                                    const std::int32_t* colVal,
                                                    const double* nzVal,
                                                    const double* l,
                                                    const double* u,
                                                    std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    double acc_min = 0.0;
    double acc_max = 0.0;
    const std::int32_t row_start = rowPtr[i];
    const std::int32_t row_stop = rowPtr[i + 1];
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const std::int32_t col = colVal[p];
      const double a = nzVal[p];
      const double lj = l[col];
      const double uj = u[col];
      if (a >= 0.0) {
        acc_min += a * lj;
        acc_max += a * uj;
      } else {
        acc_min += a * uj;
        acc_max += a * lj;
      }
    }
    row_min[i] = acc_min;
    row_max[i] = acc_max;
  }
}

__global__ void _kernel_compute_row_activity_summary(double* row_min_fin,
                                                     double* row_max_fin,
                                                     std::int32_t* row_min_neg_inf_count,
                                                     std::int32_t* row_max_pos_inf_count,
                                                     const std::int32_t* rowPtr,
                                                     const std::int32_t* colVal,
                                                     const double* nzVal,
                                                     const double* l,
                                                     const double* u,
                                                     double zero_tol,
                                                     std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    double acc_min_fin = 0.0;
    double acc_max_fin = 0.0;
    std::int32_t neg_inf_count = 0;
    std::int32_t pos_inf_count = 0;
    const std::int32_t row_start = rowPtr[i];
    const std::int32_t row_stop = rowPtr[i + 1];
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const double a = nzVal[p];
      if (fabs(a) <= zero_tol) {
        continue;
      }
      const std::int32_t col = colVal[p];
      const double lj = l[col];
      const double uj = u[col];

      const double term_min = a >= 0.0 ? (a * lj) : (a * uj);
      const double term_max = a >= 0.0 ? (a * uj) : (a * lj);

      if (isfinite(term_min)) {
        acc_min_fin += term_min;
      } else if (term_min < 0.0) {
        ++neg_inf_count;
      }

      if (isfinite(term_max)) {
        acc_max_fin += term_max;
      } else if (term_max > 0.0) {
        ++pos_inf_count;
      }
    }
    row_min_fin[i] = acc_min_fin;
    row_max_fin[i] = acc_max_fin;
    row_min_neg_inf_count[i] = neg_inf_count;
    row_max_pos_inf_count[i] = pos_inf_count;
  }
}

__global__ void _kernel_compute_col_max_abs(double* col_max_abs,
                                            const std::int32_t* rowPtr,
                                            const double* nzVal,
                                            std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    double vmax = 0.0;
    const std::int32_t row_start = rowPtr[j];
    const std::int32_t row_stop = rowPtr[j + 1];
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      vmax = fmax(vmax, fabs(nzVal[p]));
    }
    col_max_abs[j] = vmax;
  }
}

}  // namespace

void compute_row_nnz(std::int32_t* row_nnz, const DeviceCsrMatrix& A_csr) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_row_nnz<<<blocks, GPU_PRESOLVE_THREADS>>>(row_nnz, A_csr.rowPtr, A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_row_nnz");
}

void compute_col_nnz(std::int32_t* col_nnz, const DeviceCsrMatrix& AT_csr) {
  compute_row_nnz(col_nnz, AT_csr);
}

void compute_singleton_row_support(std::int32_t* singleton_col,
                                   double* singleton_val,
                                   const std::int32_t* row_nnz,
                                   const DeviceCsrMatrix& A_csr) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_row_support<<<blocks, GPU_PRESOLVE_THREADS>>>(
      singleton_col,
      singleton_val,
      row_nnz,
      A_csr.rowPtr,
      A_csr.colVal,
      A_csr.nzVal,
      A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_row_support");
}

void compute_singleton_col_support(std::int32_t* singleton_row,
                                   double* singleton_val,
                                   const std::int32_t* col_nnz,
                                   const DeviceCsrMatrix& AT_csr) {
  compute_singleton_row_support(singleton_row, singleton_val, col_nnz, AT_csr);
}

void compute_row_activity_bounds(double* row_min,
                                 double* row_max,
                                 const DeviceCsrMatrix& A_csr,
                                 const double* l,
                                 const double* u) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_row_activity_bounds<<<blocks, GPU_PRESOLVE_THREADS>>>(
      row_min,
      row_max,
      A_csr.rowPtr,
      A_csr.colVal,
      A_csr.nzVal,
      l,
      u,
      A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_row_activity_bounds");
}

void compute_row_activity_summary(double* row_min_fin,
                                  double* row_max_fin,
                                  std::int32_t* row_min_neg_inf_count,
                                  std::int32_t* row_max_pos_inf_count,
                                  const DeviceCsrMatrix& A_csr,
                                  const double* l,
                                  const double* u,
                                  double zero_tol) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_row_activity_summary<<<blocks, GPU_PRESOLVE_THREADS>>>(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      A_csr.rowPtr,
      A_csr.colVal,
      A_csr.nzVal,
      l,
      u,
      zero_tol,
      A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_row_activity_summary");
}

void compute_col_max_abs(double* col_max_abs, const DeviceCsrMatrix& AT_csr) {
  if (AT_csr.rows == 0) {
    return;
  }
  const int blocks = (AT_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_col_max_abs<<<blocks, GPU_PRESOLVE_THREADS>>>(
      col_max_abs,
      AT_csr.rowPtr,
      AT_csr.nzVal,
      AT_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_col_max_abs");
}

}  // namespace gpu_presolver::presolve
