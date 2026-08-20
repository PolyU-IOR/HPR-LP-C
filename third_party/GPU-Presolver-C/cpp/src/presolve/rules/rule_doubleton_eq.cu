#include "gpu_presolver/presolve/rules/rule_doubleton_eq.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr int DOUBLETONEQ_BATCH_MATCHING_ROUNDS = 8;
constexpr int DOUBLETONEQ_BATCH_INNER_ROUNDS = 1024;
constexpr int DOUBLETONEQ_TAPE_INDEX_STRIDE = 3;
constexpr int DOUBLETONEQ_TAPE_VALUE_STRIDE = 12;
constexpr double DOUBLETONEQ_MAX_RATIO_PIVOT = 1.0e3;
// Avoid the per-row selection sort when rebuilding AT inside doubleton batches.
// It is quadratic on dense transpose rows and Julia does not spend core time here.
constexpr std::int32_t CSR_SORT_TRANSPOSE_NNZ_LIMIT = 0;

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
  throw std::runtime_error(std::string(context) + ": cuSPARSE status " +
                           std::to_string(static_cast<int>(status)));
}

template <class T>
cudaError_t doubleton_cuda_malloc(T** ptr, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(ptr), bytes, nullptr);
}

cudaError_t doubleton_cuda_free(void* ptr) {
  return ptr == nullptr ? cudaSuccess : cudaFreeAsync(ptr, nullptr);
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string s(value);
  return !(s.empty() || s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(static_cast<std::int32_t>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<std::int32_t>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

template <class T>
std::vector<T> copy_device_vector(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(),
                                   device,
                                   sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

__global__ void _kernel_doubleton_sort_csr_row_keys(unsigned long long* keys,
                                                    const std::int32_t* row_ptr,
                                                    const std::int32_t* col_val,
                                                    std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    const unsigned long long row_key = static_cast<unsigned long long>(static_cast<std::uint32_t>(row)) << 32;
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      keys[p] = row_key | static_cast<unsigned long long>(static_cast<std::uint32_t>(col_val[p]));
    }
  }
}

__global__ void _kernel_doubleton_unpack_csr_row_keys(std::int32_t* col_val,
                                                      const unsigned long long* keys,
                                                      std::int32_t nnz) {
  const std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < nnz) {
    col_val[p] = static_cast<std::int32_t>(keys[p] & 0xffffffffULL);
  }
}

void sort_csr_rows_by_col_thrust(DeviceCsrMatrix& matrix, const char* context) {
  if (matrix.nnz <= 0) {
    return;
  }
  unsigned long long* sort_keys = nullptr;
  throw_if_cuda_error(doubleton_cuda_malloc(&sort_keys, sizeof(unsigned long long) * static_cast<std::size_t>(matrix.nnz)),
                      context);
  const int blocks_rows = (matrix.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_sort_csr_row_keys<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      sort_keys, matrix.rowPtr, matrix.colVal, matrix.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_sort_csr_row_keys");
  thrust::sort_by_key(thrust::device_pointer_cast(sort_keys),
                      thrust::device_pointer_cast(sort_keys + matrix.nnz),
                      thrust::device_pointer_cast(matrix.nzVal));
  throw_if_cuda_error(cudaGetLastError(), "thrust doubleton sort CSR rows");
  const int blocks_nnz = (matrix.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_unpack_csr_row_keys<<<blocks_nnz, GPU_PRESOLVE_THREADS>>>(
      matrix.colVal, sort_keys, matrix.nnz);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_unpack_csr_row_keys");
  throw_if_cuda_error(cudaDeviceSynchronize(), context);
  doubleton_cuda_free(sort_keys);
}

__device__ bool _is_integral_ratio_doubleton_eq(double num, double den, double tol) {
  if (fabs(den) <= tol) {
    return false;
  }
  const double ratio = fabs(num / den);
  if (!isfinite(ratio)) {
    return false;
  }
  return fabs(ratio - nearbyint(ratio)) <= tol;
}

__device__ bool _is_acceptable_doubleton_eq_pivot(double keep_val, double elim_val) {
  if (fabs(elim_val) <= 0.0) {
    return false;
  }
  const double pivot_ratio = fabs(keep_val / elim_val);
  return pivot_ratio <= DOUBLETONEQ_MAX_RATIO_PIVOT &&
         pivot_ratio >= 1.0 / DOUBLETONEQ_MAX_RATIO_PIVOT;
}

__device__ void _choose_doubleton_eq_columns(std::int32_t col1,
                                             double val1,
                                             std::int32_t col2,
                                             double val2,
                                             const std::int32_t* col_nnz,
                                             double tol,
                                             std::int32_t* elim_col,
                                             double* elim_val,
                                             std::int32_t* keep_col,
                                             double* keep_val) {
  const bool integral12 = _is_integral_ratio_doubleton_eq(val1, val2, tol);
  const bool integral21 = _is_integral_ratio_doubleton_eq(val2, val1, tol);

  if (col_nnz[col1] == 1 && col_nnz[col2] != 1) {
    *elim_col = col1;
    *elim_val = val1;
    *keep_col = col2;
    *keep_val = val2;
  } else if (col_nnz[col1] != 1 && col_nnz[col2] == 1) {
    *elim_col = col2;
    *elim_val = val2;
    *keep_col = col1;
    *keep_val = val1;
  } else if (integral12 && !integral21) {
    *elim_col = col2;
    *elim_val = val2;
    *keep_col = col1;
    *keep_val = val1;
  } else if (integral21 && !integral12) {
    *elim_col = col1;
    *elim_val = val1;
    *keep_col = col2;
    *keep_val = val2;
  } else if (col_nnz[col1] < col_nnz[col2]) {
    *elim_col = col1;
    *elim_val = val1;
    *keep_col = col2;
    *keep_val = val2;
  } else {
    *elim_col = col2;
    *elim_val = val2;
    *keep_col = col1;
    *keep_val = val1;
  }
}

__device__ std::int32_t _doubleton_eq_fill_in_proxy_device(const std::int32_t* row_ptr,
                                                           const std::int32_t* col_val,
                                                           std::int32_t keep_col,
                                                           std::int32_t elim_col) {
  std::int32_t fill_in = -1;
  std::int32_t jj = row_ptr[keep_col];
  const std::int32_t keep_stop = row_ptr[keep_col + 1];
  std::int32_t kk = row_ptr[elim_col];
  const std::int32_t elim_stop = row_ptr[elim_col + 1];
  if (keep_stop <= jj || elim_stop <= kk) {
    return fill_in;
  }
  while (jj < keep_stop && kk < elim_stop) {
    const std::int32_t keep_row = col_val[jj];
    const std::int32_t elim_row = col_val[kk];
    if (keep_row == elim_row) {
      ++jj;
      ++kk;
    } else if (elim_row < keep_row) {
      ++kk;
      ++fill_in;
    } else {
      ++jj;
    }
  }
  fill_in += elim_stop - kk;
  return fill_in;
}

__global__ void _kernel_doubleton_eq_live_col_nnz(std::int32_t* col_nnz,
                                                  const std::int32_t* AT_row_ptr,
                                                  const std::int32_t* AT_col_val,
                                                  const double* AT_nz_val,
                                                  const std::uint8_t* keep_row,
                                                  const std::uint8_t* keep_col,
                                                  double zero_tol,
                                                  std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    if (keep_col[col] == std::uint8_t{0}) {
      col_nnz[col] = 0;
      return;
    }
    std::int32_t count = 0;
    const std::int32_t start_idx = AT_row_ptr[col];
    const std::int32_t stop_idx = AT_row_ptr[col + 1];
    for (std::int32_t p = start_idx; p < stop_idx; ++p) {
      const std::int32_t row = AT_col_val[p];
      const double a = AT_nz_val[p];
      if (keep_row[row] != std::uint8_t{0} && fabs(a) > zero_tol) {
        ++count;
      }
    }
    col_nnz[col] = count;
  }
}

__global__ void _kernel_select_doubleton_eq_target_row_direct(std::int32_t* target_row_ref,
                                                              std::int32_t* target_meta_i32,
                                                              double* target_meta_f64,
                                                              const std::int32_t* row_ptr,
                                                              const std::int32_t* col_val,
                                                              const double* nz_val,
                                                              const std::uint8_t* keep_row,
                                                              const std::uint8_t* keep_col_mask,
                                                              const std::int32_t* col_nnz,
                                                              const double* AL,
                                                              const double* AU,
                                                              const double* l,
                                                              const double* u,
                                                              const double* c,
                                                              const std::int32_t* AT_row_ptr,
                                                              const std::int32_t* AT_col_val,
                                                              double zero_tol,
                                                              double tol,
                                                              double ratio_tol,
                                                              std::int32_t max_fill_in_proxy,
                                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0}) {
    const double lhs = AL[i];
    const double rhs = AU[i];
    if (!isfinite(lhs) || !isfinite(rhs) || fabs(lhs - rhs) > tol) {
      return;
    }
    const std::int32_t row_start = row_ptr[i];
    const std::int32_t row_stop = row_ptr[i + 1];
    std::int32_t live_count = 0;
    std::int32_t col1 = -1;
    std::int32_t col2 = -1;
    double val1 = 0.0;
    double val2 = 0.0;
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const std::int32_t col_j = col_val[p];
      const double a = nz_val[p];
      if (keep_col_mask[col_j] != std::uint8_t{0} && fabs(a) > zero_tol) {
        ++live_count;
        if (live_count == 1) {
          col1 = col_j;
          val1 = a;
        } else if (live_count == 2) {
          col2 = col_j;
          val2 = a;
        } else {
          return;
        }
      }
    }
    if (live_count != 2) {
      return;
    }

    std::int32_t elim_col = -1;
    std::int32_t keep_col = -1;
    double elim_val = 0.0;
    double keep_val = 0.0;
    _choose_doubleton_eq_columns(col1, val1, col2, val2, col_nnz, ratio_tol,
                                 &elim_col, &elim_val, &keep_col, &keep_val);
    if (!_is_acceptable_doubleton_eq_pivot(keep_val, elim_val)) {
      return;
    }
    const std::int32_t keep_nnz = col_nnz[keep_col];
    const std::int32_t elim_nnz = col_nnz[elim_col];
    if (elim_nnz > keep_nnz + max_fill_in_proxy + 1) {
      return;
    }
    const std::int32_t fill_in =
        _doubleton_eq_fill_in_proxy_device(AT_row_ptr, AT_col_val, keep_col, elim_col);
    if (fill_in > max_fill_in_proxy) {
      return;
    }

    const std::int32_t old = atomicMin(target_row_ref, i);
    if (i < old) {
      target_meta_i32[0] = i;
      target_meta_i32[1] = elim_col;
      target_meta_i32[2] = keep_col;
      target_meta_f64[0] = elim_val;
      target_meta_f64[1] = keep_val;
      target_meta_f64[2] = rhs;
      target_meta_f64[3] = l[elim_col];
      target_meta_f64[4] = u[elim_col];
      target_meta_f64[5] = c[elim_col];
      target_meta_f64[6] = l[keep_col];
      target_meta_f64[7] = u[keep_col];
    }
  }
}

__global__ void _kernel_doubleton_eq_apply(std::int32_t* status,
                                           double* l,
                                           double* u,
                                           double* c,
                                           double* AL,
                                           double* AU,
                                           std::uint8_t* keep_row_mask,
                                           std::uint8_t* keep_col_mask,
                                           double* obj_delta,
                                           const std::int32_t* meta_i32,
                                           const double* meta_f64,
                                           const std::int32_t* AT_row_ptr,
                                           const std::int32_t* AT_col_val,
                                           const double* AT_nz_val,
                                           double bound_tol,
                                           std::int32_t m) {
  const std::int32_t target_row = meta_i32[0];
  const std::int32_t elim_col = meta_i32[1];
  const std::int32_t keep_col = meta_i32[2];
  const double elim_val = meta_f64[0];
  const double keep_val = meta_f64[1];
  const double rhs = meta_f64[2];
  const double old_elim_l = meta_f64[3];
  const double old_elim_u = meta_f64[4];
  const double elim_obj = meta_f64[5];
  const double old_keep_l = meta_f64[6];
  const double old_keep_u = meta_f64[7];
  const double alpha = -keep_val / elim_val;
  const double beta = rhs / elim_val;

  double mapped_l = -INFINITY;
  double mapped_u = INFINITY;
  if (isfinite(old_elim_l)) {
    const double bound_val = (old_elim_l - beta) / alpha;
    if (alpha > 0.0) {
      mapped_l = bound_val;
    } else {
      mapped_u = bound_val;
    }
  }
  if (isfinite(old_elim_u)) {
    const double bound_val = (old_elim_u - beta) / alpha;
    if (alpha > 0.0) {
      mapped_u = bound_val;
    } else {
      mapped_l = bound_val;
    }
  }
  if (mapped_l > mapped_u) {
    const double tmp = mapped_l;
    mapped_l = mapped_u;
    mapped_u = tmp;
  }
  const double keep_l_new = isfinite(mapped_l) ? fmax(old_keep_l, mapped_l) : old_keep_l;
  const double keep_u_new = isfinite(mapped_u) ? fmin(old_keep_u, mapped_u) : old_keep_u;
  if (keep_l_new > keep_u_new + bound_tol) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      status[0] = 1;
    }
    return;
  }

  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < m) {
    double shift = 0.0;
    const std::int32_t start = AT_row_ptr[elim_col];
    const std::int32_t stop = AT_row_ptr[elim_col + 1];
    for (std::int32_t p = start; p < stop; ++p) {
      if (AT_col_val[p] == t) {
        shift = AT_nz_val[p] * beta;
        break;
      }
    }
    AL[t] -= shift;
    AU[t] -= shift;
  }
  if (t == 0) {
    l[keep_col] = keep_l_new;
    u[keep_col] = keep_u_new;
    c[keep_col] += elim_obj * alpha;
    keep_row_mask[target_row] = std::uint8_t{0};
    keep_col_mask[elim_col] = std::uint8_t{0};
    atomicAdd(obj_delta, elim_obj * beta);
    status[1] = 1;
  }
}

__global__ void _kernel_doubleton_eq_row_counts(std::int32_t* row_nnz_new,
                                                const std::int32_t* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                const std::int32_t* meta_i32,
                                                const double* meta_f64,
                                                double zero_tol,
                                                std::int32_t m) {
  const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= m) {
    return;
  }
  const std::int32_t target_row = meta_i32[0];
  const std::int32_t elim_col = meta_i32[1];
  const std::int32_t keep_col = meta_i32[2];
  const double alpha = -meta_f64[1] / meta_f64[0];

  const std::int32_t first = row_ptr[r];
  const std::int32_t last = row_ptr[r + 1];
  std::int32_t len = last - first;
  if (r == target_row) {
    row_nnz_new[r] = len;
    return;
  }

  bool elim_present = false;
  bool keep_present = false;
  double are = 0.0;
  double old_keep = 0.0;
  for (std::int32_t p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (col == elim_col) {
      elim_present = true;
      are = val;
    } else if (col == keep_col) {
      keep_present = true;
      old_keep = val;
    }
  }

  const double new_keep = old_keep + alpha * are;
  if (elim_present && !keep_present && fabs(new_keep) > zero_tol) {
    ++len;
  } else if (keep_present && fabs(new_keep) <= zero_tol) {
    --len;
  }
  row_nnz_new[r] = len > 0 ? len : 0;
}

__global__ void _kernel_row_ptr_from_prefix(std::int32_t* row_ptr,
                                            const std::int32_t* prefix,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) {
    row_ptr[0] = 0;
  }
  if (i < m) {
    row_ptr[i + 1] = prefix[i];
  }
}

__global__ void _kernel_inclusive_scan_i32_serial(std::int32_t* values, std::int32_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::int32_t i = 0; i < n; ++i) {
      running += values[i];
      values[i] = running;
    }
  }
}

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          temp_storage, temp_bytes, values, values, n),
                      context);
  throw_if_cuda_error(doubleton_cuda_malloc(&temp_storage, temp_bytes), context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          temp_storage, temp_bytes, values, values, n),
                      context);
  doubleton_cuda_free(temp_storage);
}

__global__ void _kernel_fill_i32(std::int32_t* data, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    data[i] = value;
  }
}

__global__ void _kernel_count_transpose_rows_local(std::int32_t* counts,
                                                   const std::int32_t* col_val,
                                                   std::int32_t nnz) {
  const std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < nnz) {
    atomicAdd(&counts[col_val[p]], 1);
  }
}

__global__ void _kernel_scatter_transpose_atomic_local(std::int32_t* at_col_val,
                                                       double* at_nz_val,
                                                       std::int32_t* write_ptr,
                                                       const std::int32_t* row_ptr,
                                                       const std::int32_t* col_val,
                                                       const double* nz_val,
                                                       std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col = col_val[p];
      const std::int32_t dst = atomicAdd(&write_ptr[col], 1);
      at_col_val[dst] = row;
      at_nz_val[dst] = nz_val[p];
    }
  }
}

__global__ void _kernel_sort_csr_rows_by_col_local(std::int32_t* col_val,
                                                   double* nz_val,
                                                   const std::int32_t* row_ptr,
                                                   std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    const std::int32_t first = row_ptr[row];
    const std::int32_t last = row_ptr[row + 1];
    for (std::int32_t i = first; i + 1 < last; ++i) {
      std::int32_t min_pos = i;
      std::int32_t min_col = col_val[i];
      for (std::int32_t j = i + 1; j < last; ++j) {
        const std::int32_t candidate = col_val[j];
        if (candidate < min_col) {
          min_col = candidate;
          min_pos = j;
        }
      }
      if (min_pos != i) {
        const std::int32_t tmp_col = col_val[i];
        col_val[i] = col_val[min_pos];
        col_val[min_pos] = tmp_col;
        const double tmp_val = nz_val[i];
        nz_val[i] = nz_val[min_pos];
        nz_val[min_pos] = tmp_val;
      }
    }
  }
}

__global__ void _kernel_u8_to_i32(std::int32_t* out, const std::uint8_t* in, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    out[i] = in[i] != std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_scatter_mask_indices(std::int32_t* indices,
                                             const std::uint8_t* mask,
                                             const std::int32_t* prefix,
                                             std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n && mask[i] != std::uint8_t{0}) {
    indices[prefix[i] - 1] = i;
  }
}

__device__ std::int32_t _doubleton_eq_next_live_col_row(const std::int32_t* row_ptr,
                                                        const std::int32_t* col_val,
                                                        const double* nz_val,
                                                        const std::uint8_t* keep_row,
                                                        double zero_tol,
                                                        std::int32_t col,
                                                        std::int32_t* pos) {
  const std::int32_t stop = row_ptr[col + 1];
  while (*pos < stop) {
    const std::int32_t row = col_val[*pos];
    const double val = nz_val[*pos];
    if (keep_row[row] != std::uint8_t{0} && fabs(val) > zero_tol) {
      return row;
    }
    ++(*pos);
  }
  return INT_MAX;
}

__device__ std::int32_t _doubleton_eq_fill_in_proxy_device_live(const std::int32_t* row_ptr,
                                                                const std::int32_t* col_val,
                                                                const double* nz_val,
                                                                const std::uint8_t* keep_row,
                                                                double zero_tol,
                                                                std::int32_t keep_col,
                                                                std::int32_t elim_col) {
  std::int32_t fill_in = -1;
  std::int32_t jj = row_ptr[keep_col];
  const std::int32_t keep_stop = row_ptr[keep_col + 1];
  std::int32_t kk = row_ptr[elim_col];
  const std::int32_t elim_stop = row_ptr[elim_col + 1];
  if (keep_stop <= jj || elim_stop <= kk) {
    return fill_in;
  }

  while (jj < keep_stop && kk < elim_stop) {
    std::int32_t jj_live = jj;
    std::int32_t kk_live = kk;
    const std::int32_t keep_row_j =
        _doubleton_eq_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, keep_col, &jj_live);
    const std::int32_t elim_row_k =
        _doubleton_eq_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, elim_col, &kk_live);
    if (keep_row_j == INT_MAX || elim_row_k == INT_MAX) {
      break;
    }
    if (keep_row_j == elim_row_k) {
      jj = jj_live + 1;
      kk = kk_live + 1;
    } else if (elim_row_k < keep_row_j) {
      kk = kk_live + 1;
      ++fill_in;
    } else {
      jj = jj_live + 1;
    }
  }
  while (kk < elim_stop) {
    std::int32_t kk_live = kk;
    const std::int32_t elim_row_k =
        _doubleton_eq_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, elim_col, &kk_live);
    if (elim_row_k == INT_MAX) {
      break;
    }
    ++fill_in;
    kk = kk_live + 1;
  }
  return fill_in;
}

__global__ void _kernel_doubleton_eq_candidates(std::uint8_t* candidate_mask,
                                                std::int32_t* candidate_elim_col,
                                                double* candidate_elim_val,
                                                std::int32_t* candidate_keep_col,
                                                double* candidate_keep_val,
                                                const std::int32_t* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* keep_col_mask,
                                                const std::int32_t* col_nnz,
                                                const double* AL,
                                                const double* AU,
                                                double zero_tol,
                                                double tol,
                                                double ratio_tol,
                                                std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= m || keep_row[i] == std::uint8_t{0}) {
    return;
  }
  const double lhs = AL[i];
  const double rhs = AU[i];
  if (!isfinite(lhs) || !isfinite(rhs) || fabs(lhs - rhs) > tol) {
    return;
  }

  std::int32_t live_count = 0;
  std::int32_t col1 = -1;
  std::int32_t col2 = -1;
  double val1 = 0.0;
  double val2 = 0.0;
  for (std::int32_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (keep_col_mask[col] != std::uint8_t{0} && fabs(val) > zero_tol) {
      ++live_count;
      if (live_count == 1) {
        col1 = col;
        val1 = val;
      } else if (live_count == 2) {
        col2 = col;
        val2 = val;
      } else {
        return;
      }
    }
  }
  if (live_count != 2) {
    return;
  }

  std::int32_t elim_col = -1;
  std::int32_t keep_col = -1;
  double elim_val = 0.0;
  double keep_val = 0.0;
  _choose_doubleton_eq_columns(col1, val1, col2, val2, col_nnz, ratio_tol,
                               &elim_col, &elim_val, &keep_col, &keep_val);
  if (!_is_acceptable_doubleton_eq_pivot(keep_val, elim_val)) {
    return;
  }
  candidate_mask[i] = std::uint8_t{1};
  candidate_elim_col[i] = elim_col;
  candidate_elim_val[i] = elim_val;
  candidate_keep_col[i] = keep_col;
  candidate_keep_val[i] = keep_val;
}

__global__ void _kernel_mark_doubleton_eq_batch_acceptable(std::uint8_t* acceptable_mask,
                                                           const std::uint8_t* candidate_mask,
                                                           const std::int32_t* candidate_keep_col,
                                                           const std::int32_t* candidate_elim_col,
                                                           const std::int32_t* col_nnz,
                                                           const std::int32_t* AT_row_ptr,
                                                           const std::int32_t* AT_col_val,
                                                           const double* AT_nz_val,
                                                           const std::uint8_t* keep_row,
                                                           double zero_tol,
                                                           std::int32_t max_fill_in_proxy,
                                                           std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && candidate_mask[i] != std::uint8_t{0}) {
    const std::int32_t keep_col = candidate_keep_col[i];
    const std::int32_t elim_col = candidate_elim_col[i];
    const std::int32_t keep_nnz = col_nnz[keep_col];
    const std::int32_t elim_nnz = col_nnz[elim_col];
    if (elim_nnz > keep_nnz + max_fill_in_proxy + 1) {
      return;
    }
    const std::int32_t fill_in = _doubleton_eq_fill_in_proxy_device_live(
        AT_row_ptr, AT_col_val, AT_nz_val, keep_row, zero_tol, keep_col, elim_col);
    if (fill_in <= max_fill_in_proxy) {
      acceptable_mask[i] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_claim_doubleton_eq_batch_columns(std::int32_t* col_owner,
                                                         const std::uint8_t* active_mask,
                                                         const std::uint8_t* blocked_col,
                                                         const std::int32_t* candidate_keep_col,
                                                         const std::int32_t* candidate_elim_col,
                                                         std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && active_mask[i] != std::uint8_t{0}) {
    const std::int32_t keep_col = candidate_keep_col[i];
    const std::int32_t elim_col = candidate_elim_col[i];
    if (blocked_col[keep_col] == std::uint8_t{0} && blocked_col[elim_col] == std::uint8_t{0}) {
      atomicMin(&col_owner[keep_col], i);
      atomicMin(&col_owner[elim_col], i);
    }
  }
}

__global__ void _kernel_select_doubleton_eq_batch_rows(std::uint8_t* selected_mask,
                                                       std::uint8_t* active_mask,
                                                       std::uint8_t* blocked_col,
                                                       const std::int32_t* candidate_keep_col,
                                                       const std::int32_t* candidate_elim_col,
                                                       const std::int32_t* col_owner,
                                                       std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && active_mask[i] != std::uint8_t{0}) {
    const std::int32_t keep_col = candidate_keep_col[i];
    const std::int32_t elim_col = candidate_elim_col[i];
    if (blocked_col[keep_col] == std::uint8_t{0} &&
        blocked_col[elim_col] == std::uint8_t{0} &&
        col_owner[keep_col] == i &&
        col_owner[elim_col] == i) {
      selected_mask[i] = std::uint8_t{1};
      active_mask[i] = std::uint8_t{0};
      blocked_col[keep_col] = std::uint8_t{1};
      blocked_col[elim_col] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_capture_doubleton_eq_batch_tape_before(
    std::int32_t* tape_indices,
    double* tape_vals,
    const std::int32_t* selected_rows,
    const std::int32_t* candidate_keep_col,
    const std::int32_t* candidate_elim_col,
    const double* candidate_keep_val,
    const double* candidate_elim_val,
    const double* old_l,
    const double* old_u,
    const double* old_c,
    const double* old_AU,
    std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const std::int32_t elim = candidate_elim_col[row];
  std::int32_t* indices = tape_indices + DOUBLETONEQ_TAPE_INDEX_STRIDE * t;
  double* vals = tape_vals + DOUBLETONEQ_TAPE_VALUE_STRIDE * t;
  indices[0] = elim;
  indices[1] = keep;
  indices[2] = row;
  vals[0] = candidate_elim_val[row];
  vals[1] = candidate_keep_val[row];
  vals[2] = old_AU[row];
  vals[3] = old_l[elim];
  vals[4] = old_u[elim];
  vals[5] = old_l[keep];
  vals[6] = old_u[keep];
  vals[9] = old_c[elim];
  vals[10] = 0.0;
  vals[11] = 0.0;
}

__global__ void _kernel_capture_doubleton_eq_batch_tape_after(
    double* tape_vals,
    const std::int32_t* tape_indices,
    const double* new_l,
    const double* new_u,
    std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t keep =
      tape_indices[DOUBLETONEQ_TAPE_INDEX_STRIDE * t + 1];
  double* vals = tape_vals + DOUBLETONEQ_TAPE_VALUE_STRIDE * t;
  vals[7] = new_l[keep];
  vals[8] = new_u[keep];
}

__global__ void _kernel_apply_doubleton_eq_batch(std::int32_t* status,
                                                 double* l,
                                                 double* u,
                                                 double* c,
                                                 double* AL,
                                                 double* AU,
                                                 std::uint8_t* keep_row_mask,
                                                 std::uint8_t* keep_col_mask,
                                                 std::uint8_t* subst_pair_mask,
                                                 std::uint8_t* fixed_pair_mask,
                                                 double* row_shift,
                                                 double* obj_delta,
                                                 double* fixed_obj_delta,
                                                 const std::int32_t* selected_rows,
                                                 const std::int32_t* candidate_keep_col,
                                                 const std::int32_t* candidate_elim_col,
                                                 const double* candidate_keep_val,
                                                 const double* candidate_elim_val,
                                                 const std::int32_t* AT_row_ptr,
                                                 const std::int32_t* AT_col_val,
                                                 const double* AT_nz_val,
                                                 const double* old_l,
                                                 const double* old_u,
                                                 const double* old_c,
                                                 const double* AU_old,
                                                 double bound_tol,
                                                 std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const std::int32_t elim = candidate_elim_col[row];
  const double keep_val = candidate_keep_val[row];
  const double elim_val = candidate_elim_val[row];
  const double rhs = AU_old[row];
  const double alpha = -keep_val / elim_val;
  const double beta = rhs / elim_val;

  double mapped_l = -INFINITY;
  double mapped_u = INFINITY;
  if (isfinite(old_l[elim])) {
    const double bound_val = (old_l[elim] - beta) / alpha;
    if (alpha > 0.0) {
      mapped_l = bound_val;
    } else {
      mapped_u = bound_val;
    }
  }
  if (isfinite(old_u[elim])) {
    const double bound_val = (old_u[elim] - beta) / alpha;
    if (alpha > 0.0) {
      mapped_u = bound_val;
    } else {
      mapped_l = bound_val;
    }
  }
  if (mapped_l > mapped_u) {
    const double tmp = mapped_l;
    mapped_l = mapped_u;
    mapped_u = tmp;
  }
  const double keep_l_new = isfinite(mapped_l) ? fmax(old_l[keep], mapped_l) : old_l[keep];
  const double keep_u_new = isfinite(mapped_u) ? fmin(old_u[keep], mapped_u) : old_u[keep];
  if (keep_l_new > keep_u_new + bound_tol) {
    status[0] = 1;
    return;
  }

  const bool keep_fixed = isfinite(keep_l_new) && isfinite(keep_u_new) && keep_u_new <= keep_l_new + bound_tol;
  if (keep_fixed) {
    const double fixed_at = 0.5 * (keep_l_new + keep_u_new);
    l[keep] = fixed_at;
    u[keep] = fixed_at;
    keep_col_mask[keep] = std::uint8_t{0};
    fixed_pair_mask[t] = std::uint8_t{1};
    atomicAdd(fixed_obj_delta, old_c[keep] * fixed_at);
  } else {
    l[keep] = keep_l_new;
    u[keep] = keep_u_new;
    c[keep] += old_c[elim] * alpha;
    keep_row_mask[row] = std::uint8_t{0};
    keep_col_mask[elim] = std::uint8_t{0};
    subst_pair_mask[t] = std::uint8_t{1};
    atomicAdd(obj_delta, old_c[elim] * beta);
  }
  status[1] = 1;
}

__global__ void _kernel_apply_row_shift(double* AL, double* AU, const double* row_shift, std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    AL[i] -= row_shift[i];
    AU[i] -= row_shift[i];
  }
}

__global__ void _kernel_doubleton_eq_batch_entry_counts(std::int32_t* entry_counts,
                                                        const std::int32_t* selected_rows,
                                                        const std::int32_t* candidate_elim_col,
                                                        const std::int32_t* AT_row_ptr,
                                                        const std::int32_t* AT_col_val,
                                                        const double* AT_nz_val,
                                                        const std::uint8_t* keep_row_mask,
                                                        double zero_tol,
                                                        std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t elim = candidate_elim_col[row];
  std::int32_t count = 0;
  for (std::int32_t p = AT_row_ptr[elim]; p < AT_row_ptr[elim + 1]; ++p) {
    const std::int32_t support_row = AT_col_val[p];
    const double a = AT_nz_val[p];
    if (support_row != row && keep_row_mask[support_row] != std::uint8_t{0} && fabs(a) > zero_tol) {
      ++count;
    }
  }
  entry_counts[t] = count;
}

__global__ void _kernel_scatter_selected_rows_by_pair_mask(std::int32_t* out_rows,
                                                           const std::int32_t* selected_rows,
                                                           const std::uint8_t* pair_mask,
                                                           const std::int32_t* prefix,
                                                           std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < k && pair_mask[t] != std::uint8_t{0}) {
    out_rows[prefix[t] - 1] = selected_rows[t];
  }
}

__global__ void _kernel_accumulate_doubleton_eq_fixed_row_shift(double* row_shift,
                                                                const std::int32_t* selected_rows,
                                                                const std::uint8_t* fixed_pair_mask,
                                                                const std::int32_t* candidate_keep_col,
                                                                const double* fixed_values_by_col,
                                                                const std::int32_t* AT_row_ptr,
                                                                const std::int32_t* AT_col_val,
                                                                const double* AT_nz_val,
                                                                const std::uint8_t* keep_row_mask,
                                                                double zero_tol,
                                                                std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= k || fixed_pair_mask[t] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const double fixed_at = fixed_values_by_col[keep];
  for (std::int32_t p = AT_row_ptr[keep]; p < AT_row_ptr[keep + 1]; ++p) {
    const std::int32_t support_row = AT_col_val[p];
    const double a = AT_nz_val[p];
    if (keep_row_mask[support_row] != std::uint8_t{0} && fabs(a) > zero_tol) {
      atomicAdd(&row_shift[support_row], a * fixed_at);
    }
  }
}

__global__ void _kernel_doubleton_eq_batch_build_delta(long long* delta_keys,
                                                       double* delta_vals,
                                                       double* row_shift,
                                                       const std::int32_t* entry_starts,
                                                       const std::int32_t* selected_rows,
                                                       const std::int32_t* candidate_keep_col,
                                                       const std::int32_t* candidate_elim_col,
                                                       const double* candidate_keep_val,
                                                       const double* candidate_elim_val,
                                                       const double* AU_old,
                                                       const std::int32_t* AT_row_ptr,
                                                       const std::int32_t* AT_col_val,
                                                       const double* AT_nz_val,
                                                       const std::uint8_t* keep_row_mask,
                                                       double zero_tol,
                                                       std::int32_t n,
                                                       std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const std::int32_t elim = candidate_elim_col[row];
  const double alpha = -candidate_keep_val[row] / candidate_elim_val[row];
  const double beta = AU_old[row] / candidate_elim_val[row];
  std::int32_t write = entry_starts[t];
  for (std::int32_t p = AT_row_ptr[elim]; p < AT_row_ptr[elim + 1]; ++p) {
    const std::int32_t support_row = AT_col_val[p];
    const double a = AT_nz_val[p];
    if (support_row != row && keep_row_mask[support_row] != std::uint8_t{0} && fabs(a) > zero_tol) {
      delta_keys[write] = static_cast<long long>(support_row) * static_cast<long long>(n) +
                          static_cast<long long>(keep);
      delta_vals[write] = alpha * a;
      atomicAdd(&row_shift[support_row], a * beta);
      ++write;
    }
  }
}

__global__ void _kernel_delta_row_counts(std::int32_t* row_counts,
                                         const long long* delta_keys,
                                         std::int32_t unique_count,
                                         std::int32_t n) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < unique_count) {
    const std::int32_t row = static_cast<std::int32_t>(delta_keys[t] / static_cast<long long>(n));
    atomicAdd(&row_counts[row], 1);
  }
}

__global__ void _kernel_scatter_delta_by_row(long long* delta_keys_by_row,
                                             double* delta_vals_by_row,
                                             std::int32_t* row_write_ptr,
                                             const long long* delta_keys,
                                             const double* delta_vals,
                                             std::int32_t nnz,
                                             std::int32_t n) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < nnz) {
    const long long key = delta_keys[t];
    const std::int32_t row = static_cast<std::int32_t>(key / static_cast<long long>(n));
    const std::int32_t slot = atomicAdd(&row_write_ptr[row], 1);
    delta_keys_by_row[slot] = key;
    delta_vals_by_row[slot] = delta_vals[t];
  }
}

__global__ void _kernel_sort_delta_rows_by_col(long long* delta_keys,
                                               double* delta_vals,
                                               const std::int32_t* delta_row_ptr,
                                               std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const std::int32_t first = delta_row_ptr[row];
  const std::int32_t last = delta_row_ptr[row + 1];
  for (std::int32_t i = first + 1; i < last; ++i) {
    const long long key = delta_keys[i];
    const double val = delta_vals[i];
    std::int32_t j = i - 1;
    while (j >= first && delta_keys[j] > key) {
      delta_keys[j + 1] = delta_keys[j];
      delta_vals[j + 1] = delta_vals[j];
      --j;
    }
    delta_keys[j + 1] = key;
    delta_vals[j + 1] = val;
  }
}

__global__ void _kernel_doubleton_eq_merged_row_counts(std::int32_t* row_nnz_new,
                                                       const std::int32_t* row_ptr,
                                                       const std::int32_t* col_val,
                                                       const double* nz_val,
                                                       const std::int32_t* delta_row_ptr,
                                                       const long long* delta_keys,
                                                       const double* delta_vals,
                                                       double zero_tol,
                                                       std::int32_t m,
                                                       std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const std::int32_t p_start = row_ptr[row];
  const std::int32_t p_stop = row_ptr[row + 1];
  const std::int32_t q_start = delta_row_ptr[row];
  const std::int32_t q_stop = delta_row_ptr[row + 1];
  std::int32_t count = 0;
  std::int32_t p = p_start;
  std::int32_t q = q_start;
  while (p < p_stop || q < q_stop) {
    const std::int32_t source_col = (p < p_stop) ? col_val[p] : INT_MAX;
    const std::int32_t delta_col =
        (q < q_stop) ? static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) : INT_MAX;
    const std::int32_t next_col = source_col < delta_col ? source_col : delta_col;
    double merged = 0.0;
    while (p < p_stop && col_val[p] == next_col) {
      merged += nz_val[p];
      ++p;
    }
    while (q < q_stop &&
           static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) == next_col) {
      merged += delta_vals[q];
      ++q;
    }
    if (fabs(merged) > zero_tol) {
      ++count;
    }
  }
  row_nnz_new[row] = count;
}

__global__ void _kernel_doubleton_eq_copy_merged_rows(std::int32_t* col_val_new,
                                                      double* nz_val_new,
                                                      const std::int32_t* row_ptr_new,
                                                      const std::int32_t* row_ptr,
                                                      const std::int32_t* col_val,
                                                      const double* nz_val,
                                                      const std::int32_t* delta_row_ptr,
                                                      const long long* delta_keys,
                                                      const double* delta_vals,
                                                      double zero_tol,
                                                      std::int32_t m,
                                                      std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const std::int32_t p_start = row_ptr[row];
  const std::int32_t p_stop = row_ptr[row + 1];
  const std::int32_t q_start = delta_row_ptr[row];
  const std::int32_t q_stop = delta_row_ptr[row + 1];
  std::int32_t write = row_ptr_new[row];
  std::int32_t p = p_start;
  std::int32_t q = q_start;
  while (p < p_stop || q < q_stop) {
    const std::int32_t source_col = (p < p_stop) ? col_val[p] : INT_MAX;
    const std::int32_t delta_col =
        (q < q_stop) ? static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) : INT_MAX;
    const std::int32_t next_col = source_col < delta_col ? source_col : delta_col;
    double merged = 0.0;
    while (p < p_stop && col_val[p] == next_col) {
      merged += nz_val[p];
      ++p;
    }
    while (q < q_stop &&
           static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) == next_col) {
      merged += delta_vals[q];
      ++q;
    }
    if (fabs(merged) > zero_tol) {
      col_val_new[write] = next_col;
      nz_val_new[write] = merged;
      ++write;
    }
  }
}

DeviceCsrMatrix build_doubleton_batch_new_A(const DeviceCsrMatrix& source,
                                            const DeviceCsrMatrix& AT_source,
                                            const PresolvePlanGpu& plan,
                                            const std::int32_t* selected_rows,
                                            const std::int32_t* candidate_keep_col,
                                            const std::int32_t* candidate_elim_col,
                                            const double* candidate_keep_val,
                                            const double* candidate_elim_val,
                                            double* row_shift,
                                            double zero_tol,
                                            std::int32_t selected_count,
                                            bool* matrix_changed) {
  *matrix_changed = false;
  DeviceCsrMatrix out;
  if (selected_count <= 0) {
    return out;
  }
  const int blocks_selected = (selected_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int blocks_rows = (source.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* entry_counts = nullptr;
  std::int32_t* entry_prefix = nullptr;
  std::int32_t* entry_starts = nullptr;
  long long* delta_keys = nullptr;
  double* delta_vals = nullptr;
  long long* delta_keys_by_row = nullptr;
  double* delta_vals_by_row = nullptr;
  std::int32_t* delta_row_counts = nullptr;
  std::int32_t* delta_row_ptr = nullptr;
  std::int32_t* row_write_ptr = nullptr;
  std::int32_t* row_counts_new = nullptr;

  throw_if_cuda_error(doubleton_cuda_malloc(&entry_counts, sizeof(std::int32_t) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc doubleton batch entry_counts");
  throw_if_cuda_error(doubleton_cuda_malloc(&entry_prefix, sizeof(std::int32_t) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc doubleton batch entry_prefix");
  _kernel_doubleton_eq_batch_entry_counts<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
      entry_counts, selected_rows, candidate_elim_col, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.keep_row_mask, zero_tol, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_batch_entry_counts");
  throw_if_cuda_error(cudaMemcpy(entry_prefix, entry_counts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(selected_count),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy doubleton batch entry_prefix");
  inclusive_scan_i32(entry_prefix, selected_count, "cub inclusive scan doubleton batch entries");
  std::int32_t total_entries = 0;
  throw_if_cuda_error(cudaMemcpy(&total_entries, entry_prefix + selected_count - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton batch total_entries");
  if (total_entries <= 0) {
    doubleton_cuda_free(entry_counts);
    doubleton_cuda_free(entry_prefix);
    return out;
  }

  throw_if_cuda_error(doubleton_cuda_malloc(&entry_starts, sizeof(std::int32_t) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc doubleton batch entry_starts");
  _kernel_row_ptr_from_prefix<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(entry_starts, entry_prefix, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix batch entry starts");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_keys, sizeof(long long) * static_cast<std::size_t>(total_entries)),
                      "cudaMalloc doubleton batch delta_keys");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_vals, sizeof(double) * static_cast<std::size_t>(total_entries)),
                      "cudaMalloc doubleton batch delta_vals");
  _kernel_doubleton_eq_batch_build_delta<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
      delta_keys, delta_vals, row_shift, entry_starts, selected_rows, candidate_keep_col, candidate_elim_col,
      candidate_keep_val, candidate_elim_val, plan.new_AU, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.keep_row_mask, zero_tol, source.cols, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_batch_build_delta");
  throw_if_cuda_error(cudaDeviceSynchronize(), "doubleton batch build delta synchronize");

  throw_if_cuda_error(doubleton_cuda_malloc(&delta_row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton batch delta_row_counts");
  throw_if_cuda_error(cudaMemset(delta_row_counts, 0, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows)),
                      "cudaMemset doubleton batch delta_row_counts");
  const int blocks_delta = (total_entries + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_delta_row_counts<<<blocks_delta, GPU_PRESOLVE_THREADS>>>(
      delta_row_counts, delta_keys, total_entries, source.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_delta_row_counts");
  inclusive_scan_i32(delta_row_counts, source.rows, "cub inclusive scan doubleton delta rows");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_row_ptr, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows + 1)),
                      "cudaMalloc doubleton batch delta_row_ptr");
  _kernel_row_ptr_from_prefix<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(delta_row_ptr, delta_row_counts, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix delta rows");
  throw_if_cuda_error(doubleton_cuda_malloc(&row_write_ptr, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton batch row_write_ptr");
  throw_if_cuda_error(cudaMemcpy(row_write_ptr, delta_row_ptr,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(source.rows),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy doubleton batch row_write_ptr");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_keys_by_row, sizeof(long long) * static_cast<std::size_t>(total_entries)),
                      "cudaMalloc doubleton batch delta_keys_by_row");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_vals_by_row, sizeof(double) * static_cast<std::size_t>(total_entries)),
                      "cudaMalloc doubleton batch delta_vals_by_row");
  _kernel_scatter_delta_by_row<<<blocks_delta, GPU_PRESOLVE_THREADS>>>(
      delta_keys_by_row, delta_vals_by_row, row_write_ptr, delta_keys, delta_vals, total_entries, source.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_delta_by_row");
  _kernel_sort_delta_rows_by_col<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      delta_keys_by_row, delta_vals_by_row, delta_row_ptr, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_sort_delta_rows_by_col");

  throw_if_cuda_error(doubleton_cuda_malloc(&row_counts_new, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton batch row_counts_new");
  _kernel_doubleton_eq_merged_row_counts<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      row_counts_new, source.rowPtr, source.colVal, source.nzVal, delta_row_ptr, delta_keys_by_row, delta_vals_by_row,
      zero_tol, source.rows, source.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_merged_row_counts");
  inclusive_scan_i32(row_counts_new, source.rows, "cub inclusive scan doubleton merged rows");
  std::int32_t nnz_new = 0;
  throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts_new + source.rows - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton batch nnz_new");

  out.rows = source.rows;
  out.cols = source.cols;
  out.nnz = nnz_new;
  throw_if_cuda_error(doubleton_cuda_malloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows + 1)),
                      "cudaMalloc doubleton batch rowPtr_new");
  _kernel_row_ptr_from_prefix<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts_new, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix merged rows");
  if (nnz_new > 0) {
    throw_if_cuda_error(doubleton_cuda_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton batch colVal_new");
    throw_if_cuda_error(doubleton_cuda_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton batch nzVal_new");
    _kernel_doubleton_eq_copy_merged_rows<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal, delta_row_ptr,
        delta_keys_by_row, delta_vals_by_row, zero_tol, source.rows, source.cols);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_copy_merged_rows");
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "doubleton batch merged A synchronize");
  *matrix_changed = true;

  doubleton_cuda_free(entry_counts); doubleton_cuda_free(entry_prefix); doubleton_cuda_free(entry_starts); doubleton_cuda_free(delta_keys); doubleton_cuda_free(delta_vals);
  doubleton_cuda_free(delta_keys_by_row); doubleton_cuda_free(delta_vals_by_row); doubleton_cuda_free(delta_row_counts); doubleton_cuda_free(delta_row_ptr);
  doubleton_cuda_free(row_write_ptr); doubleton_cuda_free(row_counts_new);
  return out;
}

std::int32_t count_mask(std::uint8_t* mask, std::int32_t n, std::int32_t* prefix) {
  if (n == 0) {
    return 0;
  }
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_u8_to_i32<<<blocks, GPU_PRESOLVE_THREADS>>>(prefix, mask, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_u8_to_i32");
  inclusive_scan_i32(prefix, n, "cub inclusive scan doubleton mask count");
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, prefix + n - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy mask count");
  return count;
}

DeviceCsrMatrix transpose_csr_local(const DeviceCsrMatrix& A) {
  DeviceCsrMatrix AT;
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.nnz = A.nnz;
  cusparseHandle_t handle = nullptr;
  void* buffer = nullptr;
  try {
    throw_if_cuda_error(
        doubleton_cuda_malloc(&AT.rowPtr,
                   sizeof(std::int32_t) * static_cast<std::size_t>(AT.rows + 1)),
        "cudaMalloc doubleton transpose rowPtr");
    if (A.nnz == 0) {
      throw_if_cuda_error(
          cudaMemset(AT.rowPtr, 0,
                     sizeof(std::int32_t) * static_cast<std::size_t>(AT.rows + 1)),
          "cudaMemset doubleton empty transpose rowPtr");
      return AT;
    }
    throw_if_cuda_error(
        doubleton_cuda_malloc(&AT.colVal,
                   sizeof(std::int32_t) * static_cast<std::size_t>(A.nnz)),
        "cudaMalloc doubleton transpose colVal");
    throw_if_cuda_error(
        doubleton_cuda_malloc(&AT.nzVal, sizeof(double) * static_cast<std::size_t>(A.nnz)),
        "cudaMalloc doubleton transpose nzVal");
    throw_if_cusparse_error(cusparseCreate(&handle),
                            "cusparseCreate doubleton transpose");
    std::size_t buffer_size = 0;
    constexpr cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
    throw_if_cusparse_error(
        cusparseCsr2cscEx2_bufferSize(
            handle, A.rows, A.cols, A.nnz, A.nzVal, A.rowPtr, A.colVal,
            AT.nzVal, AT.rowPtr, AT.colVal, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO, alg, &buffer_size),
        "cusparseCsr2cscEx2_bufferSize doubleton transpose");
    if (buffer_size > 0) {
      throw_if_cuda_error(doubleton_cuda_malloc(&buffer, buffer_size),
                          "cudaMalloc doubleton transpose buffer");
    }
    throw_if_cusparse_error(
        cusparseCsr2cscEx2(
            handle, A.rows, A.cols, A.nnz, A.nzVal, A.rowPtr, A.colVal,
            AT.nzVal, AT.rowPtr, AT.colVal, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO, alg, buffer),
        "cusparseCsr2cscEx2 doubleton transpose");
    doubleton_cuda_free(buffer);
    buffer = nullptr;
    throw_if_cusparse_error(cusparseDestroy(handle),
                            "cusparseDestroy doubleton transpose");
    handle = nullptr;
  } catch (...) {
    doubleton_cuda_free(buffer);
    if (handle != nullptr) {
      cusparseDestroy(handle);
    }
    doubleton_cuda_free(AT.rowPtr);
    doubleton_cuda_free(AT.colVal);
    doubleton_cuda_free(AT.nzVal);
    throw;
  }
  return AT;
}

void free_local_csr(DeviceCsrMatrix& A) {
  doubleton_cuda_free(A.rowPtr);
  doubleton_cuda_free(A.colVal);
  doubleton_cuda_free(A.nzVal);
  A = DeviceCsrMatrix{};
}

bool apply_rule_doubleton_eq_batch_once(PresolvePlanGpu& plan,
                                        const LPInfoGpu& lp,
                                        const DeviceCsrMatrix& AT_source,
                                        const PresolveParams& pparams) {
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return false;
  }
  const DeviceCsrMatrix& A_source = plan.has_new_A ? plan.new_A : lp.A;
  const int blocks_rows = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int blocks_cols = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const double ratio_tol = fmax(1.0e-9, pparams.bound_tol);
  const bool profile_batch = env_enabled("GPUPRESOLVER_DOUBLETON_BATCH_PROFILE");

  std::int32_t* live_col_nnz = nullptr;
  std::uint8_t* candidate_mask = nullptr;
  std::uint8_t* acceptable_mask = nullptr;
  std::uint8_t* active_mask = nullptr;
  std::uint8_t* selected_mask = nullptr;
  std::uint8_t* blocked_col = nullptr;
  std::int32_t* candidate_elim_col = nullptr;
  std::int32_t* candidate_keep_col = nullptr;
  double* candidate_elim_val = nullptr;
  double* candidate_keep_val = nullptr;
  std::int32_t* col_owner = nullptr;
  std::int32_t* prefix = nullptr;
  std::int32_t* selected_rows = nullptr;
  std::int32_t* subst_rows = nullptr;
  std::int32_t* status = nullptr;
  std::uint8_t* subst_pair_mask = nullptr;
  std::uint8_t* fixed_pair_mask = nullptr;
  double* row_shift = nullptr;
  double* obj_delta_device = nullptr;
  double* fixed_obj_delta_device = nullptr;

  throw_if_cuda_error(doubleton_cuda_malloc(&live_col_nnz, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc batch live_col_nnz");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_mask, static_cast<std::size_t>(m)), "cudaMalloc candidate_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&acceptable_mask, static_cast<std::size_t>(m)), "cudaMalloc acceptable_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&active_mask, static_cast<std::size_t>(m)), "cudaMalloc active_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&selected_mask, static_cast<std::size_t>(m)), "cudaMalloc selected_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&blocked_col, static_cast<std::size_t>(n)), "cudaMalloc blocked_col");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_elim_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_elim_col");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_keep_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_keep_col");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_elim_val, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_elim_val");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_keep_val, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_keep_val");
  throw_if_cuda_error(doubleton_cuda_malloc(&col_owner, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc col_owner");
  throw_if_cuda_error(doubleton_cuda_malloc(&prefix, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc batch prefix");
  throw_if_cuda_error(doubleton_cuda_malloc(&status, sizeof(std::int32_t) * 2), "cudaMalloc batch status");
  throw_if_cuda_error(doubleton_cuda_malloc(&subst_pair_mask, static_cast<std::size_t>(m)), "cudaMalloc subst_pair_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&fixed_pair_mask, static_cast<std::size_t>(m)), "cudaMalloc fixed_pair_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&row_shift, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc batch row_shift");
  throw_if_cuda_error(doubleton_cuda_malloc(&obj_delta_device, sizeof(double)), "cudaMalloc batch obj_delta");
  throw_if_cuda_error(doubleton_cuda_malloc(&fixed_obj_delta_device, sizeof(double)), "cudaMalloc batch fixed_obj_delta");
  throw_if_cuda_error(cudaMemset(candidate_mask, 0, static_cast<std::size_t>(m)), "cudaMemset candidate_mask");
  throw_if_cuda_error(cudaMemset(acceptable_mask, 0, static_cast<std::size_t>(m)), "cudaMemset acceptable_mask");
  throw_if_cuda_error(cudaMemset(selected_mask, 0, static_cast<std::size_t>(m)), "cudaMemset selected_mask");
  throw_if_cuda_error(cudaMemset(blocked_col, 0, static_cast<std::size_t>(n)), "cudaMemset blocked_col");
  throw_if_cuda_error(cudaMemset(status, 0, sizeof(std::int32_t) * 2), "cudaMemset batch status");
  throw_if_cuda_error(cudaMemset(subst_pair_mask, 0, static_cast<std::size_t>(m)), "cudaMemset subst_pair_mask");
  throw_if_cuda_error(cudaMemset(fixed_pair_mask, 0, static_cast<std::size_t>(m)), "cudaMemset fixed_pair_mask");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset row_shift");
  throw_if_cuda_error(cudaMemset(obj_delta_device, 0, sizeof(double)), "cudaMemset batch obj_delta");
  throw_if_cuda_error(cudaMemset(fixed_obj_delta_device, 0, sizeof(double)), "cudaMemset batch fixed_obj_delta");

  _kernel_doubleton_eq_live_col_nnz<<<blocks_cols, GPU_PRESOLVE_THREADS>>>(
      live_col_nnz, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.keep_row_mask, plan.keep_col_mask, pparams.zero_tol, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_live_col_nnz batch");

  _kernel_doubleton_eq_candidates<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      candidate_mask, candidate_elim_col, candidate_elim_val, candidate_keep_col, candidate_keep_val,
      A_source.rowPtr, A_source.colVal, A_source.nzVal, plan.keep_row_mask, plan.keep_col_mask,
      live_col_nnz, plan.new_AL, plan.new_AU, pparams.zero_tol, pparams.bound_tol, ratio_tol, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_candidates");

  _kernel_mark_doubleton_eq_batch_acceptable<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      acceptable_mask, candidate_mask, candidate_keep_col, candidate_elim_col, live_col_nnz,
      AT_source.rowPtr, AT_source.colVal, AT_source.nzVal, plan.keep_row_mask, pparams.zero_tol,
      static_cast<std::int32_t>(pparams.doubleton_eq_max_fill_in_proxy), m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_doubleton_eq_batch_acceptable");

  const std::int32_t acceptable_count = count_mask(acceptable_mask, m, prefix);
  if (acceptable_count == 0) {
    doubleton_cuda_free(live_col_nnz); doubleton_cuda_free(candidate_mask); doubleton_cuda_free(acceptable_mask); doubleton_cuda_free(active_mask);
    doubleton_cuda_free(selected_mask); doubleton_cuda_free(blocked_col); doubleton_cuda_free(candidate_elim_col); doubleton_cuda_free(candidate_keep_col);
    doubleton_cuda_free(candidate_elim_val); doubleton_cuda_free(candidate_keep_val); doubleton_cuda_free(col_owner); doubleton_cuda_free(prefix);
    doubleton_cuda_free(status); doubleton_cuda_free(subst_pair_mask); doubleton_cuda_free(fixed_pair_mask);
    doubleton_cuda_free(row_shift); doubleton_cuda_free(obj_delta_device); doubleton_cuda_free(fixed_obj_delta_device);
    return false;
  }

  throw_if_cuda_error(cudaMemcpy(active_mask, acceptable_mask, static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy active_mask");
  for (int round = 0; round < DOUBLETONEQ_BATCH_MATCHING_ROUNDS; ++round) {
    _kernel_fill_i32<<<blocks_cols, GPU_PRESOLVE_THREADS>>>(col_owner, INT_MAX, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 col_owner");
    _kernel_claim_doubleton_eq_batch_columns<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
        col_owner, active_mask, blocked_col, candidate_keep_col, candidate_elim_col, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_claim_doubleton_eq_batch_columns");
    _kernel_select_doubleton_eq_batch_rows<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
        selected_mask, active_mask, blocked_col, candidate_keep_col, candidate_elim_col, col_owner, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_select_doubleton_eq_batch_rows");
  }

  const std::int32_t selected_count = count_mask(selected_mask, m, prefix);
  const double selected_ratio =
      acceptable_count == 0 ? 0.0 : static_cast<double>(selected_count) / static_cast<double>(acceptable_count);
  if (profile_batch) {
    std::cout << ">>> [doubleton_eq batch C++] acceptable=" << acceptable_count
              << " selected=" << selected_count
              << " ratio=" << selected_ratio << "\n";
  }
  if (selected_count == 0 ||
      (pparams.doubleton_eq_scan &&
       (selected_count < pparams.doubleton_eq_min_selected_per_batch ||
        selected_ratio < pparams.doubleton_eq_min_selected_ratio))) {
    doubleton_cuda_free(live_col_nnz); doubleton_cuda_free(candidate_mask); doubleton_cuda_free(acceptable_mask); doubleton_cuda_free(active_mask);
    doubleton_cuda_free(selected_mask); doubleton_cuda_free(blocked_col); doubleton_cuda_free(candidate_elim_col); doubleton_cuda_free(candidate_keep_col);
    doubleton_cuda_free(candidate_elim_val); doubleton_cuda_free(candidate_keep_val); doubleton_cuda_free(col_owner); doubleton_cuda_free(prefix);
    doubleton_cuda_free(status); doubleton_cuda_free(subst_pair_mask); doubleton_cuda_free(fixed_pair_mask);
    doubleton_cuda_free(row_shift); doubleton_cuda_free(obj_delta_device); doubleton_cuda_free(fixed_obj_delta_device);
    return false;
  }

  throw_if_cuda_error(doubleton_cuda_malloc(&selected_rows, sizeof(std::int32_t) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc selected_rows");
  _kernel_scatter_mask_indices<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(selected_rows, selected_mask, prefix, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_mask_indices selected");

  const int blocks_selected = (selected_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* tape_indices_device = nullptr;
  double* tape_vals_device = nullptr;
  if (pparams.record_postsolve_tape) {
    throw_if_cuda_error(
        doubleton_cuda_malloc(&tape_indices_device,
                   sizeof(std::int32_t) * DOUBLETONEQ_TAPE_INDEX_STRIDE *
                       static_cast<std::size_t>(selected_count)),
        "cudaMalloc doubleton batch packed tape indices");
    throw_if_cuda_error(
        doubleton_cuda_malloc(&tape_vals_device,
                   sizeof(double) * DOUBLETONEQ_TAPE_VALUE_STRIDE *
                       static_cast<std::size_t>(selected_count)),
        "cudaMalloc doubleton batch packed tape values");
    _kernel_capture_doubleton_eq_batch_tape_before<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
        tape_indices_device, tape_vals_device, selected_rows, candidate_keep_col,
        candidate_elim_col, candidate_keep_val, candidate_elim_val, plan.new_l,
        plan.new_u, plan.new_c, plan.new_AU, selected_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_capture_doubleton_eq_batch_tape_before");
  }
  _kernel_apply_doubleton_eq_batch<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
      status, plan.new_l, plan.new_u, plan.new_c, plan.new_AL, plan.new_AU, plan.keep_row_mask,
      plan.keep_col_mask, subst_pair_mask, fixed_pair_mask, row_shift, obj_delta_device,
      fixed_obj_delta_device, selected_rows, candidate_keep_col, candidate_elim_col,
      candidate_keep_val, candidate_elim_val, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.new_l, plan.new_u, plan.new_c, plan.new_AU, pparams.bound_tol, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_doubleton_eq_batch");
  throw_if_cuda_error(cudaDeviceSynchronize(), "doubleton batch synchronize");

  std::int32_t status_host[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(status_host, status, sizeof(status_host), cudaMemcpyDeviceToHost),
                      "cudaMemcpy batch status");
  bool changed = false;
  if (status_host[0] != 0) {
    plan.has_infeasible = true;
  } else if (status_host[1] != 0) {
    const std::int32_t subst_count = count_mask(subst_pair_mask, selected_count, prefix);
    if (subst_count > 0) {
      throw_if_cuda_error(doubleton_cuda_malloc(&subst_rows, sizeof(std::int32_t) * static_cast<std::size_t>(subst_count)),
                          "cudaMalloc subst_rows");
      _kernel_scatter_selected_rows_by_pair_mask<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
          subst_rows, selected_rows, subst_pair_mask, prefix, selected_count);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_selected_rows_by_pair_mask");
      if (pparams.record_postsolve_tape) {
        _kernel_capture_doubleton_eq_batch_tape_after<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
            tape_vals_device, tape_indices_device, plan.new_l, plan.new_u,
            selected_count);
        throw_if_cuda_error(cudaGetLastError(),
                            "_kernel_capture_doubleton_eq_batch_tape_after");
        const std::vector<std::int32_t> tape_indices_host = copy_device_vector(
            tape_indices_device,
            DOUBLETONEQ_TAPE_INDEX_STRIDE * selected_count,
            "cudaMemcpy doubleton batch packed tape indices");
        const std::vector<double> tape_vals_host = copy_device_vector(
            tape_vals_device,
            DOUBLETONEQ_TAPE_VALUE_STRIDE * selected_count,
            "cudaMemcpy doubleton batch packed tape values");
        const std::vector<std::uint8_t> subst_pair_mask_host = copy_device_vector(
            subst_pair_mask, selected_count,
            "cudaMemcpy doubleton batch substitution mask for tape");
        const std::size_t old_records = plan.tape.types.size();
        const std::size_t old_indices = plan.tape.indices.size();
        const std::size_t old_values = plan.tape.vals.size();
        plan.tape.types.resize(old_records + static_cast<std::size_t>(subst_count));
        plan.tape.dual_modes.resize(old_records + static_cast<std::size_t>(subst_count));
        plan.tape.indices.resize(old_indices + 4 * static_cast<std::size_t>(subst_count));
        plan.tape.vals.resize(old_values + DOUBLETONEQ_TAPE_VALUE_STRIDE *
                                              static_cast<std::size_t>(subst_count));
        plan.tape.index_starts.resize(old_records + static_cast<std::size_t>(subst_count) + 1);
        plan.tape.value_starts.resize(old_records + static_cast<std::size_t>(subst_count) + 1);
        std::size_t out_record = 0;
        for (std::int32_t t = 0; t < selected_count; ++t) {
          if (subst_pair_mask_host[static_cast<std::size_t>(t)] ==
              std::uint8_t{0}) {
            continue;
          }
          const std::size_t index_base =
              static_cast<std::size_t>(DOUBLETONEQ_TAPE_INDEX_STRIDE * t);
          const std::size_t value_base =
              static_cast<std::size_t>(DOUBLETONEQ_TAPE_VALUE_STRIDE * t);
          const std::size_t record = old_records + out_record;
          const std::size_t index_out = old_indices + 4 * out_record;
          const std::size_t value_out =
              old_values + DOUBLETONEQ_TAPE_VALUE_STRIDE * out_record;
          plan.tape.types[record] =
              static_cast<std::int32_t>(PostsolveReductionType::DoubletonEq);
          plan.tape.dual_modes[record] =
              static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
          plan.tape.indices[index_out] = tape_indices_host[index_base];
          plan.tape.indices[index_out + 1] = tape_indices_host[index_base + 1];
          plan.tape.indices[index_out + 2] = tape_indices_host[index_base + 2];
          plan.tape.indices[index_out + 3] = 0;
          for (std::size_t j = 0; j < DOUBLETONEQ_TAPE_VALUE_STRIDE; ++j) {
            plan.tape.vals[value_out + j] = tape_vals_host[value_base + j];
          }
          plan.tape.index_starts[record + 1] =
              static_cast<std::int32_t>(index_out + 4);
          plan.tape.value_starts[record + 1] =
              static_cast<std::int32_t>(value_out + DOUBLETONEQ_TAPE_VALUE_STRIDE);
          ++out_record;
        }
        if (out_record != static_cast<std::size_t>(subst_count)) {
          throw std::runtime_error("doubleton batch substitution tape count mismatch");
        }
      }
    }
    const std::int32_t fixed_count = count_mask(fixed_pair_mask, selected_count, prefix);
    if (profile_batch) {
      std::cout << ">>> [doubleton_eq batch C++] subst=" << subst_count
                << " fixed=" << fixed_count << "\n";
    }
    if (fixed_count > 0) {
      _kernel_accumulate_doubleton_eq_fixed_row_shift<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
          row_shift, selected_rows, fixed_pair_mask, candidate_keep_col, plan.new_l, AT_source.rowPtr,
          AT_source.colVal, AT_source.nzVal, plan.keep_row_mask, pparams.zero_tol, selected_count);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_accumulate_doubleton_eq_fixed_row_shift");
    }
    const DeviceCsrMatrix source_A = plan.has_new_A ? plan.new_A : lp.A;
    bool matrix_changed = false;
    if (subst_count > 0) {
      DeviceCsrMatrix rewritten_A = build_doubleton_batch_new_A(
          source_A,
          AT_source,
          plan,
          subst_rows,
          candidate_keep_col,
          candidate_elim_col,
          candidate_keep_val,
          candidate_elim_val,
          row_shift,
          pparams.zero_tol,
          subst_count,
          &matrix_changed);
      if (matrix_changed) {
        if (plan.has_new_A) {
          doubleton_cuda_free(plan.new_A.rowPtr);
          doubleton_cuda_free(plan.new_A.colVal);
          doubleton_cuda_free(plan.new_A.nzVal);
        }
        plan.new_A = rewritten_A;
        plan.has_new_A = true;
      }
    }
    _kernel_apply_row_shift<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(plan.new_AL, plan.new_AU, row_shift, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_row_shift batch");
    throw_if_cuda_error(cudaDeviceSynchronize(), "doubleton batch finalize synchronize");
    double obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&obj_delta, obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy batch obj_delta");
    double fixed_obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&fixed_obj_delta, fixed_obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy batch fixed_obj_delta");
    plan.obj_constant_delta += obj_delta;
    plan.obj_constant_delta += fixed_obj_delta;
    plan.has_row_action = subst_count > 0;
    plan.has_col_action = true;
    plan.has_change = true;
    changed = true;
  }

  doubleton_cuda_free(live_col_nnz); doubleton_cuda_free(candidate_mask); doubleton_cuda_free(acceptable_mask); doubleton_cuda_free(active_mask);
  doubleton_cuda_free(selected_mask); doubleton_cuda_free(blocked_col); doubleton_cuda_free(candidate_elim_col); doubleton_cuda_free(candidate_keep_col);
  doubleton_cuda_free(candidate_elim_val); doubleton_cuda_free(candidate_keep_val); doubleton_cuda_free(col_owner); doubleton_cuda_free(prefix);
  doubleton_cuda_free(selected_rows); doubleton_cuda_free(subst_rows); doubleton_cuda_free(status); doubleton_cuda_free(subst_pair_mask); doubleton_cuda_free(fixed_pair_mask);
  doubleton_cuda_free(tape_indices_device); doubleton_cuda_free(tape_vals_device);
  doubleton_cuda_free(row_shift); doubleton_cuda_free(obj_delta_device); doubleton_cuda_free(fixed_obj_delta_device);
  return changed;
}

bool apply_rule_doubleton_eq_batch(PresolvePlanGpu& plan,
                                   const LPInfoGpu& lp,
                                   const PresolveParams& pparams) {
  bool changed_any = false;
  DeviceCsrMatrix AT_source = lp.AT;
  bool owns_at_source = false;
  const bool scan_enabled = pparams.doubleton_eq_scan;
  const int batch_round_limit =
      scan_enabled && pparams.doubleton_eq_max_batch_rounds > 0 &&
              pparams.doubleton_eq_max_batch_rounds < DOUBLETONEQ_BATCH_INNER_ROUNDS
          ? pparams.doubleton_eq_max_batch_rounds
          : DOUBLETONEQ_BATCH_INNER_ROUNDS;
  const bool has_batch_time_limit =
      scan_enabled && pparams.doubleton_eq_max_time > 0.0 && std::isfinite(pparams.doubleton_eq_max_time);
  const auto scan_start = std::chrono::steady_clock::now();
  const bool profile_batch = env_enabled("GPUPRESOLVER_DOUBLETON_BATCH_PROFILE");

  for (int round = 0; round < batch_round_limit; ++round) {
    if (has_batch_time_limit) {
      const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - scan_start;
      if (elapsed.count() >= pparams.doubleton_eq_max_time) {
        if (profile_batch) {
          std::cout << ">>> [doubleton_eq batch C++] scan stop: max_time\n";
        }
        break;
      }
    }
    const bool changed = apply_rule_doubleton_eq_batch_once(plan, lp, AT_source, pparams);
    if (plan.has_infeasible || plan.has_unbounded) {
      changed_any = changed_any || changed;
      break;
    }
    if (!changed) {
      break;
    }
    changed_any = true;
    if (plan.has_new_A) {
      if (owns_at_source) {
        free_local_csr(AT_source);
      }
      AT_source = transpose_csr_local(plan.new_A);
      owns_at_source = true;
    }
  }
  if (owns_at_source) {
    free_local_csr(AT_source);
  }
  return changed_any;
}

__global__ void _kernel_copy_doubleton_eq_rows(std::int32_t* col_val_new,
                                               double* nz_val_new,
                                               const std::int32_t* row_ptr_new,
                                               const std::int32_t* row_ptr,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               const std::int32_t* meta_i32,
                                               const double* meta_f64,
                                               double zero_tol,
                                               std::int32_t m) {
  const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= m) {
    return;
  }
  const std::int32_t target_row = meta_i32[0];
  const std::int32_t elim_col = meta_i32[1];
  const std::int32_t keep_col = meta_i32[2];
  const double alpha = -meta_f64[1] / meta_f64[0];

  const std::int32_t first = row_ptr[r];
  const std::int32_t last = row_ptr[r + 1];
  std::int32_t write = row_ptr_new[r];
  if (first >= last) {
    return;
  }
  if (r == target_row) {
    for (std::int32_t p = first; p < last; ++p) {
      col_val_new[write] = col_val[p];
      nz_val_new[write] = nz_val[p];
      ++write;
    }
    return;
  }

  bool elim_present = false;
  bool keep_present = false;
  double are = 0.0;
  double old_keep = 0.0;
  for (std::int32_t p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (col == elim_col) {
      elim_present = true;
      are = val;
    } else if (col == keep_col) {
      keep_present = true;
      old_keep = val;
    }
  }

  const double new_keep = old_keep + alpha * are;
  const bool insert_keep = elim_present && !keep_present && fabs(new_keep) > zero_tol;
  bool inserted = !insert_keep;
  for (std::int32_t p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (insert_keep && !inserted && col > keep_col) {
      col_val_new[write] = keep_col;
      nz_val_new[write] = new_keep;
      ++write;
      inserted = true;
    }
    if (col == keep_col) {
      if (fabs(new_keep) > zero_tol) {
        col_val_new[write] = keep_col;
        nz_val_new[write] = new_keep;
        ++write;
      }
    } else {
      col_val_new[write] = col;
      nz_val_new[write] = val;
      ++write;
    }
  }
  if (insert_keep && !inserted) {
    col_val_new[write] = keep_col;
    nz_val_new[write] = new_keep;
  }
}

DeviceCsrMatrix build_doubleton_scalar_new_A(const DeviceCsrMatrix& source,
                                             const std::int32_t* meta_i32,
                                             const double* meta_f64,
                                             double zero_tol) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  if (source.rows == 0) {
    return out;
  }

  std::int32_t* row_counts = nullptr;
  throw_if_cuda_error(doubleton_cuda_malloc(&row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton row_counts");
  const int blocks = (source.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_eq_row_counts<<<blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, source.rowPtr, source.colVal, source.nzVal, meta_i32, meta_f64, zero_tol, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_row_counts");

  inclusive_scan_i32(row_counts, source.rows, "cub inclusive scan doubleton scalar rows");

  std::int32_t nnz_new = 0;
  throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts + source.rows - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton nnz_new");
  out.nnz = nnz_new;
  throw_if_cuda_error(doubleton_cuda_malloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(source.rows + 1)),
                      "cudaMalloc doubleton rowPtr_new");
  _kernel_row_ptr_from_prefix<<<blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix doubleton");
  if (nnz_new > 0) {
    throw_if_cuda_error(doubleton_cuda_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton colVal_new");
    throw_if_cuda_error(doubleton_cuda_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton nzVal_new");
    _kernel_copy_doubleton_eq_rows<<<blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal,
        out.nzVal,
        out.rowPtr,
        source.rowPtr,
        source.colVal,
        source.nzVal,
        meta_i32,
        meta_f64,
        zero_tol,
        source.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_doubleton_eq_rows");
  }
  doubleton_cuda_free(row_counts);
  return out;
}

}  // namespace

void apply_rule_doubleton_eq(PresolvePlanGpu& plan,
                             const LPInfoGpu& lp,
                             const PresolveStatsGpu& stats,
                             const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  if (pparams.doubleton_eq_scan) {
    (void)apply_rule_doubleton_eq_batch(plan, lp, pparams);
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  std::int32_t* live_col_nnz = nullptr;
  std::int32_t* target_row_ref = nullptr;
  std::int32_t* meta_i32 = nullptr;
  double* meta_f64 = nullptr;
  std::int32_t* status = nullptr;
  double* obj_delta_device = nullptr;
  throw_if_cuda_error(doubleton_cuda_malloc(&live_col_nnz, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc doubleton live_col_nnz");
  throw_if_cuda_error(doubleton_cuda_malloc(&target_row_ref, sizeof(std::int32_t)), "cudaMalloc doubleton target_row_ref");
  throw_if_cuda_error(doubleton_cuda_malloc(&meta_i32, sizeof(std::int32_t) * 3), "cudaMalloc doubleton meta_i32");
  throw_if_cuda_error(doubleton_cuda_malloc(&meta_f64, sizeof(double) * 8), "cudaMalloc doubleton meta_f64");
  throw_if_cuda_error(doubleton_cuda_malloc(&status, sizeof(std::int32_t) * 2), "cudaMalloc doubleton status");
  throw_if_cuda_error(doubleton_cuda_malloc(&obj_delta_device, sizeof(double)), "cudaMalloc doubleton obj_delta");
  throw_if_cuda_error(cudaMemset(status, 0, sizeof(std::int32_t) * 2), "cudaMemset doubleton status");
  throw_if_cuda_error(cudaMemset(obj_delta_device, 0, sizeof(double)), "cudaMemset doubleton obj_delta");

  const int blocks_cols = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_eq_live_col_nnz<<<blocks_cols, GPU_PRESOLVE_THREADS>>>(
      live_col_nnz,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      plan.keep_row_mask,
      plan.keep_col_mask,
      pparams.zero_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_live_col_nnz");

  const std::int32_t target_init = m;
  throw_if_cuda_error(cudaMemcpy(target_row_ref, &target_init, sizeof(std::int32_t), cudaMemcpyHostToDevice),
                      "cudaMemcpy doubleton target init");
  const int blocks_rows = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const double ratio_tol = fmax(1.0e-9, pparams.bound_tol);
  _kernel_select_doubleton_eq_target_row_direct<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      target_row_ref,
      meta_i32,
      meta_f64,
      (plan.has_new_A ? plan.new_A : lp.A).rowPtr,
      (plan.has_new_A ? plan.new_A : lp.A).colVal,
      (plan.has_new_A ? plan.new_A : lp.A).nzVal,
      plan.keep_row_mask,
      plan.keep_col_mask,
      live_col_nnz,
      plan.new_AL,
      plan.new_AU,
      plan.new_l,
      plan.new_u,
      plan.new_c,
      lp.AT.rowPtr,
      lp.AT.colVal,
      pparams.zero_tol,
      pparams.bound_tol,
      ratio_tol,
      static_cast<std::int32_t>(pparams.doubleton_eq_max_fill_in_proxy),
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_select_doubleton_eq_target_row_direct");

  std::int32_t target_row = m;
  throw_if_cuda_error(cudaMemcpy(&target_row, target_row_ref, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton target");
  if (target_row >= m) {
    doubleton_cuda_free(live_col_nnz);
    doubleton_cuda_free(target_row_ref);
    doubleton_cuda_free(meta_i32);
    doubleton_cuda_free(meta_f64);
    doubleton_cuda_free(status);
    doubleton_cuda_free(obj_delta_device);
    return;
  }

  _kernel_doubleton_eq_apply<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      status,
      plan.new_l,
      plan.new_u,
      plan.new_c,
      plan.new_AL,
      plan.new_AU,
      plan.keep_row_mask,
      plan.keep_col_mask,
      obj_delta_device,
      meta_i32,
      meta_f64,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.bound_tol,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_eq_apply");
  throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_doubleton_eq synchronize");

  std::int32_t status_host[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(status_host, status, sizeof(status_host), cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton status");
  if (status_host[0] != 0) {
    plan.has_infeasible = true;
  } else if (status_host[1] != 0) {
    if (pparams.record_postsolve_tape) {
      std::int32_t meta_i32_host[3] = {0, 0, 0};
      double meta_f64_host[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      throw_if_cuda_error(cudaMemcpy(meta_i32_host, meta_i32, sizeof(meta_i32_host), cudaMemcpyDeviceToHost),
                          "cudaMemcpy doubleton meta_i32 for tape");
      throw_if_cuda_error(cudaMemcpy(meta_f64_host, meta_f64, sizeof(meta_f64_host), cudaMemcpyDeviceToHost),
                          "cudaMemcpy doubleton meta_f64 for tape");
      const std::int32_t elim_col = meta_i32_host[1];
      const std::int32_t keep_col = meta_i32_host[2];
      double new_keep_l = 0.0;
      double new_keep_u = 0.0;
      throw_if_cuda_error(cudaMemcpy(&new_keep_l, plan.new_l + keep_col, sizeof(double), cudaMemcpyDeviceToHost),
                          "cudaMemcpy doubleton new keep l for tape");
      throw_if_cuda_error(cudaMemcpy(&new_keep_u, plan.new_u + keep_col, sizeof(double), cudaMemcpyDeviceToHost),
                          "cudaMemcpy doubleton new keep u for tape");
      append_postsolve_record(
          plan.tape,
          PostsolveReductionType::DoubletonEq,
          std::vector<std::int32_t>{elim_col, keep_col, meta_i32_host[0], 0},
          std::vector<double>{meta_f64_host[0],
                              meta_f64_host[1],
                              meta_f64_host[2],
                              meta_f64_host[3],
                              meta_f64_host[4],
                              meta_f64_host[6],
                              meta_f64_host[7],
                              new_keep_l,
                              new_keep_u,
                              meta_f64_host[5],
                              0.0,
                              0.0},
          PostsolveDualMode::Minimal);
    }
    const DeviceCsrMatrix source_A = plan.has_new_A ? plan.new_A : lp.A;
    const bool free_old_new_A = plan.has_new_A;
    DeviceCsrMatrix rewritten_A = build_doubleton_scalar_new_A(source_A, meta_i32, meta_f64, pparams.zero_tol);
    if (plan.has_new_A) {
      doubleton_cuda_free(plan.new_A.rowPtr);
      doubleton_cuda_free(plan.new_A.colVal);
      doubleton_cuda_free(plan.new_A.nzVal);
    }
    (void)free_old_new_A;
    plan.new_A = rewritten_A;
    plan.has_new_A = true;
    double obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&obj_delta, obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy doubleton obj_delta");
    plan.obj_constant_delta += obj_delta;
    plan.has_row_action = true;
    plan.has_col_action = true;
    plan.has_change = true;
  }

  doubleton_cuda_free(live_col_nnz);
  doubleton_cuda_free(target_row_ref);
  doubleton_cuda_free(meta_i32);
  doubleton_cuda_free(meta_f64);
  doubleton_cuda_free(status);
  doubleton_cuda_free(obj_delta_device);
}

}  // namespace gpu_presolver::presolve
