#include "gpu_presolver/presolve/rules/rule_parallel_cols.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr unsigned long long FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr unsigned long long FNV_PRIME = 0x100000001b3ULL;

struct HashColLess {
  __host__ __device__ bool operator()(const thrust::tuple<unsigned long long, std::int32_t>& lhs,
                                      const thrust::tuple<unsigned long long, std::int32_t>& rhs) const {
    const unsigned long long lhs_hash = thrust::get<0>(lhs);
    const unsigned long long rhs_hash = thrust::get<0>(rhs);
    if (lhs_hash != rhs_hash) {
      return lhs_hash < rhs_hash;
    }
    return thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
};

void throw_if_cuda_error(cudaError_t status, const char* context);

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

void append_fixed_col_tape_from_device(PostsolveTape& tape,
                                       const std::uint8_t* fixed_mask,
                                       const double* fixed_val,
                                       const std::uint8_t* keep_row,
                                       const double* c,
                                       const DeviceCsrMatrix& AT,
                                       std::int32_t m,
                                       std::int32_t n,
                                       const char* context) {
  std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_keep_row(static_cast<std::size_t>(m));
  std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
  std::vector<double> host_c(static_cast<std::size_t>(n));
  std::vector<std::int32_t> host_at_row_ptr(static_cast<std::size_t>(n + 1));
  std::vector<std::int32_t> host_at_col_val(static_cast<std::size_t>(AT.nnz));
  std::vector<double> host_at_nz_val(static_cast<std::size_t>(AT.nnz));

  throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask, static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (m > 0) {
    throw_if_cuda_error(cudaMemcpy(host_keep_row.data(), keep_row, static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                 sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_c.data(), c, sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_at_row_ptr.data(), AT.rowPtr,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(n + 1),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (AT.nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_at_col_val.data(), AT.colVal,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(AT.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_at_nz_val.data(), AT.nzVal,
                                   sizeof(double) * static_cast<std::size_t>(AT.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }

  for (std::int32_t col = 0; col < n; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    if (host_fixed_mask[col_idx] == std::uint8_t{0}) {
      continue;
    }
    std::vector<std::int32_t> indices{col};
    std::vector<double> vals{host_fixed_val[col_idx], host_c[col_idx]};
    for (std::int32_t p = host_at_row_ptr[col_idx]; p < host_at_row_ptr[col_idx + 1]; ++p) {
      const std::int32_t row = host_at_col_val[static_cast<std::size_t>(p)];
      if (host_keep_row[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
        continue;
      }
      indices.push_back(row);
      vals.push_back(host_at_nz_val[static_cast<std::size_t>(p)]);
    }
    append_postsolve_record(
        tape, PostsolveReductionType::FixedCol, indices, vals, PostsolveDualMode::Minimal);
  }
}

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

__global__ void _kernel_parallel_col_sort_keys(unsigned long long* keys,
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

__global__ void _kernel_parallel_col_unpack_keys(std::int32_t* col_val,
                                                 const unsigned long long* keys,
                                                 std::int32_t nnz) {
  const std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < nnz) {
    col_val[p] = static_cast<std::int32_t>(keys[p] & 0xffffffffULL);
  }
}

__device__ unsigned long long _parallel_col_hash_mix(unsigned long long h,
                                                     unsigned long long x) {
  return (h ^ x) * FNV_PRIME;
}

__device__ unsigned long long _double_bits(double value) {
  return static_cast<unsigned long long>(__double_as_longlong(value));
}

__global__ void _kernel_parallel_col_hashes(unsigned long long* col_hash,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* row_idx,
                                            const double* row_val,
                                            double coeff_tol,
                                            std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0}) {
      col_hash[j] = static_cast<unsigned long long>(j);
      return;
    }
    const std::int32_t start_j = row_ptr[j];
    const std::int32_t stop_j = row_ptr[j + 1];
    std::int32_t live_len = 0;
    double pivot = 0.0;
    bool pivot_set = false;
    for (std::int32_t p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        const double a = row_val[p];
        if (fabs(a) > coeff_tol) {
          ++live_len;
          if (!pivot_set) {
            pivot = a;
            pivot_set = true;
          }
        }
      }
    }
    if (!pivot_set || live_len == 0) {
      col_hash[j] = _parallel_col_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(j));
      return;
    }
    unsigned long long h = _parallel_col_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(live_len));
    for (std::int32_t p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        const double a = row_val[p];
        if (fabs(a) > coeff_tol) {
          h = _parallel_col_hash_mix(h, static_cast<unsigned long long>(row));
          h = _parallel_col_hash_mix(h, _double_bits(a / pivot));
        }
      }
    }
    col_hash[j] = h;
  }
}

__device__ std::int32_t _next_live_parallel_col_entry(std::int32_t ptr,
                                                      std::int32_t stop,
                                                      const std::int32_t* row_idx,
                                                      const double* row_val,
                                                      const std::uint8_t* keep_row,
                                                      double zero_tol) {
  std::int32_t cur = ptr;
  while (cur < stop) {
    const std::int32_t row = row_idx[cur];
    if (keep_row[row] != std::uint8_t{0}) {
      const double a = row_val[cur];
      if (fabs(a) > zero_tol) {
        return cur;
      }
    }
    ++cur;
  }
  return stop;
}

__device__ bool _parallel_col_ratio(std::int32_t j,
                                    std::int32_t k,
                                    const std::int32_t* row_ptr,
                                    const std::int32_t* row_idx,
                                    const double* row_val,
                                    const std::uint8_t* keep_row,
                                    double zero_tol,
                                    double coeff_tol,
                                    double* ratio_out) {
  std::int32_t ptr_j = row_ptr[j];
  const std::int32_t stop_j = row_ptr[j + 1];
  std::int32_t ptr_k = row_ptr[k];
  const std::int32_t stop_k = row_ptr[k + 1];
  double ratio = 0.0;
  bool ratio_set = false;

  for (;;) {
    ptr_j = _next_live_parallel_col_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol);
    ptr_k = _next_live_parallel_col_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol);
    if (ptr_j >= stop_j || ptr_k >= stop_k) {
      break;
    }
    const std::int32_t row_j = row_idx[ptr_j];
    const std::int32_t row_k = row_idx[ptr_k];
    if (row_j != row_k) {
      return false;
    }
    const double a_j = row_val[ptr_j];
    const double a_k = row_val[ptr_k];
    if (!ratio_set) {
      if (fabs(a_j) <= zero_tol) {
        return false;
      }
      ratio = a_k / a_j;
      if (fabs(ratio) <= zero_tol || !isfinite(ratio)) {
        return false;
      }
      ratio_set = true;
    }
    if (fabs(a_k - ratio * a_j) > coeff_tol) {
      return false;
    }
    ++ptr_j;
    ++ptr_k;
  }

  ptr_j = _next_live_parallel_col_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol);
  ptr_k = _next_live_parallel_col_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol);
  if (ptr_j < stop_j || ptr_k < stop_k || !ratio_set) {
    return false;
  }
  *ratio_out = ratio;
  return true;
}

__device__ double _merged_lower_bound_parallel_cols(double target_l,
                                                    double target_u,
                                                    double source_l,
                                                    double source_u,
                                                    double ratio) {
  (void)target_u;
  if (ratio > 0.0) {
    if (isfinite(target_l) && isfinite(source_l)) {
      return target_l + ratio * source_l;
    }
  } else if (isfinite(target_l) && isfinite(source_u)) {
    return target_l + ratio * source_u;
  }
  return -INFINITY;
}

__device__ double _merged_upper_bound_parallel_cols(double target_l,
                                                    double target_u,
                                                    double source_l,
                                                    double source_u,
                                                    double ratio) {
  (void)target_l;
  if (ratio > 0.0) {
    if (isfinite(target_u) && isfinite(source_u)) {
      return target_u + ratio * source_u;
    }
  } else if (isfinite(target_u) && isfinite(source_l)) {
    return target_u + ratio * source_l;
  }
  return INFINITY;
}

__global__ void _kernel_parallel_col_groups(std::int32_t* status_flag,
                                            std::uint8_t* col_delete,
                                            std::uint8_t* fixed_mask,
                                            double* fixed_val,
                                            std::int32_t* merge_to,
                                            double* merge_ratio,
                                            double* merge_from_l,
                                            double* merge_from_u,
                                            double* merge_to_l,
                                            double* merge_to_u,
                                            const unsigned long long* sorted_hash,
                                            const std::int32_t* sorted_cols,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const double* c,
                                            double* l,
                                            double* u,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* row_idx,
                                            const double* row_val,
                                            double zero_tol,
                                            double coeff_tol,
                                            double obj_tol,
                                            std::int32_t n) {
  const std::int32_t s = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s < n) {
    const unsigned long long hs = sorted_hash[s];
    if (s > 0 && sorted_hash[s - 1] == hs) {
      return;
    }
    std::int32_t e = s;
    while (e + 1 < n && sorted_hash[e + 1] == hs) {
      ++e;
    }
    if (e <= s) {
      return;
    }

    for (std::int32_t a = s; a < e; ++a) {
      const std::int32_t j = sorted_cols[a];
      if (keep_col[j] == std::uint8_t{0} || col_delete[j] != std::uint8_t{0} || fixed_mask[j] != std::uint8_t{0}) {
        continue;
      }
      for (std::int32_t b = a + 1; b <= e; ++b) {
        const std::int32_t k = sorted_cols[b];
        if (keep_col[k] == std::uint8_t{0} || col_delete[k] != std::uint8_t{0} || fixed_mask[k] != std::uint8_t{0}) {
          continue;
        }
        double ratio = 0.0;
        if (!_parallel_col_ratio(j, k, row_ptr, row_idx, row_val, keep_row, zero_tol, coeff_tol, &ratio)) {
          continue;
        }

        const double obj_gap = c[k] - ratio * c[j];
        const double target_l = l[j];
        const double target_u = u[j];
        const double source_l = l[k];
        const double source_u = u[k];

        if (fabs(obj_gap) <= obj_tol) {
          merge_to[k] = j;
          merge_ratio[k] = ratio;
          merge_from_l[k] = source_l;
          merge_from_u[k] = source_u;
          merge_to_l[k] = target_l;
          merge_to_u[k] = target_u;
          l[j] = _merged_lower_bound_parallel_cols(target_l, target_u, source_l, source_u, ratio);
          u[j] = _merged_upper_bound_parallel_cols(target_l, target_u, source_l, source_u, ratio);
          col_delete[k] = std::uint8_t{1};
          atomicMax(&status_flag[1], 1);
          continue;
        }

        bool fix_xk_to_lower = false;
        bool fix_xk_to_upper = false;
        bool fix_xj_to_lower = false;
        bool fix_xj_to_upper = false;
        if (obj_gap > obj_tol) {
          if (ratio > 0.0) {
            fix_xk_to_lower = !isfinite(target_u);
            fix_xj_to_upper = !isfinite(source_l);
          } else {
            fix_xk_to_lower = !isfinite(target_l);
            fix_xj_to_lower = !isfinite(source_l);
          }
        } else {
          if (ratio > 0.0) {
            fix_xk_to_upper = !isfinite(target_l);
            fix_xj_to_lower = !isfinite(source_u);
          } else {
            fix_xk_to_upper = !isfinite(target_u);
            fix_xj_to_upper = !isfinite(source_u);
          }
        }

        if (fix_xk_to_lower) {
          if (!isfinite(source_l)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[k] = std::uint8_t{1};
          fixed_val[k] = source_l;
          atomicMax(&status_flag[2], 1);
          continue;
        } else if (fix_xk_to_upper) {
          if (!isfinite(source_u)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[k] = std::uint8_t{1};
          fixed_val[k] = source_u;
          atomicMax(&status_flag[2], 1);
          continue;
        }

        if (fix_xj_to_lower) {
          if (!isfinite(target_l)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[j] = std::uint8_t{1};
          fixed_val[j] = target_l;
          atomicMax(&status_flag[2], 1);
          break;
        } else if (fix_xj_to_upper) {
          if (!isfinite(target_u)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[j] = std::uint8_t{1};
          fixed_val[j] = target_u;
          atomicMax(&status_flag[2], 1);
          break;
        }
      }
    }
  }
}

__global__ void _kernel_parallel_col_fixed_row_shift(double* row_shift,
                                                     const std::uint8_t* fixed_mask,
                                                     const double* fixed_val,
                                                     const std::uint8_t* keep_row,
                                                     const std::int32_t* row_ptr,
                                                     const std::int32_t* row_idx,
                                                     const double* row_val,
                                                     std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    const double vj = fixed_val[j];
    const std::int32_t start_j = row_ptr[j];
    const std::int32_t stop_j = row_ptr[j + 1];
    for (std::int32_t p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        atomicAdd(&row_shift[row], row_val[p] * vj);
      }
    }
  }
}

__global__ void _kernel_apply_parallel_cols(std::uint8_t* keep_col,
                                            double* AL,
                                            double* AU,
                                            double* new_l,
                                            double* new_u,
                                            double* obj_delta,
                                            const double* c,
                                            const std::uint8_t* col_delete,
                                            const std::uint8_t* fixed_mask,
                                            const double* fixed_val,
                                            const double* row_shift,
                                            std::int32_t m,
                                            std::int32_t n) {
  const std::int32_t q = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q < n && (col_delete[q] != std::uint8_t{0} || fixed_mask[q] != std::uint8_t{0})) {
    keep_col[q] = std::uint8_t{0};
    if (fixed_mask[q] != std::uint8_t{0}) {
      new_l[q] = fixed_val[q];
      new_u[q] = fixed_val[q];
      atomicAdd(obj_delta, c[q] * fixed_val[q]);
    }
  }
  if (q < m) {
    AL[q] -= row_shift[q];
    AU[q] -= row_shift[q];
  }
}

}  // namespace

void apply_rule_parallel_cols(PresolvePlanGpu& plan,
                              const LPInfoGpu& lp,
                              const PresolveStatsGpu& stats,
                              const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (n <= 1) {
    return;
  }
  const auto rule_start = std::chrono::steady_clock::now();
  const bool profile = env_enabled("GPUPRESOLVER_PARALLEL_COLS_PROFILE");
  auto profile_stage = [&](const char* stage, const std::chrono::steady_clock::time_point& stage_start) {
    if (!profile) {
      return;
    }
    throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_cols profile synchronize");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - stage_start;
    std::cerr << ">>> [parallel_cols C++] " << stage << " = " << elapsed.count() << "s\n";
  };

  if (lp.AT.nnz > 0) {
    const auto sort_at_start = std::chrono::steady_clock::now();
    unsigned long long* sort_keys = nullptr;
    throw_if_cuda_error(cudaMalloc(&sort_keys, sizeof(unsigned long long) * static_cast<std::size_t>(lp.AT.nnz)),
                        "cudaMalloc parallel_cols sort keys");
    const int blocks_at_rows = (lp.AT.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_parallel_col_sort_keys<<<blocks_at_rows, GPU_PRESOLVE_THREADS>>>(
        sort_keys, lp.AT.rowPtr, lp.AT.colVal, lp.AT.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_col_sort_keys");
    thrust::sort_by_key(thrust::device_pointer_cast(sort_keys),
                        thrust::device_pointer_cast(sort_keys + lp.AT.nnz),
                        thrust::device_pointer_cast(lp.AT.nzVal));
    throw_if_cuda_error(cudaGetLastError(), "thrust parallel_cols sort AT rows");
    const int blocks_at_nnz = (lp.AT.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_parallel_col_unpack_keys<<<blocks_at_nnz, GPU_PRESOLVE_THREADS>>>(
        lp.AT.colVal, sort_keys, lp.AT.nnz);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_col_unpack_keys");
    throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_cols sort AT synchronize");
    cudaFree(sort_keys);
    profile_stage("sort_AT_rows", sort_at_start);
  }

  unsigned long long* col_hash = nullptr;
  std::int32_t* sorted_cols = nullptr;
  std::int32_t* status_flag = nullptr;
  std::uint8_t* col_delete = nullptr;
  std::uint8_t* fixed_mask = nullptr;
  double* fixed_val = nullptr;
  std::int32_t* merge_to = nullptr;
  double* merge_ratio = nullptr;
  double* merge_from_l = nullptr;
  double* merge_from_u = nullptr;
  double* merge_to_l = nullptr;
  double* merge_to_u = nullptr;
  double* row_shift = nullptr;
  double* obj_delta_device = nullptr;

  throw_if_cuda_error(cudaMalloc(&col_hash, sizeof(unsigned long long) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols col_hash");
  throw_if_cuda_error(cudaMalloc(&sorted_cols, sizeof(std::int32_t) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols sorted_cols");
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t) * 3), "cudaMalloc parallel_cols status_flag");
  throw_if_cuda_error(cudaMalloc(&col_delete, static_cast<std::size_t>(n)), "cudaMalloc parallel_cols col_delete");
  throw_if_cuda_error(cudaMalloc(&fixed_mask, static_cast<std::size_t>(n)), "cudaMalloc parallel_cols fixed_mask");
  throw_if_cuda_error(cudaMalloc(&fixed_val, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols fixed_val");
  throw_if_cuda_error(cudaMalloc(&merge_to, sizeof(std::int32_t) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols merge_to");
  throw_if_cuda_error(cudaMalloc(&merge_ratio, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols merge_ratio");
  throw_if_cuda_error(cudaMalloc(&merge_from_l, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols merge_from_l");
  throw_if_cuda_error(cudaMalloc(&merge_from_u, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols merge_from_u");
  throw_if_cuda_error(cudaMalloc(&merge_to_l, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols merge_to_l");
  throw_if_cuda_error(cudaMalloc(&merge_to_u, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc parallel_cols merge_to_u");
  throw_if_cuda_error(cudaMalloc(&row_shift, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc parallel_cols row_shift");
  throw_if_cuda_error(cudaMalloc(&obj_delta_device, sizeof(double)), "cudaMalloc parallel_cols obj_delta");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset parallel_cols status_flag");
  throw_if_cuda_error(cudaMemset(col_delete, 0, static_cast<std::size_t>(n)), "cudaMemset parallel_cols col_delete");
  throw_if_cuda_error(cudaMemset(fixed_mask, 0, static_cast<std::size_t>(n)), "cudaMemset parallel_cols fixed_mask");
  throw_if_cuda_error(cudaMemset(merge_to, 0, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMemset parallel_cols merge_to");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset parallel_cols row_shift");
  throw_if_cuda_error(cudaMemset(obj_delta_device, 0, sizeof(double)), "cudaMemset parallel_cols obj_delta");

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_parallel_col_hashes<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      col_hash,
      plan.keep_col_mask,
      plan.keep_row_mask,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_col_hashes");
  throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_cols hash synchronize");
  profile_stage("hash", rule_start);

  const auto sort_start = std::chrono::steady_clock::now();
  auto col_hash_begin = thrust::device_pointer_cast(col_hash);
  auto sorted_cols_begin = thrust::device_pointer_cast(sorted_cols);
  thrust::sequence(sorted_cols_begin, sorted_cols_begin + n, std::int32_t{0});
  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(col_hash_begin, sorted_cols_begin));
  thrust::sort(zipped_begin, zipped_begin + n, HashColLess{});
  throw_if_cuda_error(cudaGetLastError(), "thrust parallel_cols sort hashes");
  profile_stage("device_sort", sort_start);

  const double coeff_tol = fmax(pparams.zero_tol, pparams.bound_tol);
  const double obj_tol = fmax(pparams.zero_tol, pparams.bound_tol);
  const auto groups_start = std::chrono::steady_clock::now();
  _kernel_parallel_col_groups<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      col_delete,
      fixed_mask,
      fixed_val,
      merge_to,
      merge_ratio,
      merge_from_l,
      merge_from_u,
      merge_to_l,
      merge_to_u,
      col_hash,
      sorted_cols,
      plan.keep_col_mask,
      plan.keep_row_mask,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      coeff_tol,
      obj_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_col_groups");
  profile_stage("groups", groups_start);

  const auto status_start = std::chrono::steady_clock::now();
  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy parallel_cols status");
  profile_stage("status", status_start);
  if (status[0] != 0) {
    plan.has_unbounded = true;
  } else if (status[1] != 0 || status[2] != 0) {
    const auto tape_start = std::chrono::steady_clock::now();
    if (status[1] != 0 && pparams.record_postsolve_tape) {
      std::vector<std::uint8_t> host_col_delete(static_cast<std::size_t>(n));
      std::vector<std::int32_t> host_merge_to(static_cast<std::size_t>(n));
      std::vector<double> host_merge_ratio(static_cast<std::size_t>(n));
      std::vector<double> host_merge_from_l(static_cast<std::size_t>(n));
      std::vector<double> host_merge_from_u(static_cast<std::size_t>(n));
      std::vector<double> host_merge_to_l(static_cast<std::size_t>(n));
      std::vector<double> host_merge_to_u(static_cast<std::size_t>(n));
      throw_if_cuda_error(cudaMemcpy(host_col_delete.data(), col_delete, static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols col_delete D2H");
      throw_if_cuda_error(cudaMemcpy(host_merge_to.data(), merge_to,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols merge_to D2H");
      throw_if_cuda_error(cudaMemcpy(host_merge_ratio.data(), merge_ratio,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols merge_ratio D2H");
      throw_if_cuda_error(cudaMemcpy(host_merge_from_l.data(), merge_from_l,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols merge_from_l D2H");
      throw_if_cuda_error(cudaMemcpy(host_merge_from_u.data(), merge_from_u,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols merge_from_u D2H");
      throw_if_cuda_error(cudaMemcpy(host_merge_to_l.data(), merge_to_l,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols merge_to_l D2H");
      throw_if_cuda_error(cudaMemcpy(host_merge_to_u.data(), merge_to_u,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy parallel_cols merge_to_u D2H");
      for (std::int32_t col = 0; col < n; ++col) {
        if (host_col_delete[static_cast<std::size_t>(col)] == std::uint8_t{0}) {
          continue;
        }
        append_postsolve_record(
            plan.tape,
            PostsolveReductionType::ParallelCol,
            std::vector<std::int32_t>{col, host_merge_to[static_cast<std::size_t>(col)]},
            std::vector<double>{host_merge_ratio[static_cast<std::size_t>(col)],
                                host_merge_from_l[static_cast<std::size_t>(col)],
                                host_merge_from_u[static_cast<std::size_t>(col)],
                                host_merge_to_l[static_cast<std::size_t>(col)],
                                host_merge_to_u[static_cast<std::size_t>(col)]},
            PostsolveDualMode::Minimal);
      }
    }
    if (status[2] != 0) {
      if (pparams.record_postsolve_tape) {
        append_fixed_col_tape_from_device(plan.tape,
                                          fixed_mask,
                                          fixed_val,
                                          plan.keep_row_mask,
                                          plan.new_c,
                                          lp.AT,
                                          m,
                                          n,
                                          "cudaMemcpy parallel_cols fixed-col tape");
      }
      _kernel_parallel_col_fixed_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          row_shift,
          fixed_mask,
          fixed_val,
          plan.keep_row_mask,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_col_fixed_row_shift");
      plan.has_row_action = true;
    }
    profile_stage("tape", tape_start);
    const auto apply_start = std::chrono::steady_clock::now();
    const std::int32_t max_mn = m > n ? m : n;
    const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_apply_parallel_cols<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        obj_delta_device,
        plan.new_c,
        col_delete,
        fixed_mask,
        fixed_val,
        row_shift,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_parallel_cols");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_parallel_cols synchronize");
    double obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&obj_delta, obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy parallel_cols obj_delta");
    profile_stage("apply_delete", apply_start);
    plan.obj_constant_delta += obj_delta;
    plan.has_col_action = true;
    plan.has_change = true;
  }

  cudaFree(col_hash);
  cudaFree(sorted_cols);
  cudaFree(status_flag);
  cudaFree(col_delete);
  cudaFree(fixed_mask);
  cudaFree(fixed_val);
  cudaFree(merge_to);
  cudaFree(merge_ratio);
  cudaFree(merge_from_l);
  cudaFree(merge_from_u);
  cudaFree(merge_to_l);
  cudaFree(merge_to_u);
  cudaFree(row_shift);
  cudaFree(obj_delta_device);
  profile_stage("total", rule_start);
}

}  // namespace gpu_presolver::presolve
