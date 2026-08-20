#include "gpu_presolver/presolve/rules/rule_redundant_bounds.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr unsigned long long UINT64_MAX_DEVICE = 0xffffffffffffffffULL;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
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

__global__ void _kernel_fill_u64(unsigned long long* values,
                                 unsigned long long value,
                                 std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    values[j] = value;
  }
}

__global__ void _kernel_redundant_bounds_candidates(std::uint8_t* drop_lower,
                                                    std::uint8_t* drop_upper,
                                                    const double* l_cur,
                                                    const double* u_cur,
                                                    const double* AL,
                                                    const double* AU,
                                                    const std::int32_t* row_ptr,
                                                    const std::int32_t* col_val,
                                                    const double* nz_val,
                                                    const std::int32_t* at_row_ptr,
                                                    const std::int32_t* at_row_val,
                                                    const double* at_nz_val,
                                                    double zero_tol,
                                                    double tol,
                                                    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    const double lj = l_cur[j];
    const double uj = u_cur[j];

    if (lj == -INFINITY && isfinite(uj)) {
      const std::int32_t p_start = at_row_ptr[j];
      const std::int32_t p_stop = at_row_ptr[j + 1];
      for (std::int32_t p = p_start; p < p_stop; ++p) {
        const std::int32_t i = at_row_val[p];
        const double a = at_nz_val[p];
        if (fabs(a) <= zero_tol) {
          continue;
        }

        const std::int32_t row_start = row_ptr[i];
        const std::int32_t row_stop = row_ptr[i + 1];
        double implied_u = INFINITY;
        if (a > zero_tol && isfinite(AU[i])) {
          double rest_min = 0.0;
          bool finite_rest = true;
          for (std::int32_t q = row_start; q < row_stop; ++q) {
            const std::int32_t col = col_val[q];
            if (col == j) {
              continue;
            }
            const double aq = nz_val[q];
            const double term_min = aq >= 0.0 ? (aq * l_cur[col]) : (aq * u_cur[col]);
            if (!isfinite(term_min)) {
              finite_rest = false;
              break;
            }
            rest_min += term_min;
          }
          if (finite_rest) {
            implied_u = (AU[i] - rest_min) / a;
          }
        } else if (a < -zero_tol && isfinite(AL[i])) {
          double rest_max = 0.0;
          bool finite_rest = true;
          for (std::int32_t q = row_start; q < row_stop; ++q) {
            const std::int32_t col = col_val[q];
            if (col == j) {
              continue;
            }
            const double aq = nz_val[q];
            const double term_max = aq >= 0.0 ? (aq * u_cur[col]) : (aq * l_cur[col]);
            if (!isfinite(term_max)) {
              finite_rest = false;
              break;
            }
            rest_max += term_max;
          }
          if (finite_rest) {
            implied_u = (AL[i] - rest_max) / a;
          }
        }
        if (isfinite(implied_u) && implied_u <= uj + tol) {
          drop_upper[j] = std::uint8_t{1};
          break;
        }
      }
    } else if (isfinite(lj) && uj == INFINITY) {
      const std::int32_t p_start = at_row_ptr[j];
      const std::int32_t p_stop = at_row_ptr[j + 1];
      for (std::int32_t p = p_start; p < p_stop; ++p) {
        const std::int32_t i = at_row_val[p];
        const double a = at_nz_val[p];
        if (fabs(a) <= zero_tol) {
          continue;
        }

        const std::int32_t row_start = row_ptr[i];
        const std::int32_t row_stop = row_ptr[i + 1];
        double implied_l = -INFINITY;
        if (a > zero_tol && isfinite(AL[i])) {
          double rest_max = 0.0;
          bool finite_rest = true;
          for (std::int32_t q = row_start; q < row_stop; ++q) {
            const std::int32_t col = col_val[q];
            if (col == j) {
              continue;
            }
            const double aq = nz_val[q];
            const double term_max = aq >= 0.0 ? (aq * u_cur[col]) : (aq * l_cur[col]);
            if (!isfinite(term_max)) {
              finite_rest = false;
              break;
            }
            rest_max += term_max;
          }
          if (finite_rest) {
            implied_l = (AL[i] - rest_max) / a;
          }
        } else if (a < -zero_tol && isfinite(AU[i])) {
          double rest_min = 0.0;
          bool finite_rest = true;
          for (std::int32_t q = row_start; q < row_stop; ++q) {
            const std::int32_t col = col_val[q];
            if (col == j) {
              continue;
            }
            const double aq = nz_val[q];
            const double term_min = aq >= 0.0 ? (aq * l_cur[col]) : (aq * u_cur[col]);
            if (!isfinite(term_min)) {
              finite_rest = false;
              break;
            }
            rest_min += term_min;
          }
          if (finite_rest) {
            implied_l = (AU[i] - rest_min) / a;
          }
        }
        if (isfinite(implied_l) && implied_l >= lj - tol) {
          drop_lower[j] = std::uint8_t{1};
          break;
        }
      }
    }
  }
}

__global__ void _kernel_compute_redundant_bounds_row_min_keys(unsigned long long* row_min_key,
                                                              const std::uint8_t* drop_lower,
                                                              const std::uint8_t* drop_upper,
                                                              const std::int32_t* row_ptr,
                                                              const std::int32_t* col_val,
                                                              const std::int32_t* at_row_ptr,
                                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    const std::int32_t row_start = row_ptr[i];
    const std::int32_t row_stop = row_ptr[i + 1];
    unsigned long long best_key = UINT64_MAX_DEVICE;
    for (std::int32_t q = row_start; q < row_stop; ++q) {
      const std::int32_t j = col_val[q];
      if (drop_lower[j] != std::uint8_t{0} || drop_upper[j] != std::uint8_t{0}) {
        const std::int32_t nnz_j = at_row_ptr[j + 1] - at_row_ptr[j];
        const unsigned long long key =
            (static_cast<unsigned long long>(static_cast<unsigned int>(nnz_j)) << 32) |
            static_cast<unsigned long long>(static_cast<unsigned int>(j));
        if (key < best_key) {
          best_key = key;
        }
      }
    }
    row_min_key[i] = best_key;
  }
}

__global__ void _kernel_select_redundant_bounds_batch(std::int32_t* selected_any,
                                                      std::uint8_t* selected_lower,
                                                      std::uint8_t* selected_upper,
                                                      const std::uint8_t* drop_lower,
                                                      const std::uint8_t* drop_upper,
                                                      const unsigned long long* row_min_key,
                                                      const std::int32_t* at_row_ptr,
                                                      const std::int32_t* at_row_val,
                                                      std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (drop_lower[j] == std::uint8_t{0} && drop_upper[j] == std::uint8_t{0}) {
      return;
    }
    const std::int32_t p_start = at_row_ptr[j];
    const std::int32_t p_stop = at_row_ptr[j + 1];
    const std::int32_t nnz_j = at_row_ptr[j + 1] - at_row_ptr[j];
    const unsigned long long key =
        (static_cast<unsigned long long>(static_cast<unsigned int>(nnz_j)) << 32) |
        static_cast<unsigned long long>(static_cast<unsigned int>(j));
    bool choose = true;
    for (std::int32_t p = p_start; p < p_stop; ++p) {
      const std::int32_t i = at_row_val[p];
      if (row_min_key[i] != key) {
        choose = false;
        break;
      }
    }
    if (choose) {
      selected_lower[j] = drop_lower[j];
      selected_upper[j] = drop_upper[j];
      atomicMax(selected_any, 1);
    }
  }
}

__global__ void _kernel_apply_selected_redundant_bounds(double* l_cur,
                                                        double* u_cur,
                                                        const std::uint8_t* selected_lower,
                                                        const std::uint8_t* selected_upper,
                                                        std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (selected_lower[j] != std::uint8_t{0}) {
      l_cur[j] = -INFINITY;
    } else if (selected_upper[j] != std::uint8_t{0}) {
      u_cur[j] = INFINITY;
    }
  }
}

}  // namespace

void apply_rule_redundant_bounds(PresolvePlanGpu& plan,
                                 const LPInfoGpu& lp,
                                 const PresolveStatsGpu& stats,
                                 const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  std::uint8_t* drop_lower = nullptr;
  std::uint8_t* drop_upper = nullptr;
  unsigned long long* row_min_key = nullptr;
  std::uint8_t* selected_lower = nullptr;
  std::uint8_t* selected_upper = nullptr;
  std::int32_t* selected_any_device = nullptr;
  throw_if_cuda_error(cudaMalloc(&drop_lower, static_cast<std::size_t>(n)), "cudaMalloc redundant_bounds drop_lower");
  throw_if_cuda_error(cudaMalloc(&drop_upper, static_cast<std::size_t>(n)), "cudaMalloc redundant_bounds drop_upper");
  throw_if_cuda_error(cudaMalloc(&row_min_key, sizeof(unsigned long long) * static_cast<std::size_t>(m)), "cudaMalloc redundant_bounds row_min_key");
  throw_if_cuda_error(cudaMalloc(&selected_lower, static_cast<std::size_t>(n)), "cudaMalloc redundant_bounds selected_lower");
  throw_if_cuda_error(cudaMalloc(&selected_upper, static_cast<std::size_t>(n)), "cudaMalloc redundant_bounds selected_upper");
  throw_if_cuda_error(cudaMalloc(&selected_any_device, sizeof(std::int32_t)), "cudaMalloc redundant_bounds selected_any");

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  bool changed_any = false;
  for (;;) {
    throw_if_cuda_error(cudaMemset(drop_lower, 0, static_cast<std::size_t>(n)), "cudaMemset redundant_bounds drop_lower");
    throw_if_cuda_error(cudaMemset(drop_upper, 0, static_cast<std::size_t>(n)), "cudaMemset redundant_bounds drop_upper");
    throw_if_cuda_error(cudaMemset(selected_lower, 0, static_cast<std::size_t>(n)), "cudaMemset redundant_bounds selected_lower");
    throw_if_cuda_error(cudaMemset(selected_upper, 0, static_cast<std::size_t>(n)), "cudaMemset redundant_bounds selected_upper");
    throw_if_cuda_error(cudaMemset(selected_any_device, 0, sizeof(std::int32_t)), "cudaMemset redundant_bounds selected_any");
    _kernel_fill_u64<<<blocks_m, GPU_PRESOLVE_THREADS>>>(row_min_key, UINT64_MAX_DEVICE, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_u64 redundant_bounds");

    _kernel_redundant_bounds_candidates<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        drop_lower,
        drop_upper,
        plan.new_l,
        plan.new_u,
        plan.new_AL,
        plan.new_AU,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        pparams.zero_tol,
        pparams.bound_tol,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_redundant_bounds_candidates");

    _kernel_compute_redundant_bounds_row_min_keys<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
        row_min_key,
        drop_lower,
        drop_upper,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.AT.rowPtr,
        m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_redundant_bounds_row_min_keys");

    _kernel_select_redundant_bounds_batch<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        selected_any_device,
        selected_lower,
        selected_upper,
        drop_lower,
        drop_upper,
        row_min_key,
        lp.AT.rowPtr,
        lp.AT.colVal,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_select_redundant_bounds_batch");

    std::int32_t selected_any = 0;
    throw_if_cuda_error(cudaMemcpy(&selected_any, selected_any_device, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy redundant_bounds selected_any");
    if (selected_any == 0) {
      break;
    }

    if (pparams.record_postsolve_tape) {
      std::vector<std::uint8_t> host_selected_lower(static_cast<std::size_t>(n));
      std::vector<std::uint8_t> host_selected_upper(static_cast<std::size_t>(n));
      std::vector<double> host_old_l(static_cast<std::size_t>(n));
      std::vector<double> host_old_u(static_cast<std::size_t>(n));
      throw_if_cuda_error(cudaMemcpy(host_selected_lower.data(), selected_lower, static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy redundant_bounds selected_lower for tape");
      throw_if_cuda_error(cudaMemcpy(host_selected_upper.data(), selected_upper, static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy redundant_bounds selected_upper for tape");
      throw_if_cuda_error(cudaMemcpy(host_old_l.data(), plan.new_l, sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy redundant_bounds old_l for tape");
      throw_if_cuda_error(cudaMemcpy(host_old_u.data(), plan.new_u, sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy redundant_bounds old_u for tape");
      for (std::int32_t col = 0; col < n; ++col) {
        const auto idx = static_cast<std::size_t>(col);
        if (host_selected_lower[idx] != std::uint8_t{0}) {
          append_postsolve_record(plan.tape,
                                  PostsolveReductionType::BoundChangeNoRow,
                                  {col},
                                  {host_old_l[idx], host_old_u[idx], -INFINITY, host_old_u[idx]},
                                  PostsolveDualMode::Minimal);
        } else if (host_selected_upper[idx] != std::uint8_t{0}) {
          append_postsolve_record(plan.tape,
                                  PostsolveReductionType::BoundChangeNoRow,
                                  {col},
                                  {host_old_l[idx], host_old_u[idx], host_old_l[idx], INFINITY},
                                  PostsolveDualMode::Minimal);
        }
      }
    }

    _kernel_apply_selected_redundant_bounds<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        plan.new_l,
        plan.new_u,
        selected_lower,
        selected_upper,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_selected_redundant_bounds");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_redundant_bounds loop synchronize");
    changed_any = true;
  }

  if (changed_any) {
    plan.has_col_action = true;
    plan.has_change = true;
  }

  cudaFree(drop_lower);
  cudaFree(drop_upper);
  cudaFree(row_min_key);
  cudaFree(selected_lower);
  cudaFree(selected_upper);
  cudaFree(selected_any_device);
}

}  // namespace gpu_presolver::presolve
