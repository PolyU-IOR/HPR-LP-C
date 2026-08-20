#include "gpu_presolver/presolve/rules/rule_singleton_cols.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr std::uint8_t SINGLETON_COL_INEQ_NO_ACTION = 0;
constexpr std::uint8_t SINGLETON_COL_INEQ_ELIMINATE = 1;
constexpr std::uint8_t SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS = 2;
constexpr std::uint8_t SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS = 3;
constexpr std::uint8_t SINGLETON_COL_RULE_EQ = 1;
constexpr std::uint8_t SINGLETON_COL_RULE_DUAL_INFER = 2;

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

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  throw_if_cuda_error(cudaMallocAsync(&temp_storage, temp_bytes, nullptr), context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  cudaFreeAsync(temp_storage, nullptr);
}

double sum_device_double(const double* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return 0.0;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  double* result_device = nullptr;
  double result = 0.0;
  throw_if_cuda_error(cudaMallocAsync(&result_device, sizeof(double), nullptr), context);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMallocAsync(&temp_storage, temp_bytes, nullptr), context);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMemcpy(&result, result_device, sizeof(double), cudaMemcpyDeviceToHost),
                      context);
  cudaFreeAsync(temp_storage, nullptr);
  cudaFreeAsync(result_device, nullptr);
  return result;
}

void append_eq_to_ineq_tape_from_device(PostsolveTape& tape,
                                        const std::uint8_t* row_lhs_change,
                                        const std::uint8_t* row_rhs_change,
                                        std::int32_t m,
                                        const char* context) {
  std::vector<std::uint8_t> host_lhs(static_cast<std::size_t>(m));
  std::vector<std::uint8_t> host_rhs(static_cast<std::size_t>(m));
  throw_if_cuda_error(cudaMemcpy(host_lhs.data(), row_lhs_change, static_cast<std::size_t>(m),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_rhs.data(), row_rhs_change, static_cast<std::size_t>(m),
                                 cudaMemcpyDeviceToHost),
                      context);
  for (std::int32_t row = 0; row < m; ++row) {
    if (host_lhs[static_cast<std::size_t>(row)] == std::uint8_t{0} &&
        host_rhs[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
      continue;
    }
    append_postsolve_record(tape,
                            PostsolveReductionType::EqToIneq,
                            {row},
                            {0.0},
                            PostsolveDualMode::Minimal);
  }
}

__global__ void _kernel_fill_i32(std::int32_t* values, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    values[i] = value;
  }
}

__device__ bool _singleton_col_direct_unbounded(double cj,
                                                double a,
                                                double lhs,
                                                double rhs,
                                                double lb,
                                                double ub,
                                                double zero_tol) {
  return ((cj > zero_tol && a > zero_tol && !isfinite(lhs) && !isfinite(lb)) ||
          (cj > zero_tol && a < -zero_tol && !isfinite(rhs) && !isfinite(lb)) ||
          (cj < -zero_tol && a < -zero_tol && !isfinite(lhs) && !isfinite(ub)) ||
          (cj < -zero_tol && a > zero_tol && !isfinite(rhs) && !isfinite(ub)));
}

__device__ bool _singleton_col_eq_free_from_above(double implied_ub, double ub, double tol) {
  return !isfinite(ub) || implied_ub <= ub + tol;
}

__device__ bool _singleton_col_eq_free_from_below(double implied_lb, double lb, double tol) {
  return !isfinite(lb) || implied_lb >= lb - tol;
}

__device__ void _apply_singleton_col_eq_one_sided_row_update(double* AL,
                                                             double* AU,
                                                             std::int32_t row,
                                                             double a,
                                                             double bound_val,
                                                             bool keep_lower_part) {
  const double shifted_rhs = AU[row] - a * bound_val;
  if (keep_lower_part) {
    AL[row] = shifted_rhs;
    AU[row] = INFINITY;
  } else {
    AL[row] = -INFINITY;
    AU[row] = shifted_rhs;
  }
}

__device__ bool _singleton_col_activity_bounds(std::int32_t row_start,
                                               std::int32_t row_stop,
                                               std::int32_t excluded_col,
                                               const std::uint8_t* keep_col,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               const double* l,
                                               const double* u,
                                               double* rest_min,
                                               double* rest_max) {
  double min_acc = 0.0;
  double max_acc = 0.0;
  for (std::int32_t p = row_start; p < row_stop; ++p) {
    const std::int32_t col = col_val[p];
    if (col == excluded_col || keep_col[col] == std::uint8_t{0}) {
      continue;
    }
    const double a = nz_val[p];
    const double term_min = a >= 0.0 ? (a * l[col]) : (a * u[col]);
    const double term_max = a >= 0.0 ? (a * u[col]) : (a * l[col]);
    if (isnan(term_min) || isnan(term_max)) {
      return false;
    }
    min_acc += term_min;
    max_acc += term_max;
    if (isnan(min_acc) || isnan(max_acc)) {
      return false;
    }
  }
  *rest_min = min_acc;
  *rest_max = max_acc;
  return true;
}

__device__ bool _singleton_col_implied_free_from_above(double a,
                                                       double lhs,
                                                       double rhs,
                                                       double ub,
                                                       double rest_min,
                                                       double rest_max,
                                                       double tol) {
  if (!isfinite(ub)) {
    return true;
  }
  double implied_ub = INFINITY;
  if (a > 0.0 && isfinite(rhs)) {
    implied_ub = (rhs - rest_min) / a;
  } else if (a < 0.0 && isfinite(lhs)) {
    implied_ub = (lhs - rest_max) / a;
  }
  return implied_ub <= ub + tol;
}

__device__ bool _singleton_col_implied_free_from_below(double a,
                                                       double lhs,
                                                       double rhs,
                                                       double lb,
                                                       double rest_min,
                                                       double rest_max,
                                                       double tol) {
  if (!isfinite(lb)) {
    return true;
  }
  double implied_lb = -INFINITY;
  if (a > 0.0 && isfinite(lhs)) {
    implied_lb = (lhs - rest_max) / a;
  } else if (a < 0.0 && isfinite(rhs)) {
    implied_lb = (rhs - rest_min) / a;
  }
  return implied_lb >= lb - tol;
}

__device__ double _singleton_col_active_side(double cj,
                                             double a,
                                             double lhs,
                                             double rhs,
                                             double zero_tol) {
  if ((cj > zero_tol && a > 0.0) || (cj < -zero_tol && a < 0.0)) {
    return lhs;
  }
  if ((cj > zero_tol && a < 0.0) || (cj < -zero_tol && a > 0.0)) {
    return rhs;
  }
  return isfinite(lhs) ? lhs : rhs;
}

__device__ std::uint8_t _singleton_col_ineq_action(double cj,
                                                   double a,
                                                   double lhs,
                                                   double rhs,
                                                   bool impl_free_from_above,
                                                   bool impl_free_from_below,
                                                   double zero_tol,
                                                   double* action_side) {
  if (impl_free_from_above && impl_free_from_below) {
    *action_side = _singleton_col_active_side(cj, a, lhs, rhs, zero_tol);
    if (isfinite(*action_side)) {
      return SINGLETON_COL_INEQ_ELIMINATE;
    }
    return SINGLETON_COL_INEQ_NO_ACTION;
  }
  const bool tighten_lhs_to_rhs =
      ((cj < -zero_tol && a > 0.0 && impl_free_from_above) ||
       (cj > zero_tol && a < 0.0 && impl_free_from_below)) &&
      isfinite(rhs);
  if (tighten_lhs_to_rhs) {
    *action_side = rhs;
    return SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS;
  }
  const bool tighten_rhs_to_lhs =
      ((cj > zero_tol && a > 0.0 && impl_free_from_below) ||
       (cj < -zero_tol && a < 0.0 && impl_free_from_above)) &&
      isfinite(lhs);
  if (tighten_rhs_to_lhs) {
    *action_side = lhs;
    return SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS;
  }
  *action_side = 0.0;
  return SINGLETON_COL_INEQ_NO_ACTION;
}

__global__ void _kernel_singleton_col_row_owner(std::int32_t* status_flag,
                                                std::int32_t* row_owner,
                                                double* rest_min_cache,
                                                double* rest_max_cache,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* keep_col,
                                                const std::uint8_t* singleton_mask,
                                                const std::int32_t* support_row,
                                                const double* support_val,
                                                const double* c_cur,
                                                const double* l_cur,
                                                const double* u_cur,
                                                const double* AL_cur,
                                                const double* AU_cur,
                                                const std::int32_t* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                double tol,
                                                double zero_tol,
                                                std::uint8_t rule_mode,
                                                std::int32_t m,
                                                std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0} || singleton_mask[j] == std::uint8_t{0}) {
      return;
    }
    const std::int32_t row = support_row[j];
    if (row < 0 || row >= m || keep_row[row] == std::uint8_t{0}) {
      return;
    }
    const double a = support_val[j];
    if (fabs(a) <= zero_tol) {
      return;
    }

    const std::int32_t row_start = row_ptr[row];
    const std::int32_t row_stop = row_ptr[row + 1];
    std::int32_t live_row_nnz = 0;
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] != std::uint8_t{0}) {
        ++live_row_nnz;
        if (live_row_nnz > 1) {
          break;
        }
      }
    }
    if (live_row_nnz <= 1) {
      return;
    }

    const double lhs = AL_cur[row];
    const double rhs = AU_cur[row];
    const double cj = c_cur[j];
    if (rule_mode != SINGLETON_COL_RULE_EQ &&
        _singleton_col_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    double rest_min = 0.0;
    double rest_max = 0.0;
    if (!_singleton_col_activity_bounds(row_start, row_stop, j, keep_col, col_val, nz_val, l_cur, u_cur, &rest_min, &rest_max)) {
      return;
    }

    const bool is_eq_row = isfinite(lhs) && isfinite(rhs) && fabs(lhs - rhs) <= tol;
    if (is_eq_row) {
      if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
        return;
      }
      const double x1 = (rhs - rest_min) / a;
      const double x2 = (rhs - rest_max) / a;
      const double implied_lb = fmin(x1, x2);
      const double implied_ub = fmax(x1, x2);
      const bool impl_free_from_above = _singleton_col_eq_free_from_above(implied_ub, u_cur[j], tol);
      const bool impl_free_from_below = _singleton_col_eq_free_from_below(implied_lb, l_cur[j], tol);
      if (!(impl_free_from_above || impl_free_from_below)) {
        return;
      }
    } else {
      if (rule_mode == SINGLETON_COL_RULE_EQ) {
        return;
      }
      const bool impl_free_from_above =
          _singleton_col_implied_free_from_above(a, lhs, rhs, u_cur[j], rest_min, rest_max, tol);
      const bool impl_free_from_below =
          _singleton_col_implied_free_from_below(a, lhs, rhs, l_cur[j], rest_min, rest_max, tol);
      if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
        if (!(impl_free_from_above && impl_free_from_below)) {
          return;
        }
        const double chosen = _singleton_col_active_side(cj, a, lhs, rhs, zero_tol);
        if (!isfinite(chosen)) {
          return;
        }
      } else {
        double action_side = 0.0;
        if (_singleton_col_ineq_action(cj, a, lhs, rhs, impl_free_from_above, impl_free_from_below, zero_tol, &action_side) ==
            SINGLETON_COL_INEQ_NO_ACTION) {
          return;
        }
      }
    }
    rest_min_cache[j] = rest_min;
    rest_max_cache[j] = rest_max;
    atomicMin(&row_owner[row], j);
  }
}

__global__ void _kernel_process_singleton_cols(std::int32_t* status_flag,
                                               std::uint8_t* row_delete,
                                               std::uint8_t* row_lhs_change,
                                               std::uint8_t* row_rhs_change,
                                               std::uint8_t* col_delete,
                                               std::int32_t* pair_row,
                                               double* chosen_side,
                                               double* c_new,
                                               double* AL_new,
                                               double* AU_new,
                                               double* obj_contrib,
                                               const std::int32_t* row_owner,
                                               const std::uint8_t* keep_row,
                                               const std::uint8_t* keep_col,
                                               const std::uint8_t* singleton_mask,
                                               const std::int32_t* support_row,
                                               const double* support_val,
                                               const double* c_cur,
                                               const double* l_cur,
                                               const double* u_cur,
                                               const std::int32_t* row_ptr,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               double tol,
                                               double zero_tol,
                                               std::uint8_t rule_mode,
                                               std::int32_t m,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0} || singleton_mask[j] == std::uint8_t{0}) {
      return;
    }
    const std::int32_t row = support_row[j];
    if (row < 0 || row >= m || keep_row[row] == std::uint8_t{0}) {
      return;
    }
    if (row_owner[row] != j) {
      return;
    }
    const double a = support_val[j];
    if (fabs(a) <= zero_tol) {
      return;
    }
    const double lhs = AL_new[row];
    const double rhs = AU_new[row];
    const double cj = c_cur[j];
    if (rule_mode != SINGLETON_COL_RULE_EQ &&
        _singleton_col_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    const std::int32_t row_start = row_ptr[row];
    const std::int32_t row_stop = row_ptr[row + 1];
    const double rest_min = chosen_side[j];
    const double rest_max = obj_contrib[j];

    const bool is_eq_row = isfinite(lhs) && isfinite(rhs) && fabs(lhs - rhs) <= tol;
    if (is_eq_row) {
      if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
        return;
      }
      const double x1 = (rhs - rest_min) / a;
      const double x2 = (rhs - rest_max) / a;
      const double implied_lb = fmin(x1, x2);
      const double implied_ub = fmax(x1, x2);
      const bool impl_free_from_above = _singleton_col_eq_free_from_above(implied_ub, u_cur[j], tol);
      const bool impl_free_from_below = _singleton_col_eq_free_from_below(implied_lb, l_cur[j], tol);
      if (!(impl_free_from_above || impl_free_from_below)) {
        return;
      }
      chosen_side[j] = rhs;
      obj_contrib[j] = cj * rhs / a;
      for (std::int32_t p = row_start; p < row_stop; ++p) {
        const std::int32_t col = col_val[p];
        if (col == j || keep_col[col] == std::uint8_t{0}) {
          continue;
        }
        atomicAdd(&c_new[col], -(cj * nz_val[p] / a));
      }
      if (impl_free_from_above && impl_free_from_below) {
        row_delete[row] = std::uint8_t{1};
      } else if (impl_free_from_above) {
        _apply_singleton_col_eq_one_sided_row_update(AL_new, AU_new, row, a, l_cur[j], a < 0.0);
      } else {
        _apply_singleton_col_eq_one_sided_row_update(AL_new, AU_new, row, a, u_cur[j], a > 0.0);
      }
      col_delete[j] = std::uint8_t{1};
      pair_row[j] = row;
      atomicMax(&status_flag[1], 1);
      return;
    }

    if (rule_mode == SINGLETON_COL_RULE_EQ) {
      return;
    }

    const bool impl_free_from_above =
        _singleton_col_implied_free_from_above(a, lhs, rhs, u_cur[j], rest_min, rest_max, tol);
    const bool impl_free_from_below =
        _singleton_col_implied_free_from_below(a, lhs, rhs, l_cur[j], rest_min, rest_max, tol);
    if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
      if (!(impl_free_from_above && impl_free_from_below)) {
        return;
      }
      const double action_side = _singleton_col_active_side(cj, a, lhs, rhs, zero_tol);
      if (!isfinite(action_side)) {
        return;
      }
      if (!isfinite(lhs) || fabs(lhs - action_side) > tol) {
        row_lhs_change[row] = std::uint8_t{1};
      }
      if (!isfinite(rhs) || fabs(rhs - action_side) > tol) {
        row_rhs_change[row] = std::uint8_t{1};
      }
      AL_new[row] = action_side;
      AU_new[row] = action_side;
      atomicMax(&status_flag[2], 1);
    }
  }
}

__global__ void _kernel_count_singleton_col_support_eq(
    std::int32_t* counts,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::uint8_t* singleton_mask,
    const std::int32_t* support_row,
    const double* AL,
    const double* AU,
    double tol,
    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && keep_col[j] != std::uint8_t{0} && singleton_mask[j] != std::uint8_t{0}) {
    const std::int32_t row = support_row[j];
    if (row >= 0 && keep_row[row] != std::uint8_t{0}) {
      atomicAdd(&counts[0], 1);
      const double lhs = AL[row];
      const double rhs = AU[row];
      if (isfinite(lhs) && isfinite(rhs) && fabs(rhs - lhs) <= tol) {
        atomicAdd(&counts[1], 1);
      }
    }
  }
}

__global__ void _kernel_apply_singleton_cols(std::uint8_t* keep_row,
                                             std::uint8_t* keep_col,
                                             double* c_cur,
                                             double* AL_cur,
                                             double* AU_cur,
                                             const std::uint8_t* row_delete,
                                             const std::uint8_t* col_delete,
                                             const double* c_new,
                                             const double* AL_new,
                                             const double* AU_new,
                                             std::int32_t m,
                                             std::int32_t n) {
  const std::int32_t q = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q < m) {
    if (row_delete[q] != std::uint8_t{0}) {
      keep_row[q] = std::uint8_t{0};
    }
    AL_cur[q] = AL_new[q];
    AU_cur[q] = AU_new[q];
  }
  if (q < n) {
    if (col_delete[q] != std::uint8_t{0}) {
      keep_col[q] = std::uint8_t{0};
    }
    c_cur[q] = c_new[q];
  }
}

__global__ void _kernel_zero_non_deleted_obj_contrib(double* obj_contrib,
                                                     const std::uint8_t* col_delete,
                                                     std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n && col_delete[col] == std::uint8_t{0}) {
    obj_contrib[col] = 0.0;
  }
}

__global__ void _kernel_singleton_subcol_tape_counts(std::int32_t* selected_scan,
                                                     std::int32_t* support_scan,
                                                     const std::uint8_t* col_delete,
                                                     const std::int32_t* pair_row,
                                                     const double* singleton_val,
                                                     const std::int32_t* row_ptr,
                                                     double zero_tol,
                                                     std::int32_t m,
                                                     std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    const std::int32_t row = pair_row[col];
    const bool selected = col_delete[col] != std::uint8_t{0} && row >= 0 && row < m &&
                          fabs(singleton_val[col]) > zero_tol;
    selected_scan[col] = selected ? 1 : 0;
    const std::int32_t support_count = selected ? row_ptr[row + 1] - row_ptr[row] - 1 : 0;
    support_scan[col] = support_count;
  }
}

__global__ void _kernel_singleton_pack_subcol_tape(std::int32_t* types,
                                                   std::int32_t* index_starts,
                                                   std::int32_t* value_starts,
                                                   std::uint8_t* dual_modes,
                                                   std::int32_t* indices,
                                                   double* vals,
                                                   const std::int32_t* selected_scan,
                                                   const std::int32_t* support_scan,
                                                   const std::uint8_t* col_delete,
                                                   const std::uint8_t* row_delete,
                                                   const std::uint8_t* keep_col,
                                                   const std::int32_t* pair_row,
                                                   const double* singleton_val,
                                                   const double* chosen_side,
                                                   const double* c,
                                                   const double* l,
                                                   const double* u,
                                                   const std::int32_t* row_ptr,
                                                   const std::int32_t* col_val,
                                                   const double* nz_val,
                                                   std::int32_t m,
                                                   std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || col_delete[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t row = pair_row[col];
  if (row < 0 || row >= m) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const std::int32_t record = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || record < 0) {
    return;
  }

  const std::int32_t support_start = col == 0 ? 0 : support_scan[col - 1];
  const std::int32_t support_stop = support_scan[col];
  const std::int32_t support_count = support_stop - support_start;
  const std::int32_t index_start = 3 * record + support_start;
  const std::int32_t value_start = 6 * record + support_start;
  types[record] = static_cast<std::int32_t>(PostsolveReductionType::SubCol);
  index_starts[record] = index_start;
  value_starts[record] = value_start;
  dual_modes[record] = static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
  indices[index_start] = col;
  indices[index_start + 1] = row;
  indices[index_start + 2] = support_count;
  vals[value_start + 1] = chosen_side[col];
  vals[value_start + 2] = l[col];
  vals[value_start + 3] = u[col];
  vals[value_start + 4] = c[col];
  vals[value_start + 5] = row_delete[row] != std::uint8_t{0} ? 1.0 : 0.0;
  std::int32_t out = 0;
  for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    const std::int32_t row_col = col_val[p];
    const double coeff = nz_val[p];
    if (row_col != col && keep_col[row_col] != std::uint8_t{0} && out < support_count) {
      indices[index_start + 3 + out] = row_col;
      vals[value_start + 6 + out] = coeff;
      ++out;
    }
  }
  vals[value_start] = singleton_val[col];
}

void append_subcol_tape_compacted_from_device(PostsolveTapeGpu& tape,
                                              PostsolveTape& tape_cpu,
                                              const std::uint8_t* col_delete,
                                              const std::uint8_t* row_delete,
                                              const std::uint8_t* keep_col,
                                              const std::int32_t* pair_row,
                                              const double* singleton_val,
                                              const double* chosen_side,
                                              const double* c,
                                              const double* l,
                                              const double* u,
                                              const DeviceCsrMatrix& A,
                                              double zero_tol,
                                              const char* context) {
  const std::int32_t m = A.rows;
  const std::int32_t n = A.cols;
  if (n <= 0) {
    return;
  }

  std::int32_t* selected_scan = nullptr;
  std::int32_t* support_scan = nullptr;
  throw_if_cuda_error(cudaMallocAsync(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n), nullptr),
                      context);
  throw_if_cuda_error(cudaMallocAsync(&support_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n), nullptr),
                      context);

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_subcol_tape_counts<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      selected_scan,
      support_scan,
      col_delete,
      pair_row,
      singleton_val,
      A.rowPtr,
      zero_tol,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_subcol_tape_counts");
  inclusive_scan_i32(selected_scan, n, context);
  inclusive_scan_i32(support_scan, n, context);

  std::int32_t counts[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(&counts[0], selected_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(&counts[1], support_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  const std::int32_t record_count = counts[0];
  const std::int32_t support_nnz = counts[1];
  if (record_count <= 0) {
    cudaFreeAsync(selected_scan, nullptr);
    cudaFreeAsync(support_scan, nullptr);
    return;
  }

  PostsolveTapeGpu built;
  built.record_count = record_count;
  built.index_count = 3 * record_count + support_nnz;
  built.value_count = 6 * record_count + support_nnz;
  throw_if_cuda_error(cudaMalloc(&built.types,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&built.index_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1)),
                      context);
  throw_if_cuda_error(cudaMalloc(&built.value_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1)),
                      context);
  throw_if_cuda_error(cudaMalloc(&built.dual_modes,
                                 static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&built.indices,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(built.index_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&built.vals,
                                 sizeof(double) * static_cast<std::size_t>(built.value_count)),
                      context);

  _kernel_singleton_pack_subcol_tape<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      built.types,
      built.index_starts,
      built.value_starts,
      built.dual_modes,
      built.indices,
      built.vals,
      selected_scan,
      support_scan,
      col_delete,
      row_delete,
      keep_col,
      pair_row,
      singleton_val,
      chosen_side,
      c,
      l,
      u,
      A.rowPtr,
      A.colVal,
      A.nzVal,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_pack_subcol_tape");
  throw_if_cuda_error(cudaMemcpy(built.index_starts + record_count, &built.index_count,
                                 sizeof(std::int32_t), cudaMemcpyHostToDevice),
                      context);
  throw_if_cuda_error(cudaMemcpy(built.value_starts + record_count, &built.value_count,
                                 sizeof(std::int32_t), cudaMemcpyHostToDevice),
                      context);

  tape_cpu.types.resize(static_cast<std::size_t>(built.record_count));
  tape_cpu.index_starts.resize(static_cast<std::size_t>(built.record_count + 1));
  tape_cpu.value_starts.resize(static_cast<std::size_t>(built.record_count + 1));
  tape_cpu.dual_modes.resize(static_cast<std::size_t>(built.record_count));
  tape_cpu.indices.resize(static_cast<std::size_t>(built.index_count));
  tape_cpu.vals.resize(static_cast<std::size_t>(built.value_count));
  throw_if_cuda_error(cudaMemcpy(tape_cpu.types.data(), built.types,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(built.record_count),
                                 cudaMemcpyDeviceToHost), context);
  throw_if_cuda_error(cudaMemcpy(tape_cpu.index_starts.data(), built.index_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(built.record_count + 1),
                                 cudaMemcpyDeviceToHost), context);
  throw_if_cuda_error(cudaMemcpy(tape_cpu.value_starts.data(), built.value_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(built.record_count + 1),
                                 cudaMemcpyDeviceToHost), context);
  throw_if_cuda_error(cudaMemcpy(tape_cpu.dual_modes.data(), built.dual_modes,
                                 static_cast<std::size_t>(built.record_count),
                                 cudaMemcpyDeviceToHost), context);
  throw_if_cuda_error(cudaMemcpy(tape_cpu.indices.data(), built.indices,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(built.index_count),
                                 cudaMemcpyDeviceToHost), context);
  throw_if_cuda_error(cudaMemcpy(tape_cpu.vals.data(), built.vals,
                                 sizeof(double) * static_cast<std::size_t>(built.value_count),
                                 cudaMemcpyDeviceToHost), context);

  cudaFreeAsync(selected_scan, nullptr);
  cudaFreeAsync(support_scan, nullptr);
  tape = std::move(built);
}

void apply_rule_singleton_cols_mode(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    const PresolveStatsGpu& stats,
                                    const PresolveParams& pparams,
                                    std::uint8_t rule_mode) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* singleton_counts_device = nullptr;
  throw_if_cuda_error(cudaMallocAsync(&singleton_counts_device, sizeof(std::int32_t) * 2, nullptr),
                      "cudaMalloc singleton_cols counts");
  throw_if_cuda_error(cudaMemset(singleton_counts_device, 0, sizeof(std::int32_t) * 2),
                      "cudaMemset singleton_cols counts");
  _kernel_count_singleton_col_support_eq<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      singleton_counts_device,
      plan.keep_row_mask,
      plan.keep_col_mask,
      stats.singleton_col_mask,
      stats.singleton_col_row,
      plan.new_AL,
      plan.new_AU,
      pparams.bound_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_count_singleton_col_support_eq");
  std::int32_t singleton_counts[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(singleton_counts,
                                 singleton_counts_device,
                                 sizeof(singleton_counts),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy singleton_cols counts");
  cudaFreeAsync(singleton_counts_device, nullptr);
  const std::int32_t singleton_count = singleton_counts[0];
  const std::int32_t eq_support_count = singleton_counts[1];
  if (singleton_count == 0 ||
      (rule_mode == SINGLETON_COL_RULE_EQ && eq_support_count == 0) ||
      (rule_mode == SINGLETON_COL_RULE_DUAL_INFER && singleton_count == eq_support_count)) {
    return;
  }

  std::int32_t* status_flag = nullptr;
  std::int32_t* row_owner = nullptr;
  std::uint8_t* col_delete = nullptr;
  std::uint8_t* row_delete = nullptr;
  std::uint8_t* row_lhs_change = nullptr;
  std::uint8_t* row_rhs_change = nullptr;
  std::int32_t* pair_row = nullptr;
  double* chosen_side = nullptr;
  double* c_new = nullptr;
  double* AL_new = nullptr;
  double* AU_new = nullptr;
  double* obj_contrib = nullptr;

  throw_if_cuda_error(cudaMallocAsync(&status_flag, sizeof(std::int32_t) * 3, nullptr), "cudaMallocAsync singleton_cols status_flag");
  throw_if_cuda_error(cudaMallocAsync(&row_owner, sizeof(std::int32_t) * static_cast<std::size_t>(m), nullptr), "cudaMallocAsync singleton_cols row_owner");
  throw_if_cuda_error(cudaMallocAsync(&col_delete, static_cast<std::size_t>(n), nullptr), "cudaMallocAsync singleton_cols col_delete");
  throw_if_cuda_error(cudaMallocAsync(&row_delete, static_cast<std::size_t>(m), nullptr), "cudaMallocAsync singleton_cols row_delete");
  throw_if_cuda_error(cudaMallocAsync(&row_lhs_change, static_cast<std::size_t>(m), nullptr), "cudaMallocAsync singleton_cols row_lhs_change");
  throw_if_cuda_error(cudaMallocAsync(&row_rhs_change, static_cast<std::size_t>(m), nullptr), "cudaMallocAsync singleton_cols row_rhs_change");
  throw_if_cuda_error(cudaMallocAsync(&pair_row, sizeof(std::int32_t) * static_cast<std::size_t>(n), nullptr), "cudaMallocAsync singleton_cols pair_row");
  throw_if_cuda_error(cudaMallocAsync(&chosen_side, sizeof(double) * static_cast<std::size_t>(n), nullptr), "cudaMallocAsync singleton_cols chosen_side");
  throw_if_cuda_error(cudaMallocAsync(&c_new, sizeof(double) * static_cast<std::size_t>(n), nullptr), "cudaMallocAsync singleton_cols c_new");
  throw_if_cuda_error(cudaMallocAsync(&AL_new, sizeof(double) * static_cast<std::size_t>(m), nullptr), "cudaMallocAsync singleton_cols AL_new");
  throw_if_cuda_error(cudaMallocAsync(&AU_new, sizeof(double) * static_cast<std::size_t>(m), nullptr), "cudaMallocAsync singleton_cols AU_new");
  throw_if_cuda_error(cudaMallocAsync(&obj_contrib, sizeof(double) * static_cast<std::size_t>(n), nullptr), "cudaMallocAsync singleton_cols obj_contrib");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset singleton_cols status_flag");
  throw_if_cuda_error(cudaMemset(col_delete, 0, static_cast<std::size_t>(n)), "cudaMemset singleton_cols col_delete");
  throw_if_cuda_error(cudaMemset(row_delete, 0, static_cast<std::size_t>(m)), "cudaMemset singleton_cols row_delete");
  throw_if_cuda_error(cudaMemset(row_lhs_change, 0, static_cast<std::size_t>(m)), "cudaMemset singleton_cols row_lhs_change");
  throw_if_cuda_error(cudaMemset(row_rhs_change, 0, static_cast<std::size_t>(m)), "cudaMemset singleton_cols row_rhs_change");
  throw_if_cuda_error(cudaMemset(chosen_side, 0, sizeof(double) * static_cast<std::size_t>(n)), "cudaMemset singleton_cols chosen_side");
  throw_if_cuda_error(cudaMemset(obj_contrib, 0, sizeof(double) * static_cast<std::size_t>(n)), "cudaMemset singleton_cols obj_contrib");
  throw_if_cuda_error(cudaMemcpy(c_new, plan.new_c, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy singleton_cols c_new");
  throw_if_cuda_error(cudaMemcpy(AL_new, plan.new_AL, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy singleton_cols AL_new");
  throw_if_cuda_error(cudaMemcpy(AU_new, plan.new_AU, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy singleton_cols AU_new");

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_fill_i32<<<blocks_m, GPU_PRESOLVE_THREADS>>>(row_owner, INT_MAX, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 singleton row_owner");

  _kernel_singleton_col_row_owner<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      row_owner,
      chosen_side,
      obj_contrib,
      plan.keep_row_mask,
      plan.keep_col_mask,
      stats.singleton_col_mask,
      stats.singleton_col_row,
      stats.singleton_col_val,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      plan.new_AL,
      plan.new_AU,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.bound_tol,
      pparams.zero_tol,
      rule_mode,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_col_row_owner");

  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy singleton_cols status after owner");
  if (status[0] != 0) {
    plan.has_unbounded = true;
  } else {
    _kernel_process_singleton_cols<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        status_flag,
        row_delete,
        row_lhs_change,
        row_rhs_change,
        col_delete,
        pair_row,
        chosen_side,
        c_new,
        AL_new,
        AU_new,
        obj_contrib,
        row_owner,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        plan.new_c,
        plan.new_l,
        plan.new_u,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        rule_mode,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_process_singleton_cols");
    throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                        "cudaMemcpy singleton_cols status after process");
    if (status[0] != 0) {
      plan.has_unbounded = true;
    } else if (status[1] != 0 || status[2] != 0) {
      _kernel_zero_non_deleted_obj_contrib<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          obj_contrib, col_delete, n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_zero_non_deleted_obj_contrib");
      const std::int32_t max_mn = m > n ? m : n;
      const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
      if (status[1] != 0 && rule_mode == SINGLETON_COL_RULE_EQ && pparams.record_postsolve_tape) {
        append_subcol_tape_compacted_from_device(plan.tape_gpu,
                                                 plan.tape,
                                                 col_delete,
                                                 row_delete,
                                                 plan.keep_col_mask,
                                                 pair_row,
                                                 stats.singleton_col_val,
                                                 chosen_side,
                                                 plan.new_c,
                                                 plan.new_l,
                                                 plan.new_u,
                                                 lp.A,
                                                 pparams.zero_tol,
                                                 "cudaMemcpy singleton compact SubCol tape");
        plan.tape_gpu_mirrors_cpu = true;
      }
      if (status[2] != 0 && rule_mode == SINGLETON_COL_RULE_DUAL_INFER &&
          pparams.record_postsolve_tape) {
        append_eq_to_ineq_tape_from_device(plan.tape,
                                           row_lhs_change,
                                           row_rhs_change,
                                           m,
                                           "cudaMemcpy singleton eq-to-ineq tape");
      }
      _kernel_apply_singleton_cols<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
          plan.keep_row_mask,
          plan.keep_col_mask,
          plan.new_c,
          plan.new_AL,
          plan.new_AU,
          row_delete,
          col_delete,
          c_new,
          AL_new,
          AU_new,
          m,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_singleton_cols");
      throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_singleton_cols synchronize");
      const double obj_delta = sum_device_double(obj_contrib, n, "cudaMemcpy singleton_cols obj_contrib");
      plan.obj_constant_delta += obj_delta;
      plan.has_row_action = true;
      plan.has_col_action = plan.has_col_action || status[1] != 0;
      plan.has_change = true;
    }
  }

  cudaFreeAsync(status_flag, nullptr);
  cudaFreeAsync(row_owner, nullptr);
  cudaFreeAsync(col_delete, nullptr);
  cudaFreeAsync(row_delete, nullptr);
  cudaFreeAsync(row_lhs_change, nullptr);
  cudaFreeAsync(row_rhs_change, nullptr);
  cudaFreeAsync(pair_row, nullptr);
  cudaFreeAsync(chosen_side, nullptr);
  cudaFreeAsync(c_new, nullptr);
  cudaFreeAsync(AL_new, nullptr);
  cudaFreeAsync(AU_new, nullptr);
  cudaFreeAsync(obj_contrib, nullptr);
}

}  // namespace

void apply_rule_singleton_cols_eq(PresolvePlanGpu& plan,
                                  const LPInfoGpu& lp,
                                  const PresolveStatsGpu& stats,
                                  const PresolveParams& pparams) {
  apply_rule_singleton_cols_mode(plan, lp, stats, pparams, SINGLETON_COL_RULE_EQ);
}

void apply_rule_singleton_cols_dual_infer(PresolvePlanGpu& plan,
                                          const LPInfoGpu& lp,
                                          const PresolveStatsGpu& stats,
                                          const PresolveParams& pparams) {
  apply_rule_singleton_cols_mode(plan, lp, stats, pparams, SINGLETON_COL_RULE_DUAL_INFER);
}

}  // namespace gpu_presolver::presolve
