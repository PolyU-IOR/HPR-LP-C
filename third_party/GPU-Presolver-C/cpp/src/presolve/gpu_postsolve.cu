#include "gpu_presolver/presolve/gpu_postsolve.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

__device__ double _postsolve_project_single_column_dual_from_original(double* x_org,
                                                                      double* z_org,
                                                                      const std::int32_t* AT_rowPtr,
                                                                      const std::int32_t* AT_colVal,
                                                                      const double* AT_nzVal,
                                                                      const double* y_org,
                                                                      const double* c,
                                                                      const double* l,
                                                                      const double* u,
                                                                      std::int32_t col,
                                                                      double tol);

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
T* copy_vector_to_device(const std::vector<T>& values, const char* context) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  throw_if_cuda_error(cudaMalloc(&device, sizeof(T) * values.size()), context);
  throw_if_cuda_error(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
                      context);
  return device;
}

__global__ void _kernel_structural_restore_splits(double* x_org,
                                                  const std::int32_t* t_cols,
                                                  const std::int32_t* e_cols,
                                                  const double* rhos,
                                                  std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= k) {
    return;
  }
  const std::int32_t t_idx = t_cols[i];
  const std::int32_t e_idx = e_cols[i];
  const double rho = rhos[i];
  const double t_new = x_org[t_idx];
  const double e_new = x_org[e_idx];
  x_org[t_idx] = t_new + e_new;
  x_org[e_idx] = (t_new - e_new) / rho;
}

__global__ void _kernel_structural_restore_outer_pairs(double* x_org,
                                                       const std::int32_t* bound_cols,
                                                       const std::int32_t* free_cols,
                                                       std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < k) {
    x_org[free_cols[i]] = x_org[bound_cols[i]];
  }
}

__global__ void _kernel_structural_restore_linked_slacks(double* x_org,
                                                         const std::int32_t* slack_cols,
                                                         const std::int32_t* t_cols,
                                                         const double* factors,
                                                         std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < k) {
    x_org[slack_cols[i]] = factors[i] * x_org[t_cols[i]];
  }
}

__global__ void _kernel_structural_restore_max_slack(double* x_org,
                                                     std::int32_t slack_col,
                                                     const std::int32_t* t_cols,
                                                     const double* factors,
                                                     std::int32_t count,
                                                     double lower_bound) {
  const int tid = threadIdx.x;
  const int lane_count = blockDim.x;
  double partial = lower_bound;
  for (std::int32_t j = tid; j < count; j += lane_count) {
    partial = fmax(partial, factors[j] * x_org[t_cols[j]]);
  }

  __shared__ double scratch[GPU_PRESOLVE_THREADS];
  scratch[tid] = partial;
  __syncthreads();

  for (int offset = lane_count >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] = fmax(scratch[tid], scratch[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_org[slack_col] = scratch[0];
  }
}

__global__ void _kernel_scatter_reduced_to_original(double* dst,
                                                    const double* src,
                                                    const std::int32_t* red2org,
                                                    std::int32_t n_red) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n_red) {
    dst[red2org[i]] = src[i];
  }
}

__global__ void _kernel_mark_retrieved(std::uint8_t* mask,
                                       const std::int32_t* red2org,
                                       std::int32_t n_red) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n_red) {
    mask[red2org[i]] = std::uint8_t{1};
  }
}

__global__ void _kernel_fill_unretrieved_duals_from_original(double* z_org,
                                                            std::uint8_t* z_retrieved,
                                                            const double* y_org,
                                                            const std::int32_t* AT_rowPtr,
                                                            const std::int32_t* AT_colVal,
                                                            const double* AT_nzVal,
                                                            const double* c,
                                                            std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || z_retrieved[col] != std::uint8_t{0}) {
    return;
  }
  double aty = 0.0;
  if (y_org != nullptr) {
    for (std::int32_t p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
      aty += AT_nzVal[p] * y_org[AT_colVal[p]];
    }
  }
  z_org[col] = c[col] - aty;
  z_retrieved[col] = std::uint8_t{1};
}

__global__ void _kernel_project_column_duals_from_original(double* x_org,
                                                           double* z_org,
                                                           const std::uint8_t* protected_cols,
                                                           const std::int32_t* AT_rowPtr,
                                                           const std::int32_t* AT_colVal,
                                                           const double* AT_nzVal,
                                                           const double* y_org,
                                                           const double* c,
                                                           const double* l,
                                                           const double* u,
                                                           std::int32_t n,
                                                           double tol) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || (protected_cols != nullptr && protected_cols[col] != std::uint8_t{0})) {
    return;
  }
  _postsolve_project_single_column_dual_from_original(
      x_org, z_org, AT_rowPtr, AT_colVal, AT_nzVal, y_org, c, l, u, col, tol);
}

__global__ void _kernel_restore_fixed_values(double* x_org,
                                             const std::int32_t* fixed_idx,
                                             const double* fixed_val,
                                             std::int32_t fixed_count) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < fixed_count) {
    x_org[fixed_idx[i]] = fixed_val[i];
  }
}

__device__ double _choose_merged_source_value(double lo, double hi, double tol) {
  if (isfinite(lo) && isfinite(hi)) {
    return 0.5 * (lo + hi);
  }
  if (isfinite(lo)) {
    if (0.0 > lo + tol) {
      return 0.0;
    }
    return lo + fmax(1.0, fabs(lo) * fmax(tol, 1.0e-12));
  }
  if (isfinite(hi)) {
    if (0.0 < hi - tol) {
      return 0.0;
    }
    return hi - fmax(1.0, fabs(hi) * fmax(tol, 1.0e-12));
  }
  return 0.0;
}

__device__ double _parallel_col_aggregate_lower_bound(double target_l,
                                                      double source_l,
                                                      double source_u,
                                                      double ratio) {
  if (ratio > 0.0) {
    if (isfinite(target_l) && isfinite(source_l)) {
      return target_l + ratio * source_l;
    }
  } else if (isfinite(target_l) && isfinite(source_u)) {
    return target_l + ratio * source_u;
  }
  return -INFINITY;
}

__device__ double _parallel_col_aggregate_upper_bound(double target_u,
                                                      double source_l,
                                                      double source_u,
                                                      double ratio) {
  if (ratio > 0.0) {
    if (isfinite(target_u) && isfinite(source_u)) {
      return target_u + ratio * source_u;
    }
  } else if (isfinite(target_u) && isfinite(source_l)) {
    return target_u + ratio * source_l;
  }
  return INFINITY;
}

__device__ void _recover_parallel_col_primal(double y_val,
                                             double z_to,
                                             double ratio,
                                             double source_l,
                                             double source_u,
                                             double target_l,
                                             double target_u,
                                             double tol,
                                             double* x_from,
                                             double* x_to) {
  double lo = source_l;
  double hi = source_u;
  if (isfinite(target_u)) {
    const double bound = (y_val - target_u) / ratio;
    if (ratio > 0.0) {
      lo = fmax(lo, bound);
    } else {
      hi = fmin(hi, bound);
    }
  }
  if (isfinite(target_l)) {
    const double bound = (y_val - target_l) / ratio;
    if (ratio > 0.0) {
      hi = fmin(hi, bound);
    } else {
      lo = fmax(lo, bound);
    }
  }

  const double lower = _parallel_col_aggregate_lower_bound(target_l, source_l, source_u, ratio);
  const double upper = _parallel_col_aggregate_upper_bound(target_u, source_l, source_u, ratio);
  if (z_to > tol && isfinite(lower) && fabs(y_val - lower) <= tol) {
    *x_from = ratio > 0.0 ? source_l : source_u;
  } else if (z_to < -tol && isfinite(upper) && fabs(y_val - upper) <= tol) {
    *x_from = ratio > 0.0 ? source_u : source_l;
  } else {
    *x_from = _choose_merged_source_value(lo, hi, tol);
  }
  *x_to = y_val - ratio * (*x_from);
}

__device__ bool _postsolve_is_finite(double x) {
  return x > -INFINITY && x < INFINITY;
}

__device__ bool _postsolve_at_lower(double x, double l, double tol) {
  return _postsolve_is_finite(l) && x <= l + tol;
}

__device__ bool _postsolve_at_upper(double x, double u, double tol) {
  return _postsolve_is_finite(u) && x >= u - tol;
}

__device__ double _clamp_with_inf(double x, double lo, double hi) {
  double y = x;
  if (_postsolve_is_finite(lo)) {
    y = fmax(y, lo);
  }
  if (_postsolve_is_finite(hi)) {
    y = fmin(y, hi);
  }
  return y;
}

__device__ double _clamp_row_refinement_value(double x, double lo, double hi) {
  double y = x;
  if (_postsolve_is_finite(lo) && y < lo) {
    y = lo;
  }
  if (_postsolve_is_finite(hi) && y > hi) {
    y = hi;
  }
  return y;
}

__device__ void _parallel_row_scaled_interval(double lower,
                                              double upper,
                                              double ratio,
                                              double* scaled_l,
                                              double* scaled_u) {
  const double raw_l = lower * ratio;
  const double raw_u = upper * ratio;
  if (ratio >= 0.0) {
    *scaled_l = raw_l;
    *scaled_u = raw_u;
  } else {
    *scaled_l = raw_u;
    *scaled_u = raw_l;
  }
}

__device__ bool _parallel_row_interval_contains(double outer_lo,
                                                double outer_hi,
                                                double inner_lo,
                                                double inner_hi,
                                                double tol) {
  return outer_lo <= inner_lo + tol && inner_hi <= outer_hi + tol;
}

__device__ double _parallel_row_activity(std::int32_t row,
                                         const std::int32_t* A_rowPtr,
                                         const std::int32_t* A_colVal,
                                         const double* A_nzVal,
                                         const double* x_org) {
  double activity = 0.0;
  for (std::int32_t p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    activity += A_nzVal[p] * x_org[A_colVal[p]];
  }
  return activity;
}

enum : std::uint8_t {
  PARALLEL_ROW_STATE_ZERO = 0,
  PARALLEL_ROW_STATE_NONNEG = 1,
  PARALLEL_ROW_STATE_NONPOS = 2,
  PARALLEL_ROW_STATE_FREE = 3,
};

__device__ std::uint8_t _parallel_row_dual_state(double activity,
                                                 double lower,
                                                 double upper,
                                                 double tol) {
  const bool at_lower = _postsolve_is_finite(lower) && activity <= lower + tol;
  const bool at_upper = _postsolve_is_finite(upper) && activity >= upper - tol;
  if (at_lower && at_upper) {
    return PARALLEL_ROW_STATE_FREE;
  }
  if (at_lower) {
    return PARALLEL_ROW_STATE_NONNEG;
  }
  if (at_upper) {
    return PARALLEL_ROW_STATE_NONPOS;
  }
  return PARALLEL_ROW_STATE_ZERO;
}

__device__ void _parallel_row_state_interval(std::uint8_t state, double* lo, double* hi) {
  if (state == PARALLEL_ROW_STATE_FREE) {
    *lo = -INFINITY;
    *hi = INFINITY;
  } else if (state == PARALLEL_ROW_STATE_NONNEG) {
    *lo = 0.0;
    *hi = INFINITY;
  } else if (state == PARALLEL_ROW_STATE_NONPOS) {
    *lo = -INFINITY;
    *hi = 0.0;
  } else {
    *lo = 0.0;
    *hi = 0.0;
  }
}

__device__ void _parallel_row_deleted_dual_interval(double ybar_p,
                                                    double ratio,
                                                    std::uint8_t kept_state,
                                                    double* lo,
                                                    double* hi) {
  const double pivot = ratio * ybar_p;
  if (kept_state == PARALLEL_ROW_STATE_FREE) {
    *lo = -INFINITY;
    *hi = INFINITY;
  } else if (kept_state == PARALLEL_ROW_STATE_ZERO) {
    *lo = pivot;
    *hi = pivot;
  } else if (kept_state == PARALLEL_ROW_STATE_NONNEG) {
    if (ratio > 0.0) {
      *lo = -INFINITY;
      *hi = pivot;
    } else {
      *lo = pivot;
      *hi = INFINITY;
    }
  } else {
    if (ratio > 0.0) {
      *lo = pivot;
      *hi = INFINITY;
    } else {
      *lo = -INFINITY;
      *hi = pivot;
    }
  }
}

__device__ bool _parallel_row_doc_dual_split(double ybar_p,
                                             double ratio,
                                             double kept_old_AL,
                                             double kept_old_AU,
                                             double deleted_scaled_AL,
                                             double deleted_scaled_AU,
                                             double tol,
                                             double* yp,
                                             double* yd) {
  if (fabs(ybar_p) <= tol) {
    *yp = 0.0;
    *yd = 0.0;
    return true;
  }

  const double bar_l = fmax(kept_old_AL, deleted_scaled_AL);
  const double bar_u = fmin(kept_old_AU, deleted_scaled_AU);
  if (!(bar_l <= bar_u + tol)) {
    *yp = ybar_p;
    *yd = 0.0;
    return false;
  }

  const bool lower_tightened = bar_l > kept_old_AL + tol;
  const bool upper_tightened = bar_u < kept_old_AU - tol;
  if (lower_tightened && upper_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
    return false;
  }
  if (!lower_tightened && !upper_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
    return true;
  }

  if (ybar_p > 0.0) {
    if (lower_tightened) {
      *yp = 0.0;
      *yd = ratio * ybar_p;
    } else {
      *yp = ybar_p;
      *yd = 0.0;
    }
    return true;
  }

  if (lower_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
  } else {
    *yp = 0.0;
    *yd = ratio * ybar_p;
  }
  return true;
}

__device__ void _doubleton_eq_recover_z_split(double x_keep,
                                              double z_keep_hat,
                                              double alpha,
                                              double old_keep_l,
                                              double old_keep_u,
                                              double new_keep_l,
                                              double new_keep_u,
                                              double lower_source,
                                              double upper_source,
                                              double tol,
                                              double* z_keep,
                                              double* z_elim) {
  constexpr double BOUND_UNKNOWN = 0.0;
  constexpr double BOUND_INHERITED = 1.0;
  constexpr double BOUND_INDUCED = 2.0;
  if (fabs(alpha) <= tol) {
    *z_keep = z_keep_hat;
    *z_elim = 0.0;
    return;
  }

  const bool at_lower = _postsolve_is_finite(new_keep_l) && x_keep <= new_keep_l + tol;
  const bool at_upper = _postsolve_is_finite(new_keep_u) && x_keep >= new_keep_u - tol;
  if (!at_lower && !at_upper) {
    *z_keep = 0.0;
    *z_elim = 0.0;
    return;
  }

  double source = at_lower ? lower_source : upper_source;
  if (source == BOUND_UNKNOWN) {
    if (at_lower) {
      source = (_postsolve_is_finite(old_keep_l) && fabs(new_keep_l - old_keep_l) <= tol)
                   ? BOUND_INHERITED
                   : BOUND_INDUCED;
    } else {
      source = (_postsolve_is_finite(old_keep_u) && fabs(new_keep_u - old_keep_u) <= tol)
                   ? BOUND_INHERITED
                   : BOUND_INDUCED;
    }
  }

  if (source == BOUND_INDUCED) {
    *z_keep = 0.0;
    *z_elim = z_keep_hat / alpha;
  } else {
    *z_keep = z_keep_hat;
    *z_elim = 0.0;
  }
}

__device__ double _doubleton_eq_deleted_row_dual_from_original(const double* y_org,
                                                               const std::int32_t* AT_rowPtr,
                                                               const std::int32_t* AT_colVal,
                                                               const double* AT_nzVal,
                                                               const double* c,
                                                               std::int32_t row,
                                                               std::int32_t elim_col,
                                                               double elim_coeff,
                                                               double z_elim) {
  double sum_other = 0.0;
  for (std::int32_t p = AT_rowPtr[elim_col]; p < AT_rowPtr[elim_col + 1]; ++p) {
    const std::int32_t prow = AT_colVal[p];
    if (prow != row) {
      sum_other += AT_nzVal[p] * y_org[prow];
    }
  }
  return (c[elim_col] - sum_other - z_elim) / elim_coeff;
}

__device__ double _doubleton_eq_deleted_row_dual_from_tape(const double* y_org,
                                                           const std::int32_t* tape_idx,
                                                           const double* tape_val,
                                                           std::int32_t idx0,
                                                           std::int32_t val0,
                                                           double elim_coeff,
                                                           double z_elim) {
  const std::int32_t support_count = tape_idx[idx0 + 3];
  double sum_other = 0.0;
  for (std::int32_t t = 0; t < support_count; ++t) {
    const std::int32_t row = tape_idx[idx0 + 4 + t];
    const double coeff = tape_val[val0 + 12 + t];
    sum_other += coeff * y_org[row];
  }
  const double elim_obj = tape_val[val0 + 9];
  return (elim_obj - sum_other - z_elim) / elim_coeff;
}

__device__ double _postsolve_compute_aty_col(const std::int32_t* AT_rowPtr,
                                             const std::int32_t* AT_colVal,
                                             const double* AT_nzVal,
                                             const double* y_org,
                                             std::int32_t col) {
  double aty = 0.0;
  for (std::int32_t p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
    aty += AT_nzVal[p] * y_org[AT_colVal[p]];
  }
  return aty;
}

__device__ double _postsolve_compute_aty_without_row(const std::int32_t* AT_rowPtr,
                                                     const std::int32_t* AT_colVal,
                                                     const double* AT_nzVal,
                                                     const double* y_org,
                                                     std::int32_t col,
                                                     std::int32_t row) {
  double aty = 0.0;
  for (std::int32_t p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
    const std::int32_t prow = AT_colVal[p];
    if (prow != row) {
      aty += AT_nzVal[p] * y_org[prow];
    }
  }
  return aty;
}

__device__ double _postsolve_project_single_column_dual_from_original(double* x_org,
                                                                      double* z_org,
                                                                      const std::int32_t* AT_rowPtr,
                                                                      const std::int32_t* AT_colVal,
                                                                      const double* AT_nzVal,
                                                                      const double* y_org,
                                                                      const double* c,
                                                                      const double* l,
                                                                      const double* u,
                                                                      std::int32_t col,
                                                                      double tol) {
  const double r = c[col] - _postsolve_compute_aty_col(AT_rowPtr, AT_colVal, AT_nzVal, y_org, col);
  const double xj = x_org[col];
  const double lj = l[col];
  const double uj = u[col];
  const bool at_lower = _postsolve_at_lower(xj, lj, tol);
  const bool at_upper = _postsolve_at_upper(xj, uj, tol);
  double zj = 0.0;
  if (at_lower && at_upper) {
    zj = r;
  } else if (!at_lower && !at_upper) {
    zj = 0.0;
  } else if (at_lower) {
    zj = fmax(r, 0.0);
  } else {
    zj = fmin(r, 0.0);
  }
  const double dual_eps = fmin(tol, 1.0e-12);
  if (zj > dual_eps && !isfinite(lj)) {
    zj = 0.0;
  } else if (zj < -dual_eps && !isfinite(uj)) {
    zj = 0.0;
  }
  z_org[col] = zj;
  return zj;
}

__device__ __noinline__ void _postsolve_refine_live_row_dual_from_original(double* x_org,
                                                                           double* y_org,
                                                                           double* z_org,
                                                                           std::uint8_t* z_retrieved,
                                                                           const std::int32_t* A_rowPtr,
                                                                           const std::int32_t* A_colVal,
                                                                           const double* A_nzVal,
                                                                           const std::int32_t* AT_rowPtr,
                                                                           const std::int32_t* AT_colVal,
                                                                           const double* AT_nzVal,
                                                                           const double* c,
                                                                           const double* AL,
                                                                           const double* AU,
                                                                           const double* l,
                                                                           const double* u,
                                                                           std::int32_t row,
                                                                           double tol) {
  const std::int32_t row_start = A_rowPtr[row];
  const std::int32_t row_stop = A_rowPtr[row + 1];
  if (row_start >= row_stop || y_org == nullptr || z_org == nullptr) {
    return;
  }

  const double y_old = y_org[row];
  volatile double lo = -INFINITY;
  volatile double hi = INFINITY;
  volatile double exact_sum = 0.0;
  volatile std::int32_t exact_count = 0;
  volatile double activity = 0.0;

  for (std::int32_t p = row_start; p < row_stop; ++p) {
    const double coeff = A_nzVal[p];
    if (fabs(coeff) <= tol) {
      continue;
    }
    const std::int32_t col = A_colVal[p];
    const double xj = x_org[col];
    const double lj = l[col];
    const double uj = u[col];
    activity = activity + coeff * xj;

    const bool at_lower = _postsolve_at_lower(xj, lj, tol);
    const bool at_upper = _postsolve_at_upper(xj, uj, tol);
    double lo_col = -INFINITY;
    double hi_col = INFINITY;
    if (!(at_lower && at_upper)) {
      const double s = c[col] -
                       _postsolve_compute_aty_without_row(AT_rowPtr, AT_colVal, AT_nzVal, y_org, col, row);
      const double thresh = s / coeff;
      if (!at_lower && !at_upper) {
        lo_col = thresh;
        hi_col = thresh;
        exact_sum = exact_sum + thresh;
        exact_count = exact_count + 1;
      } else if (at_lower) {
        if (coeff > 0.0) {
          hi_col = thresh;
        } else {
          lo_col = thresh;
        }
      } else {
        if (coeff > 0.0) {
          lo_col = thresh;
        } else {
          hi_col = thresh;
        }
      }
    }
    lo = fmax(static_cast<double>(lo), lo_col);
    hi = fmin(static_cast<double>(hi), hi_col);
  }

  const double row_l = AL[row];
  const double row_u = AU[row];
  const bool row_at_lower = _postsolve_is_finite(row_l) && static_cast<double>(activity) <= row_l + tol;
  const bool row_at_upper = _postsolve_is_finite(row_u) && static_cast<double>(activity) >= row_u - tol;
  const bool row_exact = !row_at_lower && !row_at_upper;
  double row_lo = -INFINITY;
  double row_hi = INFINITY;
  if (row_at_lower && row_at_upper) {
    row_lo = -INFINITY;
    row_hi = INFINITY;
  } else if (row_exact) {
    row_lo = 0.0;
    row_hi = 0.0;
  } else if (row_at_lower) {
    row_lo = 0.0;
    row_hi = INFINITY;
  } else {
    row_lo = -INFINITY;
    row_hi = 0.0;
  }
  lo = fmax(static_cast<double>(lo), row_lo);
  hi = fmin(static_cast<double>(hi), row_hi);

  const double guess = row_exact ? 0.0 : y_old;
  const double final_lo = static_cast<double>(lo);
  const double final_hi = static_cast<double>(hi);
  double y_new = 0.0;
  if (exact_count > 0 && final_lo <= final_hi + tol && fabs(final_hi - final_lo) <= tol) {
    y_new = 0.5 * (final_lo + final_hi);
  } else if (row_exact) {
    y_new = 0.0;
  } else if (exact_count > 0) {
    const double target = static_cast<double>(exact_sum) / static_cast<double>(exact_count);
    if (final_lo <= final_hi + tol) {
      y_new = _clamp_row_refinement_value(target, final_lo, final_hi);
    } else {
      const double clamp_lo = fmin(final_lo, final_hi);
      const double clamp_hi = fmax(final_lo, final_hi);
      y_new = _clamp_row_refinement_value(target, clamp_lo, clamp_hi);
    }
  } else {
    if (final_lo <= final_hi + tol) {
      y_new = _clamp_row_refinement_value(guess, final_lo, final_hi);
    } else {
      const double clamp_lo = fmin(final_lo, final_hi);
      const double clamp_hi = fmax(final_lo, final_hi);
      y_new = _clamp_row_refinement_value(guess, clamp_lo, clamp_hi);
    }
  }

  const volatile double y_new_stable = y_new;
  y_org[row] = y_new_stable;
  for (std::int32_t p = row_start; p < row_stop; ++p) {
    const double coeff = A_nzVal[p];
    if (fabs(coeff) <= tol) {
      continue;
    }
    const std::int32_t col = A_colVal[p];
    z_org[col] = c[col] - _postsolve_compute_aty_col(AT_rowPtr, AT_colVal, AT_nzVal, y_org, col);
    if (z_retrieved != nullptr) {
      z_retrieved[col] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_replay_postsolve_tape(double* x_org,
                                              double* y_org,
                                              double* z_org,
                                              std::uint8_t* z_retrieved,
                                              const std::int32_t* reduction_types,
                                              const std::int32_t* idx_starts,
                                              const std::int32_t* val_starts,
                                              const std::int32_t* tape_idx,
                                              const double* tape_val,
                                              const std::int32_t* A_rowPtr,
                                              const std::int32_t* A_colVal,
                                              const double* A_nzVal,
                                              const std::int32_t* AT_rowPtr,
                                              const std::int32_t* AT_colVal,
                                              const double* AT_nzVal,
                                              const double* c,
                                              const double* AL,
                                              const double* AU,
                                              const double* l,
                                              const double* u,
                                              std::int32_t record_count,
                                              std::int32_t has_original_model,
                                              std::int32_t replay_primal,
                                              std::int32_t replay_dual,
                                              std::int32_t parallel_row_doc_recovery,
                                              double tol) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (std::int32_t k = record_count - 1; k >= 0; --k) {
    const std::int32_t reduction_type = reduction_types[k];
    const std::int32_t idx0 = idx_starts[k];
    const std::int32_t idx1 = idx_starts[k + 1] - 1;
    const std::int32_t val0 = val_starts[k];
    const std::int32_t val1 = val_starts[k + 1] - 1;
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelCol)) {
      const std::int32_t from_idx = tape_idx[idx0];
      const std::int32_t to_idx = tape_idx[idx0 + 1];
      const double ratio = tape_val[val0];
      const double lo = tape_val[val0 + 1];
      const double hi = tape_val[val0 + 2];
      const double to_l = tape_val[val0 + 3];
      const double to_u = tape_val[val0 + 4];
      if (replay_primal != 0) {
        const double y_val = x_org[to_idx];
        const double z_to = z_org != nullptr ? z_org[to_idx] : 0.0;
        double x_from = 0.0;
        double x_to = y_val;
        _recover_parallel_col_primal(y_val, z_to, ratio, lo, hi, to_l, to_u, tol, &x_from, &x_to);
        x_org[from_idx] = x_from;
        x_org[to_idx] = x_to;
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[from_idx] = ratio * z_org[to_idx];
        if (z_retrieved != nullptr) {
          z_retrieved[from_idx] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DoubletonEq)) {
      const std::int32_t elim_col = tape_idx[idx0];
      const std::int32_t keep_col = tape_idx[idx0 + 1];
      const std::int32_t row = tape_idx[idx0 + 2];
      const double elim_coeff = tape_val[val0];
      const double keep_coeff = tape_val[val0 + 1];
      const double rhs = tape_val[val0 + 2];
      const double elim_obj = tape_val[val0 + 9];
      const double alpha = -keep_coeff / elim_coeff;
      const double beta = rhs / elim_coeff;
      if (replay_primal != 0) {
        x_org[elim_col] = beta + alpha * x_org[keep_col];
      }
      if (replay_dual != 0 && z_org != nullptr && y_org != nullptr && has_original_model != 0) {
        const double old_keep_l = tape_val[val0 + 5];
        const double old_keep_u = tape_val[val0 + 6];
        const double new_keep_l = tape_val[val0 + 7];
        const double new_keep_u = tape_val[val0 + 8];
        const double lower_source = val1 >= val0 + 10 ? tape_val[val0 + 10] : 0.0;
        const double upper_source = val1 >= val0 + 11 ? tape_val[val0 + 11] : 0.0;
        double z_keep = 0.0;
        double z_elim = 0.0;
        _doubleton_eq_recover_z_split(x_org[keep_col],
                                      z_org[keep_col],
                                      alpha,
                                      old_keep_l,
                                      old_keep_u,
                                      new_keep_l,
                                      new_keep_u,
                                      lower_source,
                                      upper_source,
                                      tol,
                                      &z_keep,
                                      &z_elim);
        const bool has_support_payload = idx1 >= idx0 + 3 && val1 >= val0 + 11;
        y_org[row] = has_support_payload
                         ? _doubleton_eq_deleted_row_dual_from_tape(
                               y_org, tape_idx, tape_val, idx0, val0, elim_coeff, z_elim)
                         : _doubleton_eq_deleted_row_dual_from_original(
                               y_org, AT_rowPtr, AT_colVal, AT_nzVal, c, row, elim_col, elim_coeff, z_elim);
        z_org[keep_col] = z_keep;
        z_org[elim_col] = z_elim;
        if (z_retrieved != nullptr) {
          z_retrieved[keep_col] = std::uint8_t{1};
          z_retrieved[elim_col] = std::uint8_t{1};
        }
      } else if (replay_dual != 0) {
        if (z_org != nullptr) {
          z_org[elim_col] = 0.0;
          if (z_retrieved != nullptr) {
            z_retrieved[elim_col] = std::uint8_t{1};
          }
        }
        if (y_org != nullptr && row >= 0) {
          y_org[row] = elim_obj / elim_coeff;
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::SubCol)) {
      const std::int32_t elim_col = tape_idx[idx0];
      const std::int32_t row = tape_idx[idx0 + 1];
      const std::int32_t support_count = tape_idx[idx0 + 2];
      const double pivot_coeff = tape_val[val0];
      const double rhs = tape_val[val0 + 1];
      const double elim_obj = tape_val[val0 + 4];
      const bool row_deleted = tape_val[val0 + 5] != 0.0;
      double partial = 0.0;
      for (std::int32_t t = 0; t < support_count; ++t) {
        const std::int32_t col = tape_idx[idx0 + 3 + t];
        const double coeff = tape_val[val0 + 6 + t];
        partial += coeff * x_org[col];
      }
      if (replay_primal != 0) {
        x_org[elim_col] = (rhs - partial) / pivot_coeff;
      }
      if (replay_dual != 0 && y_org != nullptr && row >= 0) {
        if (row_deleted) {
          y_org[row] = elim_obj / pivot_coeff;
        } else {
          y_org[row] += elim_obj / pivot_coeff;
        }
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[elim_col] = row_deleted || y_org == nullptr ? 0.0 : elim_obj - pivot_coeff * y_org[row];
        if (z_retrieved != nullptr) {
          z_retrieved[elim_col] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedCol)) {
      const std::int32_t col = tape_idx[idx0];
      if (replay_primal != 0) {
        x_org[col] = tape_val[val0];
      }
      if (replay_dual != 0 && z_org != nullptr) {
        double zj = tape_val[val0 + 1];
        for (std::int32_t pos = idx0 + 1; pos <= idx1; ++pos) {
          const std::int32_t row = tape_idx[pos];
          const double coeff = tape_val[val0 + 2 + (pos - (idx0 + 1))];
          if (y_org != nullptr) {
            zj -= coeff * y_org[row];
          }
        }
        z_org[col] = zj;
        if (z_retrieved != nullptr) {
          z_retrieved[col] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedColInf)) {
      const bool fix_to_pos_inf = tape_idx[idx0] > 0;
      const std::int32_t col = tape_idx[idx0 + 1];
      const std::int32_t n_rows = static_cast<std::int32_t>(llround(tape_val[val0]));
      double extreme_val = tape_val[val0 + 1];
      std::int32_t idx_pos = idx0 + 2;
      std::int32_t val_pos = val0 + 2;
      if (replay_primal != 0) {
        for (std::int32_t r = 0; r < n_rows; ++r) {
          const std::int32_t row_len = tape_idx[idx_pos];
          double side = tape_val[val_pos];
          double coeff = 0.0;
          for (std::int32_t j = 0; j < row_len; ++j) {
            const std::int32_t row_col = tape_idx[idx_pos + 1 + j];
            const double row_coeff = tape_val[val_pos + 1 + j];
            if (row_col == col) {
              coeff = row_coeff;
            } else {
              side -= row_coeff * x_org[row_col];
            }
          }
          const double candidate = side / coeff;
          extreme_val = fix_to_pos_inf ? fmax(extreme_val, candidate) : fmin(extreme_val, candidate);
          idx_pos += row_len + 1;
          val_pos += row_len + 1;
        }
        x_org[col] = extreme_val;
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[col] = 0.0;
        if (z_retrieved != nullptr) {
          z_retrieved[col] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FmeCol)) {
      const std::int32_t K = tape_idx[idx0];
      const std::int32_t elim_col = tape_idx[idx0 + 1];
      const std::int32_t L = static_cast<std::int32_t>(llround(tape_val[val0]));
      std::int32_t col_cursor = idx0 + 3 + K;
      std::int32_t vpos = val0 + 1;
      double lo = -INFINITY;
      double hi = INFINITY;
      bool has_lo = false;
      bool has_hi = false;
      if (replay_primal != 0) {
        for (std::int32_t s = 0; s < L; ++s) {
          const double orient = tape_val[vpos];
          const std::int32_t len = static_cast<std::int32_t>(llround(tape_val[vpos + 1]));
          double cand = tape_val[vpos + 2];
          for (std::int32_t t = 0; t < len; ++t) {
            const std::int32_t row_col = tape_idx[col_cursor++];
            const double coeff = tape_val[vpos + 3 + t];
            cand += coeff * x_org[row_col];
          }
          if (orient > 0.0) {
            lo = fmax(lo, cand);
            has_lo = true;
          } else {
            hi = fmin(hi, cand);
            has_hi = true;
          }
          vpos += 3 + len;
        }
        x_org[elim_col] = has_lo ? lo : (has_hi ? hi : 0.0);
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[elim_col] = 0.0;
        if (z_retrieved != nullptr) {
          z_retrieved[elim_col] = std::uint8_t{1};
        }
      }
      if (replay_dual != 0 && y_org != nullptr) {
        for (std::int32_t t = 0; t < K; ++t) {
          y_org[tape_idx[idx0 + 2 + t]] = 0.0;
        }
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)) {
      if (y_org != nullptr) {
        const std::int32_t row = tape_idx[idx0];
        y_org[row] += tape_val[val0];
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelRow) &&
               idx1 >= idx0 + 1 && val1 >= val0 + 4) {
      const std::int32_t kept_row = tape_idx[idx0];
      const std::int32_t deleted_row = tape_idx[idx0 + 1];
      const double ratio = tape_val[val0];
      const double kept_old_AL = tape_val[val0 + 1];
      const double kept_old_AU = tape_val[val0 + 2];
      const double deleted_old_AL = tape_val[val0 + 3];
      const double deleted_old_AU = tape_val[val0 + 4];
      if (fabs(ratio) <= tol || y_org == nullptr) {
        continue;
      }
      double deleted_scaled_AL = 0.0;
      double deleted_scaled_AU = 0.0;
      _parallel_row_scaled_interval(deleted_old_AL, deleted_old_AU, ratio,
                                    &deleted_scaled_AL, &deleted_scaled_AU);
      if (_parallel_row_interval_contains(deleted_scaled_AL,
                                          deleted_scaled_AU,
                                          kept_old_AL,
                                          kept_old_AU,
                                          tol)) {
        y_org[deleted_row] = 0.0;
        continue;
      }
      if (has_original_model != 0) {
        const double alpha_p = _parallel_row_activity(kept_row, A_rowPtr, A_colVal, A_nzVal, x_org);
        const double alpha_d = _parallel_row_activity(deleted_row, A_rowPtr, A_colVal, A_nzVal, x_org);
        const std::uint8_t kept_state =
            _parallel_row_dual_state(alpha_p, kept_old_AL, kept_old_AU, tol);
        const std::uint8_t deleted_state =
            _parallel_row_dual_state(alpha_d, deleted_old_AL, deleted_old_AU, tol);
        double yd_lo = 0.0;
        double yd_hi = 0.0;
        double yp_lo = 0.0;
        double yp_hi = 0.0;
        _parallel_row_state_interval(deleted_state, &yd_lo, &yd_hi);
        _parallel_row_deleted_dual_interval(y_org[kept_row], ratio, kept_state, &yp_lo, &yp_hi);
        const double sd_lo = fmax(yd_lo, yp_lo);
        const double sd_hi = fmin(yd_hi, yp_hi);
        if (sd_lo <= sd_hi + tol) {
          const double yd_new = _clamp_with_inf(0.0, sd_lo, sd_hi);
          const double yp_new = y_org[kept_row] - yd_new / ratio;
          y_org[kept_row] = yp_new;
          y_org[deleted_row] = yd_new;
          continue;
        }

        if (parallel_row_doc_recovery != 0) {
          double yp_doc = y_org[kept_row];
          double yd_doc = 0.0;
          if (_parallel_row_doc_dual_split(y_org[kept_row],
                                           ratio,
                                           kept_old_AL,
                                           kept_old_AU,
                                           deleted_scaled_AL,
                                           deleted_scaled_AU,
                                           tol,
                                           &yp_doc,
                                           &yd_doc)) {
            y_org[kept_row] = yp_doc;
            y_org[deleted_row] = yd_doc;
          }
        }
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DeletedRow) &&
               has_original_model != 0 && idx1 >= idx0 + 1 && val1 >= val0 + 2) {
      const std::int32_t row = tape_idx[idx0];
      const std::int32_t col = tape_idx[idx0 + 1];
      const double old_AL = tape_val[val0];
      const double old_AU = tape_val[val0 + 1];
      const double aij = tape_val[val0 + 2];
      if (fabs(aij) <= tol) {
        continue;
      }
      double gamma_j = c[col];
      for (std::int32_t p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
        const std::int32_t prow = AT_colVal[p];
        if (prow != row && y_org != nullptr) {
          gamma_j -= AT_nzVal[p] * y_org[prow];
        }
      }
      const double xj = x_org[col];
      const double zbar_j = z_org[col];
      const double row_activity = aij * xj;
      const bool row_at_lower = isfinite(old_AL) && row_activity <= old_AL + tol;
      const bool row_at_upper = isfinite(old_AU) && row_activity >= old_AU - tol;
      double iz_lo = -INFINITY;
      double iz_hi = INFINITY;
      if (!(row_at_lower && row_at_upper)) {
        if (row_at_lower) {
          if (aij > 0.0) {
            iz_hi = gamma_j;
          } else {
            iz_lo = gamma_j;
          }
        } else if (row_at_upper) {
          if (aij > 0.0) {
            iz_lo = gamma_j;
          } else {
            iz_hi = gamma_j;
          }
        } else {
          iz_lo = gamma_j;
          iz_hi = gamma_j;
        }
      }
      const bool at_lower = _postsolve_at_lower(xj, l[col], tol);
      const bool at_upper = _postsolve_at_upper(xj, u[col], tol);
      double z_lo = iz_lo;
      double z_hi = iz_hi;
      if (!at_lower && !at_upper) {
        if ((isfinite(iz_lo) && iz_lo > tol) || (isfinite(iz_hi) && iz_hi < -tol)) {
          z_lo = gamma_j;
          z_hi = gamma_j;
        } else {
          z_lo = 0.0;
          z_hi = 0.0;
        }
      } else if (at_lower && !at_upper) {
        z_lo = fmax(z_lo, 0.0);
      } else if (!at_lower && at_upper) {
        z_hi = fmin(z_hi, 0.0);
      }
      if (isfinite(z_lo) && isfinite(z_hi) && z_lo > z_hi + tol) {
        continue;
      }
      const double z_new = _clamp_with_inf(zbar_j, z_lo, z_hi);
      z_org[col] = z_new;
      if (y_org != nullptr) {
        y_org[row] = (gamma_j - z_new) / aij;
      }
      if (z_retrieved != nullptr) {
        z_retrieved[col] = std::uint8_t{1};
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow) &&
               has_original_model != 0 && idx1 >= idx0 + 1) {
      const std::int32_t col = tape_idx[idx0];
      const std::int32_t row = tape_idx[idx0 + 1];
      if (z_retrieved != nullptr && z_retrieved[col] == std::uint8_t{0}) {
        continue;
      }
      const double old_l = tape_val[val0];
      const double old_u = tape_val[val0 + 1];
      const double new_l = tape_val[val0 + 2];
      const double new_u = tape_val[val0 + 3];
      const bool lower_changed = new_l > old_l + tol;
      const bool upper_changed = new_u < old_u - tol;
      if (lower_changed == upper_changed) {
        continue;
      }
      const double implied_bound = lower_changed ? new_l : new_u;
      const double original_other_bound = lower_changed ? old_u : old_l;
      const bool original_other_is_lower = !lower_changed;
      const double xj = x_org[col];
      const double zj = z_org[col];
      if (_postsolve_is_finite(original_other_bound) && fabs(xj - original_other_bound) <= tol) {
        const bool sign_ok = original_other_is_lower ? (zj >= -tol) : (zj <= tol);
        if (sign_ok) {
          continue;
        }
      }
      if (fabs(xj - implied_bound) > tol) {
        continue;
      }
      double aij = 0.0;
      for (std::int32_t p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
        if (A_colVal[p] == col) {
          aij = A_nzVal[p];
          break;
        }
      }
      if (fabs(aij) <= tol) {
        continue;
      }
      const double delta_y = zj / aij;
      if (y_org != nullptr) {
        y_org[row] += delta_y;
      }
      for (std::int32_t p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
        const std::int32_t kcol = A_colVal[p];
        if (kcol == col || (z_retrieved != nullptr && z_retrieved[kcol] == std::uint8_t{0})) {
          continue;
        }
        z_org[kcol] -= (A_nzVal[p] / aij) * zj;
      }
      z_org[col] = 0.0;
      if (z_retrieved != nullptr) {
        z_retrieved[col] = std::uint8_t{1};
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow) &&
               has_original_model != 0 && idx1 >= idx0) {
      const std::int32_t col = tape_idx[idx0];
      if (z_retrieved != nullptr && z_retrieved[col] == std::uint8_t{0}) {
        continue;
      }
      _postsolve_project_single_column_dual_from_original(x_org,
                                                          z_org,
                                                          AT_rowPtr,
                                                          AT_colVal,
                                                          AT_nzVal,
                                                          y_org,
                                                          c,
                                                          l,
                                                          u,
                                                          col,
                                                          tol);
      if (z_retrieved != nullptr) {
        z_retrieved[col] = std::uint8_t{1};
      }
    }
  }
}

void restore_splits(double* x_org, const std::vector<StructuralL1SplitRecovery>& splits) {
  if (splits.empty()) {
    return;
  }
  std::vector<std::int32_t> t_cols;
  std::vector<std::int32_t> e_cols;
  std::vector<double> rhos;
  t_cols.reserve(splits.size());
  e_cols.reserve(splits.size());
  rhos.reserve(splits.size());
  for (const StructuralL1SplitRecovery& split : splits) {
    t_cols.push_back(split.t_col);
    e_cols.push_back(split.e_col);
    rhos.push_back(split.rho);
  }
  std::int32_t* t_d = copy_vector_to_device(t_cols, "cudaMalloc/copy structural split t_cols");
  std::int32_t* e_d = copy_vector_to_device(e_cols, "cudaMalloc/copy structural split e_cols");
  double* rho_d = copy_vector_to_device(rhos, "cudaMalloc/copy structural split rhos");
  const std::int32_t k = static_cast<std::int32_t>(splits.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_splits<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, t_d, e_d, rho_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_splits");
  cudaFree(t_d);
  cudaFree(e_d);
  cudaFree(rho_d);
}

void restore_outer_pairs(double* x_org, const std::vector<StructuralOuterPairRecovery>& pairs) {
  if (pairs.empty()) {
    return;
  }
  std::vector<std::int32_t> bound_cols;
  std::vector<std::int32_t> free_cols;
  bound_cols.reserve(pairs.size());
  free_cols.reserve(pairs.size());
  for (const StructuralOuterPairRecovery& pair : pairs) {
    bound_cols.push_back(pair.bound_col);
    free_cols.push_back(pair.free_col);
  }
  std::int32_t* bound_d = copy_vector_to_device(bound_cols, "cudaMalloc/copy structural outer bound_cols");
  std::int32_t* free_d = copy_vector_to_device(free_cols, "cudaMalloc/copy structural outer free_cols");
  const std::int32_t k = static_cast<std::int32_t>(pairs.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_outer_pairs<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, bound_d, free_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_outer_pairs");
  cudaFree(bound_d);
  cudaFree(free_d);
}

void restore_linked_slacks(double* x_org, const std::vector<StructuralLinkedSlackRecovery>& slacks) {
  if (slacks.empty()) {
    return;
  }
  std::vector<std::int32_t> slack_cols;
  std::vector<std::int32_t> t_cols;
  std::vector<double> factors;
  slack_cols.reserve(slacks.size());
  t_cols.reserve(slacks.size());
  factors.reserve(slacks.size());
  for (const StructuralLinkedSlackRecovery& slack : slacks) {
    slack_cols.push_back(slack.slack_col);
    t_cols.push_back(slack.t_col);
    factors.push_back(slack.factor);
  }
  std::int32_t* slack_d = copy_vector_to_device(slack_cols, "cudaMalloc/copy structural linked slack_cols");
  std::int32_t* t_d = copy_vector_to_device(t_cols, "cudaMalloc/copy structural linked t_cols");
  double* factor_d = copy_vector_to_device(factors, "cudaMalloc/copy structural linked factors");
  const std::int32_t k = static_cast<std::int32_t>(slacks.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_linked_slacks<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, slack_d, t_d, factor_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_linked_slacks");
  cudaFree(slack_d);
  cudaFree(t_d);
  cudaFree(factor_d);
}

void restore_max_slacks(double* x_org,
                        const std::vector<StructuralMaxSlackRecovery>& slacks,
                        const double* original_l) {
  for (const StructuralMaxSlackRecovery& slack : slacks) {
    if (slack.t_cols.empty()) {
      continue;
    }
    double lower_bound = 0.0;
    if (original_l != nullptr) {
      throw_if_cuda_error(cudaMemcpy(&lower_bound, original_l + slack.slack_col, sizeof(double),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy structural max slack lower bound");
      if (!isfinite(lower_bound)) {
        lower_bound = 0.0;
      }
    }
    std::int32_t* t_d = copy_vector_to_device(slack.t_cols, "cudaMalloc/copy structural max t_cols");
    double* factor_d = copy_vector_to_device(slack.factors, "cudaMalloc/copy structural max factors");
    _kernel_structural_restore_max_slack<<<1, GPU_PRESOLVE_THREADS>>>(
        x_org, slack.slack_col, t_d, factor_d, static_cast<std::int32_t>(slack.t_cols.size()), lower_bound);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_max_slack");
    cudaFree(t_d);
    cudaFree(factor_d);
  }
}

void scatter_reduced_to_original(double* dst,
                                 const double* src,
                                 const std::vector<std::int32_t>& red2org,
                                 std::int32_t original_count,
                                 const char* context) {
  if (original_count > 0) {
    throw_if_cuda_error(cudaMemset(dst, 0, sizeof(double) * static_cast<std::size_t>(original_count)),
                        context);
  }
  if (red2org.empty() || src == nullptr) {
    return;
  }
  std::int32_t* red2org_d = copy_vector_to_device(red2org, context);
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_scatter_reduced_to_original<<<blocks, GPU_PRESOLVE_THREADS>>>(dst, src, red2org_d, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_reduced_to_original");
  cudaFree(red2org_d);
}

void restore_fixed_values(double* x_org,
                          const std::vector<std::int32_t>& fixed_idx,
                          const std::vector<double>& fixed_val) {
  if (fixed_idx.empty()) {
    return;
  }
  std::int32_t* fixed_idx_d = copy_vector_to_device(fixed_idx, "cudaMalloc/copy fixed_idx");
  double* fixed_val_d = copy_vector_to_device(fixed_val, "cudaMalloc/copy fixed_val");
  const std::int32_t fixed_count = static_cast<std::int32_t>(fixed_idx.size());
  const int blocks = (fixed_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_restore_fixed_values<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, fixed_idx_d, fixed_val_d, fixed_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_restore_fixed_values");
  cudaFree(fixed_idx_d);
  cudaFree(fixed_val_d);
}

void replay_postsolve_tape(double* x_org,
                           double* y_org,
                           double* z_org,
                           std::uint8_t* z_retrieved,
                           const PostsolveTape& tape,
                           const LPInfoGpu* original_model,
                           bool replay_primal,
                           bool replay_dual,
                           double tol) {
  if (tape.types.empty()) {
    return;
  }
  std::int32_t* types_d = copy_vector_to_device(tape.types, "cudaMalloc/copy postsolve tape types");
  std::int32_t* idx_starts_d =
      copy_vector_to_device(tape.index_starts, "cudaMalloc/copy postsolve tape index_starts");
  std::int32_t* val_starts_d =
      copy_vector_to_device(tape.value_starts, "cudaMalloc/copy postsolve tape value_starts");
  std::int32_t* indices_d = copy_vector_to_device(tape.indices, "cudaMalloc/copy postsolve tape indices");
  double* vals_d = copy_vector_to_device(tape.vals, "cudaMalloc/copy postsolve tape vals");
  _kernel_replay_postsolve_tape<<<1, 1>>>(x_org,
                                          y_org,
                                          z_org,
                                          z_retrieved,
                                          types_d,
                                          idx_starts_d,
                                          val_starts_d,
                                          indices_d,
                                          vals_d,
                                          original_model != nullptr ? original_model->A.rowPtr : nullptr,
                                          original_model != nullptr ? original_model->A.colVal : nullptr,
                                          original_model != nullptr ? original_model->A.nzVal : nullptr,
                                          original_model != nullptr ? original_model->AT.rowPtr : nullptr,
                                          original_model != nullptr ? original_model->AT.colVal : nullptr,
                                          original_model != nullptr ? original_model->AT.nzVal : nullptr,
                                          original_model != nullptr ? original_model->c : nullptr,
                                          original_model != nullptr ? original_model->AL : nullptr,
                                          original_model != nullptr ? original_model->AU : nullptr,
                                          original_model != nullptr ? original_model->l : nullptr,
                                          original_model != nullptr ? original_model->u : nullptr,
                                          static_cast<std::int32_t>(tape.types.size()),
                                          original_model != nullptr ? 1 : 0,
                                          replay_primal ? 1 : 0,
                                          replay_dual ? 1 : 0,
                                          1,
                                          tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_replay_postsolve_tape");
  cudaFree(types_d);
  cudaFree(idx_starts_d);
  cudaFree(val_starts_d);
  cudaFree(indices_d);
  cudaFree(vals_d);
}

void replay_postsolve_tape_gpu(double* x_org,
                               double* y_org,
                               double* z_org,
                               std::uint8_t* z_retrieved,
                               const PostsolveTapeGpu& tape,
                               const LPInfoGpu* original_model,
                               bool replay_primal,
                               bool replay_dual,
                               double tol) {
  if (tape.record_count == 0) {
    return;
  }
  _kernel_replay_postsolve_tape<<<1, 1>>>(x_org,
                                          y_org,
                                          z_org,
                                          z_retrieved,
                                          tape.types,
                                          tape.index_starts,
                                          tape.value_starts,
                                          tape.indices,
                                          tape.vals,
                                          original_model ? original_model->A.rowPtr : nullptr,
                                          original_model ? original_model->A.colVal : nullptr,
                                          original_model ? original_model->A.nzVal : nullptr,
                                          original_model ? original_model->AT.rowPtr : nullptr,
                                          original_model ? original_model->AT.colVal : nullptr,
                                          original_model ? original_model->AT.nzVal : nullptr,
                                          original_model ? original_model->c : nullptr,
                                          original_model ? original_model->AL : nullptr,
                                          original_model ? original_model->AU : nullptr,
                                          original_model ? original_model->l : nullptr,
                                          original_model ? original_model->u : nullptr,
                                          tape.record_count,
                                          original_model ? 1 : 0,
                                          replay_primal ? 1 : 0,
                                          replay_dual ? 1 : 0,
                                          1,
                                          tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_replay_postsolve_tape gpu");
}

void mark_retrieved_values(std::uint8_t* z_retrieved, const std::vector<std::int32_t>& red2org) {
  if (z_retrieved == nullptr || red2org.empty()) {
    return;
  }
  std::int32_t* red2org_d = copy_vector_to_device(red2org, "cudaMalloc/copy z_retrieved red2org");
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_mark_retrieved<<<blocks, GPU_PRESOLVE_THREADS>>>(z_retrieved, red2org_d, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_retrieved");
  cudaFree(red2org_d);
}

void fill_unretrieved_duals_from_original(double* z_org,
                                          double* y_org,
                                          std::uint8_t* z_retrieved,
                                          const LPInfoGpu* original_model) {
  if (z_org == nullptr || z_retrieved == nullptr || original_model == nullptr ||
      original_model->AT.rows <= 0 || original_model->AT.rowPtr == nullptr ||
      original_model->c == nullptr) {
    return;
  }
  const std::int32_t n = original_model->AT.rows;
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_fill_unretrieved_duals_from_original<<<blocks, GPU_PRESOLVE_THREADS>>>(
      z_org,
      z_retrieved,
      y_org,
      original_model->AT.rowPtr,
      original_model->AT.colVal,
      original_model->AT.nzVal,
      original_model->c,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_unretrieved_duals_from_original");
}

std::vector<std::uint8_t> build_exact_replay_column_mask(const PostsolveTape& tape, std::int32_t n) {
  std::vector<std::uint8_t> protected_cols(static_cast<std::size_t>(n), std::uint8_t{0});
  for (std::size_t k = 0; k < tape.types.size(); ++k) {
    const std::int32_t idx0 = tape.index_starts[k];
    const auto type = static_cast<PostsolveReductionType>(tape.types[k]);
    if (type == PostsolveReductionType::SubCol) {
      const std::int32_t col = tape.indices[idx0];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::DoubletonEq) {
      const std::int32_t elim_col = tape.indices[idx0];
      const std::int32_t keep_col = tape.indices[idx0 + 1];
      if (0 <= elim_col && elim_col < n) {
        protected_cols[static_cast<std::size_t>(elim_col)] = std::uint8_t{1};
      }
      if (0 <= keep_col && keep_col < n) {
        protected_cols[static_cast<std::size_t>(keep_col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::FixedColInf) {
      const std::int32_t col = tape.indices[idx0 + 1];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::ParallelCol) {
      const std::int32_t col = tape.indices[idx0];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    }
  }
  return protected_cols;
}

void project_column_duals_from_original(double* x_org,
                                        double* y_org,
                                        double* z_org,
                                        const PostsolveTape& tape,
                                        const LPInfoGpu* original_model,
                                        double tol) {
  if (x_org == nullptr || y_org == nullptr || z_org == nullptr || original_model == nullptr ||
      original_model->AT.rows <= 0 || original_model->AT.rowPtr == nullptr ||
      original_model->c == nullptr || original_model->l == nullptr || original_model->u == nullptr) {
    return;
  }
  const std::int32_t n = original_model->AT.rows;
  std::vector<std::uint8_t> protected_cols = build_exact_replay_column_mask(tape, n);
  std::uint8_t* protected_cols_d =
      copy_vector_to_device(protected_cols, "cudaMalloc/copy protected postsolve columns");
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_project_column_duals_from_original<<<blocks, GPU_PRESOLVE_THREADS>>>(
      x_org,
      z_org,
      protected_cols_d,
      original_model->AT.rowPtr,
      original_model->AT.colVal,
      original_model->AT.nzVal,
      y_org,
      original_model->c,
      original_model->l,
      original_model->u,
      n,
      tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_project_column_duals_from_original");
  cudaFree(protected_cols_d);
}

}  // namespace

void postsolve_restore_structural_primal_gpu(
    double* x_org,
    const std::vector<StructuralL1PrimalRecoveryStep>& recoveries,
    const double* original_l) {
  for (auto it = recoveries.rbegin(); it != recoveries.rend(); ++it) {
    restore_splits(x_org, it->splits);
    restore_outer_pairs(x_org, it->outer_pairs);
    restore_linked_slacks(x_org, it->linked_slacks);
    restore_max_slacks(x_org, it->max_slacks, original_l);
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_restore_structural_primal_gpu synchronize");
}

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const double* original_l) {
  return postsolve_gpu(x_red, y_red, z_red, record, nullptr, original_l);
}

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const LPInfoGpu* original_model,
                                 const double* original_l) {
  GpuPostsolveResult result;
  result.n0 = record.n0;
  result.m0 = record.m0;
  if (original_l == nullptr && original_model != nullptr) {
    original_l = original_model->l;
  }
  if (record.n0 > 0) {
    throw_if_cuda_error(cudaMalloc(&result.x_org, sizeof(double) * static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve x_org");
    throw_if_cuda_error(cudaMalloc(&result.z_org, sizeof(double) * static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve z_org");
  }
  if (record.m0 > 0) {
    throw_if_cuda_error(cudaMalloc(&result.y_org, sizeof(double) * static_cast<std::size_t>(record.m0)),
                        "cudaMalloc postsolve y_org");
  }

  std::uint8_t* z_retrieved = nullptr;
  if (record.n0 > 0) {
    throw_if_cuda_error(cudaMalloc(&z_retrieved, static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve z_retrieved");
    throw_if_cuda_error(cudaMemset(z_retrieved, 0, static_cast<std::size_t>(record.n0)),
                        "cudaMemset postsolve z_retrieved");
  }

  scatter_reduced_to_original(result.x_org, x_red, record.col_red2org, record.n0, "cudaMemset/scatter x_org");
  scatter_reduced_to_original(result.y_org, y_red, record.row_red2org, record.m0, "cudaMemset/scatter y_org");
  scatter_reduced_to_original(result.z_org, z_red, record.col_red2org, record.n0, "cudaMemset/scatter z_org");
  mark_retrieved_values(z_retrieved, record.col_red2org);
  restore_fixed_values(result.x_org, record.fixed_idx, record.fixed_val);
  if (record.tape_gpu.record_count > 0) {
    replay_postsolve_tape_gpu(
        result.x_org, result.y_org, result.z_org, z_retrieved, record.tape_gpu, original_model, true, false, 1.0e-7);
  } else {
    replay_postsolve_tape(
        result.x_org, result.y_org, result.z_org, z_retrieved, record.tape, original_model, true, false, 1.0e-7);
  }
  postsolve_restore_structural_primal_gpu(result.x_org, record.structural_primal_recoveries, original_l);
  if (record.tape_gpu.record_count > 0) {
    replay_postsolve_tape_gpu(
        result.x_org, result.y_org, result.z_org, z_retrieved, record.tape_gpu, original_model, false, true, 1.0e-7);
  } else {
    replay_postsolve_tape(
        result.x_org, result.y_org, result.z_org, z_retrieved, record.tape, original_model, false, true, 1.0e-7);
  }
  project_column_duals_from_original(result.x_org, result.y_org, result.z_org, record.tape, original_model, 1.0e-7);
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_gpu synchronize");
  cudaFree(z_retrieved);
  return result;
}

}  // namespace gpu_presolver::presolve
