#include "gpu_presolver/presolve/rules/rule_structural_l1_substitution.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr double STRUCTURAL_L1_SUB_TOL = 1.0e-12;

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

__global__ void _kernel_fill_i32(std::int32_t* values, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    values[i] = value;
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

__device__ bool _near(double lhs, double rhs) {
  return fabs(lhs - rhs) <= STRUCTURAL_L1_SUB_TOL;
}

__device__ std::int32_t _sls_outer_map_col_device(const std::int32_t* free_to_bound,
                                                  std::int32_t col) {
  const std::int32_t mapped = free_to_bound[col];
  return mapped >= 0 ? mapped : col;
}

__device__ bool _sls_outer_targets_overlap(std::int32_t a1,
                                           std::int32_t a2,
                                           std::int32_t ac,
                                           std::int32_t b1,
                                           std::int32_t b2,
                                           std::int32_t bc) {
  if (ac >= 1 && bc >= 1 && a1 == b1) return true;
  if (ac >= 1 && bc >= 2 && a1 == b2) return true;
  if (ac >= 2 && bc >= 1 && a2 == b1) return true;
  if (ac >= 2 && bc >= 2 && a2 == b2) return true;
  return false;
}

__device__ bool _sls_outer_pair_signature_device(std::int32_t c1a,
                                                 std::int32_t c2a,
                                                 double v1a,
                                                 double v2a,
                                                 std::int32_t c1b,
                                                 std::int32_t c2b,
                                                 double v1b,
                                                 double v2b,
                                                 std::int32_t c1c,
                                                 std::int32_t c2c,
                                                 double v1c,
                                                 double v2c) {
  if (c1a != c1b || c1a != c1c || c2a != c2b || c2a != c2c) {
    return false;
  }
  return _near(v1a, 1.0) && _near(v2a, -1.0) &&
         _near(v1b, 1.0) && _near(v2b, 1.0) &&
         _near(v1c, -1.0) && _near(v2c, 1.0);
}

__device__ bool _sls_extract_l1_pair_device(std::int32_t ep1_col1,
                                            std::int32_t ep1_col2,
                                            double ep1_val1,
                                            double ep1_val2,
                                            double ep2_val1,
                                            double ep2_val2,
                                            std::int32_t* q_col,
                                            std::int32_t* e_col) {
  if (!_near(ep1_val1, -1.0) || !_near(ep1_val2, 1.0) ||
      !_near(ep2_val1, 1.0) || !_near(ep2_val2, 1.0)) {
    return false;
  }
  *e_col = ep1_col1;
  *q_col = ep1_col2;
  return *q_col != *e_col;
}

__device__ bool _sls_extract_l1_split_pair_device(std::int32_t c1a,
                                                  std::int32_t c2a,
                                                  double v1a,
                                                  double v2a,
                                                  std::int32_t c1b,
                                                  std::int32_t c2b,
                                                  double v1b,
                                                  double v2b,
                                                  std::int32_t* t_col,
                                                  std::int32_t* e_col) {
  if (c1a != c1b || c2a != c2b) {
    return false;
  }
  const bool first_is_t = v1a > STRUCTURAL_L1_SUB_TOL &&
                          _near(v1a, v1b) &&
                          _near(v2a, -v1a) &&
                          _near(v2b, v1b);
  if (first_is_t) {
    *t_col = c1a;
    *e_col = c2a;
    return *t_col != *e_col;
  }
  const bool second_is_t = v2a > STRUCTURAL_L1_SUB_TOL &&
                           _near(v2a, v2b) &&
                           _near(v1a, -v2a) &&
                           _near(v1b, v2b);
  if (second_is_t) {
    *t_col = c2a;
    *e_col = c1a;
    return *t_col != *e_col;
  }
  return false;
}

__device__ bool _sls_extract_l1_orientation_device(std::int32_t c1a,
                                                   std::int32_t c2a,
                                                   double v1a,
                                                   double v2a,
                                                   double v1b,
                                                   double v2b,
                                                   std::int32_t* t_col,
                                                   std::int32_t* e_col,
                                                   double* rho) {
  if (!_sls_extract_l1_split_pair_device(c1a, c2a, v1a, v2a, c1a, c2a, v1b, v2b, t_col, e_col)) {
    return false;
  }
  if (*t_col == c1a && *e_col == c2a) {
    *rho = fabs(v2a / v1a);
  } else if (*t_col == c2a && *e_col == c1a) {
    *rho = fabs(v1a / v2a);
  } else {
    *rho = 0.0;
  }
  return *rho > STRUCTURAL_L1_SUB_TOL;
}

__global__ void _kernel_structural_row_metadata(std::uint8_t* eq_row,
                                                std::uint8_t* eq_zero_two_nnz,
                                                std::uint8_t* lower_two_nnz,
                                                std::uint8_t* zero_lower_two_nnz,
                                                std::int32_t* raw_col1,
                                                std::int32_t* raw_col2,
                                                double* raw_val1,
                                                double* raw_val2,
                                                std::int32_t* zl_col1,
                                                std::int32_t* zl_col2,
                                                double* zl_val1,
                                                double* zl_val2,
                                                const std::int32_t* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                const double* AL,
                                                const double* AU,
                                                std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }

  const double al = AL[row];
  const double au = AU[row];
  const std::int32_t lo = row_ptr[row];
  const std::int32_t hi = row_ptr[row + 1];
  const std::int32_t nnz = hi - lo;
  const bool is_eq = isfinite(al) && isfinite(au) && fabs(al - au) <= STRUCTURAL_L1_SUB_TOL;

  eq_row[row] = is_eq ? std::uint8_t{1} : std::uint8_t{0};
  eq_zero_two_nnz[row] = std::uint8_t{0};
  lower_two_nnz[row] = std::uint8_t{0};
  zero_lower_two_nnz[row] = std::uint8_t{0};
  raw_col1[row] = -1;
  raw_col2[row] = -1;
  raw_val1[row] = 0.0;
  raw_val2[row] = 0.0;
  zl_col1[row] = -1;
  zl_col2[row] = -1;
  zl_val1[row] = 0.0;
  zl_val2[row] = 0.0;

  if (nnz != 2) {
    return;
  }

  std::int32_t c1 = col_val[lo];
  std::int32_t c2 = col_val[lo + 1];
  double v1 = nz_val[lo];
  double v2 = nz_val[lo + 1];
  if (c2 < c1) {
    const std::int32_t tc = c1;
    c1 = c2;
    c2 = tc;
    const double tv = v1;
    v1 = v2;
    v2 = tv;
  }

  raw_col1[row] = c1;
  raw_col2[row] = c2;
  raw_val1[row] = v1;
  raw_val2[row] = v2;

  if (is_eq && fabs(al) <= STRUCTURAL_L1_SUB_TOL) {
    eq_zero_two_nnz[row] = std::uint8_t{1};
  }
  if (isfinite(al) && isinf(au)) {
    lower_two_nnz[row] = std::uint8_t{1};
  }

  const bool is_zero_lower = isfinite(al) && isinf(au) && fabs(al) <= STRUCTURAL_L1_SUB_TOL;
  const bool is_zero_upper = isinf(al) && isfinite(au) && fabs(au) <= STRUCTURAL_L1_SUB_TOL;
  if (is_zero_lower || is_zero_upper) {
    const double scale = is_zero_lower ? 1.0 : -1.0;
    zero_lower_two_nnz[row] = std::uint8_t{1};
    zl_col1[row] = c1;
    zl_col2[row] = c2;
    zl_val1[row] = scale * v1;
    zl_val2[row] = scale * v2;
  }
}

__global__ void _kernel_structural_outer_count_pairs(std::int32_t* status_flag,
                                                     std::int32_t* pair_count,
                                                     std::int32_t* start_row_out,
                                                     const std::uint8_t* zero_lower_two_nnz,
                                                     const std::int32_t* zl_col1,
                                                     const std::int32_t* zl_col2,
                                                     const double* zl_val1,
                                                     const double* zl_val2,
                                                     std::int32_t m) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    std::int32_t row = 0;
    std::int32_t count = 0;
    while (row + 2 < m) {
      const bool ok =
          zero_lower_two_nnz[row] != std::uint8_t{0} &&
          zero_lower_two_nnz[row + 1] != std::uint8_t{0} &&
          zero_lower_two_nnz[row + 2] != std::uint8_t{0} &&
          _sls_outer_pair_signature_device(
              zl_col1[row], zl_col2[row], zl_val1[row], zl_val2[row],
              zl_col1[row + 1], zl_col2[row + 1], zl_val1[row + 1], zl_val2[row + 1],
              zl_col1[row + 2], zl_col2[row + 2], zl_val1[row + 2], zl_val2[row + 2]);
      if (!ok) {
        break;
      }
      ++count;
      row += 3;
    }

    pair_count[0] = count;
    start_row_out[0] = row;
    if (count == 0 || row >= m || ((m - row) % 8) != 0) {
      status_flag[0] = 1;
    }
  }
}

__global__ void _kernel_structural_outer_extract_pairs(std::int32_t* bound_cols,
                                                       std::int32_t* free_cols,
                                                       const std::int32_t* zl_col1,
                                                       const std::int32_t* zl_col2,
                                                       std::int32_t pair_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < pair_count) {
    const std::int32_t row = 3 * idx;
    bound_cols[idx] = zl_col1[row];
    free_cols[idx] = zl_col2[row];
  }
}

__global__ void _kernel_structural_l1_split_extract_pairs(std::int32_t* status_flag,
                                                          std::int32_t* t_cols,
                                                          std::int32_t* e_cols,
                                                          const std::uint8_t* eq_row,
                                                          const std::uint8_t* zero_lower_two_nnz,
                                                          const std::int32_t* zl_col1,
                                                          const std::int32_t* zl_col2,
                                                          const double* zl_val1,
                                                          const double* zl_val2,
                                                          std::int32_t nblocks) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t eq = 3 * idx;
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  if (eq_row[eq] == std::uint8_t{0} ||
      zero_lower_two_nnz[r1] == std::uint8_t{0} ||
      zero_lower_two_nnz[r2] == std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  if (!_sls_extract_l1_split_pair_device(
          zl_col1[r1], zl_col2[r1], zl_val1[r1], zl_val2[r1],
          zl_col1[r2], zl_col2[r2], zl_val1[r2], zl_val2[r2],
          &t_col, &e_col)) {
    atomicMax(status_flag, 1);
    return;
  }
  t_cols[idx] = t_col;
  e_cols[idx] = e_col;
}

__global__ void _kernel_structural_l1_split_build_plan_data(std::int32_t* status_flag,
                                                            std::uint8_t* keep_row,
                                                            double* new_c,
                                                            double* new_l,
                                                            double* new_u,
                                                            const double* c,
                                                            const double* l,
                                                            const double* u,
                                                            const std::int32_t* row_ptr,
                                                            const std::int32_t* col_val,
                                                            const std::int32_t* t_cols,
                                                            const std::int32_t* e_cols,
                                                            double residual_bound_as_free_min,
                                                            std::int32_t nblocks) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const std::int32_t eq = 3 * idx;
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;

  const double c_t = c[t_col];
  const double c_e = c[e_col];
  const double l_t = l[t_col];
  const double u_t = u[t_col];
  const double l_e = l[e_col];
  const double u_e = u[e_col];
  const bool e_is_free =
      (isinf(l_e) && isinf(u_e)) ||
      (isfinite(residual_bound_as_free_min) &&
       l_e <= -residual_bound_as_free_min &&
       u_e >= residual_bound_as_free_min);
  const bool valid_bounds = e_is_free &&
                            fabs(l_t) <= STRUCTURAL_L1_SUB_TOL &&
                            isinf(u_t);
  bool found_e = false;
  for (std::int32_t p = row_ptr[eq]; p < row_ptr[eq + 1]; ++p) {
    if (col_val[p] == e_col) {
      found_e = true;
      break;
    }
  }
  if (!valid_bounds || !found_e) {
    atomicMax(status_flag, 1);
    return;
  }

  keep_row[r1] = std::uint8_t{0};
  keep_row[r2] = std::uint8_t{0};
  new_c[t_col] = c_t + c_e;
  new_c[e_col] = c_t - c_e;
  new_l[t_col] = 0.0;
  new_l[e_col] = 0.0;
  new_u[t_col] = INFINITY;
  new_u[e_col] = INFINITY;
}

__global__ void _kernel_structural_l1_split_count_rows(std::int32_t* row_nnz_new,
                                                       const std::int32_t* row_ptr,
                                                       std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  if ((row % 3) == 0) {
    row_nnz_new[row] = (row_ptr[row + 1] - row_ptr[row]) + 1;
  } else {
    row_nnz_new[row] = 0;
  }
}

__global__ void _kernel_structural_l1_split_fill(std::int32_t* col_val_new,
                                                 double* nz_val_new,
                                                 const std::int32_t* row_ptr_new,
                                                 const std::int32_t* row_ptr_org,
                                                 const std::int32_t* col_val_org,
                                                 const double* nz_val_org,
                                                 const std::int32_t* t_cols,
                                                 const std::int32_t* e_cols,
                                                 std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || (row % 3) != 0) {
    return;
  }

  const std::int32_t block_idx = row / 3;
  const std::int32_t t_col = t_cols[block_idx];
  const std::int32_t e_col = e_cols[block_idx];
  std::int32_t write_ptr = row_ptr_new[row];
  for (std::int32_t p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    const std::int32_t col = col_val_org[p];
    const double val = nz_val_org[p];
    if (col == e_col) {
      col_val_new[write_ptr] = t_col;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
      col_val_new[write_ptr] = e_col;
      nz_val_new[write_ptr] = -val;
      ++write_ptr;
    } else {
      col_val_new[write_ptr] = col;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
    }
  }
}

__global__ void _kernel_structural_graph_extract_blocks(std::int32_t* status_flag,
                                                        std::int32_t* coupling_rows,
                                                        std::int32_t* t_cols,
                                                        std::int32_t* e_cols,
                                                        double* rhos,
                                                        const std::uint8_t* eq_row,
                                                        const std::uint8_t* zero_lower_two_nnz,
                                                        const std::int32_t* zl_col1,
                                                        const std::int32_t* zl_col2,
                                                        const double* zl_val1,
                                                        const double* zl_val2,
                                                        std::int32_t nblocks) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t eq = 4 * idx;
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  const std::int32_t r3 = eq + 3;
  if (eq_row[eq] == std::uint8_t{0} ||
      zero_lower_two_nnz[r1] == std::uint8_t{0} ||
      zero_lower_two_nnz[r2] == std::uint8_t{0} ||
      zero_lower_two_nnz[r3] == std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  double rho = 0.0;
  if (!_sls_extract_l1_orientation_device(
          zl_col1[r1], zl_col2[r1], zl_val1[r1], zl_val2[r1],
          zl_val1[r2], zl_val2[r2], &t_col, &e_col, &rho)) {
    atomicMax(status_flag, 1);
    return;
  }

  double coeff_t = 0.0;
  double coeff_s = 0.0;
  if (zl_col1[r3] == t_col) {
    coeff_t = zl_val1[r3];
    coeff_s = zl_val2[r3];
  } else if (zl_col2[r3] == t_col) {
    coeff_t = zl_val2[r3];
    coeff_s = zl_val1[r3];
  } else {
    atomicMax(status_flag, 1);
    return;
  }
  if (!(coeff_t < -STRUCTURAL_L1_SUB_TOL && coeff_s > STRUCTURAL_L1_SUB_TOL)) {
    atomicMax(status_flag, 1);
    return;
  }

  coupling_rows[idx] = eq;
  t_cols[idx] = t_col;
  e_cols[idx] = e_col;
  rhos[idx] = rho;
}

__global__ void _kernel_structural_graph_validate_blocks(std::int32_t* status_flag,
                                                         std::uint8_t* block_bad,
                                                         const std::int32_t* row_ptr,
                                                         const std::int32_t* col_val,
                                                         const std::int32_t* at_row_ptr,
                                                         const std::int32_t* at_col_val,
                                                         const double* c,
                                                         const double* l,
                                                         const double* u,
                                                         const std::int32_t* coupling_rows,
                                                         const std::int32_t* t_cols,
                                                         const std::int32_t* e_cols,
                                                         std::int32_t block_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t eq = coupling_rows[idx];
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const bool valid_bounds = isinf(l[e_col]) && isinf(u[e_col]) &&
                            fabs(l[t_col]) <= STRUCTURAL_L1_SUB_TOL &&
                            isinf(u[t_col]) &&
                            c[t_col] >= -STRUCTURAL_L1_SUB_TOL;
  if (!valid_bounds) {
    block_bad[idx] = std::uint8_t{1};
    atomicMax(status_flag, 1);
    return;
  }

  bool found_e = false;
  bool duplicate_target = false;
  const std::int32_t row_start = row_ptr[eq];
  const std::int32_t row_stop = row_ptr[eq + 1];
  for (std::int32_t p = row_start; p < row_stop; ++p) {
    const std::int32_t col_p = col_val[p];
    std::int32_t p_count = 1;
    std::int32_t p_t1 = col_p;
    std::int32_t p_t2 = -1;
    if (col_p == e_col || col_p == t_col) {
      found_e = found_e || (col_p == e_col);
      p_count = 2;
      p_t1 = t_col;
      p_t2 = e_col;
      duplicate_target = duplicate_target || (p_t1 == p_t2);
    }
    for (std::int32_t q = p + 1; q < row_stop; ++q) {
      const std::int32_t col_q = col_val[q];
      std::int32_t q_count = 1;
      std::int32_t q_t1 = col_q;
      std::int32_t q_t2 = -1;
      if (col_q == e_col || col_q == t_col) {
        q_count = 2;
        q_t1 = t_col;
        q_t2 = e_col;
        duplicate_target = duplicate_target || (q_t1 == q_t2);
      }
      if (_sls_outer_targets_overlap(p_t1, p_t2, p_count, q_t1, q_t2, q_count)) {
        duplicate_target = true;
      }
    }
  }

  bool seen_eq = false;
  bool seen_r1 = false;
  bool seen_r2 = false;
  for (std::int32_t p = at_row_ptr[e_col]; p < at_row_ptr[e_col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (row == eq) {
      seen_eq = true;
    } else if (row == r1) {
      seen_r1 = true;
    } else if (row == r2) {
      seen_r2 = true;
    } else {
      block_bad[idx] = std::uint8_t{1};
      atomicMax(status_flag, 1);
      return;
    }
  }

  if (!(found_e && !duplicate_target && seen_eq && seen_r1 && seen_r2)) {
    block_bad[idx] = std::uint8_t{1};
    atomicMax(status_flag, 1);
  }
}

__global__ void _kernel_structural_graph_mark_extra_rows(std::int32_t* status_flag,
                                                         std::uint8_t* block_bad,
                                                         std::int32_t* row_to_block,
                                                         std::int32_t* row_slack_col,
                                                         double* row_factor,
                                                         const std::int32_t* row_ptr,
                                                         const std::int32_t* col_val,
                                                         const double* nz_val,
                                                         const std::int32_t* at_row_ptr,
                                                         const std::int32_t* at_col_val,
                                                         const double* AL,
                                                         const double* AU,
                                                         const std::int32_t* coupling_rows,
                                                         const std::int32_t* t_cols,
                                                         std::int32_t block_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t t_col = t_cols[idx];
  const std::int32_t r1 = coupling_rows[idx] + 1;
  const std::int32_t r2 = coupling_rows[idx] + 2;
  for (std::int32_t p = at_row_ptr[t_col]; p < at_row_ptr[t_col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (row == r1 || row == r2) {
      continue;
    }

    bool valid_row = isfinite(AL[row]) && isinf(AU[row]) &&
                     fabs(AL[row]) <= STRUCTURAL_L1_SUB_TOL;
    const std::int32_t row_start = row_ptr[row];
    const std::int32_t row_stop = row_ptr[row + 1];
    valid_row = valid_row && ((row_stop - row_start) == 2);

    double coeff_t = 0.0;
    double coeff_s = 0.0;
    std::int32_t slack_col = -1;
    if (valid_row) {
      const std::int32_t c1 = col_val[row_start];
      const std::int32_t c2 = col_val[row_start + 1];
      const double v1 = nz_val[row_start];
      const double v2 = nz_val[row_start + 1];
      if (c1 == t_col) {
        coeff_t = v1;
        coeff_s = v2;
        slack_col = c2;
      } else if (c2 == t_col) {
        coeff_t = v2;
        coeff_s = v1;
        slack_col = c1;
      } else {
        valid_row = false;
      }
    }
    valid_row = valid_row && coeff_t < -STRUCTURAL_L1_SUB_TOL && coeff_s > STRUCTURAL_L1_SUB_TOL;
    if (!valid_row) {
      block_bad[idx] = std::uint8_t{1};
      atomicMax(status_flag, 1);
      continue;
    }

    const std::int32_t existing = row_to_block[row];
    if (existing != -1 && existing != idx) {
      block_bad[idx] = std::uint8_t{1};
      atomicMax(status_flag, 1);
      continue;
    }
    row_to_block[row] = idx;
    row_slack_col[row] = slack_col;
    row_factor[row] = -coeff_t / coeff_s;
  }
}

__global__ void _kernel_structural_graph_validate_slacks(std::uint8_t* block_bad,
                                                         std::uint8_t* row_removable,
                                                         std::uint8_t* slack_removable,
                                                         const std::int32_t* row_to_block,
                                                         const std::int32_t* row_slack_col,
                                                         const std::int32_t* at_row_ptr,
                                                         const std::int32_t* at_col_val,
                                                         const double* c,
                                                         const double* u,
                                                         std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }

  const std::int32_t block_idx = row_to_block[row];
  if (block_idx < 0) {
    return;
  }
  const std::int32_t slack_col = row_slack_col[row];
  bool valid_slack = slack_col >= 0 &&
                     fabs(c[slack_col]) <= STRUCTURAL_L1_SUB_TOL &&
                     isinf(u[slack_col]);
  if (valid_slack) {
    for (std::int32_t p = at_row_ptr[slack_col]; p < at_row_ptr[slack_col + 1]; ++p) {
      const std::int32_t s_row = at_col_val[p];
      if (row_to_block[s_row] != block_idx || row_slack_col[s_row] != slack_col) {
        valid_slack = false;
        break;
      }
    }
  }
  if (valid_slack) {
    row_removable[row] = std::uint8_t{1};
    slack_removable[slack_col] = std::uint8_t{1};
  } else {
    block_bad[block_idx] = std::uint8_t{1};
  }
}

__global__ void _kernel_structural_graph_apply_plan(std::int32_t* status_flag,
                                                    std::uint8_t* keep_row,
                                                    std::uint8_t* keep_col,
                                                    double* new_c,
                                                    double* new_l,
                                                    double* new_u,
                                                    const std::uint8_t* block_bad,
                                                    const std::uint8_t* row_removable,
                                                    const std::int32_t* row_slack_col,
                                                    const std::int32_t* at_row_ptr,
                                                    const std::int32_t* at_col_val,
                                                    const double* c,
                                                    const double* l,
                                                    const double* u,
                                                    const std::int32_t* coupling_rows,
                                                    const std::int32_t* t_cols,
                                                    const std::int32_t* e_cols,
                                                    const double* rhos,
                                                    std::int32_t block_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }
  if (block_bad[idx] != std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  const std::int32_t eq = coupling_rows[idx];
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const double rho = rhos[idx];

  keep_row[r1] = std::uint8_t{0};
  keep_row[r2] = std::uint8_t{0};
  for (std::int32_t p = at_row_ptr[t_col]; p < at_row_ptr[t_col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (row == r1 || row == r2) {
      continue;
    }
    if (row_removable[row] != std::uint8_t{0}) {
      keep_row[row] = std::uint8_t{0};
      const std::int32_t slack_col = row_slack_col[row];
      if (slack_col >= 0) {
        keep_col[slack_col] = std::uint8_t{0};
      }
    } else {
      atomicMax(status_flag, 1);
      return;
    }
  }

  const double inv_rho = 1.0 / rho;
  const double c_t_old = c[t_col];
  const double c_e_old = c[e_col];
  new_c[t_col] = c_t_old + c_e_old * inv_rho;
  new_c[e_col] = c_t_old - c_e_old * inv_rho;
  new_l[t_col] = 0.0;
  new_l[e_col] = 0.0;
  new_u[t_col] = INFINITY;
  new_u[e_col] = INFINITY;
  (void)l;
  (void)u;
}

__global__ void _kernel_structural_graph_count_rows(std::int32_t* row_nnz_new,
                                                    const std::uint8_t* coupling_keep,
                                                    const std::int32_t* row_ptr,
                                                    std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  row_nnz_new[row] = coupling_keep[row] != std::uint8_t{0} ? (row_ptr[row + 1] - row_ptr[row]) + 1 : 0;
}

__global__ void _kernel_structural_graph_build_rewrite_maps(std::uint8_t* coupling_keep,
                                                            std::int32_t* row_to_block,
                                                            const std::int32_t* coupling_rows,
                                                            std::int32_t block_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < block_count) {
    const std::int32_t row = coupling_rows[idx];
    coupling_keep[row] = std::uint8_t{1};
    row_to_block[row] = idx;
  }
}

__global__ void _kernel_structural_graph_fill(std::int32_t* col_val_new,
                                              double* nz_val_new,
                                              const std::int32_t* row_ptr_new,
                                              const std::int32_t* row_ptr_org,
                                              const std::int32_t* col_val_org,
                                              const double* nz_val_org,
                                              const std::uint8_t* coupling_keep,
                                              const std::int32_t* row_to_block,
                                              const std::int32_t* t_cols,
                                              const std::int32_t* e_cols,
                                              const double* rhos,
                                              std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || coupling_keep[row] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t block_idx = row_to_block[row];
  const std::int32_t t_col = t_cols[block_idx];
  const std::int32_t e_col = e_cols[block_idx];
  const double rho = rhos[block_idx];
  std::int32_t write_ptr = row_ptr_new[row];
  for (std::int32_t p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    const std::int32_t col = col_val_org[p];
    const double val = nz_val_org[p];
    if (col == e_col) {
      const double coeff = val / rho;
      col_val_new[write_ptr] = t_col;
      nz_val_new[write_ptr] = coeff;
      ++write_ptr;
      col_val_new[write_ptr] = e_col;
      nz_val_new[write_ptr] = -coeff;
      ++write_ptr;
    } else if (col == t_col) {
      col_val_new[write_ptr] = t_col;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
      col_val_new[write_ptr] = e_col;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
    } else {
      col_val_new[write_ptr] = col;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
    }
  }
}

__global__ void _kernel_structural_outer_build_free_to_bound(std::int32_t* free_to_bound,
                                                             const std::int32_t* bound_cols,
                                                             const std::int32_t* free_cols,
                                                             std::int32_t pair_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < pair_count) {
    free_to_bound[free_cols[idx]] = bound_cols[idx];
  }
}

__global__ void _kernel_structural_outer_extract_blocks(std::int32_t* status_flag,
                                                        std::int32_t* q_cols,
                                                        std::int32_t* e_cols,
                                                        std::int32_t* s_cols,
                                                        double* alphas,
                                                        std::int32_t* local_x_cols,
                                                        const std::int32_t* free_to_bound,
                                                        const std::uint8_t* eq_row,
                                                        const std::uint8_t* eq_zero_two_nnz,
                                                        const std::uint8_t* lower_two_nnz,
                                                        const std::uint8_t* zero_lower_two_nnz,
                                                        const std::int32_t* raw_col1,
                                                        const std::int32_t* raw_col2,
                                                        const double* raw_val1,
                                                        const double* raw_val2,
                                                        const std::int32_t* zl_col1,
                                                        const std::int32_t* zl_col2,
                                                        const double* zl_val1,
                                                        const double* zl_val2,
                                                        std::int32_t start_row,
                                                        std::int32_t block_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t dense_row = start_row + idx * 8;
  const std::int32_t ep1 = dense_row + 1;
  const std::int32_t ep2 = dense_row + 2;
  const std::int32_t link = dense_row + 3;

  if (eq_row[dense_row] == std::uint8_t{0} ||
      zero_lower_two_nnz[ep1] == std::uint8_t{0} ||
      zero_lower_two_nnz[ep2] == std::uint8_t{0} ||
      eq_zero_two_nnz[link] == std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t q_col = -1;
  std::int32_t e_col = -1;
  if (!_sls_extract_l1_pair_device(
          zl_col1[ep1], zl_col2[ep1], zl_val1[ep1], zl_val2[ep1],
          zl_val1[ep2], zl_val2[ep2], &q_col, &e_col)) {
    atomicMax(status_flag, 1);
    return;
  }

  const std::int32_t link_c1 = raw_col1[link];
  const std::int32_t link_c2 = raw_col2[link];
  const double link_v1 = raw_val1[link];
  const double link_v2 = raw_val2[link];
  std::int32_t s_col = -1;
  double alpha = 0.0;
  if (link_c1 == q_col) {
    s_col = link_c2;
    alpha = link_v1;
    if (!_near(link_v2, -1.0)) {
      atomicMax(status_flag, 1);
      return;
    }
  } else if (link_c2 == q_col) {
    s_col = link_c1;
    alpha = link_v2;
    if (!_near(link_v1, -1.0)) {
      atomicMax(status_flag, 1);
      return;
    }
  } else {
    atomicMax(status_flag, 1);
    return;
  }
  if (alpha <= 0.0 || q_col == e_col) {
    atomicMax(status_flag, 1);
    return;
  }

  for (std::int32_t k = 0; k < 4; ++k) {
    const std::int32_t rr = link + 1 + k;
    if (lower_two_nnz[rr] == std::uint8_t{0}) {
      atomicMax(status_flag, 1);
      return;
    }
    const std::int32_t loc_c1 = raw_col1[rr];
    const std::int32_t loc_c2 = raw_col2[rr];
    const double loc_v1 = raw_val1[rr];
    const double loc_v2 = raw_val2[rr];
    std::int32_t x_col = -1;
    if (loc_c1 == s_col) {
      if (!_near(loc_v1, -1.0) || !_near(loc_v2, 1.0)) {
        atomicMax(status_flag, 1);
        return;
      }
      x_col = loc_c2;
    } else if (loc_c2 == s_col) {
      if (!_near(loc_v2, -1.0) || !_near(loc_v1, 1.0)) {
        atomicMax(status_flag, 1);
        return;
      }
      x_col = loc_c1;
    } else {
      atomicMax(status_flag, 1);
      return;
    }

    const std::int32_t mapped_x = _sls_outer_map_col_device(free_to_bound, x_col);
    if (mapped_x == q_col || mapped_x == e_col) {
      atomicMax(status_flag, 1);
      return;
    }
    local_x_cols[idx * 4 + k] = x_col;
  }

  q_cols[idx] = q_col;
  e_cols[idx] = e_col;
  s_cols[idx] = s_col;
  alphas[idx] = alpha;
}

__global__ void _kernel_structural_outer_pair_apply(std::int32_t* status_flag,
                                                    std::uint8_t* keep_row,
                                                    std::uint8_t* keep_col,
                                                    double* new_c,
                                                    const double* c,
                                                    const double* l,
                                                    const double* u,
                                                    const std::int32_t* bound_cols,
                                                    const std::int32_t* free_cols,
                                                    std::int32_t pair_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= pair_count) {
    return;
  }

  const std::int32_t bound_col = bound_cols[idx];
  const std::int32_t free_col = free_cols[idx];
  const bool valid_bounds = l[bound_col] == 0.0 && isfinite(u[bound_col]) &&
                            isinf(l[free_col]) && isinf(u[free_col]);
  if (!valid_bounds) {
    atomicMax(status_flag, 1);
    return;
  }

  const std::int32_t row1 = 3 * idx;
  keep_row[row1] = std::uint8_t{0};
  keep_row[row1 + 1] = std::uint8_t{0};
  keep_row[row1 + 2] = std::uint8_t{0};
  keep_col[free_col] = std::uint8_t{0};
  new_c[bound_col] = c[bound_col] + c[free_col];
}

__global__ void _kernel_structural_outer_block_apply(std::int32_t* status_flag,
                                                     std::uint8_t* keep_row,
                                                     std::uint8_t* keep_col,
                                                     double* new_c,
                                                     double* new_l,
                                                     double* new_u,
                                                     const double* c,
                                                     const double* l,
                                                     const double* u,
                                                     const std::int32_t* row_ptr,
                                                     const std::int32_t* col_val,
                                                     const std::int32_t* free_to_bound,
                                                     const std::int32_t* q_cols,
                                                     const std::int32_t* e_cols,
                                                     const std::int32_t* s_cols,
                                                     const double* alphas,
                                                     const std::int32_t* local_x_cols,
                                                     std::int32_t start_row,
                                                     std::int32_t block_count) {
  const std::int32_t idx = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t dense_row = start_row + idx * 8;
  const std::int32_t ep1 = dense_row + 1;
  const std::int32_t ep2 = dense_row + 2;
  const std::int32_t link = dense_row + 3;
  const std::int32_t q_col = q_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const std::int32_t s_col = s_cols[idx];
  const double alpha = alphas[idx];

  const bool valid_bounds = isinf(l[e_col]) && isinf(u[e_col]) &&
                            l[q_col] == 0.0 && isinf(u[q_col]) &&
                            l[s_col] == 0.0 && isinf(u[s_col]);
  if (!valid_bounds) {
    atomicMax(status_flag, 1);
    return;
  }

  for (std::int32_t k = 0; k < 4; ++k) {
    const std::int32_t x_col = local_x_cols[idx * 4 + k];
    if (!(l[x_col] == 0.0 && isinf(u[x_col]) && c[x_col] > STRUCTURAL_L1_SUB_TOL)) {
      atomicMax(status_flag, 1);
      return;
    }
  }

  bool found_e = false;
  bool duplicate_target = false;
  const std::int32_t row_start = row_ptr[dense_row];
  const std::int32_t row_stop = row_ptr[dense_row + 1];
  for (std::int32_t p = row_start; p < row_stop; ++p) {
    const std::int32_t col_p = col_val[p];
    std::int32_t p_count = 1;
    std::int32_t p_t1 = _sls_outer_map_col_device(free_to_bound, col_p);
    std::int32_t p_t2 = -1;
    if (col_p == e_col) {
      found_e = true;
      p_count = 2;
      p_t1 = q_col;
      p_t2 = e_col;
      if (p_t1 == p_t2) duplicate_target = true;
    }

    for (std::int32_t q = p + 1; q < row_stop; ++q) {
      const std::int32_t col_q = col_val[q];
      std::int32_t q_count = 1;
      std::int32_t q_t1 = _sls_outer_map_col_device(free_to_bound, col_q);
      std::int32_t q_t2 = -1;
      if (col_q == e_col) {
        q_count = 2;
        q_t1 = q_col;
        q_t2 = e_col;
        if (q_t1 == q_t2) duplicate_target = true;
      }
      if (_sls_outer_targets_overlap(p_t1, p_t2, p_count, q_t1, q_t2, q_count)) {
        duplicate_target = true;
      }
    }
  }

  if (!found_e || duplicate_target) {
    atomicMax(status_flag, 1);
    return;
  }

  keep_row[ep1] = std::uint8_t{0};
  keep_row[ep2] = std::uint8_t{0};
  keep_row[link] = std::uint8_t{0};
  keep_col[s_col] = std::uint8_t{0};
  const double c_e_old = c[e_col];
  const double c_q_old = c[q_col];
  const double c_s_old = c[s_col];
  new_c[q_col] = c_e_old + c_q_old + alpha * c_s_old;
  new_c[e_col] = -c_e_old + c_q_old + alpha * c_s_old;
  new_l[q_col] = 0.0;
  new_l[e_col] = 0.0;
  new_u[q_col] = INFINITY;
  new_u[e_col] = INFINITY;
}

__global__ void _kernel_structural_outer_count_rows(std::int32_t* row_nnz_new,
                                                    const std::int32_t* row_ptr,
                                                    std::int32_t start_row,
                                                    std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  if (row < start_row) {
    row_nnz_new[row] = 0;
  } else {
    const std::int32_t offset = row - start_row;
    const std::int32_t rem8 = offset % 8;
    if (rem8 == 0 || rem8 >= 4) {
      row_nnz_new[row] = (row_ptr[row + 1] - row_ptr[row]) + 1;
    } else {
      row_nnz_new[row] = 0;
    }
  }
}

__global__ void _kernel_structural_outer_fill(std::int32_t* col_val_new,
                                              double* nz_val_new,
                                              const std::int32_t* row_ptr_new,
                                              const std::int32_t* row_ptr_org,
                                              const std::int32_t* col_val_org,
                                              const double* nz_val_org,
                                              const std::int32_t* free_to_bound,
                                              std::int32_t start_row,
                                              const std::int32_t* q_cols,
                                              const std::int32_t* e_cols,
                                              const std::int32_t* s_cols,
                                              const double* alphas,
                                              std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || row < start_row) {
    return;
  }

  const std::int32_t offset = row - start_row;
  const std::int32_t rem8 = offset % 8;
  if (!(rem8 == 0 || rem8 >= 4)) {
    return;
  }

  const std::int32_t block_idx = offset / 8;
  const std::int32_t q_col = q_cols[block_idx];
  const std::int32_t e_col = e_cols[block_idx];
  const std::int32_t s_col = s_cols[block_idx];
  const double alpha = alphas[block_idx];
  std::int32_t write_ptr = row_ptr_new[row];

  for (std::int32_t p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    const std::int32_t col_org = col_val_org[p];
    const double val = nz_val_org[p];
    const std::int32_t mapped = _sls_outer_map_col_device(free_to_bound, col_org);
    if (rem8 == 0 && col_org == e_col) {
      col_val_new[write_ptr] = q_col;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
      col_val_new[write_ptr] = e_col;
      nz_val_new[write_ptr] = -val;
      ++write_ptr;
    } else if (rem8 >= 4 && col_org == s_col) {
      const double coeff = val * alpha;
      col_val_new[write_ptr] = q_col;
      nz_val_new[write_ptr] = coeff;
      ++write_ptr;
      col_val_new[write_ptr] = e_col;
      nz_val_new[write_ptr] = coeff;
      ++write_ptr;
    } else {
      col_val_new[write_ptr] = mapped;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
    }
  }
}

DeviceCsrMatrix build_structural_outer_new_A(const DeviceCsrMatrix& source,
                                             const std::int32_t* free_to_bound,
                                             std::int32_t start_row,
                                             const std::int32_t* q_cols,
                                             const std::int32_t* e_cols,
                                             const std::int32_t* s_cols,
                                             const double* alphas) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  throw_if_cuda_error(cudaMalloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows + 1)),
                      "cudaMalloc structural outer rowPtr");

  if (out.rows == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(std::int32_t)), "cudaMemset structural outer empty rowPtr");
    return out;
  }

  std::int32_t* row_counts = nullptr;
  throw_if_cuda_error(cudaMalloc(&row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural outer row_counts");
  const int row_blocks = (out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_outer_count_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, source.rowPtr, start_row, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_count_rows");
  _kernel_inclusive_scan_i32_serial<<<1, 1>>>(row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_inclusive_scan_i32_serial structural outer");
  throw_if_cuda_error(cudaMemcpy(&out.nnz, row_counts + out.rows - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural outer nnz");
  _kernel_row_ptr_from_prefix<<<row_blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix structural outer");

  if (out.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural outer colVal");
    throw_if_cuda_error(cudaMalloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural outer nzVal");
    _kernel_structural_outer_fill<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal,
        free_to_bound, start_row, q_cols, e_cols, s_cols, alphas, out.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_fill");
  }
  cudaFree(row_counts);
  return out;
}

DeviceCsrMatrix build_structural_l1_split_new_A(const DeviceCsrMatrix& source,
                                                const std::int32_t* t_cols,
                                                const std::int32_t* e_cols) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  throw_if_cuda_error(cudaMalloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows + 1)),
                      "cudaMalloc structural l1 split rowPtr");

  if (out.rows == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(std::int32_t)), "cudaMemset structural l1 split empty rowPtr");
    return out;
  }

  std::int32_t* row_counts = nullptr;
  throw_if_cuda_error(cudaMalloc(&row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural l1 split row_counts");
  const int row_blocks = (out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_l1_split_count_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, source.rowPtr, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_l1_split_count_rows");
  _kernel_inclusive_scan_i32_serial<<<1, 1>>>(row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_inclusive_scan_i32_serial structural l1 split");
  throw_if_cuda_error(cudaMemcpy(&out.nnz, row_counts + out.rows - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split nnz");
  _kernel_row_ptr_from_prefix<<<row_blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix structural l1 split");

  if (out.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural l1 split colVal");
    throw_if_cuda_error(cudaMalloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural l1 split nzVal");
    _kernel_structural_l1_split_fill<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal,
        t_cols, e_cols, out.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_l1_split_fill");
  }
  cudaFree(row_counts);
  return out;
}

DeviceCsrMatrix build_structural_graph_new_A(const DeviceCsrMatrix& source,
                                             const std::int32_t* coupling_rows,
                                             const std::int32_t* t_cols,
                                             const std::int32_t* e_cols,
                                             const double* rhos,
                                             std::int32_t block_count) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  throw_if_cuda_error(cudaMalloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows + 1)),
                      "cudaMalloc structural graph rowPtr");
  if (out.rows == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(std::int32_t)), "cudaMemset structural graph empty rowPtr");
    return out;
  }

  std::uint8_t* coupling_keep = nullptr;
  std::int32_t* row_to_block = nullptr;
  std::int32_t* row_counts = nullptr;
  throw_if_cuda_error(cudaMalloc(&coupling_keep, static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural graph coupling_keep");
  throw_if_cuda_error(cudaMalloc(&row_to_block, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural graph row_to_block");
  throw_if_cuda_error(cudaMalloc(&row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural graph row_counts");
  throw_if_cuda_error(cudaMemset(coupling_keep, 0, static_cast<std::size_t>(out.rows)),
                      "cudaMemset structural graph coupling_keep");
  _kernel_fill_i32<<<(out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS, GPU_PRESOLVE_THREADS>>>(
      row_to_block, -1, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph row_to_block");
  if (block_count > 0) {
    const int block_blocks = (block_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_structural_graph_build_rewrite_maps<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        coupling_keep, row_to_block, coupling_rows, block_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_build_rewrite_maps");
  }

  const int row_blocks = (out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_graph_count_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, coupling_keep, source.rowPtr, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_count_rows");
  _kernel_inclusive_scan_i32_serial<<<1, 1>>>(row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_inclusive_scan_i32_serial structural graph");
  throw_if_cuda_error(cudaMemcpy(&out.nnz, row_counts + out.rows - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph nnz");
  _kernel_row_ptr_from_prefix<<<row_blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix structural graph");

  if (out.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural graph colVal");
    throw_if_cuda_error(cudaMalloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural graph nzVal");
    _kernel_structural_graph_fill<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal,
        coupling_keep, row_to_block, t_cols, e_cols, rhos, out.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_fill");
  }

  cudaFree(coupling_keep);
  cudaFree(row_to_block);
  cudaFree(row_counts);
  return out;
}

template <class T>
std::vector<T> copy_device_vector(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

void record_l1_split_recovery(PresolvePlanGpu& plan,
                              const std::int32_t* t_cols,
                              const std::int32_t* e_cols,
                              const double* rhos,
                              std::int32_t count,
                              const char* pattern) {
  const std::vector<std::int32_t> t_h = copy_device_vector(t_cols, count, "cudaMemcpy structural recovery t_cols");
  const std::vector<std::int32_t> e_h = copy_device_vector(e_cols, count, "cudaMemcpy structural recovery e_cols");
  std::vector<double> rho_h(static_cast<std::size_t>(count), 1.0);
  if (rhos != nullptr) {
    rho_h = copy_device_vector(rhos, count, "cudaMemcpy structural recovery rhos");
  }
  plan.structural_primal_recovery.pattern = pattern;
  plan.structural_primal_recovery.splits.clear();
  plan.structural_primal_recovery.outer_pairs.clear();
  plan.structural_primal_recovery.linked_slacks.clear();
  plan.structural_primal_recovery.max_slacks.clear();
  plan.structural_primal_recovery.splits.reserve(static_cast<std::size_t>(count));
  for (std::int32_t i = 0; i < count; ++i) {
    plan.structural_primal_recovery.splits.push_back({t_h[static_cast<std::size_t>(i)],
                                                       e_h[static_cast<std::size_t>(i)],
                                                       rho_h[static_cast<std::size_t>(i)]});
  }
  plan.has_structural_primal_recovery = true;
}

bool try_apply_l1_split(PresolvePlanGpu& plan,
                        const LPInfoGpu& lp,
                        const DeviceCsrMatrix& source_A,
                        const std::uint8_t* eq_row,
                        const std::uint8_t* zero_lower_two_nnz,
                        const std::int32_t* zl_col1,
                        const std::int32_t* zl_col2,
                        const double* zl_val1,
                        const double* zl_val2,
                        double residual_bound_as_free_min,
                        bool profile) {
  const std::int32_t m = source_A.rows;
  if (m < 3 || (m % 3) != 0 || plan.has_new_A) {
    return false;
  }
  const std::int32_t nblocks = m / 3;
  const int blocks = (nblocks + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* status_flag = nullptr;
  std::int32_t* t_cols = nullptr;
  std::int32_t* e_cols = nullptr;
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), "cudaMalloc structural l1 split status");
  throw_if_cuda_error(cudaMalloc(&t_cols, sizeof(std::int32_t) * static_cast<std::size_t>(nblocks)),
                      "cudaMalloc structural l1 split t_cols");
  throw_if_cuda_error(cudaMalloc(&e_cols, sizeof(std::int32_t) * static_cast<std::size_t>(nblocks)),
                      "cudaMalloc structural l1 split e_cols");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural l1 split status");

  _kernel_structural_l1_split_extract_pairs<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, t_cols, e_cols, eq_row, zero_lower_two_nnz,
      zl_col1, zl_col2, zl_val1, zl_val2, nblocks);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_l1_split_extract_pairs");
  std::int32_t status = 0;
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split extract status");
  if (profile) {
    std::cerr << ">>> [structural_l1 C++] l1_split extract status=" << status
              << " nblocks=" << nblocks << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(t_cols);
    cudaFree(e_cols);
    return false;
  }

  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural l1 split apply status");
  _kernel_structural_l1_split_build_plan_data<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, plan.keep_row_mask, plan.new_c, plan.new_l, plan.new_u,
      plan.new_c, plan.new_l, plan.new_u, source_A.rowPtr, source_A.colVal,
      t_cols, e_cols, residual_bound_as_free_min, nblocks);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_l1_split_build_plan_data");
  throw_if_cuda_error(cudaDeviceSynchronize(), "structural l1 split apply synchronize");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split apply status");
  if (profile) {
    std::cerr << ">>> [structural_l1 C++] l1_split apply status=" << status << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(t_cols);
    cudaFree(e_cols);
    return false;
  }

  plan.new_A = build_structural_l1_split_new_A(source_A, t_cols, e_cols);
  plan.has_new_A = true;
  plan.has_change = true;
  plan.has_col_action = true;
  plan.has_row_action = true;
  record_l1_split_recovery(plan, t_cols, e_cols, nullptr, nblocks, "l1_split_3row");
  if (profile) {
    std::cerr << ">>> [structural_l1 C++] l1_split changed=1 new_A_nnz=" << plan.new_A.nnz << "\n";
  }

  cudaFree(status_flag);
  cudaFree(t_cols);
  cudaFree(e_cols);
  (void)lp;
  return true;
}

bool try_apply_graph_l1(PresolvePlanGpu& plan,
                        const LPInfoGpu& lp,
                        const DeviceCsrMatrix& source_A,
                        const std::uint8_t* eq_row,
                        const std::uint8_t* zero_lower_two_nnz,
                        const std::int32_t* zl_col1,
                        const std::int32_t* zl_col2,
                        const double* zl_val1,
                        const double* zl_val2,
                        bool profile) {
  const std::int32_t m = source_A.rows;
  const std::int32_t n = source_A.cols;
  if (m < 4 || (m % 4) != 0 || plan.has_new_A) {
    return false;
  }
  const std::int32_t block_count = m / 4;
  const int block_blocks = (block_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int row_blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;

  std::int32_t* status_flag = nullptr;
  std::int32_t* coupling_rows = nullptr;
  std::int32_t* t_cols = nullptr;
  std::int32_t* e_cols = nullptr;
  double* rhos = nullptr;
  std::uint8_t* block_bad = nullptr;
  std::int32_t* row_to_block = nullptr;
  std::int32_t* row_slack_col = nullptr;
  double* row_factor = nullptr;
  std::uint8_t* row_removable = nullptr;
  std::uint8_t* slack_removable = nullptr;

  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), "cudaMalloc structural graph status");
  throw_if_cuda_error(cudaMalloc(&coupling_rows, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph coupling_rows");
  throw_if_cuda_error(cudaMalloc(&t_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph t_cols");
  throw_if_cuda_error(cudaMalloc(&e_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph e_cols");
  throw_if_cuda_error(cudaMalloc(&rhos, sizeof(double) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph rhos");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural graph status");

  _kernel_structural_graph_extract_blocks<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, coupling_rows, t_cols, e_cols, rhos, eq_row, zero_lower_two_nnz,
      zl_col1, zl_col2, zl_val1, zl_val2, block_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_extract_blocks");
  std::int32_t status = 0;
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph extract status");
  if (profile) {
    std::cerr << ">>> [structural_l1 C++] graph extract status=" << status
              << " block_count=" << block_count << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(coupling_rows);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(rhos);
    return false;
  }

  const std::vector<std::int32_t> t_h = copy_device_vector(t_cols, block_count, "cudaMemcpy graph t_cols");
  const std::vector<std::int32_t> e_h = copy_device_vector(e_cols, block_count, "cudaMemcpy graph e_cols");
  for (std::int32_t i = 0; i < block_count; ++i) {
    for (std::int32_t j = 0; j < i; ++j) {
      if (t_h[static_cast<std::size_t>(i)] == t_h[static_cast<std::size_t>(j)] ||
          e_h[static_cast<std::size_t>(i)] == e_h[static_cast<std::size_t>(j)]) {
        cudaFree(status_flag);
        cudaFree(coupling_rows);
        cudaFree(t_cols);
        cudaFree(e_cols);
        cudaFree(rhos);
        return false;
      }
    }
  }

  throw_if_cuda_error(cudaMalloc(&block_bad, static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph block_bad");
  throw_if_cuda_error(cudaMalloc(&row_to_block, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_to_block");
  throw_if_cuda_error(cudaMalloc(&row_slack_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_slack_col");
  throw_if_cuda_error(cudaMalloc(&row_factor, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_factor");
  throw_if_cuda_error(cudaMalloc(&row_removable, static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_removable");
  throw_if_cuda_error(cudaMalloc(&slack_removable, static_cast<std::size_t>(n)),
                      "cudaMalloc structural graph slack_removable");
  throw_if_cuda_error(cudaMemset(block_bad, 0, static_cast<std::size_t>(block_count)),
                      "cudaMemset structural graph block_bad");
  _kernel_fill_i32<<<row_blocks, GPU_PRESOLVE_THREADS>>>(row_to_block, -1, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph row_to_block");
  _kernel_fill_i32<<<row_blocks, GPU_PRESOLVE_THREADS>>>(row_slack_col, -1, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph row_slack_col");
  throw_if_cuda_error(cudaMemset(row_factor, 0, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMemset structural graph row_factor");
  throw_if_cuda_error(cudaMemset(row_removable, 0, static_cast<std::size_t>(m)),
                      "cudaMemset structural graph row_removable");
  throw_if_cuda_error(cudaMemset(slack_removable, 0, static_cast<std::size_t>(n)),
                      "cudaMemset structural graph slack_removable");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural graph validate status");

  _kernel_structural_graph_validate_blocks<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, block_bad, source_A.rowPtr, source_A.colVal, lp.AT.rowPtr, lp.AT.colVal,
      plan.new_c, plan.new_l, plan.new_u, coupling_rows, t_cols, e_cols, block_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_validate_blocks");
  _kernel_structural_graph_mark_extra_rows<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, block_bad, row_to_block, row_slack_col, row_factor,
      source_A.rowPtr, source_A.colVal, source_A.nzVal, lp.AT.rowPtr, lp.AT.colVal,
      plan.new_AL, plan.new_AU, coupling_rows, t_cols, block_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_mark_extra_rows");
  _kernel_structural_graph_validate_slacks<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      block_bad, row_removable, slack_removable, row_to_block, row_slack_col,
      lp.AT.rowPtr, lp.AT.colVal, plan.new_c, plan.new_u, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_validate_slacks");
  _kernel_structural_graph_apply_plan<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, plan.keep_row_mask, plan.keep_col_mask, plan.new_c, plan.new_l, plan.new_u,
      block_bad, row_removable, row_slack_col, lp.AT.rowPtr, lp.AT.colVal,
      plan.new_c, plan.new_l, plan.new_u, coupling_rows, t_cols, e_cols, rhos, block_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_apply_plan");
  throw_if_cuda_error(cudaDeviceSynchronize(), "structural graph apply synchronize");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph apply status");
  if (profile) {
    std::cerr << ">>> [structural_l1 C++] graph apply status=" << status << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(coupling_rows);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(rhos);
    cudaFree(block_bad);
    cudaFree(row_to_block);
    cudaFree(row_slack_col);
    cudaFree(row_factor);
    cudaFree(row_removable);
    cudaFree(slack_removable);
    return false;
  }

  plan.new_A = build_structural_graph_new_A(source_A, coupling_rows, t_cols, e_cols, rhos, block_count);
  plan.has_new_A = true;
  plan.has_change = true;
  plan.has_col_action = true;
  plan.has_row_action = true;
  record_l1_split_recovery(plan, t_cols, e_cols, rhos, block_count, "graph_l1_substitution");

  const std::vector<std::int32_t> row_block_h = copy_device_vector(row_to_block, m, "cudaMemcpy graph row_to_block");
  const std::vector<std::int32_t> row_slack_h = copy_device_vector(row_slack_col, m, "cudaMemcpy graph row_slack_col");
  const std::vector<double> row_factor_h = copy_device_vector(row_factor, m, "cudaMemcpy graph row_factor");
  const std::vector<std::uint8_t> row_removable_h =
      copy_device_vector(row_removable, m, "cudaMemcpy graph row_removable");
  for (std::int32_t row = 0; row < m; ++row) {
    if (row_removable_h[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
      continue;
    }
    const std::int32_t block_idx = row_block_h[static_cast<std::size_t>(row)];
    const std::int32_t slack_col = row_slack_h[static_cast<std::size_t>(row)];
    if (block_idx >= 0 && slack_col >= 0) {
      StructuralMaxSlackRecovery* recovery = nullptr;
      for (StructuralMaxSlackRecovery& existing : plan.structural_primal_recovery.max_slacks) {
        if (existing.slack_col == slack_col) {
          recovery = &existing;
          break;
        }
      }
      if (recovery == nullptr) {
        plan.structural_primal_recovery.max_slacks.push_back({slack_col, {}, {}});
        recovery = &plan.structural_primal_recovery.max_slacks.back();
      }
      recovery->t_cols.push_back(t_h[static_cast<std::size_t>(block_idx)]);
      recovery->factors.push_back(row_factor_h[static_cast<std::size_t>(row)]);
    }
  }

  if (profile) {
    std::cerr << ">>> [structural_l1 C++] graph changed=1 new_A_nnz=" << plan.new_A.nnz
              << " max_slacks=" << plan.structural_primal_recovery.max_slacks.size() << "\n";
  }

  cudaFree(status_flag);
  cudaFree(coupling_rows);
  cudaFree(t_cols);
  cudaFree(e_cols);
  cudaFree(rhos);
  cudaFree(block_bad);
  cudaFree(row_to_block);
  cudaFree(row_slack_col);
  cudaFree(row_factor);
  cudaFree(row_removable);
  cudaFree(slack_removable);
  return true;
}

struct PrefixRow2 {
  std::uint8_t is_eq = 0;
  std::uint8_t zero_lower_two_nnz = 0;
  std::int32_t zl_col1 = -1;
  std::int32_t zl_col2 = -1;
  double zl_val1 = 0.0;
  double zl_val2 = 0.0;
};

__device__ PrefixRow2 _sls_prefix_row2_device(std::int32_t row,
                                              const std::int32_t* row_ptr,
                                              const std::int32_t* col_val,
                                              const double* nz_val,
                                              const double* AL,
                                              const double* AU) {
  PrefixRow2 out;
  const double al = AL[row];
  const double au = AU[row];
  const std::int32_t lo = row_ptr[row];
  const std::int32_t hi = row_ptr[row + 1];
  const bool is_eq = isfinite(al) && isfinite(au) && fabs(al - au) <= STRUCTURAL_L1_SUB_TOL;
  out.is_eq = is_eq ? std::uint8_t{1} : std::uint8_t{0};
  if (hi - lo != 2) {
    return out;
  }

  std::int32_t c1 = col_val[lo];
  std::int32_t c2 = col_val[lo + 1];
  double v1 = nz_val[lo];
  double v2 = nz_val[lo + 1];
  if (c2 < c1) {
    const std::int32_t tc = c1;
    c1 = c2;
    c2 = tc;
    const double tv = v1;
    v1 = v2;
    v2 = tv;
  }

  const bool is_zero_lower = isfinite(al) && isinf(au) && fabs(al) <= STRUCTURAL_L1_SUB_TOL;
  const bool is_zero_upper = isinf(al) && isfinite(au) && fabs(au) <= STRUCTURAL_L1_SUB_TOL;
  if (is_zero_lower || is_zero_upper) {
    const double scale = is_zero_lower ? 1.0 : -1.0;
    out.zero_lower_two_nnz = std::uint8_t{1};
    out.zl_col1 = c1;
    out.zl_col2 = c2;
    out.zl_val1 = scale * v1;
    out.zl_val2 = scale * v2;
  }
  return out;
}

__device__ bool _sls_prefix_has_l1_pair_device(const PrefixRow2& a, const PrefixRow2& b) {
  if (a.zl_col1 != b.zl_col1 || a.zl_col2 != b.zl_col2) {
    return false;
  }
  const bool first_is_t =
      a.zl_val1 > STRUCTURAL_L1_SUB_TOL &&
      _near(a.zl_val1, b.zl_val1) &&
      _near(a.zl_val2, -a.zl_val1) &&
      _near(b.zl_val2, b.zl_val1);
  const bool second_is_t =
      a.zl_val2 > STRUCTURAL_L1_SUB_TOL &&
      _near(a.zl_val2, b.zl_val2) &&
      _near(a.zl_val1, -a.zl_val2) &&
      _near(b.zl_val1, b.zl_val2);
  return first_is_t || second_is_t;
}

__device__ bool _sls_prefix_has_outer_pair_signature_device(const PrefixRow2& a,
                                                            const PrefixRow2& b,
                                                            const PrefixRow2& c) {
  if (a.zl_col1 != b.zl_col1 || a.zl_col1 != c.zl_col1 ||
      a.zl_col2 != b.zl_col2 || a.zl_col2 != c.zl_col2) {
    return false;
  }
  return _near(a.zl_val1, 1.0) && _near(a.zl_val2, -1.0) &&
         _near(b.zl_val1, 1.0) && _near(b.zl_val2, 1.0) &&
         _near(c.zl_val1, -1.0) && _near(c.zl_val2, 1.0);
}

__device__ bool _sls_prefix_possible_l1_split_device(const std::int32_t* row_ptr,
                                                     const std::int32_t* col_val,
                                                     const double* nz_val,
                                                     const double* AL,
                                                     const double* AU,
                                                     std::int32_t m) {
  if (m < 3 || (m % 3) != 0) {
    return false;
  }
  const PrefixRow2 eq = _sls_prefix_row2_device(0, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r1 = _sls_prefix_row2_device(1, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r2 = _sls_prefix_row2_device(2, row_ptr, col_val, nz_val, AL, AU);
  return eq.is_eq != 0 &&
         r1.zero_lower_two_nnz != 0 &&
         r2.zero_lower_two_nnz != 0 &&
         _sls_prefix_has_l1_pair_device(r1, r2);
}

__device__ bool _sls_prefix_possible_outer_pair_device(const std::int32_t* row_ptr,
                                                       const std::int32_t* col_val,
                                                       const double* nz_val,
                                                       const double* AL,
                                                       const double* AU,
                                                       std::int32_t m) {
  if (m < 11) {
    return false;
  }
  const PrefixRow2 r1 = _sls_prefix_row2_device(0, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r2 = _sls_prefix_row2_device(1, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r3 = _sls_prefix_row2_device(2, row_ptr, col_val, nz_val, AL, AU);
  return r1.zero_lower_two_nnz != 0 &&
         r2.zero_lower_two_nnz != 0 &&
         r3.zero_lower_two_nnz != 0 &&
         _sls_prefix_has_outer_pair_signature_device(r1, r2, r3);
}

__device__ bool _sls_prefix_possible_graph_device(const std::int32_t* row_ptr,
                                                  const std::int32_t* col_val,
                                                  const double* nz_val,
                                                  const double* AL,
                                                  const double* AU,
                                                  std::int32_t m) {
  if (m < 4 || (m % 4) != 0) {
    return false;
  }
  const PrefixRow2 eq = _sls_prefix_row2_device(0, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r1 = _sls_prefix_row2_device(1, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r2 = _sls_prefix_row2_device(2, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r3 = _sls_prefix_row2_device(3, row_ptr, col_val, nz_val, AL, AU);
  const bool has_pos = r3.zl_val1 > STRUCTURAL_L1_SUB_TOL || r3.zl_val2 > STRUCTURAL_L1_SUB_TOL;
  const bool has_neg = r3.zl_val1 < -STRUCTURAL_L1_SUB_TOL || r3.zl_val2 < -STRUCTURAL_L1_SUB_TOL;
  const bool overlaps = r3.zl_col1 == r1.zl_col1 || r3.zl_col1 == r1.zl_col2 ||
                        r3.zl_col2 == r1.zl_col1 || r3.zl_col2 == r1.zl_col2;
  return eq.is_eq != 0 &&
         r1.zero_lower_two_nnz != 0 &&
         r2.zero_lower_two_nnz != 0 &&
         _sls_prefix_has_l1_pair_device(r1, r2) &&
         r3.zero_lower_two_nnz != 0 &&
         has_pos && has_neg && overlaps;
}

__global__ void _kernel_structural_prefix_auto_screen(std::uint8_t* screen_flag,
                                                      const std::int32_t* row_ptr,
                                                      const std::int32_t* col_val,
                                                      const double* nz_val,
                                                      const double* AL,
                                                      const double* AU,
                                                      std::int32_t m,
                                                      std::int32_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const bool passed =
        m > 0 && n > 0 &&
        (_sls_prefix_possible_graph_device(row_ptr, col_val, nz_val, AL, AU, m) ||
         _sls_prefix_possible_outer_pair_device(row_ptr, col_val, nz_val, AL, AU, m) ||
         _sls_prefix_possible_l1_split_device(row_ptr, col_val, nz_val, AL, AU, m));
    screen_flag[0] = passed ? std::uint8_t{1} : std::uint8_t{0};
  }
}

}  // namespace

bool structural_l1_prefix_screen_passes(const LPInfoGpu& lp) {
  if (lp.A.rows <= 0 || lp.A.cols <= 0) {
    return false;
  }
  std::uint8_t* screen_flag = nullptr;
  throw_if_cuda_error(cudaMalloc(&screen_flag, sizeof(std::uint8_t)), "cudaMalloc structural prefix screen");
  _kernel_structural_prefix_auto_screen<<<1, 1>>>(
      screen_flag, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, lp.AL, lp.AU, lp.A.rows, lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_prefix_auto_screen");
  std::uint8_t host_flag = 0;
  throw_if_cuda_error(cudaMemcpy(&host_flag, screen_flag, sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural prefix screen");
  cudaFree(screen_flag);
  return host_flag != std::uint8_t{0};
}

void apply_rule_structural_l1_substitution(PresolvePlanGpu& plan,
                                           const LPInfoGpu& lp,
                                           const PresolveStatsGpu& stats,
                                           const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded || !pparams.enable_structural_l1_substitution) {
    return;
  }

  const DeviceCsrMatrix& source_A = plan.has_new_A ? plan.new_A : lp.A;
  const std::int32_t m = source_A.rows;
  const std::int32_t n = source_A.cols;
  const bool profile = env_enabled("GPUPRESOLVER_STRUCTURAL_L1_PROFILE");
  if (m < 3 || n < 2) {
    if (profile) {
      std::cerr << ">>> [structural_l1 C++] skip small m=" << m << " n=" << n << "\n";
    }
    return;
  }

  const int row_blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::uint8_t* eq_row = nullptr;
  std::uint8_t* eq_zero_two_nnz = nullptr;
  std::uint8_t* lower_two_nnz = nullptr;
  std::uint8_t* zero_lower_two_nnz = nullptr;
  std::int32_t* raw_col1 = nullptr;
  std::int32_t* raw_col2 = nullptr;
  double* raw_val1 = nullptr;
  double* raw_val2 = nullptr;
  std::int32_t* zl_col1 = nullptr;
  std::int32_t* zl_col2 = nullptr;
  double* zl_val1 = nullptr;
  double* zl_val2 = nullptr;
  std::int32_t* status_flag = nullptr;
  std::int32_t* pair_count_d = nullptr;
  std::int32_t* start_row_d = nullptr;
  std::int32_t* bound_cols = nullptr;
  std::int32_t* free_cols = nullptr;
  std::int32_t* free_to_bound = nullptr;
  std::int32_t* q_cols = nullptr;
  std::int32_t* e_cols = nullptr;
  std::int32_t* s_cols = nullptr;
  double* alphas = nullptr;
  std::int32_t* local_x_cols = nullptr;
  std::int32_t status = 0;
  std::int32_t pair_count = 0;
  std::int32_t start_row = 0;

  throw_if_cuda_error(cudaMalloc(&eq_row, static_cast<std::size_t>(m)), "cudaMalloc structural eq_row");
  throw_if_cuda_error(cudaMalloc(&eq_zero_two_nnz, static_cast<std::size_t>(m)), "cudaMalloc structural eq_zero");
  throw_if_cuda_error(cudaMalloc(&lower_two_nnz, static_cast<std::size_t>(m)), "cudaMalloc structural lower_two");
  throw_if_cuda_error(cudaMalloc(&zero_lower_two_nnz, static_cast<std::size_t>(m)), "cudaMalloc structural zero_lower");
  throw_if_cuda_error(cudaMalloc(&raw_col1, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_col1");
  throw_if_cuda_error(cudaMalloc(&raw_col2, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_col2");
  throw_if_cuda_error(cudaMalloc(&raw_val1, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_val1");
  throw_if_cuda_error(cudaMalloc(&raw_val2, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_val2");
  throw_if_cuda_error(cudaMalloc(&zl_col1, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_col1");
  throw_if_cuda_error(cudaMalloc(&zl_col2, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_col2");
  throw_if_cuda_error(cudaMalloc(&zl_val1, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_val1");
  throw_if_cuda_error(cudaMalloc(&zl_val2, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_val2");
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), "cudaMalloc structural status");
  throw_if_cuda_error(cudaMalloc(&pair_count_d, sizeof(std::int32_t)), "cudaMalloc structural pair_count");
  throw_if_cuda_error(cudaMalloc(&start_row_d, sizeof(std::int32_t)), "cudaMalloc structural start_row");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural status");

  _kernel_structural_row_metadata<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      eq_row, eq_zero_two_nnz, lower_two_nnz, zero_lower_two_nnz,
      raw_col1, raw_col2, raw_val1, raw_val2, zl_col1, zl_col2, zl_val1, zl_val2,
      source_A.rowPtr, source_A.colVal, source_A.nzVal, plan.new_AL, plan.new_AU, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_row_metadata");

  if (try_apply_l1_split(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                         zl_col1, zl_col2, zl_val1, zl_val2,
                         pparams.structural_l1_residual_bound_as_free_min, profile)) {
    goto cleanup;
  }

  _kernel_structural_outer_count_pairs<<<1, 1>>>(
      status_flag, pair_count_d, start_row_d, zero_lower_two_nnz,
      zl_col1, zl_col2, zl_val1, zl_val2, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_count_pairs");

  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural status count_pairs");
  throw_if_cuda_error(cudaMemcpy(&pair_count, pair_count_d, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural pair_count");
  throw_if_cuda_error(cudaMemcpy(&start_row, start_row_d, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural start_row");
  if (profile) {
    std::cerr << ">>> [structural_l1 C++] count_pairs status=" << status
              << " pair_count=" << pair_count << " start_row0=" << start_row
              << " m=" << m << " n=" << n << "\n";
  }
  if (status != 0 || pair_count <= 0) {
    (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                             zl_col1, zl_col2, zl_val1, zl_val2, profile);
    goto cleanup;
  }

  {
    const int pair_blocks = (pair_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    const std::int32_t block_count = (m - start_row) / 8;
    const int block_blocks = (block_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;

    throw_if_cuda_error(cudaMalloc(&bound_cols, sizeof(std::int32_t) * static_cast<std::size_t>(pair_count)),
                        "cudaMalloc structural bound_cols");
    throw_if_cuda_error(cudaMalloc(&free_cols, sizeof(std::int32_t) * static_cast<std::size_t>(pair_count)),
                        "cudaMalloc structural free_cols");
    _kernel_structural_outer_extract_pairs<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        bound_cols, free_cols, zl_col1, zl_col2, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_extract_pairs");

    throw_if_cuda_error(cudaMalloc(&free_to_bound, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                        "cudaMalloc structural free_to_bound");
    _kernel_fill_i32<<<(n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS, GPU_PRESOLVE_THREADS>>>(
        free_to_bound, -1, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural free_to_bound");
    _kernel_structural_outer_build_free_to_bound<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        free_to_bound, bound_cols, free_cols, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_build_free_to_bound");

    if (block_count <= 0) {
      if (profile) {
        std::cerr << ">>> [structural_l1 C++] skip block_count=" << block_count << "\n";
      }
      (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                               zl_col1, zl_col2, zl_val1, zl_val2, profile);
      goto cleanup;
    }
    throw_if_cuda_error(cudaMalloc(&q_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural q_cols");
    throw_if_cuda_error(cudaMalloc(&e_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural e_cols");
    throw_if_cuda_error(cudaMalloc(&s_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural s_cols");
    throw_if_cuda_error(cudaMalloc(&alphas, sizeof(double) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural alphas");
    throw_if_cuda_error(cudaMalloc(&local_x_cols, sizeof(std::int32_t) * static_cast<std::size_t>(4 * block_count)),
                        "cudaMalloc structural local_x_cols");
    throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural status blocks");
    _kernel_structural_outer_extract_blocks<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, q_cols, e_cols, s_cols, alphas, local_x_cols, free_to_bound,
        eq_row, eq_zero_two_nnz, lower_two_nnz, zero_lower_two_nnz,
        raw_col1, raw_col2, raw_val1, raw_val2, zl_col1, zl_col2, zl_val1, zl_val2,
        start_row, block_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_extract_blocks");
    throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy structural status extract_blocks");
    if (profile) {
      std::cerr << ">>> [structural_l1 C++] extract_blocks status=" << status
                << " block_count=" << block_count << "\n";
    }
    if (status != 0) {
      (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                               zl_col1, zl_col2, zl_val1, zl_val2, profile);
      goto cleanup;
    }

    throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural status apply");
    _kernel_structural_outer_pair_apply<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, plan.keep_row_mask, plan.keep_col_mask, plan.new_c,
        plan.new_c, plan.new_l, plan.new_u, bound_cols, free_cols, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_pair_apply");
    _kernel_structural_outer_block_apply<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, plan.keep_row_mask, plan.keep_col_mask, plan.new_c, plan.new_l, plan.new_u,
        plan.new_c, plan.new_l, plan.new_u, source_A.rowPtr, source_A.colVal, free_to_bound,
        q_cols, e_cols, s_cols, alphas, local_x_cols, start_row, block_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_block_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "structural outer apply synchronize");
    throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy structural status apply");
    if (profile) {
      std::cerr << ">>> [structural_l1 C++] apply status=" << status << "\n";
    }
    if (status != 0) {
      goto cleanup;
    }

    plan.new_A = build_structural_outer_new_A(source_A, free_to_bound, start_row, q_cols, e_cols, s_cols, alphas);
    plan.has_new_A = true;
    plan.has_change = true;
    plan.has_col_action = true;
    plan.has_row_action = true;
    record_l1_split_recovery(plan, q_cols, e_cols, nullptr, block_count, "outer_pair_linked_l1");
    {
      const std::vector<std::int32_t> bound_h =
          copy_device_vector(bound_cols, pair_count, "cudaMemcpy structural outer bound_cols recovery");
      const std::vector<std::int32_t> free_h =
          copy_device_vector(free_cols, pair_count, "cudaMemcpy structural outer free_cols recovery");
      for (std::int32_t i = 0; i < pair_count; ++i) {
        plan.structural_primal_recovery.outer_pairs.push_back(
            {bound_h[static_cast<std::size_t>(i)], free_h[static_cast<std::size_t>(i)]});
      }
      const std::vector<std::int32_t> s_h =
          copy_device_vector(s_cols, block_count, "cudaMemcpy structural outer s_cols recovery");
      const std::vector<std::int32_t> q_h =
          copy_device_vector(q_cols, block_count, "cudaMemcpy structural outer q_cols recovery");
      const std::vector<double> alpha_h =
          copy_device_vector(alphas, block_count, "cudaMemcpy structural outer alphas recovery");
      for (std::int32_t i = 0; i < block_count; ++i) {
        plan.structural_primal_recovery.linked_slacks.push_back(
            {s_h[static_cast<std::size_t>(i)], q_h[static_cast<std::size_t>(i)],
             alpha_h[static_cast<std::size_t>(i)]});
      }
    }
    if (profile) {
      std::cerr << ">>> [structural_l1 C++] changed=1 new_A_nnz=" << plan.new_A.nnz << "\n";
    }
  }

cleanup:
  cudaFree(eq_row);
  cudaFree(eq_zero_two_nnz);
  cudaFree(lower_two_nnz);
  cudaFree(zero_lower_two_nnz);
  cudaFree(raw_col1);
  cudaFree(raw_col2);
  cudaFree(raw_val1);
  cudaFree(raw_val2);
  cudaFree(zl_col1);
  cudaFree(zl_col2);
  cudaFree(zl_val1);
  cudaFree(zl_val2);
  cudaFree(status_flag);
  cudaFree(pair_count_d);
  cudaFree(start_row_d);
  cudaFree(bound_cols);
  cudaFree(free_cols);
  cudaFree(free_to_bound);
  cudaFree(q_cols);
  cudaFree(e_cols);
  cudaFree(s_cols);
  cudaFree(alphas);
  cudaFree(local_x_cols);
}

}  // namespace gpu_presolver::presolve
