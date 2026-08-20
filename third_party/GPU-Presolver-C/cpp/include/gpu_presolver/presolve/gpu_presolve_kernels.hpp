#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void compute_row_nnz(std::int32_t* row_nnz, const DeviceCsrMatrix& A_csr);

void compute_col_nnz(std::int32_t* col_nnz, const DeviceCsrMatrix& AT_csr);

void compute_singleton_row_support(std::int32_t* singleton_col,
                                   double* singleton_val,
                                   const std::int32_t* row_nnz,
                                   const DeviceCsrMatrix& A_csr);

void compute_singleton_col_support(std::int32_t* singleton_row,
                                   double* singleton_val,
                                   const std::int32_t* col_nnz,
                                   const DeviceCsrMatrix& AT_csr);

void compute_row_activity_bounds(double* row_min,
                                 double* row_max,
                                 const DeviceCsrMatrix& A_csr,
                                 const double* l,
                                 const double* u);

void compute_row_activity_summary(double* row_min_fin,
                                  double* row_max_fin,
                                  std::int32_t* row_min_neg_inf_count,
                                  std::int32_t* row_max_pos_inf_count,
                                  const DeviceCsrMatrix& A_csr,
                                  const double* l,
                                  const double* u,
                                  double zero_tol);

void compute_col_max_abs(double* col_max_abs, const DeviceCsrMatrix& AT_csr);

}  // namespace gpu_presolver::presolve
