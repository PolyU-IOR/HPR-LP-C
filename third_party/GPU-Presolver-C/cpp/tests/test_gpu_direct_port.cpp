#include "gpu_presolver/presolve/gpu_presolve.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

void test_post_propagation_activity_closure(bool tiered_scheduler) {
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  lp.A = {2, 3, 5,
          copy_to_device<std::int32_t>({0, 2, 5}),
          copy_to_device<std::int32_t>({0, 1, 0, 1, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0})};
  lp.AT = {3, 2, 5,
           copy_to_device<std::int32_t>({0, 2, 4, 5}),
           copy_to_device<std::int32_t>({0, 1, 0, 1, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({10.0, 1.0, inf});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({2.0, 4.0});

  PresolveParams params;
  params.max_iters = tiered_scheduler ? 2 : 1;
  params.use_tiered_scheduler = tiered_scheduler;
  params.enable_tiered_bootstrap = false;
  params.record_postsolve_tape = false;
  params.enable_close_bounds = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_parallel_rows = false;
  params.enable_empty_cols = false;
  params.enable_singleton_cols_eq = false;
  params.enable_singleton_cols_dual_infer = false;
  params.enable_doubleton_eq = false;
  params.enable_dual_fix = false;
  params.enable_parallel_cols = false;
  params.enable_structural_l1_substitution = false;
  params.enable_redundant_bounds = false;

  const GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_fixed_order(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 1);
  assert(summary.reduced_cols == 3);
  assert(summary.reduced_nnz == 2);
  assert(summary.record.row_red2org == std::vector<std::int32_t>{0});
  assert(summary.record.row_org2red == std::vector<std::int32_t>({0, -1}));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

}  // namespace

int main() {
  const gpu_presolver::presolve::GpuRuntimeInfo runtime =
      gpu_presolver::presolve::query_gpu_runtime();
  assert(runtime.device_count > 0);
  assert(runtime.selected_device >= 0);

  gpu_presolver::presolve::LPInfoGpu lp;
  lp.A.nnz = 123;
  assert(gpu_presolver::presolve::_presolve_nnz(lp) == 123);
  assert(!gpu_presolver::presolve::_has_good_nnz_progress(0, 0, 0.95));
  assert(gpu_presolver::presolve::_has_good_nnz_progress(100, 94, 0.95));
  assert(!gpu_presolver::presolve::_has_good_nnz_progress(100, 95, 0.95));

  const std::int32_t counted = gpu_presolver::presolve::run_cuda_smoke_count(1000);
  assert(counted == 1000);

  // Propagation tightens x0 first; only a following activity pass can prove
  // the second row redundant.  Exercise both scheduler paths.
  test_post_propagation_activity_closure(false);
  test_post_propagation_activity_closure(true);

  const double inf = std::numeric_limits<double>::infinity();
  gpu_presolver::presolve::LPInfoGpu sched_lp;
  sched_lp.A = {3, 3, 2,
                copy_to_device<std::int32_t>({0, 1, 2, 2}),
                copy_to_device<std::int32_t>({0, 1}),
                copy_to_device<double>({1.0, 1.0})};
  sched_lp.AT = {3, 3, 2,
                 copy_to_device<std::int32_t>({0, 1, 2, 2}),
                 copy_to_device<std::int32_t>({0, 1}),
                 copy_to_device<double>({1.0, 1.0})};
  sched_lp.c = copy_to_device<double>({0.0, 0.0, 2.0});
  sched_lp.l = copy_to_device<double>({5.0, 0.0, 1.0});
  sched_lp.u = copy_to_device<double>({5.0, 10.0, 3.0});
  sched_lp.AL = copy_to_device<double>({5.0, -inf, -1.0});
  sched_lp.AU = copy_to_device<double>({5.0, inf, 1.0});

  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 4;
  const gpu_presolver::presolve::GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_fixed_order(sched_lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 0);
  assert(summary.reduced_cols == 0);

  gpu_presolver::presolve::LPInfoGpu fixed_sched_lp;
  fixed_sched_lp.A = {3, 3, 2,
                      copy_to_device<std::int32_t>({0, 1, 2, 2}),
                      copy_to_device<std::int32_t>({0, 1}),
                      copy_to_device<double>({1.0, 1.0})};
  fixed_sched_lp.AT = {3, 3, 2,
                       copy_to_device<std::int32_t>({0, 1, 2, 2}),
                       copy_to_device<std::int32_t>({0, 1}),
                       copy_to_device<double>({1.0, 1.0})};
  fixed_sched_lp.c = copy_to_device<double>({0.0, 0.0, 2.0});
  fixed_sched_lp.l = copy_to_device<double>({5.0, 0.0, 1.0});
  fixed_sched_lp.u = copy_to_device<double>({5.0, 10.0, 3.0});
  fixed_sched_lp.AL = copy_to_device<double>({5.0, -inf, -1.0});
  fixed_sched_lp.AU = copy_to_device<double>({5.0, inf, 1.0});

  gpu_presolver::presolve::PresolveParams fixed_params;
  fixed_params.max_iters = 4;
  fixed_params.use_tiered_scheduler = false;
  const gpu_presolver::presolve::GpuPresolveSummary fixed_summary =
      gpu_presolver::presolve::run_gpu_presolve_fixed_order(fixed_sched_lp, fixed_params);
  assert(!fixed_summary.has_infeasible);
  assert(!fixed_summary.has_unbounded);
  assert(fixed_summary.reduced_rows == 0);
  assert(fixed_summary.reduced_cols == 0);

  cudaFree(fixed_sched_lp.A.rowPtr);
  cudaFree(fixed_sched_lp.A.colVal);
  cudaFree(fixed_sched_lp.A.nzVal);
  cudaFree(fixed_sched_lp.AT.rowPtr);
  cudaFree(fixed_sched_lp.AT.colVal);
  cudaFree(fixed_sched_lp.AT.nzVal);
  cudaFree(fixed_sched_lp.c);
  cudaFree(fixed_sched_lp.l);
  cudaFree(fixed_sched_lp.u);
  cudaFree(fixed_sched_lp.AL);
  cudaFree(fixed_sched_lp.AU);

  cudaFree(sched_lp.A.rowPtr);
  cudaFree(sched_lp.A.colVal);
  cudaFree(sched_lp.A.nzVal);
  cudaFree(sched_lp.AT.rowPtr);
  cudaFree(sched_lp.AT.colVal);
  cudaFree(sched_lp.AT.nzVal);
  cudaFree(sched_lp.c);
  cudaFree(sched_lp.l);
  cudaFree(sched_lp.u);
  cudaFree(sched_lp.AL);
  cudaFree(sched_lp.AU);

  constexpr std::int32_t doubleton_rows = 300;
  constexpr std::int32_t doubleton_cols = doubleton_rows * 2;
  std::vector<std::int32_t> doubleton_row_ptr(static_cast<std::size_t>(doubleton_rows + 1));
  std::vector<std::int32_t> doubleton_col_val(static_cast<std::size_t>(doubleton_rows * 2));
  std::vector<double> doubleton_a(static_cast<std::size_t>(doubleton_rows * 2), 1.0);
  for (std::int32_t row = 0; row < doubleton_rows; ++row) {
    doubleton_row_ptr[static_cast<std::size_t>(row)] = 2 * row;
    doubleton_col_val[static_cast<std::size_t>(2 * row)] = 2 * row;
    doubleton_col_val[static_cast<std::size_t>(2 * row + 1)] = 2 * row + 1;
  }
  doubleton_row_ptr[static_cast<std::size_t>(doubleton_rows)] = doubleton_rows * 2;
  std::vector<std::int32_t> doubleton_at_row_ptr(static_cast<std::size_t>(doubleton_cols + 1));
  std::vector<std::int32_t> doubleton_at_col_val(static_cast<std::size_t>(doubleton_rows * 2));
  for (std::int32_t col = 0; col < doubleton_cols; ++col) {
    doubleton_at_row_ptr[static_cast<std::size_t>(col)] = col;
    doubleton_at_col_val[static_cast<std::size_t>(col)] = col / 2;
  }
  doubleton_at_row_ptr[static_cast<std::size_t>(doubleton_cols)] = doubleton_cols;

  gpu_presolver::presolve::LPInfoGpu doubleton_only_lp;
  doubleton_only_lp.A = {doubleton_rows, doubleton_cols, doubleton_rows * 2,
                         copy_to_device<std::int32_t>(doubleton_row_ptr),
                         copy_to_device<std::int32_t>(doubleton_col_val),
                         copy_to_device<double>(doubleton_a)};
  doubleton_only_lp.AT = {doubleton_cols, doubleton_rows, doubleton_rows * 2,
                          copy_to_device<std::int32_t>(doubleton_at_row_ptr),
                          copy_to_device<std::int32_t>(doubleton_at_col_val),
                          copy_to_device<double>(doubleton_a)};
  doubleton_only_lp.c = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 0.0));
  doubleton_only_lp.l = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 0.0));
  doubleton_only_lp.u = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 10.0));
  doubleton_only_lp.AL = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_rows), 1.0));
  doubleton_only_lp.AU = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_rows), 1.0));

  gpu_presolver::presolve::PresolveParams doubleton_only_params;
  doubleton_only_params.max_iters = 4;
  doubleton_only_params.enable_close_bounds = false;
  doubleton_only_params.enable_empty_rows = false;
  doubleton_only_params.enable_singleton_rows = false;
  doubleton_only_params.enable_activity_checks = false;
  doubleton_only_params.enable_primal_propagation = false;
  doubleton_only_params.enable_parallel_rows = false;
  doubleton_only_params.enable_empty_cols = false;
  doubleton_only_params.enable_singleton_cols_eq = false;
  doubleton_only_params.enable_singleton_cols_dual_infer = false;
  doubleton_only_params.enable_dual_fix = false;
  doubleton_only_params.enable_parallel_cols = false;
  doubleton_only_params.enable_structural_l1_substitution = false;
  doubleton_only_params.enable_redundant_bounds = false;
  const gpu_presolver::presolve::GpuPresolveSummary doubleton_only_summary =
      gpu_presolver::presolve::run_gpu_presolve_fixed_order(doubleton_only_lp, doubleton_only_params);
  assert(!doubleton_only_summary.has_infeasible);
  assert(!doubleton_only_summary.has_unbounded);
  assert(doubleton_only_summary.reduced_rows == 0);
  assert(doubleton_only_summary.reduced_cols == doubleton_rows);
  assert(doubleton_only_summary.reduced_nnz == 0);
  assert(doubleton_only_summary.iterations > 1);

  cudaFree(doubleton_only_lp.A.rowPtr);
  cudaFree(doubleton_only_lp.A.colVal);
  cudaFree(doubleton_only_lp.A.nzVal);
  cudaFree(doubleton_only_lp.AT.rowPtr);
  cudaFree(doubleton_only_lp.AT.colVal);
  cudaFree(doubleton_only_lp.AT.nzVal);
  cudaFree(doubleton_only_lp.c);
  cudaFree(doubleton_only_lp.l);
  cudaFree(doubleton_only_lp.u);
  cudaFree(doubleton_only_lp.AL);
  cudaFree(doubleton_only_lp.AU);

  std::cout << "test_gpu_direct_port passed on " << runtime.device_count << " CUDA device(s)\n";
  return 0;
}
