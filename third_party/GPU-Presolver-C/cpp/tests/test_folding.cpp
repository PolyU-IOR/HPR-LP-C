#include "gpu_presolver/folding/folding.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
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
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  return values;
}

void free_lp(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
  cudaFree(lp.l);
  cudaFree(lp.u);
  lp = gpu_presolver::presolve::LPInfoGpu{};
}

gpu_presolver::presolve::LPInfoGpu make_symmetric_2x2_lp() {
  using gpu_presolver::presolve::LPInfoGpu;
  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {2, 2, 4,
           copy_to_device<std::int32_t>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.c = copy_to_device<double>({2.0, 2.0});
  lp.AL = copy_to_device<double>({4.0, 4.0});
  lp.AU = copy_to_device<double>({4.0, 4.0});
  lp.l = copy_to_device<double>({0.0, 0.0});
  lp.u = copy_to_device<double>({10.0, 10.0});
  return lp;
}

void test_fold_initial_vector_uses_first_representative() {
  const std::vector<double> value{3.0, 7.0, 11.0, 13.0};
  const std::vector<std::int32_t> color_index{1, 2, 1, 2};
  const std::vector<double> folded =
      gpu_presolver::folding::fold_initial_vector(&value, color_index, 2);
  assert((folded == std::vector<double>{3.0, 7.0}));
}

void test_run_folding_reduces_identical_rows_and_cols() {
  gpu_presolver::presolve::LPInfoGpu lp = make_symmetric_2x2_lp();
  gpu_presolver::folding::FoldingRunSummary summary =
      gpu_presolver::folding::run_folding(lp, 1.0e-8, false);
  assert(summary.applied);
  assert(summary.original_rows == 2);
  assert(summary.original_cols == 2);
  assert(summary.folded_rows == 1);
  assert(summary.folded_cols == 1);
  const gpu_presolver::folding::FoldingMapHost map =
      gpu_presolver::folding::copy_map_to_host(summary.map);
  assert((map.row_color_id == std::vector<std::int32_t>{1, 1}));
  assert((map.col_color_id == std::vector<std::int32_t>{1, 1}));
  assert(map.row_scale.size() == 2);
  assert(map.col_scale.size() == 2);
  assert(std::fabs(map.row_scale[0] - 0.5) < 1.0e-12);
  assert(std::fabs(map.col_scale[0] - 0.5) < 1.0e-12);
  assert(summary.folded_lp.A.rows == 1);
  assert(summary.folded_lp.A.cols == 1);
  assert(summary.folded_lp.A.nnz == 1);
  assert(summary.profile.reduce.copy_unique_to_host_seconds == 0.0);
  assert(summary.profile.reduce.host_filter_rebuild_seconds == 0.0);
  assert(summary.profile.reduce.host_csr_build_seconds == 0.0);
  gpu_presolver::folding::free_folded_lp(summary);
  free_lp(lp);
}

void test_unfold_solution_scales_duals_and_slacks() {
  gpu_presolver::presolve::LPInfoGpu lp = make_symmetric_2x2_lp();
  gpu_presolver::folding::FoldingRunSummary summary =
      gpu_presolver::folding::run_folding(lp, 1.0e-8, false);
  assert(summary.applied);
  assert(summary.map.original_rows == 2);
  assert(summary.map.original_cols == 2);
  assert(summary.map.reduced_rows == 1);
  assert(summary.map.reduced_cols == 1);

  const gpu_presolver::folding::UnfoldedSolutionHost unfolded =
      gpu_presolver::folding::unfold_solution_to_host(summary.map, {4.0}, {6.0}, {8.0});

  const std::vector<double>& x = unfolded.x;
  const std::vector<double>& y = unfolded.y;
  const std::vector<double>& z = unfolded.z;
  assert((x == std::vector<double>{4.0, 4.0}));
  assert((y == std::vector<double>{3.0, 3.0}));
  assert((z == std::vector<double>{4.0, 4.0}));

  gpu_presolver::folding::free_folded_lp(summary);
  free_lp(lp);
}

void test_postsolve_and_unfold_restores_original_dimensions() {
  gpu_presolver::presolve::LPInfoGpu lp = make_symmetric_2x2_lp();

  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 1;
  params.enable_close_bounds = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_activity_checks = false;
  params.enable_primal_propagation = false;
  params.enable_parallel_rows = false;
  params.enable_empty_cols = false;
  params.enable_singleton_cols_eq = false;
  params.enable_singleton_cols_dual_infer = false;
  params.enable_doubleton_eq = false;
  params.enable_dual_fix = false;
  params.enable_parallel_cols = false;
  params.enable_structural_l1_substitution = false;
  params.enable_redundant_bounds = false;
  params.enable_folding = true;

  gpu_presolver::folding::FoldingPipelineSummary summary =
      gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, false, true);
  assert(summary.folding_applied);
  assert(summary.folded_rows == 1);
  assert(summary.folded_cols == 1);
  assert(summary.presolve.reduced_rows == 1);
  assert(summary.presolve.reduced_cols == 1);

  double* x_red = copy_to_device<double>({4.0});
  double* y_red = copy_to_device<double>({6.0});
  double* z_red = copy_to_device<double>({8.0});
  const gpu_presolver::folding::UnfoldedSolutionHost unfolded =
      gpu_presolver::folding::postsolve_and_unfold_to_host(x_red, y_red, z_red, summary);

  assert((unfolded.x == std::vector<double>{4.0, 4.0}));
  assert(unfolded.y.size() == 2);
  assert(unfolded.z.size() == 2);
  assert(std::isfinite(unfolded.y[0]));
  assert(std::isfinite(unfolded.y[1]));
  assert(std::isfinite(unfolded.z[0]));
  assert(std::isfinite(unfolded.z[1]));

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  gpu_presolver::folding::free_folded_lp(summary.folding);
  free_lp(lp);
}

}  // namespace

int main() {
  test_fold_initial_vector_uses_first_representative();
  test_run_folding_reduces_identical_rows_and_cols();
  test_unfold_solution_scales_duals_and_slacks();
  test_postsolve_and_unfold_restores_original_dimensions();
  return 0;
}
