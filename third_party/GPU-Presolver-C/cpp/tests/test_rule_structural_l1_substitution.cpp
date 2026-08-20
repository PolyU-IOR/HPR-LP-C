#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/rules/rule_structural_l1_substitution.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
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

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  return values;
}

void test_l1_split_treats_large_residual_bounds_as_free_like_julia_config() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {3, 2, 6,
          copy_to_device<std::int32_t>({0, 2, 4, 6}),
          copy_to_device<std::int32_t>({0, 1, 0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, -1.0, 1.0, 1.0})};
  lp.AT = {2, 3, 6,
           copy_to_device<std::int32_t>({0, 3, 6}),
           copy_to_device<std::int32_t>({0, 1, 2, 0, 1, 2}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({2.0, 0.5});
  plan.new_l = copy_to_device<double>({0.0, -1.0e9});
  plan.new_u = copy_to_device<double>({inf, 1.0e9});
  plan.new_AL = copy_to_device<double>({0.0, 0.0, 0.0});
  plan.new_AU = copy_to_device<double>({0.0, inf, inf});

  PresolveStatsGpu stats;
  PresolveParams params;
  params.structural_l1_residual_bound_as_free_min = 1.0e8;
  gpu_presolver::presolve::apply_rule_structural_l1_substitution(plan, lp, stats, params);

  assert(plan.has_change);
  assert(plan.has_structural_primal_recovery);
  assert(plan.structural_primal_recovery.pattern == "l1_split_3row");
  assert(plan.structural_primal_recovery.splits.size() == 1);
  assert(plan.structural_primal_recovery.splits[0].t_col == 0);
  assert(plan.structural_primal_recovery.splits[0].e_col == 1);

  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 3);
  const std::vector<double> c = copy_to_host(plan.new_c, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert((keep_row == std::vector<std::uint8_t>{1, 0, 0}));
  assert(std::fabs(c[0] - 2.5) < 1.0e-12);
  assert(std::fabs(c[1] - 1.5) < 1.0e-12);
  assert(l[0] == 0.0);
  assert(l[1] == 0.0);
  assert(std::isinf(u[0]));
  assert(std::isinf(u[1]));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_A.rowPtr);
  cudaFree(plan.new_A.colVal);
  cudaFree(plan.new_A.nzVal);
}

void test_graph_l1_rewrites_four_row_block_and_records_postsolve() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {4, 3, 7,
          copy_to_device<std::int32_t>({0, 1, 3, 5, 7}),
          copy_to_device<std::int32_t>({1, 0, 1, 0, 1, 0, 2}),
          copy_to_device<double>({3.0, 1.0, -1.0, 1.0, 1.0, -2.0, 1.0})};
  lp.AT = {3, 4, 7,
           copy_to_device<std::int32_t>({0, 3, 6, 7}),
           copy_to_device<std::int32_t>({1, 2, 3, 0, 1, 2, 3}),
           copy_to_device<double>({1.0, 1.0, -2.0, 3.0, -1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.5, 2.0, 0.0});
  plan.new_l = copy_to_device<double>({0.0, -inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, inf, inf});
  plan.new_AL = copy_to_device<double>({7.0, 0.0, 0.0, 0.0});
  plan.new_AU = copy_to_device<double>({7.0, inf, inf, inf});

  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_structural_l1_substitution(plan, lp, stats, params);

  assert(plan.has_change);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_new_A);
  assert(plan.new_A.rows == 4);
  assert(plan.new_A.cols == 3);
  assert(plan.new_A.nnz == 2);

  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 4);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 3);
  const std::vector<double> c = copy_to_host(plan.new_c, 3);
  const std::vector<double> l = copy_to_host(plan.new_l, 3);
  const std::vector<double> u = copy_to_host(plan.new_u, 3);
  const std::vector<std::int32_t> row_ptr = copy_to_host(plan.new_A.rowPtr, 5);
  const std::vector<std::int32_t> col_val = copy_to_host(plan.new_A.colVal, 2);
  const std::vector<double> nz_val = copy_to_host(plan.new_A.nzVal, 2);

  assert((keep_row == std::vector<std::uint8_t>{1, 0, 0, 0}));
  assert((keep_col == std::vector<std::uint8_t>{1, 1, 0}));
  assert(std::fabs(c[0] - 2.5) < 1.0e-12);
  assert(std::fabs(c[1] - (-1.5)) < 1.0e-12);
  assert(l[0] == 0.0);
  assert(l[1] == 0.0);
  assert(std::isinf(u[0]));
  assert(std::isinf(u[1]));
  assert((row_ptr == std::vector<std::int32_t>{0, 2, 2, 2, 2}));
  assert((col_val == std::vector<std::int32_t>{0, 1}));
  assert(std::fabs(nz_val[0] - 3.0) < 1.0e-12);
  assert(std::fabs(nz_val[1] - (-3.0)) < 1.0e-12);

  assert(plan.has_structural_primal_recovery);
  assert(plan.structural_primal_recovery.pattern == "graph_l1_substitution");
  assert(plan.structural_primal_recovery.splits.size() == 1);
  assert(plan.structural_primal_recovery.max_slacks.size() == 1);
  assert(plan.structural_primal_recovery.max_slacks[0].slack_col == 2);

  double* x_org = copy_to_device<double>({4.0, 1.0, 0.0});
  double* original_l = copy_to_device<double>({0.0, -inf, 0.0});
  gpu_presolver::presolve::postsolve_restore_structural_primal_gpu(
      x_org, std::vector<gpu_presolver::presolve::StructuralL1PrimalRecoveryStep>{plan.structural_primal_recovery},
      original_l);
  const std::vector<double> x = copy_to_host(x_org, 3);
  assert(std::fabs(x[0] - 5.0) < 1.0e-12);
  assert(std::fabs(x[1] - 3.0) < 1.0e-12);
  assert(std::fabs(x[2] - 10.0) < 1.0e-12);

  cudaFree(x_org);
  cudaFree(original_l);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_A.rowPtr);
  cudaFree(plan.new_A.colVal);
  cudaFree(plan.new_A.nzVal);
}

}  // namespace

int main() {
  test_l1_split_treats_large_residual_bounds_as_free_like_julia_config();
  test_graph_l1_rewrites_four_row_block_and_records_postsolve();
  std::cout << "test_rule_structural_l1_substitution passed\n";
  return 0;
}
