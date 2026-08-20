#include "gpu_presolver/presolve/rules/rule_empty_cols.hpp"

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

}  // namespace

void test_fixes_empty_columns() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.rows = 0;
  lp.A.cols = 4;

  PresolvePlanGpu plan;
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1, 0});
  plan.new_c = copy_to_device<double>({2.0, -3.0, 0.0, 4.0});
  plan.new_l = copy_to_device<double>({1.0, -10.0, -inf, 5.0});
  plan.new_u = copy_to_device<double>({9.0, 7.0, 6.0, 5.0});

  PresolveStatsGpu stats;
  stats.empty_col_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1});

  PresolveParams params;
  params.bound_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_empty_cols(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_col_mask, 4);
  assert(keep[0] == 0);
  assert(keep[1] == 0);
  assert(keep[2] == 0);
  assert(keep[3] == 0);
  assert(std::fabs(plan.obj_constant_delta - (2.0 * 1.0 - 3.0 * 7.0)) < 1.0e-12);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert(plan.tape.types.empty());

  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.empty_col_mask);
}

void test_marks_unbounded_empty_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.cols = 1;
  PresolvePlanGpu plan;
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1});
  plan.new_c = copy_to_device<double>({2.0});
  plan.new_l = copy_to_device<double>({-inf});
  plan.new_u = copy_to_device<double>({10.0});
  PresolveStatsGpu stats;
  stats.empty_col_mask = copy_to_device<std::uint8_t>({1});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_empty_cols(plan, lp, stats, params);
  assert(plan.has_unbounded);

  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.empty_col_mask);
}

void test_marks_infeasible_empty_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.cols = 1;
  PresolvePlanGpu plan;
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1});
  plan.new_c = copy_to_device<double>({0.0});
  plan.new_l = copy_to_device<double>({3.0});
  plan.new_u = copy_to_device<double>({1.0});
  PresolveStatsGpu stats;
  stats.empty_col_mask = copy_to_device<std::uint8_t>({1});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_empty_cols(plan, lp, stats, params);
  assert(plan.has_infeasible);

  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.empty_col_mask);
}

int main() {
  test_fixes_empty_columns();
  test_marks_unbounded_empty_column();
  test_marks_infeasible_empty_column();
  std::cout << "test_rule_empty_cols passed\n";
  return 0;
}
