#include "gpu_presolver/presolve/rules/rule_redundant_bounds.hpp"

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

void free_lp(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
}

}  // namespace

void test_drops_redundant_upper_bound() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<std::int32_t>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({5.0, 5.0});
  plan.new_AL = copy_to_device<double>({-inf});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_redundant_bounds(plan, lp, stats, params);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(plan.has_col_action);
  assert(plan.has_change);

  free_lp(lp);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

void test_drops_redundant_lower_bound() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<std::int32_t>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({inf, 0.0});
  plan.new_AL = copy_to_device<double>({0.0});
  plan.new_AU = copy_to_device<double>({inf});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_redundant_bounds(plan, lp, stats, params);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  assert(std::isinf(l[0]) && l[0] < 0.0);
  assert(plan.has_col_action);
  assert(plan.has_change);

  free_lp(lp);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

int main() {
  test_drops_redundant_upper_bound();
  test_drops_redundant_lower_bound();
  std::cout << "test_rule_redundant_bounds passed\n";
  return 0;
}
