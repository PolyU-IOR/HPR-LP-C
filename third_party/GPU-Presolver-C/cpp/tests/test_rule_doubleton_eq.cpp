#include "gpu_presolver/presolve/rules/rule_doubleton_eq.hpp"

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

void test_eliminates_doubleton_equality_row() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 2.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<std::int32_t>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({3.0, 5.0});
  plan.new_l = copy_to_device<double>({0.0, -inf});
  plan.new_u = copy_to_device<double>({10.0, inf});
  plan.new_AL = copy_to_device<double>({4.0});
  plan.new_AU = copy_to_device<double>({4.0});

  PresolveStatsGpu stats;
  stats.col_nnz = nullptr;
  PresolveParams params;
  params.doubleton_eq_scan = false;
  params.doubleton_eq_max_fill_in_proxy = 10;

  gpu_presolver::presolve::apply_rule_doubleton_eq(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 1);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> c = copy_to_host(plan.new_c, 2);
  assert(keep_row[0] == 0);
  assert(keep_col[0] == 0);
  assert(keep_col[1] == 1);
  assert(std::fabs(c[1] - (-1.0)) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 12.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_change);

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

int main() {
  test_eliminates_doubleton_equality_row();
  std::cout << "test_rule_doubleton_eq passed\n";
  return 0;
}
