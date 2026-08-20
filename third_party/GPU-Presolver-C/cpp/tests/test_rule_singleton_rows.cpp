#include "gpu_presolver/presolve/rules/rule_singleton_rows.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
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

int main() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_l = copy_to_device<double>({0.0, -10.0});
  plan.new_u = copy_to_device<double>({100.0, 10.0});
  plan.new_AL = copy_to_device<double>({4.0, -6.0});
  plan.new_AU = copy_to_device<double>({8.0, -2.0});

  PresolveStatsGpu stats;
  stats.singleton_row_mask = copy_to_device<std::uint8_t>({1, 1});
  stats.singleton_row_col = copy_to_device<std::int32_t>({0, 1});
  stats.singleton_row_val = copy_to_device<double>({2.0, -2.0});

  PresolveParams params;
  params.zero_tol = 1.0e-12;
  params.bound_tol = 1.0e-9;

  gpu_presolver::presolve::apply_rule_singleton_rows(plan, lp, stats, params);

  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);

  assert(keep_row[0] == 0);
  assert(keep_row[1] == 0);
  assert(std::fabs(l[0] - 2.0) < 1.0e-12);
  assert(std::fabs(u[0] - 4.0) < 1.0e-12);
  assert(std::fabs(l[1] - 1.0) < 1.0e-12);
  assert(std::fabs(u[1] - 3.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_change);
  assert(plan.tape.types.empty());

  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_row_mask);
  cudaFree(stats.singleton_row_col);
  cudaFree(stats.singleton_row_val);

  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_l = copy_to_device<double>({0.0, -10.0});
  plan.new_u = copy_to_device<double>({100.0, 10.0});
  plan.new_AL = copy_to_device<double>({4.0, -6.0});
  plan.new_AU = copy_to_device<double>({8.0, -2.0});
  stats.singleton_row_mask = copy_to_device<std::uint8_t>({1, 1});
  stats.singleton_row_col = copy_to_device<std::int32_t>({0, 1});
  stats.singleton_row_val = copy_to_device<double>({2.0, -2.0});
  params.record_postsolve_tape_cpu = true;

  gpu_presolver::presolve::apply_rule_singleton_rows(plan, lp, stats, params);

  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::DeletedRow),
                                 static_cast<std::int32_t>(PostsolveReductionType::DeletedRow)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 2, 4}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 3, 6}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0, 1, 1}));
  assert(std::fabs(plan.tape.vals[0] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 8.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - -6.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - -2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - -2.0) < 1.0e-12);

  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_row_mask);
  cudaFree(stats.singleton_row_col);
  cudaFree(stats.singleton_row_val);

  std::cout << "test_rule_singleton_rows passed\n";
  return 0;
}
