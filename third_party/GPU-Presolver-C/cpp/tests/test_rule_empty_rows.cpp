#include "gpu_presolver/presolve/rules/rule_empty_rows.hpp"

#include <cuda_runtime.h>

#include <cassert>
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

void test_removes_feasible_empty_rows() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 3;
  lp.A.cols = 2;

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_AL = copy_to_device<double>({-1.0, -2.0, 0.0});
  plan.new_AU = copy_to_device<double>({1.0, 3.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({0, 0, 1});

  PresolveParams params;
  params.feasibility_tol = 1.0e-9;

  gpu_presolver::presolve::apply_rule_empty_rows(plan, lp, stats, params);
  std::vector<std::uint8_t> keep = copy_to_host(plan.keep_row_mask, 3);
  assert(keep[0] == 0);
  assert(keep[1] == 0);
  assert(keep[2] == 1);
  assert(!plan.has_infeasible);
  assert(plan.has_row_action);
  assert(plan.has_change);

  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.row_nnz);
}

void test_marks_infeasible_empty_row_without_applying_removals() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 1;

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_AL = copy_to_device<double>({-1.0, 2.0});
  plan.new_AU = copy_to_device<double>({1.0, 3.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({0, 0});

  PresolveParams params;
  params.feasibility_tol = 1.0e-9;

  gpu_presolver::presolve::apply_rule_empty_rows(plan, lp, stats, params);
  std::vector<std::uint8_t> keep = copy_to_host(plan.keep_row_mask, 2);
  assert(keep[0] == 1);
  assert(keep[1] == 1);
  assert(plan.has_infeasible);

  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.row_nnz);
}

int main() {
  test_removes_feasible_empty_rows();
  test_marks_infeasible_empty_row_without_applying_removals();

  std::cout << "test_rule_empty_rows passed\n";
  return 0;
}
