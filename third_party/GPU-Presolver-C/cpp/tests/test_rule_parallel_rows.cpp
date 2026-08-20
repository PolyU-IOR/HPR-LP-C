#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"

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

void free_plan(gpu_presolver::presolve::PresolvePlanGpu& plan) {
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

void free_matrix(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
}

}  // namespace

void test_deletes_parallel_row_and_tightens_representative() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 2.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_AL = copy_to_device<double>({0.0, 4.0});
  plan.new_AU = copy_to_device<double>({10.0, 12.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_rows(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_row_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 2);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 2);
  assert((keep[0] == 0 && keep[1] == 1) || (keep[0] == 1 && keep[1] == 0));
  const int rep = keep[0] == 1 ? 0 : 1;
  assert(AL[rep] <= AU[rep]);
  assert(plan.has_row_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::ParallelRow)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 2}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 5}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Exact)}));
  assert(plan.tape.indices.size() == 2);
  assert(plan.tape.indices[0] == rep);
  assert(plan.tape.indices[1] == (rep == 0 ? 1 : 0));
  assert(plan.tape.vals.size() == 5);

  free_plan(plan);
  free_matrix(lp);
}

void test_detects_disjoint_parallel_row_intervals() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_AL = copy_to_device<double>({0.0, 3.0});
  plan.new_AU = copy_to_device<double>({1.0, 4.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_rows(plan, lp, stats, params);
  assert(plan.has_infeasible);

  free_plan(plan);
  free_matrix(lp);
}

int main() {
  test_deletes_parallel_row_and_tightens_representative();
  test_detects_disjoint_parallel_row_intervals();
  std::cout << "test_rule_parallel_rows passed\n";
  return 0;
}
