#include "gpu_presolver/presolve/rules/rule_activity_checks.hpp"

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

void test_activity_actions_match_julia_classification() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.rows = 5;
  lp.A.cols = 2;
  lp.A.rowPtr = copy_to_device<std::int32_t>({0, 2, 4, 6, 8, 9});
  lp.A.colVal = copy_to_device<std::int32_t>({0, 1, 0, 1, 0, 1, 0, 1, 0});
  lp.A.nzVal = copy_to_device<double>({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0});

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1});
  plan.new_AL = copy_to_device<double>({-inf, -1.0, 2.0, -10.0, -100.0});
  plan.new_AU = copy_to_device<double>({10.0, 5.0, 10.0, 10.0, 100.0});
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({4.0, 4.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2, 2, 2, 1});

  PresolveParams params;
  params.bound_tol = 1.0e-9;

  gpu_presolver::presolve::apply_rule_activity_checks(plan, lp, stats, params);

  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_row_mask, 5);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 5);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 5);
  assert(keep[0] == 0);
  assert(keep[1] == 1);
  assert(keep[2] == 1);
  assert(keep[3] == 0);
  assert(keep[4] == 1);
  assert(std::isinf(AL[1]) && AL[1] < 0.0);
  assert(std::isinf(AU[2]) && AU[2] > 0.0);
  assert(plan.has_row_action);
  assert(plan.has_change);
  assert(!plan.has_infeasible);
  assert(plan.tape.types.empty());

  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);

  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1});
  plan.new_AL = copy_to_device<double>({-inf, -1.0, 2.0, -10.0, -100.0});
  plan.new_AU = copy_to_device<double>({10.0, 5.0, 10.0, 10.0, 100.0});
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({4.0, 4.0});
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2, 2, 2, 1});
  params.record_postsolve_tape_cpu = true;

  gpu_presolver::presolve::apply_rule_activity_checks(plan, lp, stats, params);

  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::DeletedRow),
                                 static_cast<std::int32_t>(PostsolveReductionType::DeletedRow),
                                 static_cast<std::int32_t>(PostsolveReductionType::LhsChange),
                                 static_cast<std::int32_t>(PostsolveReductionType::RhsChange)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 1, 2, 3, 4}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 2, 4, 8, 12}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 3, 1, 2}));
  assert(std::fabs(plan.tape.vals[0] - -inf) < 1.0e-12 || (std::isinf(plan.tape.vals[0]) && plan.tape.vals[0] < 0.0));
  assert(std::fabs(plan.tape.vals[1] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - -10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - -1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - 5.0) < 1.0e-12);
  assert(std::isinf(plan.tape.vals[6]) && plan.tape.vals[6] < 0.0);
  assert(std::fabs(plan.tape.vals[7] - 5.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[8] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[9] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[10] - 2.0) < 1.0e-12);
  assert(std::isinf(plan.tape.vals[11]) && plan.tape.vals[11] > 0.0);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_activity_infeasible_short_circuits_apply() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 1;
  lp.A.cols = 2;
  lp.A.rowPtr = copy_to_device<std::int32_t>({0, 2});
  lp.A.colVal = copy_to_device<std::int32_t>({0, 1});
  lp.A.nzVal = copy_to_device<double>({1.0, 1.0});

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.new_AL = copy_to_device<double>({9.0});
  plan.new_AU = copy_to_device<double>({20.0});
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({4.0, 4.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_activity_checks(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_row_mask, 1);
  assert(plan.has_infeasible);
  assert(keep[0] == 1);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

int main() {
  test_activity_actions_match_julia_classification();
  test_activity_infeasible_short_circuits_apply();
  std::cout << "test_rule_activity_checks passed\n";
  return 0;
}
