#include "gpu_presolver/presolve/rules/rule_dual_fix.hpp"

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

void test_fixes_improving_unlocked_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.rows = 1;
  lp.A.cols = 1;
  lp.AT = {1, 1, 1,
           copy_to_device<std::int32_t>({0, 1}),
           copy_to_device<std::int32_t>({0}),
           copy_to_device<double>({1.0})};

  PresolvePlanGpu plan;
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.new_c = copy_to_device<double>({3.0});
  plan.new_l = copy_to_device<double>({2.0});
  plan.new_u = copy_to_device<double>({inf});
  plan.new_AL = copy_to_device<double>({-inf});
  plan.new_AU = copy_to_device<double>({10.0});
  PresolveStatsGpu stats;
  PresolveParams params;
  params.record_postsolve_tape_cpu = true;

  gpu_presolver::presolve::apply_rule_dual_fix(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 1);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 1);
  assert(keep_col[0] == 0);
  assert(std::fabs(AU[0] - 8.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 6.0) < 1.0e-12);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert(plan.tape.types.size() == 1);
  assert(plan.tape.types[0] == static_cast<std::int32_t>(PostsolveReductionType::FixedCol));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0}));
  assert(std::fabs(plan.tape.vals[0] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 1.0) < 1.0e-12);

  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

void test_marks_unbounded_improving_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.rows = 0;
  lp.A.cols = 1;
  lp.AT = {1, 0, 0,
           copy_to_device<std::int32_t>({0, 0}),
           nullptr,
           nullptr};
  PresolvePlanGpu plan;
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_row_mask = nullptr;
  plan.new_c = copy_to_device<double>({1.0});
  plan.new_l = copy_to_device<double>({-inf});
  plan.new_u = copy_to_device<double>({10.0});
  plan.new_AL = nullptr;
  plan.new_AU = nullptr;
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_dual_fix(plan, lp, stats, params);
  assert(plan.has_unbounded);

  cudaFree(lp.AT.rowPtr);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
}

void test_records_infinite_zero_objective_fix() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 1, 1,
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<std::int32_t>({0}),
          copy_to_device<double>({1.0})};
  lp.AT = {1, 1, 1,
           copy_to_device<std::int32_t>({0, 1}),
           copy_to_device<std::int32_t>({0}),
           copy_to_device<double>({1.0})};

  PresolvePlanGpu plan;
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.new_c = copy_to_device<double>({0.0});
  plan.new_l = copy_to_device<double>({-inf});
  plan.new_u = copy_to_device<double>({10.0});
  plan.new_AL = copy_to_device<double>({-inf});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_dual_fix(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 1);
  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 1);
  assert(keep_col[0] == 0);
  assert(keep_row[0] == 0);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::FixedColInf)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 4}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 4}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Exact)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{-1, 0, 1, 0}));
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 5.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 1.0) < 1.0e-12);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

int main() {
  test_fixes_improving_unlocked_column();
  test_marks_unbounded_improving_column();
  test_records_infinite_zero_objective_fix();
  std::cout << "test_rule_dual_fix passed\n";
  return 0;
}
