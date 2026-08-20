#include "gpu_presolver/presolve/rules/rule_parallel_cols.hpp"

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

void test_merges_objective_compatible_parallel_columns() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.AT = {2, 2, 4,
           copy_to_device<std::int32_t>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 2.0, 2.0, 4.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 2.0});
  plan.new_l = copy_to_device<double>({0.0, 1.0});
  plan.new_u = copy_to_device<double>({10.0, 3.0});
  plan.new_AL = copy_to_device<double>({0.0, 0.0});
  plan.new_AU = copy_to_device<double>({10.0, 10.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_cols(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(keep[0] == 1);
  assert(keep[1] == 0);
  assert(std::fabs(l[0] - 2.0) < 1.0e-12);
  assert(std::fabs(u[0] - 16.0) < 1.0e-12);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::ParallelCol)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 2}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 5}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{1, 0}));
  assert(std::fabs(plan.tape.vals[0] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 10.0) < 1.0e-12);

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
}

void test_fixed_parallel_column_records_tape() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.AT = {2, 2, 4,
           copy_to_device<std::int32_t>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 2.0, 1.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 2.0});
  plan.new_l = copy_to_device<double>({0.0, 3.0});
  plan.new_u = copy_to_device<double>({inf, 5.0});
  plan.new_AL = copy_to_device<double>({0.0, 0.0});
  plan.new_AU = copy_to_device<double>({10.0, 20.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_cols(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 2);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 2);
  assert(keep[0] == 1);
  assert(keep[1] == 0);
  assert(std::fabs(l[1] - 3.0) < 1.0e-12);
  assert(std::fabs(u[1] - 3.0) < 1.0e-12);
  assert(std::fabs(AL[0] - -3.0) < 1.0e-12);
  assert(std::fabs(AU[0] - 7.0) < 1.0e-12);
  assert(std::fabs(AL[1] - -6.0) < 1.0e-12);
  assert(std::fabs(AU[1] - 14.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 6.0) < 1.0e-12);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::FixedCol)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 3}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 4}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{1, 0, 1}));
  assert(std::fabs(plan.tape.vals[0] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 2.0) < 1.0e-12);

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
}

int main() {
  test_merges_objective_compatible_parallel_columns();
  test_fixed_parallel_column_records_tape();
  std::cout << "test_rule_parallel_cols passed\n";
  return 0;
}
