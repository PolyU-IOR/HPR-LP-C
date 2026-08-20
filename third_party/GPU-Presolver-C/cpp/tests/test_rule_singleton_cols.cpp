#include "gpu_presolver/presolve/rules/rule_singleton_cols.hpp"

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

void test_eliminates_eq_singleton_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({4.0, 1.0});
  plan.new_l = copy_to_device<double>({0.0, 2.0});
  plan.new_u = copy_to_device<double>({10.0, 2.0});
  plan.new_AL = copy_to_device<double>({5.0});
  plan.new_AU = copy_to_device<double>({5.0});

  PresolveStatsGpu stats;
  stats.singleton_col_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_col_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_col_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_cols_eq(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 1);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> c = copy_to_host(plan.new_c, 2);
  assert(keep_row[0] == 0);
  assert(keep_col[0] == 0);
  assert(keep_col[1] == 1);
  assert(std::fabs(c[1] + 3.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 20.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::SubCol)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0, 1, 1}));
  assert(plan.tape.vals.size() == 7);
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 5.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[6] - 1.0) < 1.0e-12);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_col_mask);
  cudaFree(stats.singleton_col_row);
  cudaFree(stats.singleton_col_val);
}

void test_dual_infer_marks_direct_unbounded() {
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
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({10.0, 1.0});
  plan.new_AL = copy_to_device<double>({-inf});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  stats.singleton_col_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_col_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_col_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_cols_dual_infer(plan, lp, stats, params);
  assert(plan.has_unbounded);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_col_mask);
  cudaFree(stats.singleton_col_row);
  cudaFree(stats.singleton_col_val);
}

void test_dual_infer_records_eq_to_ineq_tape() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, 0.0});
  plan.new_AL = copy_to_device<double>({1.0});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  stats.singleton_col_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_col_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_col_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_cols_dual_infer(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 1);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 1);
  assert(keep_col[0] == 1);
  assert(std::fabs(AL[0] - 1.0) < 1.0e-12);
  assert(std::fabs(AU[0] - 1.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 1}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 1}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0}));
  assert(plan.tape.vals.size() == 1);
  assert(std::fabs(plan.tape.vals[0]) < 1.0e-12);
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_col_mask);
  cudaFree(stats.singleton_col_row);
  cudaFree(stats.singleton_col_val);
}

int main() {
  test_eliminates_eq_singleton_column();
  test_dual_infer_marks_direct_unbounded();
  test_dual_infer_records_eq_to_ineq_tape();
  std::cout << "test_rule_singleton_cols passed\n";
  return 0;
}
