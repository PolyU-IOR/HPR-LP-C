#include "gpu_presolver/presolve/rules/rule_primal_propagation.hpp"

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

void test_tightens_bounds_from_row_activity() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {2, 3, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<std::int32_t>({0, 2, 3, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0, 0.0});
  plan.new_AL = copy_to_device<double>({5.0, -inf});
  plan.new_AU = copy_to_device<double>({inf, 6.0});
  plan.new_l = copy_to_device<double>({0.0, 2.0, 0.0});
  plan.new_u = copy_to_device<double>({10.0, 2.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});

  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_primal_propagation(plan, lp, stats, params);
  const std::vector<double> l = copy_to_host(plan.new_l, 3);
  const std::vector<double> u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0] - 3.0) < 1.0e-12);
  assert(std::fabs(u[0] - 6.0) < 1.0e-12);
  assert(!plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert(!plan.has_infeasible);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_records_fixed_column_from_row_activity() {
  using gpu_presolver::presolve::LPInfoGpu;
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
  lp.AT = {2, 1, 2,
           copy_to_device<std::int32_t>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({4.0, 0.0});
  plan.new_AL = copy_to_device<double>({6.0});
  plan.new_AU = copy_to_device<double>({inf});
  plan.new_l = copy_to_device<double>({0.0, 5.0});
  plan.new_u = copy_to_device<double>({1.0, 5.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_primal_propagation(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(keep_col[0] == 0);
  assert(std::fabs(l[0] - 1.0) < 1.0e-12);
  assert(std::fabs(u[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 4.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.tape.types.size() == 1);
  assert(plan.tape.types[0] == static_cast<std::int32_t>(PostsolveReductionType::FixedCol));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0}));
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 1.0) < 1.0e-12);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_detects_infeasible_tightening() {
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
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0});
  plan.new_AL = copy_to_device<double>({10.0});
  plan.new_AU = copy_to_device<double>({inf});
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({4.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_primal_propagation(plan, lp, stats, params);
  assert(plan.has_infeasible);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

int main() {
  test_tightens_bounds_from_row_activity();
  test_records_fixed_column_from_row_activity();
  test_detects_infeasible_tightening();
  std::cout << "test_rule_primal_propagation passed\n";
  return 0;
}
