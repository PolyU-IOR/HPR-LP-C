#include "gpu_presolver/presolve/rules/rule_close_bounds.hpp"

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
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc copy_to_device");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy copy_to_device");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy copy_to_host");
  return values;
}

void free_all(std::initializer_list<void*> pointers) {
  for (void* pointer : pointers) {
    if (pointer != nullptr) {
      cudaFree(pointer);
    }
  }
}

}  // namespace

int main() {
  using gpu_presolver::presolve::DeviceCsrMatrix;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;

  // Julia indexing equivalent:
  // AT has 2 columns-as-rows. Column 1 appears in row 1 with coeff 2.0 and row 2 with coeff -1.0.
  // Column 2 appears in row 2 with coeff 3.0.
  std::int32_t* at_row_ptr = copy_to_device<std::int32_t>({0, 2, 3});
  std::int32_t* at_col_val = copy_to_device<std::int32_t>({0, 1, 1});
  double* at_nz_val = copy_to_device<double>({2.0, -1.0, 3.0});

  std::uint8_t* keep_col = copy_to_device<std::uint8_t>({1, 1});
  std::uint8_t* keep_row = copy_to_device<std::uint8_t>({1, 1});
  double* new_c = copy_to_device<double>({4.0, 5.0});
  double* new_l = copy_to_device<double>({2.0, 0.0});
  double* new_u = copy_to_device<double>({2.0, 10.0});
  double* new_AL = copy_to_device<double>({5.0, -3.0});
  double* new_AU = copy_to_device<double>({7.0, 9.0});

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.A.nnz = 3;
  lp.AT = DeviceCsrMatrix{2, 2, 3, at_row_ptr, at_col_val, at_nz_val};

  PresolvePlanGpu plan;
  plan.keep_row_mask = keep_row;
  plan.keep_col_mask = keep_col;
  plan.new_c = new_c;
  plan.new_l = new_l;
  plan.new_u = new_u;
  plan.new_AL = new_AL;
  plan.new_AU = new_AU;

  PresolveParams params;
  params.bound_tol = 1.0e-9;

  gpu_presolver::presolve::apply_rule_close_bounds(plan, lp, params);

  const std::vector<std::uint8_t> keep_col_after = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> al_after = copy_to_host(plan.new_AL, 2);
  const std::vector<double> au_after = copy_to_host(plan.new_AU, 2);

  assert(keep_col_after[0] == 0);
  assert(keep_col_after[1] == 1);
  assert(std::fabs(al_after[0] - 1.0) < 1.0e-12);
  assert(std::fabs(au_after[0] - 3.0) < 1.0e-12);
  assert(std::fabs(al_after[1] - (-1.0)) < 1.0e-12);
  assert(std::fabs(au_after[1] - 11.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 8.0) < 1.0e-12);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert(plan.tape.types.empty());

  free_all({
      at_row_ptr,
      at_col_val,
      at_nz_val,
      keep_col,
      keep_row,
      new_c,
      new_l,
      new_u,
      new_AL,
      new_AU,
  });

  std::cout << "test_rule_close_bounds passed\n";
  return 0;
}
