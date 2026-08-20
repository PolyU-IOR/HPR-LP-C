#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

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

int main() {
  using gpu_presolver::presolve::DeviceCsrMatrix;
  const double inf = std::numeric_limits<double>::infinity();

  std::int32_t* row_ptr = copy_to_device<std::int32_t>({0, 0, 1, 3});
  std::int32_t* col_val = copy_to_device<std::int32_t>({2, 0, 1});
  double* nz_val = copy_to_device<double>({4.0, 5.0, 6.0});

  DeviceCsrMatrix matrix{3, 3, 3, row_ptr, col_val, nz_val};

  std::int32_t* row_nnz = nullptr;
  std::int32_t* singleton_col = nullptr;
  double* singleton_val = nullptr;
  check(cudaMalloc(&row_nnz, sizeof(std::int32_t) * 3), "cudaMalloc row_nnz");
  check(cudaMalloc(&singleton_col, sizeof(std::int32_t) * 3), "cudaMalloc singleton_col");
  check(cudaMalloc(&singleton_val, sizeof(double) * 3), "cudaMalloc singleton_val");

  gpu_presolver::presolve::compute_row_nnz(row_nnz, matrix);
  gpu_presolver::presolve::compute_singleton_row_support(singleton_col, singleton_val, row_nnz, matrix);
  check(cudaDeviceSynchronize(), "sync");

  const std::vector<std::int32_t> nnz = copy_to_host(row_nnz, 3);
  const std::vector<std::int32_t> support_col = copy_to_host(singleton_col, 3);
  const std::vector<double> support_val = copy_to_host(singleton_val, 3);

  assert((nnz == std::vector<std::int32_t>{0, 1, 2}));
  assert((support_col == std::vector<std::int32_t>{-1, 2, -1}));
  assert(support_val[0] == 0.0);
  assert(support_val[1] == 4.0);
  assert(support_val[2] == 0.0);

  double* l = copy_to_device<double>({-1.0, -inf, 2.0});
  double* u = copy_to_device<double>({3.0, 5.0, inf});
  double* row_min = nullptr;
  double* row_max = nullptr;
  double* row_min_fin = nullptr;
  double* row_max_fin = nullptr;
  std::int32_t* row_min_neg_inf_count = nullptr;
  std::int32_t* row_max_pos_inf_count = nullptr;
  check(cudaMalloc(&row_min, sizeof(double) * 3), "cudaMalloc row_min");
  check(cudaMalloc(&row_max, sizeof(double) * 3), "cudaMalloc row_max");
  check(cudaMalloc(&row_min_fin, sizeof(double) * 3), "cudaMalloc row_min_fin");
  check(cudaMalloc(&row_max_fin, sizeof(double) * 3), "cudaMalloc row_max_fin");
  check(cudaMalloc(&row_min_neg_inf_count, sizeof(std::int32_t) * 3), "cudaMalloc row_min_neg_inf_count");
  check(cudaMalloc(&row_max_pos_inf_count, sizeof(std::int32_t) * 3), "cudaMalloc row_max_pos_inf_count");

  gpu_presolver::presolve::compute_row_activity_bounds(row_min, row_max, matrix, l, u);
  gpu_presolver::presolve::compute_row_activity_summary(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      matrix,
      l,
      u,
      1.0e-12);
  check(cudaDeviceSynchronize(), "activity sync");

  const std::vector<double> h_row_min = copy_to_host(row_min, 3);
  const std::vector<double> h_row_max = copy_to_host(row_max, 3);
  const std::vector<double> h_row_min_fin = copy_to_host(row_min_fin, 3);
  const std::vector<double> h_row_max_fin = copy_to_host(row_max_fin, 3);
  const std::vector<std::int32_t> h_neg = copy_to_host(row_min_neg_inf_count, 3);
  const std::vector<std::int32_t> h_pos = copy_to_host(row_max_pos_inf_count, 3);
  assert(h_row_min[0] == 0.0);
  assert(h_row_max[0] == 0.0);
  assert(h_row_min[1] == 8.0);
  assert(std::isinf(h_row_max[1]) && h_row_max[1] > 0.0);
  assert(std::isinf(h_row_min[2]) && h_row_min[2] < 0.0);
  assert(h_row_max[2] == 45.0);
  assert(h_row_min_fin[2] == -5.0);
  assert(h_row_max_fin[2] == 45.0);
  assert(h_neg[2] == 1);
  assert(h_pos[2] == 0);

  std::int32_t* at_row_ptr = copy_to_device<std::int32_t>({0, 1, 2, 3});
  std::int32_t* at_col_val = copy_to_device<std::int32_t>({2, 2, 1});
  double* at_nz_val = copy_to_device<double>({5.0, -6.0, 4.0});
  DeviceCsrMatrix transpose{3, 3, 3, at_row_ptr, at_col_val, at_nz_val};
  double* col_max_abs = nullptr;
  check(cudaMalloc(&col_max_abs, sizeof(double) * 3), "cudaMalloc col_max_abs");
  gpu_presolver::presolve::compute_col_max_abs(col_max_abs, transpose);
  check(cudaDeviceSynchronize(), "col max sync");
  const std::vector<double> h_col_max_abs = copy_to_host(col_max_abs, 3);
  assert((h_col_max_abs == std::vector<double>{5.0, 6.0, 4.0}));

  cudaFree(row_ptr);
  cudaFree(col_val);
  cudaFree(nz_val);
  cudaFree(row_nnz);
  cudaFree(singleton_col);
  cudaFree(singleton_val);
  cudaFree(l);
  cudaFree(u);
  cudaFree(row_min);
  cudaFree(row_max);
  cudaFree(row_min_fin);
  cudaFree(row_max_fin);
  cudaFree(row_min_neg_inf_count);
  cudaFree(row_max_pos_inf_count);
  cudaFree(at_row_ptr);
  cudaFree(at_col_val);
  cudaFree(at_nz_val);
  cudaFree(col_max_abs);

  std::cout << "test_gpu_presolve_kernels passed\n";
  return 0;
}
