#include "gpu_presolver/presolve/rules/rule_close_bounds.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(static_cast<std::int32_t>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<std::int32_t>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

double sum_device_double(const double* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return 0.0;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  double* result_device = nullptr;
  double result = 0.0;
  throw_if_cuda_error(cudaMalloc(&result_device, sizeof(double)), context);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMalloc(&temp_storage, temp_bytes), context);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMemcpy(&result, result_device, sizeof(double), cudaMemcpyDeviceToHost),
                      context);
  cudaFree(temp_storage);
  cudaFree(result_device);
  return result;
}

__global__ void _kernel_close_bounds_candidates(std::uint8_t* fixed_mask,
                                                double* fixed_val,
                                                double* obj_contrib,
                                                double* row_shift,
                                                const std::uint8_t* keep_col,
                                                const double* l,
                                                const double* u,
                                                const double* c,
                                                const std::int32_t* at_row_ptr,
                                                const std::int32_t* at_col_val,
                                                const double* at_nz_val,
                                                double bound_tol,
                                                std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    fixed_mask[j] = std::uint8_t{0};
    fixed_val[j] = 0.0;
    obj_contrib[j] = 0.0;
    if (keep_col[j] == std::uint8_t{0}) {
      return;
    }
    const double lj = l[j];
    const double uj = u[j];
    if (isfinite(lj) && isfinite(uj) && fabs(uj - lj) < bound_tol) {
      const double vj = 0.5 * (lj + uj);
      fixed_mask[j] = std::uint8_t{1};
      fixed_val[j] = vj;
      obj_contrib[j] = c[j] * vj;

      const std::int32_t p_start = at_row_ptr[j];
      const std::int32_t p_stop = at_row_ptr[j + 1];
      for (std::int32_t p = p_start; p < p_stop; ++p) {
        const std::int32_t row = at_col_val[p];
        const double shift = at_nz_val[p] * vj;
        atomicAdd(&row_shift[row], shift);
      }
    }
  }
}

__global__ void _kernel_close_bounds_apply(std::uint8_t* keep_col,
                                           double* new_AL,
                                           double* new_AU,
                                           const std::uint8_t* fixed_mask,
                                           const double* row_shift,
                                           std::int32_t n,
                                           std::int32_t m) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    keep_col[j] = std::uint8_t{0};
  }
  if (j < m) {
    new_AL[j] -= row_shift[j];
    new_AU[j] -= row_shift[j];
  }
}

__global__ void _kernel_count_mask(const std::uint8_t* mask,
                                   std::int32_t n,
                                   std::int32_t* count) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && mask[j] != std::uint8_t{0}) {
    atomicAdd(count, 1);
  }
}

}  // namespace

void apply_rule_close_bounds(PresolvePlanGpu& plan,
                             const LPInfoGpu& lp,
                             const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t n = lp.A.cols;
  const std::int32_t m = lp.A.rows;
  if (n == 0) {
    return;
  }

  std::uint8_t* fixed_mask = nullptr;
  double* fixed_val = nullptr;
  double* obj_contrib = nullptr;
  double* row_shift = nullptr;
  std::int32_t* fixed_count_device = nullptr;

  throw_if_cuda_error(cudaMalloc(&fixed_mask, static_cast<std::size_t>(n)), "cudaMalloc fixed_mask");
  throw_if_cuda_error(cudaMalloc(&fixed_val, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc fixed_val");
  throw_if_cuda_error(cudaMalloc(&obj_contrib, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc obj_contrib");
  throw_if_cuda_error(cudaMalloc(&row_shift, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc row_shift");
  throw_if_cuda_error(cudaMalloc(&fixed_count_device, sizeof(std::int32_t)), "cudaMalloc fixed_count");
  throw_if_cuda_error(cudaMemset(fixed_mask, 0, static_cast<std::size_t>(n)), "cudaMemset fixed_mask");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset row_shift");
  throw_if_cuda_error(cudaMemset(fixed_count_device, 0, sizeof(std::int32_t)), "cudaMemset fixed_count");

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const std::int32_t max_mn = n > m ? n : m;
  const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;

  _kernel_close_bounds_candidates<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      fixed_mask,
      fixed_val,
      obj_contrib,
      row_shift,
      plan.keep_col_mask,
      plan.new_l,
      plan.new_u,
      plan.new_c,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.bound_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_close_bounds_candidates");

  _kernel_count_mask<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      fixed_mask,
      n,
      fixed_count_device);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_count_mask close_bounds");

  std::int32_t fixed_count = 0;
  throw_if_cuda_error(cudaMemcpy(&fixed_count, fixed_count_device, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy fixed_count");
  if (fixed_count == 0) {
    cudaFree(fixed_mask);
    cudaFree(fixed_val);
    cudaFree(obj_contrib);
    cudaFree(row_shift);
    cudaFree(fixed_count_device);
    return;
  }

  if (pparams.record_postsolve_tape && pparams.record_postsolve_tape_cpu) {
    std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> host_keep_row(static_cast<std::size_t>(m));
    std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
    std::vector<double> host_c(static_cast<std::size_t>(n));
    std::vector<std::int32_t> host_at_row_ptr(static_cast<std::size_t>(n + 1));
    std::vector<std::int32_t> host_at_col_val(static_cast<std::size_t>(lp.AT.nnz));
    std::vector<double> host_at_nz_val(static_cast<std::size_t>(lp.AT.nnz));
    throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask, static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy close_bounds fixed_mask for tape");
    throw_if_cuda_error(cudaMemcpy(host_keep_row.data(), plan.keep_row_mask, static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy close_bounds keep_row for tape");
    throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                   sizeof(double) * static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy close_bounds fixed_val for tape");
    throw_if_cuda_error(cudaMemcpy(host_c.data(), plan.new_c, sizeof(double) * static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy close_bounds c for tape");
    throw_if_cuda_error(cudaMemcpy(host_at_row_ptr.data(), lp.AT.rowPtr,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(n + 1),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy close_bounds AT rowPtr for tape");
    if (lp.AT.nnz > 0) {
      throw_if_cuda_error(cudaMemcpy(host_at_col_val.data(), lp.AT.colVal,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(lp.AT.nnz),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy close_bounds AT colVal for tape");
      throw_if_cuda_error(cudaMemcpy(host_at_nz_val.data(), lp.AT.nzVal,
                                     sizeof(double) * static_cast<std::size_t>(lp.AT.nnz),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy close_bounds AT nzVal for tape");
    }
    for (std::int32_t col = 0; col < n; ++col) {
      if (host_fixed_mask[static_cast<std::size_t>(col)] == std::uint8_t{0}) {
        continue;
      }
      std::vector<std::int32_t> indices{col};
      std::vector<double> vals{host_fixed_val[static_cast<std::size_t>(col)],
                               host_c[static_cast<std::size_t>(col)]};
      for (std::int32_t p = host_at_row_ptr[static_cast<std::size_t>(col)];
           p < host_at_row_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
        const std::int32_t row = host_at_col_val[static_cast<std::size_t>(p)];
        if (host_keep_row[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
          continue;
        }
        indices.push_back(row);
        vals.push_back(host_at_nz_val[static_cast<std::size_t>(p)]);
      }
      append_postsolve_record(plan.tape,
                              PostsolveReductionType::FixedCol,
                              indices,
                              vals,
                              PostsolveDualMode::Minimal);
    }
  }

  _kernel_close_bounds_apply<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
      plan.keep_col_mask,
      plan.new_AL,
      plan.new_AU,
      fixed_mask,
      row_shift,
      n,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_close_bounds_apply");
  throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_close_bounds synchronize");

  const double obj_delta = sum_device_double(obj_contrib, n, "cudaMemcpy close_bounds obj_contrib");
  plan.obj_constant_delta += obj_delta;
  plan.has_col_action = true;
  plan.has_change = true;

  cudaFree(fixed_mask);
  cudaFree(fixed_val);
  cudaFree(obj_contrib);
  cudaFree(row_shift);
  cudaFree(fixed_count_device);
}

}  // namespace gpu_presolver::presolve
