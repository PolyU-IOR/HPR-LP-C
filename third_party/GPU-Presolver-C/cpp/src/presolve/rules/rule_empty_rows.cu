#include "gpu_presolver/presolve/rules/rule_empty_rows.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

__global__ void _kernel_empty_rows_classify(std::int32_t* status_flags,
                                            const std::uint8_t* keep_row,
                                            const double* AL,
                                            const double* AU,
                                            const std::int32_t* row_nnz,
                                            double tol,
                                            std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m) {
    const bool live = keep_row[row] != std::uint8_t{0};
    const bool is_empty = row_nnz[row] == 0;
    if (live && is_empty) {
      const double al = AL[row];
      const double au = AU[row];
      if (al <= tol && au >= -tol) {
        atomicMax(&status_flags[1], 1);
      } else {
        atomicMax(&status_flags[0], 1);
      }
    }
  }
}

__global__ void _kernel_empty_rows_apply(std::uint8_t* keep_row,
                                         const double* AL,
                                         const double* AU,
                                         const std::int32_t* row_nnz,
                                         double tol,
                                         std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m) {
    const bool live = keep_row[row] != std::uint8_t{0};
    const bool is_empty = row_nnz[row] == 0;
    if (live && is_empty) {
      const double al = AL[row];
      const double au = AU[row];
      if (al <= tol && au >= -tol) {
        keep_row[row] = std::uint8_t{0};
      }
    }
  }
}

}  // namespace

void apply_rule_empty_rows(PresolvePlanGpu& plan,
                           const LPInfoGpu& lp,
                           const PresolveStatsGpu& stats,
                           const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  if (m == 0) {
    return;
  }

  std::int32_t* status_flags = nullptr;
  throw_if_cuda_error(cudaMalloc(&status_flags, sizeof(std::int32_t) * 2), "cudaMalloc empty_rows status_flags");
  throw_if_cuda_error(cudaMemset(status_flags, 0, sizeof(std::int32_t) * 2), "cudaMemset empty_rows status_flags");

  const int blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_empty_rows_classify<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flags,
      plan.keep_row_mask,
      plan.new_AL,
      plan.new_AU,
      stats.row_nnz,
      pparams.feasibility_tol,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_empty_rows_classify");

  std::int32_t flags[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, status_flags, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy empty_rows flags");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
    cudaFree(status_flags);
    return;
  }

  if (flags[1] != 0) {
    _kernel_empty_rows_apply<<<blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        stats.row_nnz,
        pparams.feasibility_tol,
        m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_empty_rows_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_empty_rows synchronize");
    plan.has_row_action = true;
    plan.has_change = true;
  }

  cudaFree(status_flags);
}

}  // namespace gpu_presolver::presolve
