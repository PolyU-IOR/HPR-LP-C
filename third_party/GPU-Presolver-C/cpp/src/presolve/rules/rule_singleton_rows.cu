#include "gpu_presolver/presolve/rules/rule_singleton_rows.hpp"

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

__device__ double atomic_max_double(double* address, double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current >= value) {
      break;
    }
    old = atomicCAS(
        address_as_ull,
        assumed,
        static_cast<unsigned long long int>(__double_as_longlong(value)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ double atomic_min_double(double* address, double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current <= value) {
      break;
    }
    old = atomicCAS(
        address_as_ull,
        assumed,
        static_cast<unsigned long long int>(__double_as_longlong(value)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}

__global__ void _kernel_singleton_row_bounds(double* candidate_l,
                                             double* candidate_u,
                                             std::uint8_t* row_remove,
                                             const std::uint8_t* keep_row,
                                             const std::uint8_t* singleton_row_mask,
                                             const std::int32_t* support_col,
                                             const double* support_val,
                                             const double* AL,
                                             const double* AU,
                                             std::int32_t n,
                                             double zero_tol,
                                             std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0} && singleton_row_mask[i] != std::uint8_t{0}) {
    const std::int32_t col = support_col[i];
    const double a = support_val[i];
    if (col >= 0 && col < n && fabs(a) > zero_tol) {
      const double lower = AL[i];
      const double upper = AU[i];
      const double implied_l = a > 0.0 ? (lower / a) : (upper / a);
      const double implied_u = a > 0.0 ? (upper / a) : (lower / a);
      atomic_max_double(&candidate_l[col], implied_l);
      atomic_min_double(&candidate_u[col], implied_u);
      row_remove[i] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_singleton_rows_apply_and_check(std::uint8_t* keep_row,
                                                       double* new_l,
                                                       double* new_u,
                                                       const double* candidate_l,
                                                       const double* candidate_u,
                                                       const std::uint8_t* row_remove,
                                                       double tol,
                                                       std::int32_t m,
                                                       std::int32_t n,
                                                       std::int32_t* flags) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  (void)keep_row;
  if (j < n) {
    if (candidate_l[j] > candidate_u[j] + tol) {
      atomicMax(&flags[0], 1);
    }
    if (candidate_l[j] != new_l[j] || candidate_u[j] != new_u[j]) {
      atomicMax(&flags[2], 1);
    }
  }
  if (j < m && row_remove[j] != std::uint8_t{0}) {
    atomicMax(&flags[1], 1);
  }
}

__global__ void _kernel_singleton_rows_commit(std::uint8_t* keep_row,
                                              double* new_l,
                                              double* new_u,
                                              const double* candidate_l,
                                              const double* candidate_u,
                                              const std::uint8_t* row_remove,
                                              std::int32_t m,
                                              std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < m && row_remove[j] != std::uint8_t{0}) {
    keep_row[j] = std::uint8_t{0};
  }
  if (j < n) {
    new_l[j] = candidate_l[j];
    new_u[j] = candidate_u[j];
  }
}

}  // namespace

void apply_rule_singleton_rows(PresolvePlanGpu& plan,
                               const LPInfoGpu& lp,
                               const PresolveStatsGpu& stats,
                               const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0) {
    return;
  }

  double* candidate_l = nullptr;
  double* candidate_u = nullptr;
  std::uint8_t* row_remove = nullptr;
  std::int32_t* flags_device = nullptr;
  throw_if_cuda_error(cudaMalloc(&candidate_l, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc candidate_l");
  throw_if_cuda_error(cudaMalloc(&candidate_u, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc candidate_u");
  throw_if_cuda_error(cudaMalloc(&row_remove, static_cast<std::size_t>(m)), "cudaMalloc row_remove");
  throw_if_cuda_error(cudaMalloc(&flags_device, sizeof(std::int32_t) * 3), "cudaMalloc singleton_rows flags");
  throw_if_cuda_error(cudaMemcpy(candidate_l, plan.new_l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy candidate_l");
  throw_if_cuda_error(cudaMemcpy(candidate_u, plan.new_u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy candidate_u");
  throw_if_cuda_error(cudaMemset(row_remove, 0, static_cast<std::size_t>(m)), "cudaMemset row_remove");
  throw_if_cuda_error(cudaMemset(flags_device, 0, sizeof(std::int32_t) * 3), "cudaMemset singleton_rows flags");

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_row_bounds<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      candidate_l,
      candidate_u,
      row_remove,
      plan.keep_row_mask,
      stats.singleton_row_mask,
      stats.singleton_row_col,
      stats.singleton_row_val,
      plan.new_AL,
      plan.new_AU,
      n,
      pparams.zero_tol,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_row_bounds");

  const std::int32_t max_mn = m > n ? m : n;
  const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_rows_apply_and_check<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
      plan.keep_row_mask,
      plan.new_l,
      plan.new_u,
      candidate_l,
      candidate_u,
      row_remove,
      pparams.bound_tol,
      m,
      n,
      flags_device);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_rows_apply_and_check");

  std::int32_t flags[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, flags_device, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy singleton_rows flags");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
  } else if (flags[1] != 0 || flags[2] != 0) {
    if (pparams.record_postsolve_tape && pparams.record_postsolve_tape_cpu && flags[1] != 0) {
      std::vector<std::uint8_t> host_row_remove(static_cast<std::size_t>(m));
      std::vector<std::int32_t> host_support_col(static_cast<std::size_t>(m));
      std::vector<double> host_support_val(static_cast<std::size_t>(m));
      std::vector<double> host_AL(static_cast<std::size_t>(m));
      std::vector<double> host_AU(static_cast<std::size_t>(m));
      throw_if_cuda_error(cudaMemcpy(host_row_remove.data(), row_remove, static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy singleton_rows row_remove for tape");
      throw_if_cuda_error(cudaMemcpy(host_support_col.data(), stats.singleton_row_col,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy singleton_rows support cols for tape");
      throw_if_cuda_error(cudaMemcpy(host_support_val.data(), stats.singleton_row_val,
                                     sizeof(double) * static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy singleton_rows support vals for tape");
      throw_if_cuda_error(cudaMemcpy(host_AL.data(), plan.new_AL, sizeof(double) * static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy singleton_rows AL for tape");
      throw_if_cuda_error(cudaMemcpy(host_AU.data(), plan.new_AU, sizeof(double) * static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy singleton_rows AU for tape");
      for (std::int32_t row = 0; row < m; ++row) {
        if (host_row_remove[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
          continue;
        }
        append_postsolve_record(plan.tape,
                                PostsolveReductionType::DeletedRow,
                                {row, host_support_col[static_cast<std::size_t>(row)]},
                                {host_AL[static_cast<std::size_t>(row)],
                                 host_AU[static_cast<std::size_t>(row)],
                                 host_support_val[static_cast<std::size_t>(row)]},
                                PostsolveDualMode::Minimal);
      }
    }
    _kernel_singleton_rows_commit<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        plan.new_l,
        plan.new_u,
        candidate_l,
        candidate_u,
        row_remove,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_rows_commit");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_singleton_rows synchronize");
    plan.has_row_action = true;
    plan.has_change = true;
  }

  cudaFree(candidate_l);
  cudaFree(candidate_u);
  cudaFree(row_remove);
  cudaFree(flags_device);
}

}  // namespace gpu_presolver::presolve
