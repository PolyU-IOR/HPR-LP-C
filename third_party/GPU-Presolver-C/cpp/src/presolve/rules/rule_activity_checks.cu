#include "gpu_presolver/presolve/rules/rule_activity_checks.hpp"

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

__global__ void _kernel_activity_checks_classify(std::int32_t* status_flags,
                                                 std::uint8_t* full_redundant,
                                                 std::uint8_t* drop_lower,
                                                 std::uint8_t* drop_upper,
                                                 const std::uint8_t* keep_row,
                                                 const double* AL,
                                                 const double* AU,
                                                 const double* l,
                                                 const double* u,
                                                 const std::int32_t* row_nnz,
                                                 const std::int32_t* row_ptr,
                                                 const std::int32_t* col_val,
                                                 const double* nz_val,
                                                 double tol,
                                                 std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    full_redundant[i] = std::uint8_t{0};
    drop_lower[i] = std::uint8_t{0};
    drop_upper[i] = std::uint8_t{0};

    const bool keep_i = keep_row[i] != std::uint8_t{0};
    const std::int32_t row_nnz_i = row_nnz[i];
    if (!keep_i || row_nnz_i <= 1) {
      return;
    }

    const double lower_i = AL[i];
    const double upper_i = AU[i];
    const bool lower_finite = isfinite(lower_i);
    const bool upper_finite = isfinite(upper_i);
    if (lower_finite && upper_finite && fabs(upper_i - lower_i) <= tol) {
      return;
    }

    double row_min = 0.0;
    double row_max = 0.0;
    const std::int32_t row_start = row_ptr[i];
    const std::int32_t row_stop = row_ptr[i + 1];
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const std::int32_t col = col_val[p];
      const double a = nz_val[p];
      const double lj = l[col];
      const double uj = u[col];
      if (a >= 0.0) {
        row_min += a * lj;
        row_max += a * uj;
      } else {
        row_min += a * uj;
        row_max += a * lj;
      }
    }

    const bool infeasible =
        (lower_finite && row_max < lower_i - tol) ||
        (upper_finite && row_min > upper_i + tol);
    if (infeasible) {
      atomicMax(&status_flags[0], 1);
      return;
    }

    const bool lower_implied = !lower_finite || row_min >= lower_i - tol;
    const bool upper_implied = !upper_finite || row_max <= upper_i + tol;
    if (lower_implied && upper_implied) {
      full_redundant[i] = std::uint8_t{1};
      atomicMax(&status_flags[1], 1);
    } else if (lower_finite && lower_implied) {
      drop_lower[i] = std::uint8_t{1};
      atomicMax(&status_flags[2], 1);
    } else if (upper_finite && upper_implied) {
      drop_upper[i] = std::uint8_t{1};
      atomicMax(&status_flags[3], 1);
    }
  }
}

__global__ void _kernel_activity_checks_apply(std::uint8_t* keep_row,
                                              double* AL,
                                              double* AU,
                                              const std::uint8_t* full_redundant,
                                              const std::uint8_t* drop_lower,
                                              const std::uint8_t* drop_upper,
                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    if (full_redundant[i] != std::uint8_t{0}) {
      keep_row[i] = std::uint8_t{0};
    }
    if (drop_lower[i] != std::uint8_t{0}) {
      AL[i] = -INFINITY;
    }
    if (drop_upper[i] != std::uint8_t{0}) {
      AU[i] = INFINITY;
    }
  }
}

}  // namespace

void apply_rule_activity_checks(PresolvePlanGpu& plan,
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
  std::uint8_t* full_redundant = nullptr;
  std::uint8_t* drop_lower = nullptr;
  std::uint8_t* drop_upper = nullptr;
  throw_if_cuda_error(cudaMalloc(&status_flags, sizeof(std::int32_t) * 4), "cudaMalloc activity_checks status_flags");
  throw_if_cuda_error(cudaMalloc(&full_redundant, static_cast<std::size_t>(m)), "cudaMalloc activity_checks full_redundant");
  throw_if_cuda_error(cudaMalloc(&drop_lower, static_cast<std::size_t>(m)), "cudaMalloc activity_checks drop_lower");
  throw_if_cuda_error(cudaMalloc(&drop_upper, static_cast<std::size_t>(m)), "cudaMalloc activity_checks drop_upper");
  throw_if_cuda_error(cudaMemset(status_flags, 0, sizeof(std::int32_t) * 4), "cudaMemset activity_checks status_flags");

  const int blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_activity_checks_classify<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flags,
      full_redundant,
      drop_lower,
      drop_upper,
      plan.keep_row_mask,
      plan.new_AL,
      plan.new_AU,
      plan.new_l,
      plan.new_u,
      stats.row_nnz,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.bound_tol,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_activity_checks_classify");

  std::int32_t flags[4] = {0, 0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, status_flags, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy activity_checks flags");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
  } else if (flags[1] != 0 || flags[2] != 0 || flags[3] != 0) {
    if (pparams.record_postsolve_tape && pparams.record_postsolve_tape_cpu &&
        (flags[1] != 0 || flags[2] != 0 || flags[3] != 0)) {
      std::vector<std::uint8_t> host_full_redundant(static_cast<std::size_t>(m));
      std::vector<std::uint8_t> host_drop_lower(static_cast<std::size_t>(m));
      std::vector<std::uint8_t> host_drop_upper(static_cast<std::size_t>(m));
      std::vector<double> host_AL(static_cast<std::size_t>(m));
      std::vector<double> host_AU(static_cast<std::size_t>(m));
      throw_if_cuda_error(cudaMemcpy(host_full_redundant.data(), full_redundant, static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy activity_checks full_redundant for tape");
      throw_if_cuda_error(cudaMemcpy(host_drop_lower.data(), drop_lower, static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy activity_checks drop_lower for tape");
      throw_if_cuda_error(cudaMemcpy(host_drop_upper.data(), drop_upper, static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy activity_checks drop_upper for tape");
      throw_if_cuda_error(cudaMemcpy(host_AL.data(), plan.new_AL, sizeof(double) * static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy activity_checks AL for tape");
      throw_if_cuda_error(cudaMemcpy(host_AU.data(), plan.new_AU, sizeof(double) * static_cast<std::size_t>(m),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy activity_checks AU for tape");
      if (flags[1] != 0) {
        for (std::int32_t row = 0; row < m; ++row) {
          if (host_full_redundant[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
            continue;
          }
          append_postsolve_record(plan.tape,
                                  PostsolveReductionType::DeletedRow,
                                  {row},
                                  {host_AL[static_cast<std::size_t>(row)],
                                   host_AU[static_cast<std::size_t>(row)]},
                                  PostsolveDualMode::Minimal);
        }
      }
      for (std::int32_t row = 0; row < m; ++row) {
        if (host_drop_lower[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
          append_postsolve_record(plan.tape,
                                  PostsolveReductionType::LhsChange,
                                  {row},
                                  {host_AL[static_cast<std::size_t>(row)],
                                   host_AU[static_cast<std::size_t>(row)],
                                   -INFINITY,
                                   host_AU[static_cast<std::size_t>(row)]},
                                  PostsolveDualMode::Minimal);
        }
        if (host_drop_upper[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
          append_postsolve_record(plan.tape,
                                  PostsolveReductionType::RhsChange,
                                  {row},
                                  {host_AL[static_cast<std::size_t>(row)],
                                   host_AU[static_cast<std::size_t>(row)],
                                   host_AL[static_cast<std::size_t>(row)],
                                   INFINITY},
                                  PostsolveDualMode::Minimal);
        }
      }
    }
    _kernel_activity_checks_apply<<<blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        full_redundant,
        drop_lower,
        drop_upper,
        m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_activity_checks_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_activity_checks synchronize");
    plan.has_row_action = true;
    plan.has_change = true;
  }

  cudaFree(status_flags);
  cudaFree(full_redundant);
  cudaFree(drop_lower);
  cudaFree(drop_upper);
}

}  // namespace gpu_presolver::presolve
