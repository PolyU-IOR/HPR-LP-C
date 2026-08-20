#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

struct GpuRuntimeInfo {
  int device_count = 0;
  int selected_device = -1;
};

struct GpuPresolveSummary {
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t reduced_rows = 0;
  std::int32_t reduced_cols = 0;
  std::int32_t reduced_nnz = 0;
  std::int32_t iterations = 0;
  double elapsed_seconds = 0.0;
  double obj_constant_delta = 0.0;
  bool has_infeasible = false;
  bool has_unbounded = false;
  PresolveRecordGpu record;
  LPInfoGpu reduced_lp;
  bool owns_reduced_lp = false;
};

GpuRuntimeInfo query_gpu_runtime();

// Mirrors Julia `_presolve_nnz(lp::LP_info_gpu)`.
std::int32_t _presolve_nnz(const LPInfoGpu& lp);

// Mirrors Julia `_has_good_nnz_progress(nnz_before, nnz_after, ratio)`.
bool _has_good_nnz_progress(std::int32_t nnz_before, std::int32_t nnz_after, double ratio);

std::int32_t run_cuda_smoke_count(std::int32_t n);

GpuPresolveSummary run_gpu_presolve_fixed_order(const LPInfoGpu& lp,
                                                const PresolveParams& params);

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params);

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params);

void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary);

void free_gpu_presolve_record_resources(PresolveRecordGpu& record);

}  // namespace gpu_presolver::presolve
