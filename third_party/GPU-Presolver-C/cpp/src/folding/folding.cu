#include "folding_internal.cuh"

#include "gpu_presolver/folding/folding.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_presolver::folding {
namespace {

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
std::vector<T> copy_to_host(const T* device, std::size_t size, const char* context) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost), context);
  return values;
}

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

template <typename Cleanup>
struct ScopeExit {
  explicit ScopeExit(Cleanup cleanup_in) : cleanup(std::move(cleanup_in)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() {
    if (active) {
      cleanup();
    }
  }
  void release() { active = false; }

  Cleanup cleanup;
  bool active = true;
};

template <typename Cleanup>
ScopeExit<Cleanup> make_scope_exit(Cleanup cleanup) {
  return ScopeExit<Cleanup>(std::move(cleanup));
}

}  // namespace

FoldingRunSummary run_folding(const presolve::LPInfoGpu& original_model,
                              double tolerance,
                              bool verbose) {
  const auto total_start = std::chrono::steady_clock::now();
  FoldingRunSummary summary;
  summary.original_rows = original_model.A.rows;
  summary.original_cols = original_model.A.cols;
  summary.folded_rows = original_model.A.rows;
  summary.folded_cols = original_model.A.cols;

  FoldingWorkspace workspace;
  auto workspace_guard = make_scope_exit([&]() { free_workspace(workspace); });
  auto summary_guard = make_scope_exit([&]() { free_folded_lp(summary); });
  init_workspace(workspace, original_model.A.rows, original_model.A.cols);
  const bool refinement_ok =
      refine_color(original_model, workspace, tolerance, verbose, &summary.profile.refinement);
  if (!refinement_ok) {
    free_workspace(workspace);
    workspace_guard.release();
    summary.profile.total_seconds =
        seconds_since(total_start, std::chrono::steady_clock::now());
    summary_guard.release();
    return summary;
  }

  if (workspace.num_row_color == original_model.A.rows &&
      workspace.num_col_color == original_model.A.cols) {
    free_workspace(workspace);
    workspace_guard.release();
    summary.profile.total_seconds =
        seconds_since(total_start, std::chrono::steady_clock::now());
    summary_guard.release();
    return summary;
  }

  summary.folded_lp = reduce_size(workspace, original_model, tolerance, &summary.profile.reduce);
  summary.owns_folded_lp = true;
  summary.map = build_device_map(workspace, &summary.profile.map);
  summary.applied = summary.folded_lp.A.rows != original_model.A.rows ||
                    summary.folded_lp.A.cols != original_model.A.cols;
  summary.folded_rows = summary.folded_lp.A.rows;
  summary.folded_cols = summary.folded_lp.A.cols;
  if (!summary.applied) {
    free_lp_device(summary.folded_lp);
    summary.owns_folded_lp = false;
  }
  free_workspace(workspace);
  workspace_guard.release();
  summary.profile.total_seconds =
      seconds_since(total_start, std::chrono::steady_clock::now());
  summary_guard.release();
  return summary;
}

void free_folded_lp(FoldingRunSummary& summary) {
  free_folding_map_device(summary.map);
  if (!summary.owns_folded_lp) {
    summary.folded_lp = presolve::LPInfoGpu{};
    return;
  }
  free_lp_device(summary.folded_lp);
  summary.owns_folded_lp = false;
}

FoldingPipelineSummary run_gpu_presolve_with_folding(const presolve::LPInfoGpu& lp,
                                                     const presolve::PresolveParams& params,
                                                     bool keep_reduced_lp,
                                                     bool keep_folded_lp) {
  FoldingPipelineSummary summary;
  auto summary_guard = make_scope_exit([&]() {
    presolve::free_gpu_presolve_reduced_lp(summary.presolve);
    free_folded_lp(summary.folding);
  });
  summary.original_rows = lp.A.rows;
  summary.original_cols = lp.A.cols;
  summary.folded_rows = lp.A.rows;
  summary.folded_cols = lp.A.cols;

  if (!params.enable_folding) {
    if (keep_reduced_lp) {
      summary.presolve = presolve::run_gpu_presolve_with_reduced_lp(lp, params);
    } else {
      summary.presolve = presolve::run_gpu_presolve_with_record(lp, params);
    }
    summary_guard.release();
    return summary;
  }

  summary.folding = run_folding(lp, params.folding_tolerance, params.verbose);
  summary.folding_applied = summary.folding.applied;
  if (summary.folding_applied) {
    summary.folded_rows = summary.folding.folded_rows;
    summary.folded_cols = summary.folding.folded_cols;
    if (keep_reduced_lp) {
      summary.presolve =
          presolve::run_gpu_presolve_with_reduced_lp(summary.folding.folded_lp, params);
    } else {
      summary.presolve =
          presolve::run_gpu_presolve_with_record(summary.folding.folded_lp, params);
    }
    if (!keep_folded_lp) {
      free_lp_device(summary.folding.folded_lp);
      summary.folding.owns_folded_lp = false;
    }
  } else {
    if (keep_reduced_lp) {
      summary.presolve = presolve::run_gpu_presolve_with_reduced_lp(lp, params);
    } else {
      summary.presolve = presolve::run_gpu_presolve_with_record(lp, params);
    }
  }
  summary_guard.release();
  return summary;
}

UnfoldedSolutionHost postsolve_and_unfold_to_host(double* x_red,
                                                  double* y_red,
                                                  double* z_red,
                                                  const FoldingPipelineSummary& summary) {
  presolve::GpuPostsolveResult post =
      summary.folding_applied
          ? presolve::postsolve_gpu(x_red, y_red, z_red, summary.presolve.record, &summary.folding.folded_lp)
          : presolve::postsolve_gpu(x_red, y_red, z_red, summary.presolve.record);

  UnfoldedSolutionHost unfolded;
  if (!summary.folding_applied) {
    unfolded.x =
        copy_to_host(post.x_org, static_cast<std::size_t>(post.n0), "cudaMemcpy postsolve x");
    unfolded.y =
        copy_to_host(post.y_org, static_cast<std::size_t>(post.m0), "cudaMemcpy postsolve y");
    unfolded.z =
        copy_to_host(post.z_org, static_cast<std::size_t>(post.n0), "cudaMemcpy postsolve z");
    cudaFree(post.x_org);
    cudaFree(post.y_org);
    cudaFree(post.z_org);
    return unfolded;
  }
  try {
    unfolded = unfold_solution_device_to_host(summary.folding.map, post.x_org, post.y_org, post.z_org);
  } catch (...) {
    cudaFree(post.x_org);
    cudaFree(post.y_org);
    cudaFree(post.z_org);
    throw;
  }
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  return unfolded;
}

}  // namespace gpu_presolver::folding
