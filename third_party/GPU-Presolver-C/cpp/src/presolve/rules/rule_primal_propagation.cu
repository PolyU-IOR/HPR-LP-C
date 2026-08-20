#include "gpu_presolver/presolve/rules/rule_primal_propagation.hpp"

#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
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

  Cleanup cleanup;
  bool active = true;
};

template <typename Cleanup>
ScopeExit<Cleanup> make_scope_exit(Cleanup cleanup) {
  return ScopeExit<Cleanup>(std::move(cleanup));
}

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  throw_if_cuda_error(cudaMalloc(&temp_storage, temp_bytes), context);
  auto temp_storage_guard = make_scope_exit([&]() { cudaFree(temp_storage); });
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
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

__global__ void _kernel_primal_fixed_tape_counts(
    std::int32_t* selected_scan,
    std::int32_t* active_nnz_scan,
    const std::uint8_t* fixed_mask,
    const std::uint8_t* keep_row,
    const std::int32_t* at_row_ptr,
    const std::int32_t* at_col_val,
    std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n) {
    return;
  }
  const bool selected = fixed_mask[col] != std::uint8_t{0};
  selected_scan[col] = selected ? 1 : 0;
  std::int32_t active_nnz = 0;
  if (selected) {
    for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
      active_nnz += keep_row[at_col_val[p]] != std::uint8_t{0} ? 1 : 0;
    }
  }
  active_nnz_scan[col] = active_nnz;
}

__global__ void _kernel_primal_pack_fixed_tape(
    std::int32_t* packed_cols,
    double* packed_fixed_vals,
    double* packed_c,
    std::int32_t* packed_nnz_starts,
    std::int32_t* packed_rows,
    double* packed_coeffs,
    const std::int32_t* selected_scan,
    const std::int32_t* active_nnz_scan,
    const std::uint8_t* fixed_mask,
    const double* fixed_val,
    const std::uint8_t* keep_row,
    const double* c,
    const std::int32_t* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || fixed_mask[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const std::int32_t record = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || record < 0) {
    return;
  }
  const std::int32_t nnz_start = col == 0 ? 0 : active_nnz_scan[col - 1];
  const std::int32_t nnz_stop = active_nnz_scan[col];
  packed_cols[record] = col;
  packed_fixed_vals[record] = fixed_val[col];
  packed_c[record] = c[col];
  packed_nnz_starts[record + 1] = nnz_stop;
  std::int32_t out = nnz_start;
  for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (keep_row[row] == std::uint8_t{0}) {
      continue;
    }
    packed_rows[out] = row;
    packed_coeffs[out] = at_nz_val[p];
    ++out;
  }
}

double append_primal_fixed_col_tape_compacted_from_device(
    PostsolveTape* tape,
    const std::uint8_t* fixed_mask,
    const double* fixed_val,
    const std::uint8_t* keep_row,
    const double* c,
    const DeviceCsrMatrix& AT,
    std::int32_t n,
    const char* context) {
  if (n <= 0) {
    return 0.0;
  }
  std::int32_t* selected_scan = nullptr;
  std::int32_t* active_nnz_scan = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  throw_if_cuda_error(cudaMalloc(&active_nnz_scan,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_primal_fixed_tape_counts<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      selected_scan, active_nnz_scan, fixed_mask, keep_row, AT.rowPtr, AT.colVal, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_primal_fixed_tape_counts");
  inclusive_scan_i32(selected_scan, n, context);
  inclusive_scan_i32(active_nnz_scan, n, context);
  std::int32_t counts[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(&counts[0], selected_scan + n - 1,
                                 sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(&counts[1], active_nnz_scan + n - 1,
                                 sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      context);
  const std::int32_t fixed_count = counts[0];
  const std::int32_t active_nnz = counts[1];
  if (fixed_count <= 0) {
    cudaFree(selected_scan);
    cudaFree(active_nnz_scan);
    return 0.0;
  }

  std::int32_t* packed_cols = nullptr;
  double* packed_fixed_vals = nullptr;
  double* packed_c = nullptr;
  std::int32_t* packed_nnz_starts = nullptr;
  std::int32_t* packed_rows = nullptr;
  double* packed_coeffs = nullptr;
  throw_if_cuda_error(cudaMalloc(&packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(fixed_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_fixed_vals,
                                 sizeof(double) * static_cast<std::size_t>(fixed_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_c,
                                 sizeof(double) * static_cast<std::size_t>(fixed_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_nnz_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(fixed_count + 1)),
                      context);
  throw_if_cuda_error(cudaMemset(packed_nnz_starts, 0, sizeof(std::int32_t)), context);
  if (active_nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&packed_rows,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(active_nnz)),
                        context);
    throw_if_cuda_error(cudaMalloc(&packed_coeffs,
                                   sizeof(double) * static_cast<std::size_t>(active_nnz)),
                        context);
  }
  _kernel_primal_pack_fixed_tape<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      packed_cols, packed_fixed_vals, packed_c, packed_nnz_starts, packed_rows,
      packed_coeffs, selected_scan, active_nnz_scan, fixed_mask, fixed_val,
      keep_row, c, AT.rowPtr, AT.colVal, AT.nzVal, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_primal_pack_fixed_tape");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(fixed_count));
  std::vector<double> host_fixed_vals(static_cast<std::size_t>(fixed_count));
  std::vector<double> host_c(static_cast<std::size_t>(fixed_count));
  std::vector<std::int32_t> host_nnz_starts(
      tape == nullptr ? 0 : static_cast<std::size_t>(fixed_count + 1));
  std::vector<std::int32_t> host_rows(
      tape == nullptr ? 0 : static_cast<std::size_t>(active_nnz));
  std::vector<double> host_coeffs(
      tape == nullptr ? 0 : static_cast<std::size_t>(active_nnz));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(fixed_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_fixed_vals.data(), packed_fixed_vals,
                                 sizeof(double) * static_cast<std::size_t>(fixed_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_c.data(), packed_c,
                                 sizeof(double) * static_cast<std::size_t>(fixed_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (tape != nullptr) {
    throw_if_cuda_error(cudaMemcpy(host_nnz_starts.data(), packed_nnz_starts,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(fixed_count + 1),
                                   cudaMemcpyDeviceToHost),
                        context);
    if (active_nnz > 0) {
      throw_if_cuda_error(cudaMemcpy(host_rows.data(), packed_rows,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(active_nnz),
                                     cudaMemcpyDeviceToHost),
                          context);
      throw_if_cuda_error(cudaMemcpy(host_coeffs.data(), packed_coeffs,
                                     sizeof(double) * static_cast<std::size_t>(active_nnz),
                                     cudaMemcpyDeviceToHost),
                          context);
    }
  }

  long double objective_delta = 0.0L;
  for (std::int32_t rec = 0; rec < fixed_count; ++rec) {
    objective_delta +=
        static_cast<long double>(host_c[static_cast<std::size_t>(rec)]) *
        static_cast<long double>(host_fixed_vals[static_cast<std::size_t>(rec)]);
    if (tape == nullptr) {
      continue;
    }
    const std::int32_t start = host_nnz_starts[static_cast<std::size_t>(rec)];
    const std::int32_t stop = host_nnz_starts[static_cast<std::size_t>(rec + 1)];
    std::vector<std::int32_t> indices;
    std::vector<double> vals;
    indices.reserve(static_cast<std::size_t>(1 + stop - start));
    vals.reserve(static_cast<std::size_t>(2 + stop - start));
    indices.push_back(host_cols[static_cast<std::size_t>(rec)]);
    vals.push_back(host_fixed_vals[static_cast<std::size_t>(rec)]);
    vals.push_back(host_c[static_cast<std::size_t>(rec)]);
    for (std::int32_t p = start; p < stop; ++p) {
      indices.push_back(host_rows[static_cast<std::size_t>(p)]);
      vals.push_back(host_coeffs[static_cast<std::size_t>(p)]);
    }
    append_postsolve_record(
        *tape, PostsolveReductionType::FixedCol, indices, vals,
        PostsolveDualMode::Minimal);
  }

  cudaFree(selected_scan);
  cudaFree(active_nnz_scan);
  cudaFree(packed_cols);
  cudaFree(packed_fixed_vals);
  cudaFree(packed_c);
  cudaFree(packed_nnz_starts);
  cudaFree(packed_rows);
  cudaFree(packed_coeffs);
  return static_cast<double>(objective_delta);
}

__global__ void _kernel_primal_bound_change_record_counts(std::int32_t* counts,
                                                          const std::uint8_t* lower_changed,
                                                          const std::uint8_t* upper_changed,
                                                          std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    counts[j] = (lower_changed[j] != std::uint8_t{0} ? 1 : 0) +
                (upper_changed[j] != std::uint8_t{0} ? 1 : 0);
  }
}

__global__ void _kernel_pack_primal_bound_change_tape(std::int32_t* types,
                                                      std::int32_t* index_starts,
                                                      std::int32_t* value_starts,
                                                      std::uint8_t* dual_modes,
                                                      std::int32_t* indices,
                                                      double* vals,
                                                      const std::int32_t* counts_scan,
                                                      const std::uint8_t* lower_changed,
                                                      const std::uint8_t* upper_changed,
                                                      const std::int32_t* support_l_row,
                                                      const std::int32_t* support_u_row,
                                                      const double* old_l,
                                                      const double* old_u,
                                                      const double* new_l,
                                                      const double* new_u,
                                                      std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n) {
    return;
  }
  const std::int32_t prev = col == 0 ? 0 : counts_scan[col - 1];
  const std::int32_t total = counts_scan[col];
  if (total == prev) {
    return;
  }

  double cur_l = old_l[col];
  const double cur_u = old_u[col];
  std::int32_t rec = prev;
  if (lower_changed[col] != std::uint8_t{0}) {
    const std::int32_t row = support_l_row[col];
    types[rec] = row != INT_MAX
        ? static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow)
        : static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow);
    dual_modes[rec] = static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
    index_starts[rec] = 2 * rec;
    value_starts[rec] = 4 * rec;
    indices[2 * rec] = col;
    indices[2 * rec + 1] = row != INT_MAX ? row : -1;
    vals[4 * rec] = cur_l;
    vals[4 * rec + 1] = cur_u;
    vals[4 * rec + 2] = new_l[col];
    vals[4 * rec + 3] = cur_u;
    cur_l = new_l[col];
    ++rec;
  }
  if (upper_changed[col] != std::uint8_t{0}) {
    const std::int32_t row = support_u_row[col];
    types[rec] = row != INT_MAX
        ? static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow)
        : static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow);
    dual_modes[rec] = static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
    index_starts[rec] = 2 * rec;
    value_starts[rec] = 4 * rec;
    indices[2 * rec] = col;
    indices[2 * rec + 1] = row != INT_MAX ? row : -1;
    vals[4 * rec] = cur_l;
    vals[4 * rec + 1] = cur_u;
    vals[4 * rec + 2] = cur_l;
    vals[4 * rec + 3] = new_u[col];
  }
}

void append_primal_bound_change_tape_gpu(PostsolveTapeGpu& tape,
                                         const std::uint8_t* lower_changed,
                                         const std::uint8_t* upper_changed,
                                         const std::int32_t* support_l_row,
                                         const std::int32_t* support_u_row,
                                         const double* old_l,
                                         const double* old_u,
                                         const double* new_l,
                                         const double* new_u,
                                         std::int32_t n) {
  if (n <= 0) {
    return;
  }
  std::int32_t* counts = nullptr;
  auto counts_guard = make_scope_exit([&]() { cudaFree(counts); });
  throw_if_cuda_error(cudaMalloc(&counts, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc primal bound-change counts");
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_primal_bound_change_record_counts<<<blocks, GPU_PRESOLVE_THREADS>>>(
      counts, lower_changed, upper_changed, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_primal_bound_change_record_counts");
  inclusive_scan_i32(counts, n, "cub scan primal bound-change counts");
  std::int32_t record_count = 0;
  throw_if_cuda_error(cudaMemcpy(&record_count, counts + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy primal bound-change record count");
  if (record_count == 0) {
    cudaFree(counts);
    counts = nullptr;
    return;
  }

  PostsolveTapeGpu built;
  built.record_count = record_count;
  built.index_count = 2 * record_count;
  built.value_count = 4 * record_count;
  throw_if_cuda_error(cudaMalloc(&built.types, sizeof(std::int32_t) * static_cast<std::size_t>(record_count)),
                      "cudaMalloc primal bound-change tape types");
  throw_if_cuda_error(cudaMalloc(&built.index_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1)),
                      "cudaMalloc primal bound-change tape index_starts");
  throw_if_cuda_error(cudaMalloc(&built.value_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1)),
                      "cudaMalloc primal bound-change tape value_starts");
  throw_if_cuda_error(cudaMalloc(&built.dual_modes, static_cast<std::size_t>(record_count)),
                      "cudaMalloc primal bound-change tape dual_modes");
  throw_if_cuda_error(cudaMalloc(&built.indices,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(built.index_count)),
                      "cudaMalloc primal bound-change tape indices");
  throw_if_cuda_error(cudaMalloc(&built.vals, sizeof(double) * static_cast<std::size_t>(built.value_count)),
                      "cudaMalloc primal bound-change tape vals");

  _kernel_pack_primal_bound_change_tape<<<blocks, GPU_PRESOLVE_THREADS>>>(
      built.types,
      built.index_starts,
      built.value_starts,
      built.dual_modes,
      built.indices,
      built.vals,
      counts,
      lower_changed,
      upper_changed,
      support_l_row,
      support_u_row,
      old_l,
      old_u,
      new_l,
      new_u,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_primal_bound_change_tape");
  const std::int32_t final_index_start = built.index_count;
  const std::int32_t final_value_start = built.value_count;
  throw_if_cuda_error(cudaMemcpy(built.index_starts + record_count, &final_index_start,
                                 sizeof(std::int32_t), cudaMemcpyHostToDevice),
                      "cudaMemcpy primal bound-change final index_start");
  throw_if_cuda_error(cudaMemcpy(built.value_starts + record_count, &final_value_start,
                                 sizeof(std::int32_t), cudaMemcpyHostToDevice),
                      "cudaMemcpy primal bound-change final value_start");
  cudaFree(counts);
  counts = nullptr;

  tape = std::move(built);
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

__device__ void _term_interval_primal_propagation(double aij,
                                                  double lj,
                                                  double uj,
                                                  double* term_min,
                                                  double* term_max) {
  if (aij >= 0.0) {
    *term_min = aij * lj;
    *term_max = aij * uj;
  } else {
    *term_min = aij * uj;
    *term_max = aij * lj;
  }
}

__device__ double _row_residual_min_from_summary(double row_min_fin,
                                                 std::int32_t row_min_neg_inf_count,
                                                 double term_min) {
  if (isfinite(term_min)) {
    return row_min_neg_inf_count == 0 ? (row_min_fin - term_min) : -INFINITY;
  }
  if (term_min < 0.0) {
    return row_min_neg_inf_count > 1 ? -INFINITY : row_min_fin;
  }
  return row_min_fin;
}

__device__ double _row_residual_max_from_summary(double row_max_fin,
                                                 std::int32_t row_max_pos_inf_count,
                                                 double term_max) {
  if (isfinite(term_max)) {
    return row_max_pos_inf_count == 0 ? (row_max_fin - term_max) : INFINITY;
  }
  if (term_max > 0.0) {
    return row_max_pos_inf_count > 1 ? INFINITY : row_max_fin;
  }
  return row_max_fin;
}

__global__ void _kernel_primal_propagation_candidates(std::int32_t* infeasible_flag,
                                                      double* candidate_l,
                                                      double* candidate_u,
                                                      const double* row_min_fin,
                                                      const double* row_max_fin,
                                                      const std::int32_t* row_min_neg_inf_count,
                                                      const std::int32_t* row_max_pos_inf_count,
                                                      const std::uint8_t* keep_row,
                                                      const std::int32_t* row_nnz,
                                                      const double* AL,
                                                      const double* AU,
                                                      const double* l_cur,
                                                      const double* u_cur,
                                                      const std::int32_t* row_ptr,
                                                      const std::int32_t* col_val,
                                                      const double* nz_val,
                                                      double zero_tol,
                                                      std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0} && row_nnz[i] > 1) {
    const std::int32_t row_start = row_ptr[i];
    const std::int32_t row_stop = row_ptr[i + 1];
    const double lower_i = AL[i];
    const double upper_i = AU[i];
    const double row_min_fin_i = row_min_fin[i];
    const double row_max_fin_i = row_max_fin[i];
    const std::int32_t row_min_neg_inf_count_i = row_min_neg_inf_count[i];
    const std::int32_t row_max_pos_inf_count_i = row_max_pos_inf_count[i];
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const std::int32_t j = col_val[p];
      const double a = nz_val[p];
      if (fabs(a) <= zero_tol) {
        continue;
      }

      const double old_l = l_cur[j];
      const double old_u = u_cur[j];
      double term_min = 0.0;
      double term_max = 0.0;
      _term_interval_primal_propagation(a, old_l, old_u, &term_min, &term_max);

      const double rest_min = _row_residual_min_from_summary(
          row_min_fin_i,
          row_min_neg_inf_count_i,
          term_min);
      const double rest_max = _row_residual_max_from_summary(
          row_max_fin_i,
          row_max_pos_inf_count_i,
          term_max);

      double implied_l = -INFINITY;
      double implied_u = INFINITY;
      if (a > 0.0) {
        if (isfinite(lower_i) && isfinite(rest_max)) {
          implied_l = (lower_i - rest_max) / a;
        }
        if (isfinite(upper_i) && isfinite(rest_min)) {
          implied_u = (upper_i - rest_min) / a;
        }
      } else {
        if (isfinite(upper_i) && isfinite(rest_min)) {
          implied_l = (upper_i - rest_min) / a;
        }
        if (isfinite(lower_i) && isfinite(rest_max)) {
          implied_u = (lower_i - rest_max) / a;
        }
      }

      double new_l = old_l;
      double new_u = old_u;
      if (isfinite(implied_l)) {
        new_l = fmax(new_l, implied_l);
      }
      if (isfinite(implied_u)) {
        new_u = fmin(new_u, implied_u);
      }
      if (new_l > old_l) {
        atomic_max_double(&candidate_l[j], new_l);
      }
      if (new_u < old_u) {
        atomic_min_double(&candidate_u[j], new_u);
      }
    }
  }
  (void)infeasible_flag;
}

__global__ void _kernel_primal_propagation_candidates_block(
    std::int32_t* infeasible_flag,
    double* candidate_l,
    double* candidate_u,
    const double* row_min_fin,
    const double* row_max_fin,
    const std::int32_t* row_min_neg_inf_count,
    const std::int32_t* row_max_pos_inf_count,
    const std::uint8_t* keep_row,
    const std::int32_t* row_nnz,
    const double* AL,
    const double* AU,
    const double* l_cur,
    const double* u_cur,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    double zero_tol,
    std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x);
  if (i >= m || keep_row[i] == std::uint8_t{0} || row_nnz[i] <= 1) {
    return;
  }

  const std::int32_t row_start = row_ptr[i];
  const std::int32_t row_stop = row_ptr[i + 1];
  const double lower_i = AL[i];
  const double upper_i = AU[i];
  const double row_min_fin_i = row_min_fin[i];
  const double row_max_fin_i = row_max_fin[i];
  const std::int32_t row_min_neg_inf_count_i = row_min_neg_inf_count[i];
  const std::int32_t row_max_pos_inf_count_i = row_max_pos_inf_count[i];
  for (std::int32_t p = row_start + static_cast<std::int32_t>(threadIdx.x);
       p < row_stop;
       p += static_cast<std::int32_t>(blockDim.x)) {
    const std::int32_t j = col_val[p];
    const double a = nz_val[p];
    if (fabs(a) <= zero_tol) {
      continue;
    }

    const double old_l = l_cur[j];
    const double old_u = u_cur[j];
    double term_min = 0.0;
    double term_max = 0.0;
    _term_interval_primal_propagation(a, old_l, old_u, &term_min, &term_max);

    const double rest_min = _row_residual_min_from_summary(
        row_min_fin_i, row_min_neg_inf_count_i, term_min);
    const double rest_max = _row_residual_max_from_summary(
        row_max_fin_i, row_max_pos_inf_count_i, term_max);

    double implied_l = -INFINITY;
    double implied_u = INFINITY;
    if (a > 0.0) {
      if (isfinite(lower_i) && isfinite(rest_max)) {
        implied_l = (lower_i - rest_max) / a;
      }
      if (isfinite(upper_i) && isfinite(rest_min)) {
        implied_u = (upper_i - rest_min) / a;
      }
    } else {
      if (isfinite(upper_i) && isfinite(rest_min)) {
        implied_l = (upper_i - rest_min) / a;
      }
      if (isfinite(lower_i) && isfinite(rest_max)) {
        implied_u = (lower_i - rest_max) / a;
      }
    }

    double new_l = old_l;
    double new_u = old_u;
    if (isfinite(implied_l)) {
      new_l = fmax(new_l, implied_l);
    }
    if (isfinite(implied_u)) {
      new_u = fmin(new_u, implied_u);
    }
    if (new_l > old_l) {
      atomic_max_double(&candidate_l[j], new_l);
    }
    if (new_u < old_u) {
      atomic_min_double(&candidate_u[j], new_u);
    }
  }
  (void)infeasible_flag;
}

__global__ void _kernel_finalize_primal_propagation(std::int32_t* flags,
                                                    double* finalized_l,
                                                    double* finalized_u,
                                                    std::uint8_t* lower_changed,
                                                    std::uint8_t* upper_changed,
                                                    std::uint8_t* fixed_mask,
                                                    double* fixed_val,
                                                    const double* candidate_l,
                                                    const double* candidate_u,
                                                    const double* old_l,
                                                    const double* old_u,
                                                    const double* col_max_abs,
                                                    double feas_tol,
                                                    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    const double lj = old_l[j];
    const double uj = old_u[j];
    double cand_l = candidate_l[j];
    double cand_u = candidate_u[j];
    const double vmax = col_max_abs[j];

    double new_l = lj;
    double new_u = uj;
    std::uint8_t lower_changed_j = std::uint8_t{0};
    std::uint8_t upper_changed_j = std::uint8_t{0};
    std::uint8_t fixed_j = std::uint8_t{0};
    double fixed_at = 0.0;

    double exact_l = lj;
    double exact_u = uj;
    if (isfinite(cand_l)) {
      exact_l = fmax(exact_l, cand_l);
    }
    if (isfinite(cand_u)) {
      exact_u = fmin(exact_u, cand_u);
    }
    if (exact_l > exact_u + feas_tol) {
      atomicMax(&flags[0], 1);
      return;
    }

    if (isfinite(cand_l) && cand_l > lj) {
      if (isfinite(uj)) {
        if (cand_l >= uj + feas_tol) {
          atomicMax(&flags[0], 1);
          return;
        }
        if (cand_l >= uj || (uj - cand_l) * vmax <= feas_tol) {
          fixed_j = std::uint8_t{1};
          fixed_at = uj;
          new_l = uj;
          new_u = uj;
        }
      }
    }

    if (fixed_j == std::uint8_t{0} && isfinite(cand_l) && cand_l > lj) {
      const bool finite_lb_tightening =
          !isfinite(lj) ||
          ((cand_l - lj > feas_tol * 1.0e4) &&
           (cand_l - lj > 1.0e-2 * fabs(lj)));
      if (finite_lb_tightening) {
        if (cand_l != nearbyint(cand_l)) {
          cand_l -= 0.5 * feas_tol * fabs(cand_l);
        }
        new_l = cand_l;
        lower_changed_j = std::uint8_t{1};
      }
    }

    if (fixed_j == std::uint8_t{0} && isfinite(cand_u) && cand_u < uj) {
      if (isfinite(new_l)) {
        if (cand_u <= new_l - feas_tol) {
          atomicMax(&flags[0], 1);
          return;
        }
        if (cand_u <= new_l || (cand_u - new_l) * vmax <= feas_tol) {
          fixed_j = std::uint8_t{1};
          fixed_at = new_l;
          new_u = new_l;
        }
      }
    }

    if (fixed_j == std::uint8_t{0} && isfinite(cand_u) && cand_u < uj) {
      const bool finite_ub_tightening =
          !isfinite(uj) ||
          ((uj - cand_u > feas_tol * 1.0e4) &&
           (uj - cand_u > 1.0e-2 * fabs(uj)));
      if (finite_ub_tightening) {
        if (cand_u != nearbyint(cand_u)) {
          cand_u += 0.5 * feas_tol * fabs(cand_u);
        }
        new_u = cand_u;
        upper_changed_j = std::uint8_t{1};
      }
    }

    if (fixed_j != std::uint8_t{0}) {
      lower_changed_j = std::uint8_t{0};
      upper_changed_j = std::uint8_t{0};
      atomicMax(&flags[2], 1);
    }
    if (lower_changed_j != std::uint8_t{0} || upper_changed_j != std::uint8_t{0}) {
      atomicMax(&flags[1], 1);
    }

    finalized_l[j] = new_l;
    finalized_u[j] = new_u;
    lower_changed[j] = lower_changed_j;
    upper_changed[j] = upper_changed_j;
    fixed_mask[j] = fixed_j;
    fixed_val[j] = fixed_at;
  }
}

__global__ void _kernel_apply_primal_bounds_and_fixed(std::uint8_t* keep_col,
                                                      double* new_l,
                                                      double* new_u,
                                                      const double* finalized_l,
                                                      const double* finalized_u,
                                                      const std::uint8_t* fixed_mask,
                                                      std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    new_l[j] = finalized_l[j];
    new_u[j] = finalized_u[j];
    if (fixed_mask[j] != std::uint8_t{0}) {
      keep_col[j] = std::uint8_t{0};
    }
  }
}

__global__ void _kernel_capture_primal_propagation_support_rows_by_col(
    std::int32_t* support_l_row,
    std::int32_t* support_u_row,
    const std::uint8_t* lower_changed,
    const std::uint8_t* upper_changed,
    const double* candidate_l,
    const double* candidate_u,
    const double* row_min_fin,
    const double* row_max_fin,
    const std::int32_t* row_min_neg_inf_count,
    const std::int32_t* row_max_pos_inf_count,
    const std::uint8_t* keep_row,
    const std::int32_t* row_nnz,
    const double* AL,
    const double* AU,
    const double* l_cur,
    const double* u_cur,
    const std::int32_t* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    double zero_tol,
    double tol,
    double support_tol,
    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j >= n) {
    return;
  }
  std::int32_t support_l = INT_MAX;
  std::int32_t support_u = INT_MAX;
  const bool need_l = lower_changed[j] != std::uint8_t{0};
  const bool need_u = upper_changed[j] != std::uint8_t{0};
  if (need_l || need_u) {
    const double old_l = l_cur[j];
    const double old_u = u_cur[j];
    const double cand_l = candidate_l[j];
    const double cand_u = candidate_u[j];
    for (std::int32_t p = at_row_ptr[j]; p < at_row_ptr[j + 1]; ++p) {
      const std::int32_t i = at_col_val[p];
      if (keep_row[i] == std::uint8_t{0} || row_nnz[i] <= 1) {
        continue;
      }
      const double a = at_nz_val[p];
      if (fabs(a) <= zero_tol) {
        continue;
      }
      double term_min = 0.0;
      double term_max = 0.0;
      _term_interval_primal_propagation(a, old_l, old_u, &term_min, &term_max);
      const double rest_min = _row_residual_min_from_summary(
          row_min_fin[i], row_min_neg_inf_count[i], term_min);
      const double rest_max = _row_residual_max_from_summary(
          row_max_fin[i], row_max_pos_inf_count[i], term_max);
      double implied_l = -INFINITY;
      double implied_u = INFINITY;
      if (a > 0.0) {
        if (isfinite(AL[i]) && isfinite(rest_max)) {
          implied_l = (AL[i] - rest_max) / a;
        }
        if (isfinite(AU[i]) && isfinite(rest_min)) {
          implied_u = (AU[i] - rest_min) / a;
        }
      } else {
        if (isfinite(AU[i]) && isfinite(rest_min)) {
          implied_l = (AU[i] - rest_min) / a;
        }
        if (isfinite(AL[i]) && isfinite(rest_max)) {
          implied_u = (AL[i] - rest_max) / a;
        }
      }
      if (need_l && cand_l > old_l + tol && isfinite(implied_l) &&
          fabs(implied_l - cand_l) <= support_tol) {
        support_l = min(support_l, i);
      }
      if (need_u && cand_u < old_u - tol && isfinite(implied_u) &&
          fabs(implied_u - cand_u) <= support_tol) {
        support_u = min(support_u, i);
      }
    }
  }
  support_l_row[j] = support_l;
  support_u_row[j] = support_u;
}

__global__ void _kernel_primal_fixed_row_shift(double* row_shift,
                                               const std::uint8_t* fixed_mask,
                                               const double* fixed_val,
                                               const std::uint8_t* keep_row,
                                               const std::int32_t* at_row_ptr,
                                               const std::int32_t* at_col_val,
                                               const double* at_nz_val,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    const double vj = fixed_val[j];
    const std::int32_t p_start = at_row_ptr[j];
    const std::int32_t p_stop = at_row_ptr[j + 1];
    for (std::int32_t p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] != std::uint8_t{0}) {
        atomicAdd(&row_shift[row], at_nz_val[p] * vj);
      }
    }
  }
}

__global__ void _kernel_apply_row_shift(double* AL,
                                        double* AU,
                                        const double* row_shift,
                                        std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    AL[i] -= row_shift[i];
    AU[i] -= row_shift[i];
  }
}

__global__ void _kernel_row_nnz_after_fixed(std::int32_t* row_nnz_after,
                                            const std::uint8_t* keep_row,
                                            const std::uint8_t* keep_col_after,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* col_val,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    std::int32_t count = 0;
    if (keep_row[i] != std::uint8_t{0}) {
      const std::int32_t p_start = row_ptr[i];
      const std::int32_t p_stop = row_ptr[i + 1];
      for (std::int32_t p = p_start; p < p_stop; ++p) {
        if (keep_col_after[col_val[p]] != std::uint8_t{0}) {
          ++count;
        }
      }
    }
    row_nnz_after[i] = count;
  }
}

__global__ void _kernel_remove_feasible_empty_rows_after_fixed(std::int32_t* flags,
                                                               std::uint8_t* keep_row,
                                                               const double* AL,
                                                               const double* AU,
                                                               const std::int32_t* row_nnz_after,
                                                               double feas_tol,
                                                               std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0} && row_nnz_after[i] == 0) {
    if (AL[i] <= feas_tol && AU[i] >= -feas_tol) {
      keep_row[i] = std::uint8_t{0};
      atomicMax(&flags[4], 1);
    } else {
      atomicMax(&flags[3], 1);
    }
  }
}

}  // namespace

void apply_rule_primal_propagation(PresolvePlanGpu& plan,
                                   const LPInfoGpu& lp,
                                   const PresolveStatsGpu& stats,
                                   const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }
  double* candidate_l = nullptr;
  double* candidate_u = nullptr;
  double* finalized_l = nullptr;
  double* finalized_u = nullptr;
  double* row_min_fin = nullptr;
  double* row_max_fin = nullptr;
  std::int32_t* row_min_neg_inf_count = nullptr;
  std::int32_t* row_max_pos_inf_count = nullptr;
  double* col_max_abs = nullptr;
  std::uint8_t* lower_changed = nullptr;
  std::uint8_t* upper_changed = nullptr;
  std::uint8_t* fixed_mask = nullptr;
  double* fixed_val = nullptr;
  std::int32_t* flags_device = nullptr;
  double* row_shift = nullptr;
  std::int32_t* row_nnz_after = nullptr;

  throw_if_cuda_error(cudaMalloc(&candidate_l, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc primal candidate_l");
  throw_if_cuda_error(cudaMalloc(&candidate_u, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc primal candidate_u");
  throw_if_cuda_error(cudaMalloc(&finalized_l, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc primal finalized_l");
  throw_if_cuda_error(cudaMalloc(&finalized_u, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc primal finalized_u");
  throw_if_cuda_error(cudaMalloc(&row_min_fin, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc primal row_min_fin");
  throw_if_cuda_error(cudaMalloc(&row_max_fin, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc primal row_max_fin");
  throw_if_cuda_error(cudaMalloc(&row_min_neg_inf_count, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc primal row_min_neg_inf_count");
  throw_if_cuda_error(cudaMalloc(&row_max_pos_inf_count, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc primal row_max_pos_inf_count");
  throw_if_cuda_error(cudaMalloc(&col_max_abs, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc primal col_max_abs");
  throw_if_cuda_error(cudaMalloc(&lower_changed, static_cast<std::size_t>(n)), "cudaMalloc primal lower_changed");
  throw_if_cuda_error(cudaMalloc(&upper_changed, static_cast<std::size_t>(n)), "cudaMalloc primal upper_changed");
  throw_if_cuda_error(cudaMalloc(&fixed_mask, static_cast<std::size_t>(n)), "cudaMalloc primal fixed_mask");
  throw_if_cuda_error(cudaMalloc(&fixed_val, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc primal fixed_val");
  throw_if_cuda_error(cudaMalloc(&flags_device, sizeof(std::int32_t) * 5), "cudaMalloc primal flags");
  throw_if_cuda_error(cudaMalloc(&row_shift, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc primal row_shift");
  throw_if_cuda_error(cudaMalloc(&row_nnz_after, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc primal row_nnz_after");

  throw_if_cuda_error(cudaMemcpy(candidate_l, plan.new_l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy primal candidate_l");
  throw_if_cuda_error(cudaMemcpy(candidate_u, plan.new_u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy primal candidate_u");
  throw_if_cuda_error(cudaMemcpy(finalized_l, plan.new_l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy primal finalized_l");
  throw_if_cuda_error(cudaMemcpy(finalized_u, plan.new_u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy primal finalized_u");
  throw_if_cuda_error(cudaMemset(flags_device, 0, sizeof(std::int32_t) * 5), "cudaMemset primal flags");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset primal row_shift");

  compute_row_activity_summary(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      lp.A,
      plan.new_l,
      plan.new_u,
      pparams.zero_tol);

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const bool use_block_per_row =
      static_cast<std::int64_t>(lp.A.nnz) > 32LL * static_cast<std::int64_t>(m);
  if (use_block_per_row) {
    _kernel_primal_propagation_candidates_block<<<m, GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_primal_propagation_candidates_block");
  } else {
    _kernel_primal_propagation_candidates<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_primal_propagation_candidates");
  }

  compute_col_max_abs(col_max_abs, lp.AT);

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_finalize_primal_propagation<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      flags_device,
      finalized_l,
      finalized_u,
      lower_changed,
      upper_changed,
      fixed_mask,
      fixed_val,
      candidate_l,
      candidate_u,
      plan.new_l,
      plan.new_u,
      col_max_abs,
      pparams.feasibility_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_finalize_primal_propagation");

  std::int32_t flags[5] = {0, 0, 0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, flags_device, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy primal flags after finalize");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
    cudaFree(candidate_l);
    cudaFree(candidate_u);
    cudaFree(finalized_l);
    cudaFree(finalized_u);
    cudaFree(row_min_fin);
    cudaFree(row_max_fin);
    cudaFree(row_min_neg_inf_count);
    cudaFree(row_max_pos_inf_count);
    cudaFree(col_max_abs);
    cudaFree(lower_changed);
    cudaFree(upper_changed);
    cudaFree(fixed_mask);
    cudaFree(fixed_val);
    cudaFree(flags_device);
    cudaFree(row_shift);
    cudaFree(row_nnz_after);
    return;
  }

  if (flags[1] != 0 || flags[2] != 0) {
    if (pparams.record_postsolve_tape) {
      std::int32_t* support_l_row = nullptr;
      std::int32_t* support_u_row = nullptr;
      throw_if_cuda_error(cudaMalloc(&support_l_row, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                          "cudaMalloc primal support_l_row");
      throw_if_cuda_error(cudaMalloc(&support_u_row, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                          "cudaMalloc primal support_u_row");
      const double support_tol = fmax(10.0 * pparams.bound_tol, 1.0e-10);
      _kernel_capture_primal_propagation_support_rows_by_col<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          support_l_row,
          support_u_row,
          lower_changed,
          upper_changed,
          finalized_l,
          finalized_u,
          row_min_fin,
          row_max_fin,
          row_min_neg_inf_count,
          row_max_pos_inf_count,
          plan.keep_row_mask,
          stats.row_nnz,
          plan.new_AL,
          plan.new_AU,
          plan.new_l,
          plan.new_u,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          pparams.zero_tol,
          pparams.bound_tol,
          support_tol,
          n);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_capture_primal_propagation_support_rows_by_col");
      throw_if_cuda_error(cudaDeviceSynchronize(), "capture primal support rows synchronize");

      if (!pparams.record_postsolve_tape_cpu) {
        append_primal_bound_change_tape_gpu(plan.tape_gpu,
                                            lower_changed,
                                            upper_changed,
                                            support_l_row,
                                            support_u_row,
                                            plan.new_l,
                                            plan.new_u,
                                            finalized_l,
                                            finalized_u,
                                            n);
      } else {
      std::vector<std::uint8_t> host_lower_changed(static_cast<std::size_t>(n));
      std::vector<std::uint8_t> host_upper_changed(static_cast<std::size_t>(n));
      std::vector<std::int32_t> host_support_l(static_cast<std::size_t>(n));
      std::vector<std::int32_t> host_support_u(static_cast<std::size_t>(n));
      std::vector<double> host_old_l(static_cast<std::size_t>(n));
      std::vector<double> host_old_u(static_cast<std::size_t>(n));
      std::vector<double> host_new_l(static_cast<std::size_t>(n));
      std::vector<double> host_new_u(static_cast<std::size_t>(n));
      throw_if_cuda_error(cudaMemcpy(host_lower_changed.data(), lower_changed, static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal lower_changed for tape");
      throw_if_cuda_error(cudaMemcpy(host_upper_changed.data(), upper_changed, static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal upper_changed for tape");
      throw_if_cuda_error(cudaMemcpy(host_support_l.data(), support_l_row,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal support_l for tape");
      throw_if_cuda_error(cudaMemcpy(host_support_u.data(), support_u_row,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal support_u for tape");
      throw_if_cuda_error(cudaMemcpy(host_old_l.data(), plan.new_l, sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal old_l for tape");
      throw_if_cuda_error(cudaMemcpy(host_old_u.data(), plan.new_u, sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal old_u for tape");
      throw_if_cuda_error(cudaMemcpy(host_new_l.data(), finalized_l, sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal new_l for tape");
      throw_if_cuda_error(cudaMemcpy(host_new_u.data(), finalized_u, sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy primal new_u for tape");
      for (std::int32_t col = 0; col < n; ++col) {
        const auto idx = static_cast<std::size_t>(col);
        if (host_lower_changed[idx] == std::uint8_t{0} && host_upper_changed[idx] == std::uint8_t{0}) {
          continue;
        }
        double cur_l = host_old_l[idx];
        const double cur_u = host_old_u[idx];
        if (host_lower_changed[idx] != std::uint8_t{0}) {
          const double new_l = host_new_l[idx];
          const std::int32_t row = host_support_l[idx];
          if (row != INT_MAX) {
            append_postsolve_record(plan.tape,
                                    PostsolveReductionType::BoundChangeTheRow,
                                    {col, row},
                                    {cur_l, cur_u, new_l, cur_u},
                                    PostsolveDualMode::Minimal);
          } else {
            append_postsolve_record(plan.tape,
                                    PostsolveReductionType::BoundChangeNoRow,
                                    {col},
                                    {cur_l, cur_u, new_l, cur_u},
                                    PostsolveDualMode::Minimal);
          }
          cur_l = new_l;
        }
        if (host_upper_changed[idx] != std::uint8_t{0}) {
          const double new_u = host_new_u[idx];
          const std::int32_t row = host_support_u[idx];
          if (row != INT_MAX) {
            append_postsolve_record(plan.tape,
                                    PostsolveReductionType::BoundChangeTheRow,
                                    {col, row},
                                    {cur_l, cur_u, cur_l, new_u},
                                    PostsolveDualMode::Minimal);
          } else {
            append_postsolve_record(plan.tape,
                                    PostsolveReductionType::BoundChangeNoRow,
                                    {col},
                                    {cur_l, cur_u, cur_l, new_u},
                                    PostsolveDualMode::Minimal);
          }
        }
      }
      }
      cudaFree(support_l_row);
      cudaFree(support_u_row);
    }
    _kernel_apply_primal_bounds_and_fixed<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.new_l,
        plan.new_u,
        finalized_l,
        finalized_u,
        fixed_mask,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_primal_bounds_and_fixed");

    if (flags[2] != 0) {
      _kernel_primal_fixed_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          row_shift,
          fixed_mask,
          fixed_val,
          plan.keep_row_mask,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_primal_fixed_row_shift");
      _kernel_apply_row_shift<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
          plan.new_AL,
          plan.new_AU,
          row_shift,
          m);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_row_shift");
      _kernel_row_nnz_after_fixed<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
          row_nnz_after,
          plan.keep_row_mask,
          plan.keep_col_mask,
          lp.A.rowPtr,
          lp.A.colVal,
          m);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_row_nnz_after_fixed");
      _kernel_remove_feasible_empty_rows_after_fixed<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
          flags_device,
          plan.keep_row_mask,
          plan.new_AL,
          plan.new_AU,
          row_nnz_after,
          pparams.feasibility_tol,
          m);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_remove_feasible_empty_rows_after_fixed");
    }

    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_primal_propagation synchronize");
    throw_if_cuda_error(cudaMemcpy(flags, flags_device, sizeof(flags), cudaMemcpyDeviceToHost),
                        "cudaMemcpy primal flags after apply");
    if (flags[3] != 0) {
      plan.has_infeasible = true;
    } else {
      const double obj_delta = flags[2] != 0
          ? append_primal_fixed_col_tape_compacted_from_device(
                pparams.record_postsolve_tape ? &plan.tape : nullptr,
                fixed_mask,
                fixed_val,
                plan.keep_row_mask,
                plan.new_c,
                lp.AT,
                n,
                "primal compacted fixed-col tape/objective")
          : 0.0;
      plan.obj_constant_delta += obj_delta;
      plan.has_row_action = plan.has_row_action || flags[2] != 0;
      plan.has_col_action = plan.has_col_action || flags[2] != 0;
      plan.has_change = true;
    }
  }

  cudaFree(candidate_l);
  cudaFree(candidate_u);
  cudaFree(finalized_l);
  cudaFree(finalized_u);
  cudaFree(row_min_fin);
  cudaFree(row_max_fin);
  cudaFree(row_min_neg_inf_count);
  cudaFree(row_max_pos_inf_count);
  cudaFree(col_max_abs);
  cudaFree(lower_changed);
  cudaFree(upper_changed);
  cudaFree(fixed_mask);
  cudaFree(fixed_val);
  cudaFree(flags_device);
  cudaFree(row_shift);
  cudaFree(row_nnz_after);
}

}  // namespace gpu_presolver::presolve
