#include "gpu_presolver/presolve/gpu_presolve.hpp"

#include "gpu_presolver/presolve/rules/rule_activity_checks.hpp"
#include "gpu_presolver/presolve/rules/rule_close_bounds.hpp"
#include "gpu_presolver/presolve/rules/rule_doubleton_eq.hpp"
#include "gpu_presolver/presolve/rules/rule_dual_fix.hpp"
#include "gpu_presolver/presolve/rules/rule_empty_cols.hpp"
#include "gpu_presolver/presolve/rules/rule_empty_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_cols.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_primal_propagation.hpp"
#include "gpu_presolver/presolve/rules/rule_redundant_bounds.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_cols.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_structural_l1_substitution.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

// The per-row selection sort is quadratic on dense transpose rows. Julia's
// presolve path does not spend core time here, so keep transpose construction
// unsorted until a parallel sort is available.
constexpr std::int32_t CSR_SORT_TRANSPOSE_NNZ_LIMIT = 0;
constexpr std::int32_t SPARSE_FIXED_BOUND_RECORD_THRESHOLD = 65536;
constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(
      std::string(context) + ": " + cudaGetErrorString(status));
}

void throw_if_cusparse_error(cusparseStatus_t status, const char* context) {
  if (status == CUSPARSE_STATUS_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": cusparse status " +
                           std::to_string(static_cast<int>(status)));
}

template <class T>
cudaError_t compact_cuda_malloc(T** ptr, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(ptr), bytes, nullptr);
}

cudaError_t compact_cuda_free(void* ptr) {
  return ptr == nullptr ? cudaSuccess : cudaFreeAsync(ptr, nullptr);
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

void free_device_csr(DeviceCsrMatrix& matrix) {
  cudaFree(matrix.rowPtr);
  cudaFree(matrix.colVal);
  cudaFree(matrix.nzVal);
  matrix = DeviceCsrMatrix{};
}

void free_postsolve_tape_gpu(PostsolveTapeGpu& tape) {
  tape.reset();
}

template <typename T>
T* copy_host_vector_to_device(const std::vector<T>& values, const char* context) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  throw_if_cuda_error(cudaMalloc(&device, sizeof(T) * values.size()), context);
  throw_if_cuda_error(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
                      context);
  return device;
}

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  auto cleanup = make_scope_exit([&]() { compact_cuda_free(temp_storage); });
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
      context);
  throw_if_cuda_error(cudaMallocAsync(&temp_storage, temp_bytes, nullptr), context);
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
      context);
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void append_rule_name(std::string& rules, const char* name) {
  if (!rules.empty()) {
    rules += ",";
  }
  rules += name;
}

__global__ void _kernel_smoke_count(std::int32_t n, std::int32_t* out) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    atomicAdd(out, 1);
  }
}

__global__ void _kernel_fill_u8(std::uint8_t* data, std::uint8_t value, std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    data[j] = value;
  }
}

__global__ void _kernel_globalize_postsolve_tape_gpu(std::int32_t* indices,
                                                     const std::int32_t* types,
                                                     const std::int32_t* index_starts,
                                                     const std::int32_t* row_red2org,
                                                     const std::int32_t* col_red2org,
                                                     std::int32_t record_count) {
  const std::int32_t k = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= record_count) {
    return;
  }
  const std::int32_t idx0 = index_starts[k];
  const std::int32_t type = types[k];
  switch (static_cast<PostsolveReductionType>(type)) {
    case PostsolveReductionType::SubCol: {
      indices[idx0] = col_red2org[indices[idx0]];
      indices[idx0 + 1] = row_red2org[indices[idx0 + 1]];
      const std::int32_t support_count = indices[idx0 + 2];
      for (std::int32_t t = 0; t < support_count; ++t) {
        indices[idx0 + 3 + t] = col_red2org[indices[idx0 + 3 + t]];
      }
      break;
    }
    case PostsolveReductionType::BoundChangeNoRow:
      indices[idx0] = col_red2org[indices[idx0]];
      break;
    case PostsolveReductionType::BoundChangeTheRow:
      indices[idx0] = col_red2org[indices[idx0]];
      indices[idx0 + 1] = row_red2org[indices[idx0 + 1]];
      break;
    default:
      break;
  }
}

__global__ void _kernel_compute_live_row_stats(std::int32_t* row_nnz,
                                               std::uint8_t* empty_row_mask,
                                               std::uint8_t* singleton_row_mask,
                                               std::int32_t* singleton_row_col,
                                               double* singleton_row_val,
                                               const std::uint8_t* keep_row,
                                               const std::uint8_t* keep_col,
                                               const std::int32_t* row_ptr,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               double zero_tol,
                                               std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    std::int32_t count = 0;
    std::int32_t support_col = -1;
    double support_val = 0.0;
    if (keep_row[i] != std::uint8_t{0}) {
      for (std::int32_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
        const std::int32_t col = col_val[p];
        const double a = nz_val[p];
        if (keep_col[col] != std::uint8_t{0} && fabs(a) > zero_tol) {
          ++count;
          support_col = col;
          support_val = a;
        }
      }
    }
    row_nnz[i] = count;
    empty_row_mask[i] = count == 0 ? std::uint8_t{1} : std::uint8_t{0};
    singleton_row_mask[i] = count == 1 ? std::uint8_t{1} : std::uint8_t{0};
    singleton_row_col[i] = count == 1 ? support_col : -1;
    singleton_row_val[i] = count == 1 ? support_val : 0.0;
  }
}

__global__ void _kernel_compute_live_col_stats(std::int32_t* col_nnz,
                                               std::uint8_t* empty_col_mask,
                                               std::uint8_t* singleton_col_mask,
                                               std::int32_t* singleton_col_row,
                                               double* singleton_col_val,
                                               const std::uint8_t* keep_row,
                                               const std::uint8_t* keep_col,
                                               const std::int32_t* row_ptr,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               double zero_tol,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    std::int32_t count = 0;
    std::int32_t support_row = -1;
    double support_val = 0.0;
    if (keep_col[j] != std::uint8_t{0}) {
      for (std::int32_t p = row_ptr[j]; p < row_ptr[j + 1]; ++p) {
        const std::int32_t row = col_val[p];
        const double a = nz_val[p];
        if (keep_row[row] != std::uint8_t{0} && fabs(a) > zero_tol) {
          ++count;
          support_row = row;
          support_val = a;
        }
      }
    }
    col_nnz[j] = count;
    empty_col_mask[j] = count == 0 ? std::uint8_t{1} : std::uint8_t{0};
    singleton_col_mask[j] = count == 1 ? std::uint8_t{1} : std::uint8_t{0};
    singleton_col_row[j] = count == 1 ? support_row : -1;
    singleton_col_val[j] = count == 1 ? support_val : 0.0;
  }
}

__global__ void _kernel_compute_csr_row_stats(std::int32_t* row_nnz,
                                              std::uint8_t* empty_row_mask,
                                              std::uint8_t* singleton_row_mask,
                                              std::int32_t* singleton_row_col,
                                              double* singleton_row_val,
                                              const std::int32_t* row_ptr,
                                              const std::int32_t* col_val,
                                              const double* nz_val,
                                              std::int32_t rows) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < rows) {
    const std::int32_t first = row_ptr[i];
    const std::int32_t count = row_ptr[i + 1] - first;
    row_nnz[i] = count;
    empty_row_mask[i] = count == 0 ? std::uint8_t{1} : std::uint8_t{0};
    singleton_row_mask[i] = count == 1 ? std::uint8_t{1} : std::uint8_t{0};
    singleton_row_col[i] = count == 1 ? col_val[first] : -1;
    singleton_row_val[i] = count == 1 ? nz_val[first] : 0.0;
  }
}

__global__ void _kernel_count_keep_mask(const std::uint8_t* keep_mask,
                                        std::int32_t n,
                                        std::int32_t* out) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && keep_mask[j] != std::uint8_t{0}) {
    atomicAdd(out, 1);
  }
}

__global__ void _kernel_count_live_nnz(const std::uint8_t* keep_row,
                                       const std::uint8_t* keep_col,
                                       const std::int32_t* row_ptr,
                                       const std::int32_t* col_val,
                                       std::int32_t m,
                                       std::int32_t* out) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0}) {
    std::int32_t count = 0;
    for (std::int32_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      if (keep_col[col_val[p]] != std::uint8_t{0}) {
        ++count;
      }
    }
    atomicAdd(out, count);
  }
}

__global__ void _kernel_mask_to_prefix_i32(std::int32_t* out,
                                           const std::uint8_t* mask,
                                           std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    out[j] = mask[j] != std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_old_to_new_from_prefix(std::int32_t* old_to_new,
                                               const std::uint8_t* mask,
                                               const std::int32_t* prefix,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    old_to_new[j] = mask[j] != std::uint8_t{0} ? prefix[j] - 1 : -1;
  }
}

__global__ void _kernel_count_compacted_rows(std::int32_t* row_counts_new,
                                             const std::int32_t* row_old_to_new,
                                             const std::int32_t* col_old_to_new,
                                             const std::int32_t* row_ptr,
                                             const std::int32_t* col_val,
                                             std::int32_t max_row_nnz,
                                             std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m_old) {
    const std::int32_t row_new = row_old_to_new[row];
    if (row_new < 0) {
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > max_row_nnz) {
      return;
    }
    std::int32_t count = 0;
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      if (col_old_to_new[col_val[p]] >= 0) {
        ++count;
      }
    }
    row_counts_new[row_new] = count;
  }
}

__global__ void _kernel_row_ptr_from_prefix(std::int32_t* row_ptr,
                                            const std::int32_t* prefix,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) {
    row_ptr[0] = 0;
  }
  if (i < m) {
    row_ptr[i + 1] = prefix[i];
  }
}

__global__ void _kernel_inclusive_scan_i32_serial(std::int32_t* values, std::int32_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::int32_t i = 0; i < n; ++i) {
      running += values[i];
      values[i] = running;
    }
  }
}

__global__ void _kernel_copy_compacted_rows(std::int32_t* col_val_new,
                                            double* nz_val_new,
                                            const std::int32_t* row_ptr_new,
                                            const std::int32_t* row_old_to_new,
                                            const std::int32_t* col_old_to_new,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            std::int32_t max_row_nnz,
                                            std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m_old) {
    const std::int32_t row_new = row_old_to_new[row];
    if (row_new < 0) {
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > max_row_nnz) {
      return;
    }
    std::int32_t write = row_ptr_new[row_new];
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col_new = col_old_to_new[col_val[p]];
      if (col_new >= 0) {
        col_val_new[write] = col_new;
        nz_val_new[write] = nz_val[p];
        ++write;
      }
    }
  }
}

__global__ void _kernel_mark_dense_compact_rows(
    std::int32_t* selected_scan,
    const std::int32_t* row_old_to_new,
    const std::int32_t* row_ptr,
    std::int32_t dense_threshold,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m_old) {
    selected_scan[row] =
        row_old_to_new[row] >= 0 && row_ptr[row + 1] - row_ptr[row] > dense_threshold
            ? 1
            : 0;
  }
}

__global__ void _kernel_pack_dense_compact_rows(
    std::int32_t* dense_rows,
    const std::int32_t* selected_scan,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m_old) {
    return;
  }
  const std::int32_t prev_selected = row == 0 ? 0 : selected_scan[row - 1];
  const std::int32_t out = selected_scan[row] - 1;
  if (selected_scan[row] != prev_selected && out >= 0) {
    dense_rows[out] = row;
  }
}

__global__ void _kernel_count_compacted_dense_rows_block(
    std::int32_t* row_counts_new,
    const std::int32_t* dense_rows,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    std::int32_t dense_count) {
  const std::int32_t dense_idx = static_cast<std::int32_t>(blockIdx.x);
  if (dense_idx >= dense_count) {
    return;
  }
  const std::int32_t row = dense_rows[dense_idx];
  const std::int32_t row_new = row_old_to_new[row];
  std::int32_t local_count = 0;
  for (std::int32_t p = row_ptr[row] + static_cast<std::int32_t>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    local_count += col_old_to_new[col_val[p]] >= 0 ? 1 : 0;
  }
  using BlockReduce = cub::BlockReduce<std::int32_t, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t count = BlockReduce(reduce_storage).Sum(local_count);
  if (threadIdx.x == 0) {
    row_counts_new[row_new] = count;
  }
}

__global__ void _kernel_copy_compacted_dense_rows_block(
    std::int32_t* col_val_new,
    double* nz_val_new,
    const std::int32_t* row_ptr_new,
    const std::int32_t* dense_rows,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t dense_count) {
  const std::int32_t dense_idx = static_cast<std::int32_t>(blockIdx.x);
  if (dense_idx >= dense_count) {
    return;
  }
  const std::int32_t row = dense_rows[dense_idx];
  const std::int32_t row_new = row_old_to_new[row];
  using BlockScan = cub::BlockScan<std::int32_t, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::int32_t row_write_offset;
  if (threadIdx.x == 0) {
    row_write_offset = 0;
  }
  __syncthreads();
  const std::int32_t row_start = row_ptr[row];
  const std::int32_t row_stop = row_ptr[row + 1];
  for (std::int32_t chunk = row_start; chunk < row_stop;
       chunk += GPU_PRESOLVE_THREADS) {
    const std::int32_t p = chunk + static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t col_new = p < row_stop ? col_old_to_new[col_val[p]] : -1;
    const std::int32_t keep = col_new >= 0 ? 1 : 0;
    std::int32_t prefix = 0;
    std::int32_t chunk_count = 0;
    BlockScan(scan_storage).ExclusiveSum(keep, prefix, chunk_count);
    if (keep != 0) {
      const std::int32_t write = row_ptr_new[row_new] + row_write_offset + prefix;
      col_val_new[write] = col_new;
      nz_val_new[write] = nz_val[p];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      row_write_offset += chunk_count;
    }
    __syncthreads();
  }
}

__global__ void _kernel_count_compacted_rows_block(
    std::int32_t* row_counts_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= m_old) {
    return;
  }
  const std::int32_t row_new = row_old_to_new[row];
  if (row_new < 0) {
    return;
  }
  std::int32_t local_count = 0;
  for (std::int32_t p = row_ptr[row] + static_cast<std::int32_t>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    local_count += col_old_to_new[col_val[p]] >= 0 ? 1 : 0;
  }
  using BlockReduce = cub::BlockReduce<std::int32_t, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t count = BlockReduce(reduce_storage).Sum(local_count);
  if (threadIdx.x == 0) {
    row_counts_new[row_new] = count;
  }
}

__global__ void _kernel_copy_compacted_rows_block(
    std::int32_t* col_val_new,
    double* nz_val_new,
    const std::int32_t* row_ptr_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= m_old) {
    return;
  }
  const std::int32_t row_new = row_old_to_new[row];
  if (row_new < 0) {
    return;
  }
  using BlockScan = cub::BlockScan<std::int32_t, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::int32_t row_write_offset;
  if (threadIdx.x == 0) {
    row_write_offset = 0;
  }
  __syncthreads();
  const std::int32_t row_start = row_ptr[row];
  const std::int32_t row_stop = row_ptr[row + 1];
  for (std::int32_t chunk = row_start; chunk < row_stop;
       chunk += GPU_PRESOLVE_THREADS) {
    const std::int32_t p = chunk + static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t col_new =
        p < row_stop ? col_old_to_new[col_val[p]] : -1;
    const std::int32_t keep = col_new >= 0 ? 1 : 0;
    std::int32_t prefix = 0;
    std::int32_t chunk_count = 0;
    BlockScan(scan_storage).ExclusiveSum(keep, prefix, chunk_count);
    if (keep != 0) {
      const std::int32_t write =
          row_ptr_new[row_new] + row_write_offset + prefix;
      col_val_new[write] = col_new;
      nz_val_new[write] = nz_val[p];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      row_write_offset += chunk_count;
    }
    __syncthreads();
  }
}

__global__ void _kernel_gather_by_old_to_new(double* dst,
                                             const double* src,
                                             const std::int32_t* old_to_new,
                                             std::int32_t n_old) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n_old) {
    const std::int32_t mapped = old_to_new[j];
    if (mapped >= 0) {
      dst[mapped] = src[j];
    }
  }
}

__global__ void _kernel_scatter_transpose_atomic(std::int32_t* at_col_val,
                                                 double* at_nz_val,
                                                 std::int32_t* write_ptr,
                                                 const std::int32_t* row_ptr,
                                                 const std::int32_t* col_val,
                                                 const double* nz_val,
                                                 std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col = col_val[p];
      const std::int32_t dst = atomicAdd(&write_ptr[col], 1);
      at_col_val[dst] = row;
      at_nz_val[dst] = nz_val[p];
    }
  }
}

__global__ void _kernel_sort_csr_rows_by_col(std::int32_t* col_val,
                                             double* nz_val,
                                             const std::int32_t* row_ptr,
                                             std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    const std::int32_t first = row_ptr[row];
    const std::int32_t last = row_ptr[row + 1];
    for (std::int32_t i = first; i + 1 < last; ++i) {
      std::int32_t min_pos = i;
      std::int32_t min_col = col_val[i];
      for (std::int32_t j = i + 1; j < last; ++j) {
        const std::int32_t candidate = col_val[j];
        if (candidate < min_col) {
          min_col = candidate;
          min_pos = j;
        }
      }
      if (min_pos != i) {
        const std::int32_t tmp_col = col_val[i];
        col_val[i] = col_val[min_pos];
        col_val[min_pos] = tmp_col;
        const double tmp_val = nz_val[i];
        nz_val[i] = nz_val[min_pos];
        nz_val[min_pos] = tmp_val;
      }
    }
  }
}

__global__ void _kernel_count_transpose_rows(std::int32_t* counts,
                                             const std::int32_t* col_val,
                                             std::int32_t nnz) {
  const std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < nnz) {
    atomicAdd(&counts[col_val[p]], 1);
  }
}

void free_stats(PresolveStatsGpu& stats);
void free_plan(PresolvePlanGpu& plan);

void allocate_stats(PresolveStatsGpu& stats,
                    std::int32_t m,
                    std::int32_t n,
                    bool need_row_stats,
                    bool need_col_stats) {
  try {
    if (need_row_stats) {
      throw_if_cuda_error(compact_cuda_malloc(&stats.row_nnz, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMallocAsync stats row_nnz");
      throw_if_cuda_error(compact_cuda_malloc(&stats.empty_row_mask, static_cast<std::size_t>(m)), "cudaMallocAsync stats empty_row_mask");
      throw_if_cuda_error(compact_cuda_malloc(&stats.singleton_row_mask, static_cast<std::size_t>(m)), "cudaMallocAsync stats singleton_row_mask");
      throw_if_cuda_error(compact_cuda_malloc(&stats.singleton_row_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMallocAsync stats singleton_row_col");
      throw_if_cuda_error(compact_cuda_malloc(&stats.singleton_row_val, sizeof(double) * static_cast<std::size_t>(m)), "cudaMallocAsync stats singleton_row_val");
    }
    if (need_col_stats) {
      throw_if_cuda_error(compact_cuda_malloc(&stats.col_nnz, sizeof(std::int32_t) * static_cast<std::size_t>(n)), "cudaMallocAsync stats col_nnz");
      throw_if_cuda_error(compact_cuda_malloc(&stats.empty_col_mask, static_cast<std::size_t>(n)), "cudaMallocAsync stats empty_col_mask");
      throw_if_cuda_error(compact_cuda_malloc(&stats.singleton_col_mask, static_cast<std::size_t>(n)), "cudaMallocAsync stats singleton_col_mask");
      throw_if_cuda_error(compact_cuda_malloc(&stats.singleton_col_row, sizeof(std::int32_t) * static_cast<std::size_t>(n)), "cudaMallocAsync stats singleton_col_row");
      throw_if_cuda_error(compact_cuda_malloc(&stats.singleton_col_val, sizeof(double) * static_cast<std::size_t>(n)), "cudaMallocAsync stats singleton_col_val");
    }
  } catch (...) {
    free_stats(stats);
    throw;
  }
}

void free_stats(PresolveStatsGpu& stats) {
  compact_cuda_free(stats.row_nnz);
  compact_cuda_free(stats.empty_row_mask);
  compact_cuda_free(stats.singleton_row_mask);
  compact_cuda_free(stats.singleton_row_col);
  compact_cuda_free(stats.singleton_row_val);
  compact_cuda_free(stats.col_nnz);
  compact_cuda_free(stats.empty_col_mask);
  compact_cuda_free(stats.singleton_col_mask);
  compact_cuda_free(stats.singleton_col_row);
  compact_cuda_free(stats.singleton_col_val);
  stats = PresolveStatsGpu{};
}

void recompute_stats(PresolveStatsGpu& stats,
                     const PresolvePlanGpu& plan,
                     const LPInfoGpu& lp,
                     double zero_tol) {
  constexpr int threads = 256;
  const int row_blocks = (lp.A.rows + threads - 1) / threads;
  const int col_blocks = (lp.A.cols + threads - 1) / threads;
  if (lp.A.rows > 0) {
    _kernel_compute_live_row_stats<<<row_blocks, threads>>>(
        stats.row_nnz,
        stats.empty_row_mask,
        stats.singleton_row_mask,
        stats.singleton_row_col,
        stats.singleton_row_val,
        plan.keep_row_mask,
        plan.keep_col_mask,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        zero_tol,
        lp.A.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_live_row_stats");
  }
  if (lp.A.cols > 0) {
    _kernel_compute_live_col_stats<<<col_blocks, threads>>>(
        stats.col_nnz,
        stats.empty_col_mask,
        stats.singleton_col_mask,
        stats.singleton_col_row,
        stats.singleton_col_val,
        plan.keep_row_mask,
        plan.keep_col_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        zero_tol,
        lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_live_col_stats");
  }
}

void recompute_row_stats_from_csr(PresolveStatsGpu& stats, const LPInfoGpu& lp) {
  if (lp.A.rows <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int row_blocks = (lp.A.rows + threads - 1) / threads;
  _kernel_compute_csr_row_stats<<<row_blocks, threads>>>(
      stats.row_nnz,
      stats.empty_row_mask,
      stats.singleton_row_mask,
      stats.singleton_row_col,
      stats.singleton_row_val,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      lp.A.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_csr_row_stats row");
}

void recompute_col_stats_from_csr(PresolveStatsGpu& stats, const LPInfoGpu& lp) {
  if (lp.A.cols <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int col_blocks = (lp.A.cols + threads - 1) / threads;
  _kernel_compute_csr_row_stats<<<col_blocks, threads>>>(
      stats.col_nnz,
      stats.empty_col_mask,
      stats.singleton_col_mask,
      stats.singleton_col_row,
      stats.singleton_col_val,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_csr_row_stats col");
}

void allocate_plan(PresolvePlanGpu& plan, const LPInfoGpu& lp) {
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  const bool profile_alloc = env_enabled("GPUPRESOLVER_PRESOLVE_ALLOC_PROFILE");
  const auto alloc_start = std::chrono::steady_clock::now();
  auto after_mask_alloc = alloc_start;
  auto after_vector_alloc = alloc_start;
  auto after_mask_fill = alloc_start;
  auto after_row_copy = alloc_start;
  auto after_col_copy = alloc_start;
  try {
    throw_if_cuda_error(compact_cuda_malloc(&plan.keep_row_mask, static_cast<std::size_t>(m)), "cudaMallocAsync plan keep_row_mask");
    throw_if_cuda_error(compact_cuda_malloc(&plan.keep_col_mask, static_cast<std::size_t>(n)), "cudaMallocAsync plan keep_col_mask");
    if (profile_alloc) {
      after_mask_alloc = std::chrono::steady_clock::now();
    }
    throw_if_cuda_error(compact_cuda_malloc(&plan.new_c, sizeof(double) * static_cast<std::size_t>(n)), "cudaMallocAsync plan new_c");
    throw_if_cuda_error(compact_cuda_malloc(&plan.new_l, sizeof(double) * static_cast<std::size_t>(n)), "cudaMallocAsync plan new_l");
    throw_if_cuda_error(compact_cuda_malloc(&plan.new_u, sizeof(double) * static_cast<std::size_t>(n)), "cudaMallocAsync plan new_u");
    throw_if_cuda_error(compact_cuda_malloc(&plan.new_AL, sizeof(double) * static_cast<std::size_t>(m)), "cudaMallocAsync plan new_AL");
    throw_if_cuda_error(compact_cuda_malloc(&plan.new_AU, sizeof(double) * static_cast<std::size_t>(m)), "cudaMallocAsync plan new_AU");
    if (profile_alloc) {
      after_vector_alloc = std::chrono::steady_clock::now();
    }
    constexpr int threads = 256;
    if (m > 0) {
      const int blocks_m = (m + threads - 1) / threads;
      _kernel_fill_u8<<<blocks_m, threads>>>(plan.keep_row_mask, std::uint8_t{1}, m);
      throw_if_cuda_error(cudaGetLastError(), "fill keep_row_mask");
    }
    if (n > 0) {
      const int blocks_n = (n + threads - 1) / threads;
      _kernel_fill_u8<<<blocks_n, threads>>>(plan.keep_col_mask, std::uint8_t{1}, n);
      throw_if_cuda_error(cudaGetLastError(), "fill keep_col_mask");
    }
    if (profile_alloc) {
      throw_if_cuda_error(cudaDeviceSynchronize(), "profile plan mask fill synchronize");
      after_mask_fill = std::chrono::steady_clock::now();
    }
    if (m > 0) {
      throw_if_cuda_error(cudaMemcpy(plan.new_AL, lp.AL, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy plan AL");
      throw_if_cuda_error(cudaMemcpy(plan.new_AU, lp.AU, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy plan AU");
    }
    if (profile_alloc) {
      after_row_copy = std::chrono::steady_clock::now();
    }
    if (n > 0) {
      throw_if_cuda_error(cudaMemcpy(plan.new_c, lp.c, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy plan c");
      throw_if_cuda_error(cudaMemcpy(plan.new_l, lp.l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy plan l");
      throw_if_cuda_error(cudaMemcpy(plan.new_u, lp.u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                          "cudaMemcpy plan u");
    }
    if (profile_alloc) {
      after_col_copy = std::chrono::steady_clock::now();
      const std::chrono::duration<double> mask_alloc_elapsed = after_mask_alloc - alloc_start;
      const std::chrono::duration<double> vector_alloc_elapsed = after_vector_alloc - after_mask_alloc;
      const std::chrono::duration<double> mask_fill_elapsed = after_mask_fill - after_vector_alloc;
      const std::chrono::duration<double> row_copy_elapsed = after_row_copy - after_mask_fill;
      const std::chrono::duration<double> col_copy_elapsed = after_col_copy - after_row_copy;
      const std::chrono::duration<double> total_elapsed = after_col_copy - alloc_start;
      std::cerr << ">>> [GPU Presolve C++ alloc] rows=" << m << " cols=" << n
                << " total=" << total_elapsed.count() << "s"
                << " mask_malloc=" << mask_alloc_elapsed.count() << "s"
                << " vector_malloc=" << vector_alloc_elapsed.count() << "s"
                << " mask_fill=" << mask_fill_elapsed.count() << "s"
                << " row_copy=" << row_copy_elapsed.count() << "s"
                << " col_copy=" << col_copy_elapsed.count() << "s\n";
    }
  } catch (...) {
    free_plan(plan);
    throw;
  }
}

void free_plan(PresolvePlanGpu& plan) {
  const bool profile_alloc = env_enabled("GPUPRESOLVER_PRESOLVE_ALLOC_PROFILE");
  const auto free_start = std::chrono::steady_clock::now();
  compact_cuda_free(plan.keep_row_mask);
  compact_cuda_free(plan.keep_col_mask);
  compact_cuda_free(plan.new_c);
  compact_cuda_free(plan.new_l);
  compact_cuda_free(plan.new_u);
  compact_cuda_free(plan.new_AL);
  compact_cuda_free(plan.new_AU);
  if (plan.has_new_A) {
    free_device_csr(plan.new_A);
  }
  free_postsolve_tape_gpu(plan.tape_gpu);
  if (profile_alloc) {
    throw_if_cuda_error(cudaDeviceSynchronize(), "profile plan free synchronize");
    const std::chrono::duration<double> free_elapsed =
        std::chrono::steady_clock::now() - free_start;
    std::cerr << ">>> [GPU Presolve C++ alloc] free=" << free_elapsed.count() << "s\n";
  }
  plan = PresolvePlanGpu{};
}

std::int32_t build_prefix_from_mask(std::int32_t* prefix,
                                    const std::uint8_t* mask,
                                    std::int32_t n) {
  if (n == 0) {
    return 0;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_mask_to_prefix_i32<<<blocks, threads>>>(prefix, mask, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mask_to_prefix_i32");
  inclusive_scan_i32(prefix, n, "cub inclusive scan mask");
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, prefix + n - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy prefix count");
  return count;
}

std::int32_t* build_old_to_new(const std::uint8_t* mask, std::int32_t n, std::int32_t* count_out) {
  if (n == 0) {
    *count_out = 0;
    return nullptr;
  }
  std::int32_t* prefix = nullptr;
  std::int32_t* old_to_new = nullptr;
  throw_if_cuda_error(cudaMalloc(&prefix, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc prefix");
  throw_if_cuda_error(cudaMalloc(&old_to_new, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc old_to_new");
  const std::int32_t count = build_prefix_from_mask(prefix, mask, n);
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_old_to_new_from_prefix<<<blocks, threads>>>(old_to_new, mask, prefix, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_old_to_new_from_prefix");
  cudaFree(prefix);
  *count_out = count;
  return old_to_new;
}

double* compact_vector(const double* src,
                       const std::int32_t* old_to_new,
                       std::int32_t n_old,
                       std::int32_t n_new) {
  double* dst = nullptr;
  if (n_new > 0) {
    throw_if_cuda_error(cudaMalloc(&dst, sizeof(double) * static_cast<std::size_t>(n_new)),
                        "cudaMalloc compact vector");
  }
  if (n_old > 0 && n_new > 0) {
    constexpr int threads = 256;
    const int blocks = (n_old + threads - 1) / threads;
    _kernel_gather_by_old_to_new<<<blocks, threads>>>(dst, src, old_to_new, n_old);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_gather_by_old_to_new");
  }
  return dst;
}

double* clone_device_vector(const double* src, std::int32_t n, const char* context) {
  double* dst = nullptr;
  if (n > 0) {
    throw_if_cuda_error(cudaMalloc(&dst, sizeof(double) * static_cast<std::size_t>(n)),
                        context);
    throw_if_cuda_error(cudaMemcpy(dst, src, sizeof(double) * static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToDevice),
                        context);
  }
  return dst;
}

template <class T>
std::vector<T> copy_device_array(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

__global__ void _kernel_removed_fixed_bound_record_counts(std::int32_t* selected_scan,
                                                          const std::uint8_t* keep_col,
                                                          const double* l,
                                                          const double* u,
                                                          double bound_tol,
                                                          std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    const bool removed = keep_col[col] == std::uint8_t{0};
    const bool fixed = isfinite(l[col]) && isfinite(u[col]) && fabs(l[col] - u[col]) <= bound_tol;
    selected_scan[col] = removed && fixed ? 1 : 0;
  }
}

__global__ void _kernel_pack_removed_fixed_bound_records(std::int32_t* packed_cols,
                                                         double* packed_vals,
                                                         const std::int32_t* selected_scan,
                                                         const std::uint8_t* keep_col,
                                                         const double* l,
                                                         const double* u,
                                                         double bound_tol,
                                                         std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n) {
    return;
  }
  const bool removed = keep_col[col] == std::uint8_t{0};
  const bool fixed = isfinite(l[col]) && isfinite(u[col]) && fabs(l[col] - u[col]) <= bound_tol;
  if (!removed || !fixed) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const std::int32_t out = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || out < 0) {
    return;
  }
  packed_cols[out] = col;
  packed_vals[out] = 0.5 * (l[col] + u[col]);
}

__global__ void _kernel_gather_removed_bound_values(double* lower_out,
                                                    double* upper_out,
                                                    const std::int32_t* local_cols,
                                                    const double* l,
                                                    const double* u,
                                                    std::int32_t count) {
  const std::int32_t k = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < count) {
    const std::int32_t col = local_cols[k];
    lower_out[k] = l[col];
    upper_out[k] = u[col];
  }
}

__global__ void _kernel_removed_col_counts(std::int32_t* selected_scan,
                                           const std::uint8_t* keep_col,
                                           std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    selected_scan[col] = keep_col[col] == std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_pack_removed_cols(std::int32_t* packed_cols,
                                          const std::int32_t* selected_scan,
                                          const std::uint8_t* keep_col,
                                          std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || keep_col[col] != std::uint8_t{0}) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const std::int32_t out = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || out < 0) {
    return;
  }
  packed_cols[out] = col;
}

std::vector<std::int32_t> pack_removed_cols_sparse(const std::uint8_t* keep_col,
                                                   std::int32_t n,
                                                   std::int32_t removed_count,
                                                   const char* context) {
  if (n <= 0 || removed_count <= 0) {
    return {};
  }

  std::int32_t* selected_scan = nullptr;
  std::int32_t* packed_cols = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(removed_count)),
                      context);

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_removed_col_counts<<<blocks, threads>>>(selected_scan, keep_col, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_removed_col_counts");
  inclusive_scan_i32(selected_scan, n, context);
  _kernel_pack_removed_cols<<<blocks, threads>>>(packed_cols, selected_scan, keep_col, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_removed_cols");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(removed_count));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(removed_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  cudaFree(selected_scan);
  cudaFree(packed_cols);
  return host_cols;
}

void append_removed_fixed_bound_records_sparse(
    PresolveRecordGpu& record,
    const std::vector<std::int32_t>& removed_local_cols,
    const double* l,
    const double* u,
    const std::vector<std::int32_t>& old_col_red2org,
    const char* context) {
  const std::int32_t count = static_cast<std::int32_t>(removed_local_cols.size());
  if (count <= 0) {
    return;
  }

  std::int32_t* local_cols_d = nullptr;
  double* lower_d = nullptr;
  double* upper_d = nullptr;
  const std::size_t cols_bytes = sizeof(std::int32_t) * static_cast<std::size_t>(count);
  const std::size_t vals_bytes = sizeof(double) * static_cast<std::size_t>(count);
  throw_if_cuda_error(cudaMalloc(&local_cols_d, cols_bytes), context);
  throw_if_cuda_error(cudaMalloc(&lower_d, vals_bytes), context);
  throw_if_cuda_error(cudaMalloc(&upper_d, vals_bytes), context);
  throw_if_cuda_error(cudaMemcpy(local_cols_d, removed_local_cols.data(), cols_bytes,
                                 cudaMemcpyHostToDevice),
                      context);

  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  _kernel_gather_removed_bound_values<<<blocks, threads>>>(
      lower_d, upper_d, local_cols_d, l, u, count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_gather_removed_bound_values");

  std::vector<double> lower(static_cast<std::size_t>(count));
  std::vector<double> upper(static_cast<std::size_t>(count));
  throw_if_cuda_error(cudaMemcpy(lower.data(), lower_d, vals_bytes, cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(upper.data(), upper_d, vals_bytes, cudaMemcpyDeviceToHost),
                      context);

  for (std::int32_t k = 0; k < count; ++k) {
    const double lo = lower[static_cast<std::size_t>(k)];
    const double hi = upper[static_cast<std::size_t>(k)];
    if (!isfinite(lo) || !isfinite(hi) || fabs(lo - hi) > 1.0e-9) {
      continue;
    }
    const std::int32_t local_col = removed_local_cols[static_cast<std::size_t>(k)];
    if (local_col >= 0 && local_col < static_cast<std::int32_t>(old_col_red2org.size())) {
      record.fixed_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
      record.fixed_val.push_back(0.5 * (lo + hi));
    }
  }

  cudaFree(local_cols_d);
  cudaFree(lower_d);
  cudaFree(upper_d);
}

void append_removed_fixed_bound_records(PresolveRecordGpu& record,
                                        const std::uint8_t* keep_col,
                                        const double* l,
                                        const double* u,
                                        const std::vector<std::int32_t>& old_col_red2org,
                                        std::int32_t n,
                                        const char* context) {
  if (n <= 0) {
    return;
  }
  std::int32_t* selected_scan = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_removed_fixed_bound_record_counts<<<blocks, threads>>>(
      selected_scan, keep_col, l, u, 1.0e-9, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_removed_fixed_bound_record_counts");
  inclusive_scan_i32(selected_scan, n, context);
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, selected_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (count <= 0) {
    cudaFree(selected_scan);
    return;
  }

  std::int32_t* packed_cols = nullptr;
  double* packed_vals = nullptr;
  throw_if_cuda_error(cudaMalloc(&packed_cols, sizeof(std::int32_t) * static_cast<std::size_t>(count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_vals, sizeof(double) * static_cast<std::size_t>(count)),
                      context);
  _kernel_pack_removed_fixed_bound_records<<<blocks, threads>>>(
      packed_cols, packed_vals, selected_scan, keep_col, l, u, 1.0e-9, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_removed_fixed_bound_records");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(count));
  std::vector<double> host_vals(static_cast<std::size_t>(count));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_vals.data(), packed_vals,
                                 sizeof(double) * static_cast<std::size_t>(count),
                                 cudaMemcpyDeviceToHost),
                      context);
  record.fixed_idx.reserve(record.fixed_idx.size() + static_cast<std::size_t>(count));
  record.fixed_val.reserve(record.fixed_val.size() + static_cast<std::size_t>(count));
  for (std::int32_t k = 0; k < count; ++k) {
    const std::int32_t local_col = host_cols[static_cast<std::size_t>(k)];
    if (local_col >= 0 && local_col < static_cast<std::int32_t>(old_col_red2org.size())) {
      record.fixed_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
      record.fixed_val.push_back(host_vals[static_cast<std::size_t>(k)]);
    }
  }
  cudaFree(selected_scan);
  cudaFree(packed_cols);
  cudaFree(packed_vals);
}

std::vector<std::int32_t> make_identity_i32(std::int32_t n) {
  std::vector<std::int32_t> values(static_cast<std::size_t>(n));
  for (std::int32_t i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = i;
  }
  return values;
}

PresolveRecordGpu make_initial_record(const LPInfoGpu& lp) {
  PresolveRecordGpu record;
  record.m0 = lp.A.rows;
  record.n0 = lp.A.cols;
  record.m1 = lp.A.rows;
  record.n1 = lp.A.cols;
  record.row_org2red = make_identity_i32(lp.A.rows);
  record.row_red2org = make_identity_i32(lp.A.rows);
  record.col_org2red = make_identity_i32(lp.A.cols);
  record.col_red2org = make_identity_i32(lp.A.cols);
  record.obj_constant_old = lp.obj_constant;
  record.obj_constant_new = lp.obj_constant;
  return record;
}

void rebuild_org2red(std::vector<std::int32_t>& org2red,
                     const std::vector<std::int32_t>& red2org,
                     std::int32_t original_count) {
  org2red.assign(static_cast<std::size_t>(original_count), -1);
  for (std::int32_t red = 0; red < static_cast<std::int32_t>(red2org.size()); ++red) {
    const std::int32_t org = red2org[static_cast<std::size_t>(red)];
    if (org >= 0 && org < original_count) {
      org2red[static_cast<std::size_t>(org)] = red;
    }
  }
}

StructuralL1PrimalRecoveryStep globalize_structural_recovery(
    const StructuralL1PrimalRecoveryStep& local,
    const std::vector<std::int32_t>& col_red2org) {
  StructuralL1PrimalRecoveryStep global = local;
  for (StructuralL1SplitRecovery& split : global.splits) {
    split.t_col = col_red2org[static_cast<std::size_t>(split.t_col)];
    split.e_col = col_red2org[static_cast<std::size_t>(split.e_col)];
  }
  for (StructuralOuterPairRecovery& pair : global.outer_pairs) {
    pair.bound_col = col_red2org[static_cast<std::size_t>(pair.bound_col)];
    pair.free_col = col_red2org[static_cast<std::size_t>(pair.free_col)];
  }
  for (StructuralLinkedSlackRecovery& slack : global.linked_slacks) {
    slack.slack_col = col_red2org[static_cast<std::size_t>(slack.slack_col)];
    slack.t_col = col_red2org[static_cast<std::size_t>(slack.t_col)];
  }
  for (StructuralMaxSlackRecovery& slack : global.max_slacks) {
    slack.slack_col = col_red2org[static_cast<std::size_t>(slack.slack_col)];
    for (std::int32_t& t_col : slack.t_cols) {
      t_col = col_red2org[static_cast<std::size_t>(t_col)];
    }
  }
  return global;
}

void append_postsolve_record(PostsolveTape& tape,
                             std::int32_t type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             std::uint8_t dual_mode) {
  tape.types.push_back(type);
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(static_cast<std::int32_t>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<std::int32_t>(tape.vals.size()));
  tape.dual_modes.push_back(dual_mode);
}

std::int32_t map_local_col_to_global(std::int32_t col,
                                     const std::vector<std::int32_t>& col_red2org) {
  if (col < 0 || col >= static_cast<std::int32_t>(col_red2org.size())) {
    return col;
  }
  return col_red2org[static_cast<std::size_t>(col)];
}

std::int32_t map_local_row_to_global(std::int32_t row,
                                     const std::vector<std::int32_t>& row_red2org) {
  if (row < 0 || row >= static_cast<std::int32_t>(row_red2org.size())) {
    return row;
  }
  return row_red2org[static_cast<std::size_t>(row)];
}

void append_globalized_postsolve_tape(PostsolveTape& dest,
                                      const PostsolveTape& src,
                                      const std::vector<std::int32_t>& row_red2org,
                                      const std::vector<std::int32_t>& col_red2org) {
  for (std::int32_t k = 0; k < static_cast<std::int32_t>(src.types.size()); ++k) {
    const std::int32_t idx0 = src.index_starts[static_cast<std::size_t>(k)];
    const std::int32_t idx1 = src.index_starts[static_cast<std::size_t>(k + 1)];
    const std::int32_t val0 = src.value_starts[static_cast<std::size_t>(k)];
    const std::int32_t val1 = src.value_starts[static_cast<std::size_t>(k + 1)];
    std::vector<std::int32_t> indices(src.indices.begin() + idx0, src.indices.begin() + idx1);
    std::vector<double> vals(src.vals.begin() + val0, src.vals.begin() + val1);

    switch (static_cast<PostsolveReductionType>(src.types[static_cast<std::size_t>(k)])) {
      case PostsolveReductionType::FixedCol:
        if (!indices.empty()) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          for (std::size_t p = 1; p < indices.size(); ++p) {
            indices[p] = map_local_row_to_global(indices[p], row_red2org);
          }
        }
        break;
      case PostsolveReductionType::FixedColInf:
        if (indices.size() >= 2) {
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          std::size_t p = 2;
          while (p < indices.size()) {
            const std::int32_t row_len = indices[p++];
            for (std::int32_t t = 0; t < row_len && p < indices.size(); ++t, ++p) {
              indices[p] = map_local_col_to_global(indices[p], col_red2org);
            }
          }
        }
        break;
      case PostsolveReductionType::SubCol:
        if (indices.size() >= 3) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
          const std::int32_t support_count = indices[2];
          for (std::int32_t t = 0; t < support_count && 3 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(3 + t)] =
                map_local_col_to_global(indices[static_cast<std::size_t>(3 + t)], col_red2org);
          }
        }
        break;
      case PostsolveReductionType::ParallelCol:
        if (indices.size() >= 2) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
        }
        break;
      case PostsolveReductionType::ParallelRow:
        if (indices.size() >= 2) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
        }
        break;
      case PostsolveReductionType::DeletedRow:
        if (!indices.empty()) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
          if (indices.size() >= 2) {
            indices[1] = map_local_col_to_global(indices[1], col_red2org);
          }
        }
        break;
      case PostsolveReductionType::LhsChange:
      case PostsolveReductionType::RhsChange:
      case PostsolveReductionType::EqToIneq:
        if (!indices.empty()) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
        }
        break;
      case PostsolveReductionType::BoundChangeNoRow:
        if (!indices.empty()) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
        }
        break;
      case PostsolveReductionType::BoundChangeTheRow:
        if (indices.size() >= 2) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
        }
        break;
      case PostsolveReductionType::DoubletonEq:
        if (indices.size() >= 4) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          indices[2] = map_local_row_to_global(indices[2], row_red2org);
          const std::int32_t support_count = indices[3];
          for (std::int32_t t = 0; t < support_count && 4 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(4 + t)] =
                map_local_row_to_global(indices[static_cast<std::size_t>(4 + t)], row_red2org);
          }
        }
        break;
      case PostsolveReductionType::FmeCol:
        if (indices.size() >= 3) {
          const std::int32_t involved_count = indices[0];
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          for (std::int32_t t = 0; t < involved_count && 2 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(2 + t)] =
                map_local_row_to_global(indices[static_cast<std::size_t>(2 + t)], row_red2org);
          }
          const std::int32_t side_count_pos = 2 + involved_count;
          if (side_count_pos < static_cast<std::int32_t>(indices.size())) {
            const std::int32_t side_col_count = indices[static_cast<std::size_t>(side_count_pos)];
            for (std::int32_t t = 0; t < side_col_count &&
                                     side_count_pos + 1 + t < static_cast<std::int32_t>(indices.size()); ++t) {
              indices[static_cast<std::size_t>(side_count_pos + 1 + t)] =
                  map_local_col_to_global(indices[static_cast<std::size_t>(side_count_pos + 1 + t)], col_red2org);
            }
          }
        }
        break;
      case PostsolveReductionType::AddedRow:
      case PostsolveReductionType::AddedRows:
        break;
    }

    append_postsolve_record(dest,
                            src.types[static_cast<std::size_t>(k)],
                            indices,
                            vals,
                            src.dual_modes[static_cast<std::size_t>(k)]);
  }
}

PostsolveTapeGpu postsolve_tape_to_gpu(const PostsolveTape& tape, const char* context) {
  PostsolveTapeGpu out;
  out.record_count = static_cast<std::int32_t>(tape.types.size());
  out.index_count = static_cast<std::int32_t>(tape.indices.size());
  out.value_count = static_cast<std::int32_t>(tape.vals.size());
  if (out.record_count == 0) {
    return out;
  }
  out.types = copy_host_vector_to_device(tape.types, context);
  out.index_starts = copy_host_vector_to_device(tape.index_starts, context);
  out.value_starts = copy_host_vector_to_device(tape.value_starts, context);
  out.dual_modes = copy_host_vector_to_device(tape.dual_modes, context);
  out.indices = copy_host_vector_to_device(tape.indices, context);
  out.vals = copy_host_vector_to_device(tape.vals, context);
  return out;
}

__global__ void _kernel_copy_offset_starts(std::int32_t* dest,
                                           const std::int32_t* src,
                                           std::int32_t count,
                                           std::int32_t offset) {
  const std::int32_t k = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < count) {
    dest[k] = offset + src[k];
  }
}

std::int32_t grow_tape_capacity(std::int32_t current,
                                std::int32_t required,
                                std::int32_t minimum) {
  std::int64_t capacity = std::max<std::int64_t>(current, minimum);
  while (capacity < required) {
    capacity = std::min<std::int64_t>(
        static_cast<std::int64_t>(INT32_MAX), capacity + capacity / 2 + 1);
  }
  return static_cast<std::int32_t>(capacity);
}

void reserve_postsolve_tape_gpu(PostsolveTapeGpu& tape,
                                std::int32_t required_records,
                                std::int32_t required_indices,
                                std::int32_t required_values) {
  const std::int32_t current_record_capacity =
      std::max(tape.record_capacity, tape.record_count);
  const std::int32_t current_index_capacity =
      std::max(tape.index_capacity, tape.index_count);
  const std::int32_t current_value_capacity =
      std::max(tape.value_capacity, tape.value_count);
  if (required_records <= current_record_capacity &&
      required_indices <= current_index_capacity &&
      required_values <= current_value_capacity) {
    tape.record_capacity = current_record_capacity;
    tape.index_capacity = current_index_capacity;
    tape.value_capacity = current_value_capacity;
    return;
  }

  const std::int32_t new_record_capacity =
      grow_tape_capacity(current_record_capacity, required_records, 1024);
  const std::int32_t new_index_capacity =
      grow_tape_capacity(current_index_capacity, required_indices, 4096);
  const std::int32_t new_value_capacity =
      grow_tape_capacity(current_value_capacity, required_values, 4096);
  PostsolveTapeGpu grown;
  grown.record_count = tape.record_count;
  grown.index_count = tape.index_count;
  grown.value_count = tape.value_count;
  grown.record_capacity = new_record_capacity;
  grown.index_capacity = new_index_capacity;
  grown.value_capacity = new_value_capacity;
  throw_if_cuda_error(cudaMalloc(&grown.types,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(new_record_capacity)),
                      "cudaMalloc reserve GPU tape types");
  throw_if_cuda_error(cudaMalloc(&grown.index_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(new_record_capacity + 1)),
                      "cudaMalloc reserve GPU tape index_starts");
  throw_if_cuda_error(cudaMalloc(&grown.value_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(new_record_capacity + 1)),
                      "cudaMalloc reserve GPU tape value_starts");
  throw_if_cuda_error(cudaMalloc(&grown.dual_modes,
                                 static_cast<std::size_t>(new_record_capacity)),
                      "cudaMalloc reserve GPU tape dual_modes");
  throw_if_cuda_error(cudaMalloc(&grown.indices,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(new_index_capacity)),
                      "cudaMalloc reserve GPU tape indices");
  throw_if_cuda_error(cudaMalloc(&grown.vals,
                                 sizeof(double) * static_cast<std::size_t>(new_value_capacity)),
                      "cudaMalloc reserve GPU tape vals");
  if (tape.record_count > 0) {
    throw_if_cuda_error(cudaMemcpy(grown.types, tape.types,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(tape.record_count),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy reserve GPU tape types");
    throw_if_cuda_error(cudaMemcpy(grown.index_starts, tape.index_starts,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(tape.record_count + 1),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy reserve GPU tape index_starts");
    throw_if_cuda_error(cudaMemcpy(grown.value_starts, tape.value_starts,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(tape.record_count + 1),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy reserve GPU tape value_starts");
    throw_if_cuda_error(cudaMemcpy(grown.dual_modes, tape.dual_modes,
                                   static_cast<std::size_t>(tape.record_count),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy reserve GPU tape dual_modes");
  }
  if (tape.index_count > 0) {
    throw_if_cuda_error(cudaMemcpy(grown.indices, tape.indices,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(tape.index_count),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy reserve GPU tape indices");
  }
  if (tape.value_count > 0) {
    throw_if_cuda_error(cudaMemcpy(grown.vals, tape.vals,
                                   sizeof(double) * static_cast<std::size_t>(tape.value_count),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy reserve GPU tape vals");
  }
  tape = std::move(grown);
}

void append_postsolve_tape_gpu(PostsolveTapeGpu& dest, const PostsolveTapeGpu& src) {
  if (src.record_count == 0) {
    return;
  }
  const std::int32_t old_record_count = dest.record_count;
  const std::int32_t old_index_count = dest.index_count;
  const std::int32_t old_value_count = dest.value_count;
  const std::int32_t new_record_count = old_record_count + src.record_count;
  const std::int32_t new_index_count = old_index_count + src.index_count;
  const std::int32_t new_value_count = old_value_count + src.value_count;
  reserve_postsolve_tape_gpu(dest, new_record_count, new_index_count, new_value_count);
  throw_if_cuda_error(cudaMemcpy(dest.types + old_record_count, src.types,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(src.record_count),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy append GPU tape new types");
  throw_if_cuda_error(cudaMemcpy(dest.dual_modes + old_record_count, src.dual_modes,
                                 static_cast<std::size_t>(src.record_count),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy append GPU tape new dual_modes");
  if (src.index_count > 0) {
    throw_if_cuda_error(cudaMemcpy(dest.indices + old_index_count, src.indices,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(src.index_count),
                                 cudaMemcpyDeviceToDevice),
                        "cudaMemcpy append GPU tape new indices");
  }
  if (src.value_count > 0) {
    throw_if_cuda_error(cudaMemcpy(dest.vals + old_value_count, src.vals,
                                 sizeof(double) * static_cast<std::size_t>(src.value_count),
                                 cudaMemcpyDeviceToDevice),
                        "cudaMemcpy append GPU tape new vals");
  }
  const int blocks = (src.record_count + 1 + GPU_PRESOLVE_THREADS - 1) /
                     GPU_PRESOLVE_THREADS;
  _kernel_copy_offset_starts<<<blocks, GPU_PRESOLVE_THREADS>>>(
      dest.index_starts + old_record_count, src.index_starts,
      src.record_count + 1, old_index_count);
  _kernel_copy_offset_starts<<<blocks, GPU_PRESOLVE_THREADS>>>(
      dest.value_starts + old_record_count, src.value_starts,
      src.record_count + 1, old_value_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_offset_starts");
  dest.record_count = new_record_count;
  dest.index_count = new_index_count;
  dest.value_count = new_value_count;
}

void append_globalized_postsolve_tape_gpu(PostsolveTapeGpu& dest,
                                          const PostsolveTapeGpu& src,
                                          const std::vector<std::int32_t>& row_red2org,
                                          const std::vector<std::int32_t>& col_red2org) {
  if (src.record_count == 0) {
    return;
  }
  std::int32_t* row_map_d = nullptr;
  std::int32_t* col_map_d = nullptr;
  auto map_guard = make_scope_exit([&]() {
    cudaFree(row_map_d);
    cudaFree(col_map_d);
  });
  row_map_d = copy_host_vector_to_device(row_red2org, "cudaMalloc/copy GPU tape row_red2org");
  col_map_d = copy_host_vector_to_device(col_red2org, "cudaMalloc/copy GPU tape col_red2org");
  const int blocks = (src.record_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_globalize_postsolve_tape_gpu<<<blocks, GPU_PRESOLVE_THREADS>>>(
      src.indices, src.types, src.index_starts, row_map_d, col_map_d, src.record_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_globalize_postsolve_tape_gpu");
  cudaFree(row_map_d);
  row_map_d = nullptr;
  cudaFree(col_map_d);
  col_map_d = nullptr;
  append_postsolve_tape_gpu(dest, src);
}

void update_record_from_plan(PresolveRecordGpu& record,
                             const PresolvePlanGpu& plan,
                             const LPInfoGpu& old_lp,
                             std::int32_t m_new,
                             std::int32_t n_new,
                             double obj_constant_new) {
  const bool profile_record =
      env_enabled("GPUPRESOLVER_PROFILE") || env_enabled("GPUPRESOLVER_PRESOLVE_RECORD_PROFILE");
  const auto record_start = std::chrono::steady_clock::now();
  const bool has_removed_rows = m_new < old_lp.A.rows;
  const bool has_removed_cols = n_new < old_lp.A.cols;
  const std::int32_t removed_row_count = has_removed_rows ? (old_lp.A.rows - m_new) : 0;
  const std::int32_t removed_col_count = has_removed_cols ? (old_lp.A.cols - n_new) : 0;
  const bool sparse_row_update =
      has_removed_rows && removed_row_count <= SPARSE_FIXED_BOUND_RECORD_THRESHOLD;
  const bool sparse_col_update =
      has_removed_cols && removed_col_count <= SPARSE_FIXED_BOUND_RECORD_THRESHOLD;
  const bool record_removed_indices = env_enabled("GPUPRESOLVER_RECORD_REMOVED_INDICES");
  const std::vector<std::uint8_t> keep_row = has_removed_rows && !sparse_row_update
      ? copy_device_array(plan.keep_row_mask, old_lp.A.rows, "cudaMemcpy record keep_row")
      : std::vector<std::uint8_t>{};
  const std::vector<std::uint8_t> keep_col = has_removed_cols && !sparse_col_update
      ? copy_device_array(plan.keep_col_mask, old_lp.A.cols, "cudaMemcpy record keep_col")
      : std::vector<std::uint8_t>{};
  const auto after_keep_copy = std::chrono::steady_clock::now();

  const std::vector<std::int32_t>& old_row_red2org = record.row_red2org;
  const std::vector<std::int32_t>& old_col_red2org = record.col_red2org;
  const auto after_old_mapping_copy = std::chrono::steady_clock::now();
  std::vector<std::int32_t> next_row_red2org;
  std::vector<std::int32_t> next_col_red2org;
  std::vector<std::int32_t> removed_local_rows;
  std::vector<std::int32_t> removed_local_cols;
  if (has_removed_rows) {
    next_row_red2org.reserve(static_cast<std::size_t>(m_new));
    if (record_removed_indices) {
      record.removed_row_idx.reserve(record.removed_row_idx.size() +
                                     static_cast<std::size_t>(old_lp.A.rows - m_new));
    }
    if (sparse_row_update) {
      removed_local_rows.reserve(static_cast<std::size_t>(removed_row_count));
    }
  }
  if (has_removed_cols) {
    next_col_red2org.reserve(static_cast<std::size_t>(n_new));
    if (record_removed_indices) {
      record.removed_col_idx.reserve(record.removed_col_idx.size() +
                                     static_cast<std::size_t>(old_lp.A.cols - n_new));
    }
    if (sparse_col_update) {
      removed_local_cols.reserve(static_cast<std::size_t>(removed_col_count));
    }
  }

  if (has_removed_rows) {
    if (sparse_row_update) {
      removed_local_rows = pack_removed_cols_sparse(plan.keep_row_mask,
                                                    old_lp.A.rows,
                                                    removed_row_count,
                                                    "cudaMemcpy sparse removed rows");
      std::int32_t range_begin = 0;
      for (const std::int32_t local_row : removed_local_rows) {
        if (local_row < range_begin ||
            local_row >= static_cast<std::int32_t>(old_row_red2org.size())) {
          continue;
        }
        next_row_red2org.insert(next_row_red2org.end(),
                                old_row_red2org.begin() + range_begin,
                                old_row_red2org.begin() + local_row);
        if (record_removed_indices) {
          record.removed_row_idx.push_back(old_row_red2org[static_cast<std::size_t>(local_row)]);
        }
        range_begin = local_row + 1;
      }
      next_row_red2org.insert(next_row_red2org.end(),
                              old_row_red2org.begin() + range_begin,
                              old_row_red2org.end());
    } else {
      for (std::int32_t row = 0; row < old_lp.A.rows; ++row) {
        const std::int32_t global = old_row_red2org[static_cast<std::size_t>(row)];
        if (keep_row[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
          next_row_red2org.push_back(global);
        } else if (record_removed_indices) {
          record.removed_row_idx.push_back(global);
        }
      }
    }
  }
  if (has_removed_cols) {
    if (sparse_col_update) {
      removed_local_cols = pack_removed_cols_sparse(plan.keep_col_mask,
                                                    old_lp.A.cols,
                                                    removed_col_count,
                                                    "cudaMemcpy sparse removed cols");
      std::int32_t range_begin = 0;
      for (const std::int32_t local_col : removed_local_cols) {
        if (local_col < range_begin ||
            local_col >= static_cast<std::int32_t>(old_col_red2org.size())) {
          continue;
        }
        next_col_red2org.insert(next_col_red2org.end(),
                                old_col_red2org.begin() + range_begin,
                                old_col_red2org.begin() + local_col);
        if (record_removed_indices) {
          record.removed_col_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
        }
        range_begin = local_col + 1;
      }
      next_col_red2org.insert(next_col_red2org.end(),
                              old_col_red2org.begin() + range_begin,
                              old_col_red2org.end());
    } else {
      for (std::int32_t col = 0; col < old_lp.A.cols; ++col) {
        const std::int32_t global = old_col_red2org[static_cast<std::size_t>(col)];
        if (keep_col[static_cast<std::size_t>(col)] != std::uint8_t{0}) {
          next_col_red2org.push_back(global);
        } else if (record_removed_indices) {
          record.removed_col_idx.push_back(global);
        }
      }
    }
    if (sparse_col_update) {
      append_removed_fixed_bound_records_sparse(record,
                                               removed_local_cols,
                                               plan.new_l,
                                               plan.new_u,
                                               old_col_red2org,
                                               "cudaMemcpy sparse record fixed bounds");
    } else {
      append_removed_fixed_bound_records(record,
                                         plan.keep_col_mask,
                                         plan.new_l,
                                         plan.new_u,
                                         old_col_red2org,
                                         old_lp.A.cols,
                                         "cudaMemcpy record fixed bounds");
    }
  }
  const auto after_removed_mapping = std::chrono::steady_clock::now();

  if (plan.has_structural_primal_recovery) {
    record.structural_primal_recoveries.push_back(
        globalize_structural_recovery(plan.structural_primal_recovery, old_col_red2org));
  }
  append_globalized_postsolve_tape_gpu(record.tape_gpu, plan.tape_gpu, old_row_red2org, old_col_red2org);
  PostsolveTape globalized_cpu_tape;
  append_globalized_postsolve_tape(globalized_cpu_tape, plan.tape, old_row_red2org, old_col_red2org);
  if (!globalized_cpu_tape.types.empty()) {
    for (std::int32_t type : globalized_cpu_tape.types) {
      record.tape.types.push_back(type);
    }
    record.tape.indices.insert(record.tape.indices.end(),
                               globalized_cpu_tape.indices.begin(),
                               globalized_cpu_tape.indices.end());
    record.tape.vals.insert(record.tape.vals.end(),
                            globalized_cpu_tape.vals.begin(),
                            globalized_cpu_tape.vals.end());
    const std::int32_t index_offset =
        record.tape.index_starts.empty() ? 0 : record.tape.index_starts.back();
    const std::int32_t value_offset =
        record.tape.value_starts.empty() ? 0 : record.tape.value_starts.back();
    for (std::size_t k = 1; k < globalized_cpu_tape.index_starts.size(); ++k) {
      record.tape.index_starts.push_back(index_offset + globalized_cpu_tape.index_starts[k]);
    }
    for (std::size_t k = 1; k < globalized_cpu_tape.value_starts.size(); ++k) {
      record.tape.value_starts.push_back(value_offset + globalized_cpu_tape.value_starts[k]);
    }
    record.tape.dual_modes.insert(record.tape.dual_modes.end(),
                                  globalized_cpu_tape.dual_modes.begin(),
                                  globalized_cpu_tape.dual_modes.end());
    if (!plan.tape_gpu_mirrors_cpu) {
      PostsolveTapeGpu cpu_tape_gpu =
          postsolve_tape_to_gpu(globalized_cpu_tape, "cudaMalloc/copy globalized CPU tape to GPU");
      append_postsolve_tape_gpu(record.tape_gpu, cpu_tape_gpu);
      free_postsolve_tape_gpu(cpu_tape_gpu);
    }
  }
  const auto after_tape = std::chrono::steady_clock::now();

  if (has_removed_rows) {
    record.row_red2org = std::move(next_row_red2org);
  }
  if (has_removed_cols) {
    record.col_red2org = std::move(next_col_red2org);
  }
  record.m1 = m_new;
  record.n1 = n_new;
  record.obj_constant_new = obj_constant_new;
  const auto after_rebuild = std::chrono::steady_clock::now();
  if (profile_record) {
    const std::chrono::duration<double> keep_copy_elapsed = after_keep_copy - record_start;
    const std::chrono::duration<double> old_mapping_copy_elapsed = after_old_mapping_copy - after_keep_copy;
    const std::chrono::duration<double> removed_mapping_elapsed = after_removed_mapping - after_old_mapping_copy;
    const std::chrono::duration<double> tape_elapsed = after_tape - after_removed_mapping;
    const std::chrono::duration<double> rebuild_elapsed = after_rebuild - after_tape;
    const std::chrono::duration<double> total_elapsed = after_rebuild - record_start;
    std::cerr << ">>> [GPU Presolve C++ record-profile]"
              << " dims=(" << old_lp.A.rows << "," << old_lp.A.cols << ")->("
              << m_new << "," << n_new << ")"
              << " removed_rows=" << (old_lp.A.rows - m_new)
              << " removed_cols=" << (old_lp.A.cols - n_new)
              << " tape_records=" << plan.tape.types.size()
              << " tape_gpu_records=" << plan.tape_gpu.record_count
              << " tape_indices=" << plan.tape.indices.size()
              << " tape_gpu_indices=" << plan.tape_gpu.index_count
              << " tape_vals=" << plan.tape.vals.size()
              << " tape_gpu_vals=" << plan.tape_gpu.value_count
              << " keep_copy=" << keep_copy_elapsed.count() << "s"
              << " old_mapping_copy=" << old_mapping_copy_elapsed.count() << "s"
              << " removed_mapping=" << removed_mapping_elapsed.count() << "s"
              << " tape=" << tape_elapsed.count() << "s"
              << " rebuild=" << rebuild_elapsed.count() << "s"
              << " total=" << total_elapsed.count() << "s\n";
  }
}

DeviceCsrMatrix transpose_csr_gpu(const DeviceCsrMatrix& A) {
  DeviceCsrMatrix AT;
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.nnz = A.nnz;
  cusparseHandle_t handle = nullptr;
  void* buffer = nullptr;
  try {
    throw_if_cuda_error(cudaMalloc(&AT.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(AT.rows + 1)),
                        "cudaMalloc transpose rowPtr");
    if (A.nnz == 0) {
      throw_if_cuda_error(cudaMemset(AT.rowPtr, 0, sizeof(std::int32_t) * static_cast<std::size_t>(AT.rows + 1)),
                          "cudaMemset empty transpose rowPtr");
      return AT;
    }
    throw_if_cuda_error(cudaMalloc(&AT.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(A.nnz)),
                        "cudaMalloc transpose colVal");
    throw_if_cuda_error(cudaMalloc(&AT.nzVal, sizeof(double) * static_cast<std::size_t>(A.nnz)),
                        "cudaMalloc transpose nzVal");

    throw_if_cusparse_error(cusparseCreate(&handle), "cusparseCreate transpose");
    std::size_t buffer_size = 0;
    constexpr cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
    throw_if_cusparse_error(
        cusparseCsr2cscEx2_bufferSize(handle,
                                      A.rows,
                                      A.cols,
                                      A.nnz,
                                      A.nzVal,
                                      A.rowPtr,
                                      A.colVal,
                                      AT.nzVal,
                                      AT.rowPtr,
                                      AT.colVal,
                                      CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      alg,
                                      &buffer_size),
        "cusparseCsr2cscEx2_bufferSize transpose");
    if (buffer_size > 0) {
      throw_if_cuda_error(cudaMalloc(&buffer, buffer_size), "cudaMalloc cusparse transpose buffer");
    }
    throw_if_cusparse_error(cusparseCsr2cscEx2(handle,
                                              A.rows,
                                              A.cols,
                                              A.nnz,
                                              A.nzVal,
                                              A.rowPtr,
                                              A.colVal,
                                              AT.nzVal,
                                              AT.rowPtr,
                                              AT.colVal,
                                              CUDA_R_64F,
                                              CUSPARSE_ACTION_NUMERIC,
                                              CUSPARSE_INDEX_BASE_ZERO,
                                              alg,
                                              buffer),
                           "cusparseCsr2cscEx2 transpose");
    cudaFree(buffer);
    buffer = nullptr;
    throw_if_cusparse_error(cusparseDestroy(handle), "cusparseDestroy transpose");
    handle = nullptr;
  } catch (...) {
    cudaFree(buffer);
    if (handle != nullptr) {
      cusparseDestroy(handle);
    }
    free_device_csr(AT);
    throw;
  }
  return AT;
}

DeviceCsrMatrix compact_csr_by_masks_gpu(const DeviceCsrMatrix& A,
                                         const std::int32_t* row_old_to_new,
                                         const std::int32_t* col_old_to_new,
                                         std::int32_t m_new,
                                         std::int32_t n_new) {
  DeviceCsrMatrix out;
  out.rows = m_new;
  out.cols = n_new;
  std::int32_t* row_counts = nullptr;
  std::int32_t* dense_scan = nullptr;
  std::int32_t* dense_rows = nullptr;
  try {
    throw_if_cuda_error(compact_cuda_malloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(m_new + 1)),
                        "cudaMalloc compact rowPtr");
    if (m_new == 0 || n_new == 0) {
      throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(std::int32_t) * static_cast<std::size_t>(m_new + 1)),
                          "cudaMemset empty compact rowPtr");
      return out;
    }
    throw_if_cuda_error(compact_cuda_malloc(&row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(m_new)),
                        "cudaMalloc compact row_counts");
    throw_if_cuda_error(cudaMemset(row_counts, 0, sizeof(std::int32_t) * static_cast<std::size_t>(m_new)),
                        "cudaMemset compact row_counts");
    constexpr int threads = 256;
    const int old_row_blocks = (A.rows + threads - 1) / threads;
    const bool use_block_per_row =
        static_cast<std::int64_t>(A.nnz) >
        8LL * static_cast<std::int64_t>(A.rows);
    constexpr std::int32_t dense_threshold = 32;
    std::int32_t dense_count = 0;
    if (use_block_per_row) {
      _kernel_count_compacted_rows_block<<<A.rows, threads>>>(
          row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal, A.rows);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_count_compacted_rows_block");
    } else {
      throw_if_cuda_error(compact_cuda_malloc(
                              &dense_scan,
                              sizeof(std::int32_t) * static_cast<std::size_t>(A.rows)),
                          "cudaMalloc compact dense_scan");
      _kernel_mark_dense_compact_rows<<<old_row_blocks, threads>>>(
          dense_scan, row_old_to_new, A.rowPtr, dense_threshold, A.rows);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_dense_compact_rows");
      inclusive_scan_i32(dense_scan, A.rows, "cub inclusive scan dense compact rows");
      throw_if_cuda_error(cudaMemcpy(&dense_count, dense_scan + A.rows - 1,
                                     sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                          "cudaMemcpy compact dense row count");
      if (dense_count > 0) {
        throw_if_cuda_error(compact_cuda_malloc(
                                &dense_rows,
                                sizeof(std::int32_t) * static_cast<std::size_t>(dense_count)),
                            "cudaMalloc compact dense_rows");
        _kernel_pack_dense_compact_rows<<<old_row_blocks, threads>>>(
            dense_rows, dense_scan, A.rows);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_dense_compact_rows");
      }
      _kernel_count_compacted_rows<<<old_row_blocks, threads>>>(
          row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal,
          dense_threshold, A.rows);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_rows");
      if (dense_count > 0) {
        _kernel_count_compacted_dense_rows_block<<<dense_count, threads>>>(
            row_counts, dense_rows, row_old_to_new, col_old_to_new,
            A.rowPtr, A.colVal, dense_count);
        throw_if_cuda_error(cudaGetLastError(),
                            "_kernel_count_compacted_dense_rows_block");
      }
    }
    inclusive_scan_i32(row_counts, m_new, "cub inclusive scan compact");
    std::int32_t nnz_new = 0;
    throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts + m_new - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy compact nnz");
    out.nnz = nnz_new;
    const int new_row_blocks = (m_new + threads - 1) / threads;
    _kernel_row_ptr_from_prefix<<<new_row_blocks, threads>>>(out.rowPtr, row_counts, m_new);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix compact");
    if (nnz_new > 0) {
      throw_if_cuda_error(compact_cuda_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                          "cudaMalloc compact colVal");
      throw_if_cuda_error(compact_cuda_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                          "cudaMalloc compact nzVal");
      if (use_block_per_row) {
        _kernel_copy_compacted_rows_block<<<A.rows, threads>>>(
            out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
            A.rowPtr, A.colVal, A.nzVal, A.rows);
        throw_if_cuda_error(cudaGetLastError(),
                            "_kernel_copy_compacted_rows_block");
      } else {
        _kernel_copy_compacted_rows<<<old_row_blocks, threads>>>(
            out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
            A.rowPtr, A.colVal, A.nzVal, dense_threshold, A.rows);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_rows");
        if (dense_count > 0) {
          _kernel_copy_compacted_dense_rows_block<<<dense_count, threads>>>(
              out.colVal, out.nzVal, out.rowPtr, dense_rows, row_old_to_new,
              col_old_to_new, A.rowPtr, A.colVal, A.nzVal, dense_count);
          throw_if_cuda_error(cudaGetLastError(),
                              "_kernel_copy_compacted_dense_rows_block");
        }
      }
    }
    compact_cuda_free(row_counts);
    row_counts = nullptr;
    compact_cuda_free(dense_scan);
    dense_scan = nullptr;
    compact_cuda_free(dense_rows);
    dense_rows = nullptr;
  } catch (...) {
    compact_cuda_free(row_counts);
    compact_cuda_free(dense_scan);
    compact_cuda_free(dense_rows);
    free_device_csr(out);
    throw;
  }
  return out;
}

struct WorkingLp {
  LPInfoGpu lp;
  bool owns = false;

  ~WorkingLp() { free_owned(); }

  void free_owned() {
    if (!owns) {
      return;
    }
    cudaFree(lp.A.rowPtr);
    cudaFree(lp.A.colVal);
    cudaFree(lp.A.nzVal);
    cudaFree(lp.AT.rowPtr);
    cudaFree(lp.AT.colVal);
    cudaFree(lp.AT.nzVal);
    cudaFree(lp.c);
    cudaFree(lp.AL);
    cudaFree(lp.AU);
    cudaFree(lp.l);
    cudaFree(lp.u);
    owns = false;
  }

  void free_owned_vectors() {
    if (!owns) {
      return;
    }
    cudaFree(lp.c);
    cudaFree(lp.AL);
    cudaFree(lp.AU);
    cudaFree(lp.l);
    cudaFree(lp.u);
  }
};

void apply_plan_to_working_lp(WorkingLp& current, const PresolvePlanGpu& plan, PresolveRecordGpu* record) {
  const bool profile_apply =
      env_enabled("GPUPRESOLVER_PROFILE") || env_enabled("GPUPRESOLVER_PRESOLVE_APPLY_PROFILE");
  const auto apply_start = std::chrono::steady_clock::now();
  auto sync_for_profile = [&](const char* context) {
    if (profile_apply) {
      throw_if_cuda_error(cudaDeviceSynchronize(), context);
    }
  };
  std::int32_t m_new = 0;
  std::int32_t n_new = 0;
  std::int32_t* row_old_to_new = nullptr;
  std::int32_t* col_old_to_new = nullptr;
  DeviceCsrMatrix A_new;
  DeviceCsrMatrix AT_new;
  double* c_new = nullptr;
  double* l_new = nullptr;
  double* u_new = nullptr;
  double* AL_new = nullptr;
  double* AU_new = nullptr;
  auto cleanup = make_scope_exit([&]() {
    cudaFree(row_old_to_new);
    cudaFree(col_old_to_new);
    free_device_csr(A_new);
    free_device_csr(AT_new);
    cudaFree(c_new);
    cudaFree(l_new);
    cudaFree(u_new);
    cudaFree(AL_new);
    cudaFree(AU_new);
  });
  row_old_to_new = build_old_to_new(plan.keep_row_mask, current.lp.A.rows, &m_new);
  col_old_to_new = build_old_to_new(plan.keep_col_mask, current.lp.A.cols, &n_new);
  sync_for_profile("profile apply build_old_to_new synchronize");
  const auto after_old_to_new = std::chrono::steady_clock::now();
  const bool structural_unchanged =
      current.owns && !plan.has_new_A && m_new == current.lp.A.rows && n_new == current.lp.A.cols;
  if (structural_unchanged) {
    c_new = clone_device_vector(plan.new_c, current.lp.A.cols, "cudaMalloc/copy no-struct c");
    l_new = clone_device_vector(plan.new_l, current.lp.A.cols, "cudaMalloc/copy no-struct l");
    u_new = clone_device_vector(plan.new_u, current.lp.A.cols, "cudaMalloc/copy no-struct u");
    AL_new = clone_device_vector(plan.new_AL, current.lp.A.rows, "cudaMalloc/copy no-struct AL");
    AU_new = clone_device_vector(plan.new_AU, current.lp.A.rows, "cudaMalloc/copy no-struct AU");
    sync_for_profile("profile apply clone no-struct vectors synchronize");
    const auto after_vectors = std::chrono::steady_clock::now();
    if (record != nullptr) {
      update_record_from_plan(*record, plan, current.lp, m_new, n_new,
                              current.lp.obj_constant + plan.obj_constant_delta);
    }
    sync_for_profile("profile apply no-struct record synchronize");
    const auto after_record = std::chrono::steady_clock::now();
    cudaFree(row_old_to_new);
    row_old_to_new = nullptr;
    cudaFree(col_old_to_new);
    col_old_to_new = nullptr;
    current.free_owned_vectors();
    sync_for_profile("profile apply no-struct free synchronize");
    const auto after_free = std::chrono::steady_clock::now();
    current.lp.c = c_new;
    c_new = nullptr;
    current.lp.l = l_new;
    l_new = nullptr;
    current.lp.u = u_new;
    u_new = nullptr;
    current.lp.AL = AL_new;
    AL_new = nullptr;
    current.lp.AU = AU_new;
    AU_new = nullptr;
    current.lp.obj_constant += plan.obj_constant_delta;
    current.owns = true;
    if (profile_apply) {
      const std::chrono::duration<double> old_to_new_elapsed = after_old_to_new - apply_start;
      const std::chrono::duration<double> vectors_elapsed = after_vectors - after_old_to_new;
      const std::chrono::duration<double> record_elapsed = after_record - after_vectors;
      const std::chrono::duration<double> free_elapsed = after_free - after_record;
      const std::chrono::duration<double> total_elapsed = after_free - apply_start;
      std::cerr << ">>> [GPU Presolve C++ apply-profile] structural_unchanged=true"
                << " dims=(" << current.lp.A.rows << "," << current.lp.A.cols << ")->("
                << m_new << "," << n_new << ")"
                << " old_to_new=" << old_to_new_elapsed.count() << "s"
                << " vectors=" << vectors_elapsed.count() << "s"
                << " record=" << record_elapsed.count() << "s"
                << " free=" << free_elapsed.count() << "s"
                << " total=" << total_elapsed.count() << "s\n";
    }
    cleanup.release();
    return;
  }
  const DeviceCsrMatrix& source_A = plan.has_new_A ? plan.new_A : current.lp.A;
  A_new = compact_csr_by_masks_gpu(source_A, row_old_to_new, col_old_to_new, m_new, n_new);
  const std::int32_t new_nnz = A_new.nnz;
  sync_for_profile("profile apply compact A synchronize");
  const auto after_compact_a = std::chrono::steady_clock::now();
  AT_new = transpose_csr_gpu(A_new);
  sync_for_profile("profile apply transpose synchronize");
  const auto after_transpose = std::chrono::steady_clock::now();
  c_new = compact_vector(plan.new_c, col_old_to_new, current.lp.A.cols, n_new);
  l_new = compact_vector(plan.new_l, col_old_to_new, current.lp.A.cols, n_new);
  u_new = compact_vector(plan.new_u, col_old_to_new, current.lp.A.cols, n_new);
  AL_new = compact_vector(plan.new_AL, row_old_to_new, current.lp.A.rows, m_new);
  AU_new = compact_vector(plan.new_AU, row_old_to_new, current.lp.A.rows, m_new);
  sync_for_profile("profile apply compact vectors synchronize");
  const auto after_vectors = std::chrono::steady_clock::now();
  if (record != nullptr) {
    update_record_from_plan(*record, plan, current.lp, m_new, n_new,
                            current.lp.obj_constant + plan.obj_constant_delta);
  }
  sync_for_profile("profile apply record synchronize");
  const auto after_record = std::chrono::steady_clock::now();
  cudaFree(row_old_to_new);
  row_old_to_new = nullptr;
  cudaFree(col_old_to_new);
  col_old_to_new = nullptr;

  current.free_owned();
  sync_for_profile("profile apply free synchronize");
  const auto after_free = std::chrono::steady_clock::now();
  current.lp.A = A_new;
  A_new = DeviceCsrMatrix{};
  current.lp.AT = AT_new;
  AT_new = DeviceCsrMatrix{};
  current.lp.c = c_new;
  c_new = nullptr;
  current.lp.l = l_new;
  l_new = nullptr;
  current.lp.u = u_new;
  u_new = nullptr;
  current.lp.AL = AL_new;
  AL_new = nullptr;
  current.lp.AU = AU_new;
  AU_new = nullptr;
  current.lp.obj_constant += plan.obj_constant_delta;
  current.owns = true;
  if (profile_apply) {
    const std::chrono::duration<double> old_to_new_elapsed = after_old_to_new - apply_start;
    const std::chrono::duration<double> compact_a_elapsed = after_compact_a - after_old_to_new;
    const std::chrono::duration<double> transpose_elapsed = after_transpose - after_compact_a;
    const std::chrono::duration<double> vectors_elapsed = after_vectors - after_transpose;
    const std::chrono::duration<double> record_elapsed = after_record - after_vectors;
    const std::chrono::duration<double> free_elapsed = after_free - after_record;
    const std::chrono::duration<double> total_elapsed = after_free - apply_start;
    std::cerr << ">>> [GPU Presolve C++ apply-profile] structural_unchanged=false"
              << " dims=(" << source_A.rows << "," << source_A.cols << ")->("
              << m_new << "," << n_new << ")"
              << " nnz=" << source_A.nnz << "->" << new_nnz
              << " old_to_new=" << old_to_new_elapsed.count() << "s"
              << " compact_A=" << compact_a_elapsed.count() << "s"
              << " transpose=" << transpose_elapsed.count() << "s"
              << " vectors=" << vectors_elapsed.count() << "s"
              << " record=" << record_elapsed.count() << "s"
              << " free=" << free_elapsed.count() << "s"
              << " total=" << total_elapsed.count() << "s\n";
  }
  cleanup.release();
}

std::int32_t count_keep_mask(const std::uint8_t* keep_mask, std::int32_t n) {
  std::int32_t* device_count = nullptr;
  throw_if_cuda_error(cudaMalloc(&device_count, sizeof(std::int32_t)), "cudaMalloc count keep");
  throw_if_cuda_error(cudaMemset(device_count, 0, sizeof(std::int32_t)), "cudaMemset count keep");
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_count_keep_mask<<<blocks, threads>>>(keep_mask, n, device_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_count_keep_mask");
  std::int32_t host_count = 0;
  throw_if_cuda_error(cudaMemcpy(&host_count, device_count, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy count keep");
  cudaFree(device_count);
  return host_count;
}

std::int32_t count_live_nnz(const PresolvePlanGpu& plan, const LPInfoGpu& lp) {
  std::int32_t* device_count = nullptr;
  throw_if_cuda_error(cudaMalloc(&device_count, sizeof(std::int32_t)), "cudaMalloc count nnz");
  throw_if_cuda_error(cudaMemset(device_count, 0, sizeof(std::int32_t)), "cudaMemset count nnz");
  constexpr int threads = 256;
  const int blocks = (lp.A.rows + threads - 1) / threads;
  _kernel_count_live_nnz<<<blocks, threads>>>(
      plan.keep_row_mask,
      plan.keep_col_mask,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.rows,
      device_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_count_live_nnz");
  std::int32_t host_count = 0;
  throw_if_cuda_error(cudaMemcpy(&host_count, device_count, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy count nnz");
  cudaFree(device_count);
  return host_count;
}

}  // namespace

GpuRuntimeInfo query_gpu_runtime() {
  GpuRuntimeInfo info;
  throw_if_cuda_error(cudaGetDeviceCount(&info.device_count), "cudaGetDeviceCount");
  if (info.device_count > 0) {
    throw_if_cuda_error(cudaGetDevice(&info.selected_device), "cudaGetDevice");
  }
  return info;
}

std::int32_t _presolve_nnz(const LPInfoGpu& lp) {
  return lp.A.nnz;
}

bool _has_good_nnz_progress(std::int32_t nnz_before,
                            std::int32_t nnz_after,
                            double ratio) {
  if (nnz_before <= 0) {
    return false;
  }
  return static_cast<double>(nnz_after) < ratio * static_cast<double>(nnz_before);
}

std::int32_t run_cuda_smoke_count(std::int32_t n) {
  std::int32_t* device_count = nullptr;
  throw_if_cuda_error(cudaMalloc(&device_count, sizeof(std::int32_t)), "cudaMalloc smoke count");
  throw_if_cuda_error(cudaMemset(device_count, 0, sizeof(std::int32_t)), "cudaMemset smoke count");

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_smoke_count<<<blocks, threads>>>(n, device_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_smoke_count launch");
  throw_if_cuda_error(cudaDeviceSynchronize(), "_kernel_smoke_count synchronize");

  std::int32_t host_count = 0;
  throw_if_cuda_error(
      cudaMemcpy(&host_count, device_count, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy smoke count");
  throw_if_cuda_error(cudaFree(device_count), "cudaFree smoke count");
  return host_count;
}

namespace {

GpuPresolveSummary run_gpu_presolve_impl(const LPInfoGpu& lp,
                                         const PresolveParams& params,
                                         bool keep_reduced_lp);

}  // namespace

GpuPresolveSummary run_gpu_presolve_fixed_order(const LPInfoGpu& lp,
                                                const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, false);
}

namespace {

GpuPresolveSummary run_gpu_presolve_impl(const LPInfoGpu& lp,
                                         const PresolveParams& params,
                                         bool keep_reduced_lp) {
  GpuPresolveSummary summary;
  summary.original_rows = lp.A.rows;
  summary.original_cols = lp.A.cols;
  summary.record = make_initial_record(lp);
  auto summary_guard = make_scope_exit([&]() {
    free_gpu_presolve_record_resources(summary.record);
    free_gpu_presolve_reduced_lp(summary);
  });

  WorkingLp current;
  current.lp = lp;

  bool terminal = false;
  bool post_propagation_phase_barrier_hit = false;
  const bool profile_rules =
      env_enabled("GPUPRESOLVER_PROFILE") || env_enabled("GPUPRESOLVER_PRESOLVE_RULE_PROFILE");
  const auto presolve_start = std::chrono::steady_clock::now();
  const bool has_time_limit = params.max_time > 0.0 && std::isfinite(params.max_time);
  bool singleton_cols_dual_clean = false;
  bool singleton_cols_eq_clean = false;
  auto invalidate_singleton_col_clean = [&]() {
    singleton_cols_dual_clean = false;
    singleton_cols_eq_clean = false;
  };
  auto time_exceeded = [&]() -> bool {
    if (!has_time_limit) {
      return false;
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - presolve_start;
    return elapsed.count() >= params.max_time;
  };

  auto run_row_phase = [&](bool empty_rows,
                           bool singleton_rows,
                           bool activity_checks,
                           bool primal_propagation,
                           bool parallel_rows) -> bool {
    if (terminal || time_exceeded()) {
      return false;
    }
    const auto phase_start = std::chrono::steady_clock::now();
    PresolvePlanGpu plan;
    PresolveStatsGpu stats;
    auto plan_guard = make_scope_exit([&]() { free_plan(plan); });
    auto stats_guard = make_scope_exit([&]() { free_stats(stats); });
    allocate_plan(plan, current.lp);
    const bool needs_row_stats =
        (empty_rows && params.enable_empty_rows) ||
        (singleton_rows && params.enable_singleton_rows) ||
        (activity_checks && params.enable_activity_checks) ||
        (primal_propagation && params.enable_primal_propagation);
    allocate_stats(stats, current.lp.A.rows, current.lp.A.cols, needs_row_stats, false);
    if (needs_row_stats) {
      recompute_row_stats_from_csr(stats, current.lp);
    }
    auto after_stats = std::chrono::steady_clock::now();
    const std::int32_t m_before = current.lp.A.rows;
    const std::int32_t n_before = current.lp.A.cols;
    const std::int32_t nnz_before = current.lp.A.nnz;
    std::string rules;
    if (empty_rows && params.enable_empty_rows) append_rule_name(rules, "empty_rows");
    if (singleton_rows && params.enable_singleton_rows) append_rule_name(rules, "singleton_rows");
    if (activity_checks && params.enable_activity_checks) append_rule_name(rules, "activity_checks");
    if (primal_propagation && params.enable_primal_propagation) append_rule_name(rules, "primal_propagation");
    if (parallel_rows && params.enable_parallel_rows) append_rule_name(rules, "parallel_rows");

    if (!time_exceeded() && empty_rows && params.enable_empty_rows) apply_rule_empty_rows(plan, current.lp, stats, params);
    if (!time_exceeded() && singleton_rows && params.enable_singleton_rows) apply_rule_singleton_rows(plan, current.lp, stats, params);
    if (!time_exceeded() && activity_checks && params.enable_activity_checks) apply_rule_activity_checks(plan, current.lp, stats, params);
    if (!time_exceeded() && primal_propagation && params.enable_primal_propagation) apply_rule_primal_propagation(plan, current.lp, stats, params);
    if (!time_exceeded() && parallel_rows && params.enable_parallel_rows) apply_rule_parallel_rows(plan, current.lp, stats, params);
    auto after_plan = std::chrono::steady_clock::now();

    summary.has_infeasible = plan.has_infeasible;
    summary.has_unbounded = plan.has_unbounded;
    terminal = plan.has_infeasible || plan.has_unbounded;
    const bool changed = !terminal && plan.has_change;
    if (changed) {
      apply_plan_to_working_lp(current, plan, &summary.record);
      invalidate_singleton_col_clean();
    }
    auto after_apply = std::chrono::steady_clock::now();
    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(), "profile row phase synchronize");
      const std::chrono::duration<double> phase_elapsed =
          std::chrono::steady_clock::now() - phase_start;
      const std::chrono::duration<double> stats_elapsed = after_stats - phase_start;
      const std::chrono::duration<double> plan_elapsed = after_plan - after_stats;
      const std::chrono::duration<double> apply_elapsed = after_apply - after_plan;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=row rules=[" << rules << "] changed="
                << (changed ? "true" : "false") << " dims=(" << m_before << "," << n_before
                << ")->(" << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                << nnz_before << "->" << current.lp.A.nnz
                << " time=" << phase_elapsed.count() << "s stats=" << stats_elapsed.count()
                << "s plan=" << plan_elapsed.count() << "s apply=" << apply_elapsed.count()
                << "s\n";
    }
    free_stats(stats);
    stats_guard.release();
    free_plan(plan);
    plan_guard.release();
    return changed;
  };

  auto run_col_phase = [&](bool close_bounds,
                           bool empty_cols,
                           bool singleton_cols_dual,
                           bool singleton_cols_eq,
                           bool doubleton_eq,
                           bool structural_l1,
                           bool dual_fix,
                           bool parallel_cols,
                           bool redundant_bounds) -> bool {
    if (terminal || time_exceeded()) {
      return false;
    }
    const auto phase_start = std::chrono::steady_clock::now();
    const bool structural_only =
        !close_bounds && !empty_cols && !singleton_cols_dual && !singleton_cols_eq &&
        !doubleton_eq && structural_l1 && !dual_fix && !parallel_cols && !redundant_bounds &&
        params.enable_structural_l1_substitution;
    if (structural_only && !structural_l1_prefix_screen_passes(current.lp)) {
      if (profile_rules) {
        throw_if_cuda_error(cudaDeviceSynchronize(), "profile structural prefix screen synchronize");
        const std::chrono::duration<double> phase_elapsed =
            std::chrono::steady_clock::now() - phase_start;
        std::cerr << ">>> [GPU Presolve C++ profile] phase=col rules=[structural_l1_substitution] changed=false"
                  << " dims=(" << current.lp.A.rows << "," << current.lp.A.cols << ")->("
                  << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                  << current.lp.A.nnz << "->" << current.lp.A.nnz
                  << " time=" << phase_elapsed.count() << "s\n";
      }
      return false;
    }
    PresolvePlanGpu plan;
    PresolveStatsGpu stats;
    auto plan_guard = make_scope_exit([&]() { free_plan(plan); });
    auto stats_guard = make_scope_exit([&]() { free_stats(stats); });
    allocate_plan(plan, current.lp);
    const bool needs_col_stats =
        (empty_cols && params.enable_empty_cols) ||
        (singleton_cols_dual && params.enable_singleton_cols_dual_infer) ||
        (singleton_cols_eq && params.enable_singleton_cols_eq);
    allocate_stats(stats, current.lp.A.rows, current.lp.A.cols, false, needs_col_stats);
    if (needs_col_stats) {
      recompute_col_stats_from_csr(stats, current.lp);
    }
    auto after_stats = std::chrono::steady_clock::now();
    const std::int32_t m_before = current.lp.A.rows;
    const std::int32_t n_before = current.lp.A.cols;
    const std::int32_t nnz_before = current.lp.A.nnz;
    std::string rules;
    if (close_bounds && params.enable_close_bounds) append_rule_name(rules, "close_bounds");
    if (empty_cols && params.enable_empty_cols) append_rule_name(rules, "empty_cols");
    if (singleton_cols_dual && params.enable_singleton_cols_dual_infer) append_rule_name(rules, "singleton_cols_dual_infer");
    if (singleton_cols_eq && params.enable_singleton_cols_eq) append_rule_name(rules, "singleton_cols_eq");
    if (doubleton_eq && params.enable_doubleton_eq) append_rule_name(rules, "doubleton_eq");
    if (structural_l1 && params.enable_structural_l1_substitution) append_rule_name(rules, "structural_l1_substitution");
    if (dual_fix && params.enable_dual_fix) append_rule_name(rules, "dual_fix");
    if (parallel_cols && params.enable_parallel_cols) append_rule_name(rules, "parallel_cols");
    if (redundant_bounds && params.enable_redundant_bounds) append_rule_name(rules, "redundant_bounds");

    if (!time_exceeded() && close_bounds && params.enable_close_bounds) apply_rule_close_bounds(plan, current.lp, params);
    if (!time_exceeded() && empty_cols && params.enable_empty_cols) apply_rule_empty_cols(plan, current.lp, stats, params);
    if (!time_exceeded() && singleton_cols_dual && params.enable_singleton_cols_dual_infer) {
      apply_rule_singleton_cols_dual_infer(plan, current.lp, stats, params);
    }
    if (!time_exceeded() && singleton_cols_eq && params.enable_singleton_cols_eq) {
      apply_rule_singleton_cols_eq(plan, current.lp, stats, params);
    }
    if (!time_exceeded() && doubleton_eq && params.enable_doubleton_eq) apply_rule_doubleton_eq(plan, current.lp, stats, params);
    if (!time_exceeded() && structural_l1 && params.enable_structural_l1_substitution) {
      apply_rule_structural_l1_substitution(plan, current.lp, stats, params);
    }
    if (!time_exceeded() && dual_fix && params.enable_dual_fix) apply_rule_dual_fix(plan, current.lp, stats, params);
    if (!time_exceeded() && parallel_cols && params.enable_parallel_cols) apply_rule_parallel_cols(plan, current.lp, stats, params);
    if (!time_exceeded() && redundant_bounds && params.enable_redundant_bounds) apply_rule_redundant_bounds(plan, current.lp, stats, params);
    auto after_plan = std::chrono::steady_clock::now();

    summary.has_infeasible = plan.has_infeasible;
    summary.has_unbounded = plan.has_unbounded;
    terminal = plan.has_infeasible || plan.has_unbounded;
    const bool changed = !terminal && plan.has_change;
    if (changed) {
      apply_plan_to_working_lp(current, plan, &summary.record);
      invalidate_singleton_col_clean();
    } else {
      const bool singleton_dual_only =
          !close_bounds && !empty_cols && singleton_cols_dual && !singleton_cols_eq &&
          !doubleton_eq && !structural_l1 && !dual_fix && !parallel_cols && !redundant_bounds &&
          params.enable_singleton_cols_dual_infer;
      const bool singleton_eq_only =
          !close_bounds && !empty_cols && !singleton_cols_dual && singleton_cols_eq &&
          !doubleton_eq && !structural_l1 && !dual_fix && !parallel_cols && !redundant_bounds &&
          params.enable_singleton_cols_eq;
      if (singleton_dual_only) {
        singleton_cols_dual_clean = true;
      }
      if (singleton_eq_only) {
        singleton_cols_eq_clean = true;
      }
    }
    auto after_apply = std::chrono::steady_clock::now();
    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(), "profile col phase synchronize");
      const std::chrono::duration<double> phase_elapsed =
          std::chrono::steady_clock::now() - phase_start;
      const std::chrono::duration<double> stats_elapsed = after_stats - phase_start;
      const std::chrono::duration<double> plan_elapsed = after_plan - after_stats;
      const std::chrono::duration<double> apply_elapsed = after_apply - after_plan;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=col rules=[" << rules << "] changed="
                << (changed ? "true" : "false") << " dims=(" << m_before << "," << n_before
                << ")->(" << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                << nnz_before << "->" << current.lp.A.nnz
                << " time=" << phase_elapsed.count() << "s stats=" << stats_elapsed.count()
                << "s plan=" << plan_elapsed.count() << "s apply=" << apply_elapsed.count()
                << "s\n";
    }
    free_stats(stats);
    stats_guard.release();
    free_plan(plan);
    plan_guard.release();
    return changed;
  };

  auto run_singleton_rows_to_exhaustion = [&]() -> bool {
    bool changed_any = false;
    while (!terminal && !time_exceeded()) {
      const bool changed = run_row_phase(false, true, false, false, false);
      changed_any = changed_any || changed;
      if (!changed) {
        break;
      }
    }
    return changed_any;
  };

  auto run_trivial_cleanup_recirculation = [&]() -> bool {
    bool changed_any = false;
    while (!terminal && !time_exceeded()) {
      bool changed_pass = false;
      changed_pass = run_col_phase(true, true, false, false, false, false, false, false, false) || changed_pass;
      changed_pass = run_row_phase(true, true, false, false, false) || changed_pass;
      changed_pass = run_col_phase(false, true, false, false, false, false, false, false, false) || changed_pass;
      changed_any = changed_pass || changed_any;
      if (!changed_pass) {
        break;
      }
    }
    return changed_any;
  };

  auto run_cleanup = [&]() -> bool {
    bool changed_any = false;
    changed_any = run_col_phase(true, true, false, false, false, false, false, false, false) || changed_any;
    changed_any = run_col_phase(false, false, false, false, false, false, true, false, false) || changed_any;
    changed_any = run_singleton_rows_to_exhaustion() || changed_any;
    changed_any = run_row_phase(true, false, false, false, false) || changed_any;
    changed_any = run_col_phase(false, true, false, false, false, false, false, false, false) || changed_any;
    return changed_any;
  };

  auto run_repeated_doubleton = [&]() -> bool {
    bool changed_any = false;
    while (!terminal && !time_exceeded()) {
      const bool changed = run_col_phase(false, false, false, false, true, false, false, false, false);
      changed_any = changed_any || changed;
      if (!changed || params.doubleton_eq_single_batch_per_iter) {
        break;
      }
    }
    return changed_any;
  };

  auto run_fast_phase = [&]() -> bool {
    bool changed_any = false;
    bool changed_singleton = false;
    while (!terminal && !time_exceeded()) {
      const bool changed_dual =
          singleton_cols_dual_clean ? false
                                    : run_col_phase(false, false, true, false, false, false, false, false, false);
      const bool changed_eq =
          singleton_cols_eq_clean ? false
                                  : run_col_phase(false, false, false, true, false, false, false, false, false);
      const bool changed = changed_dual || changed_eq;
      changed_any = changed_any || changed;
      changed_singleton = changed_singleton || changed;
      if (changed) {
        changed_any = run_cleanup() || changed_any;
      }
      if (!changed) {
        break;
      }
    }
    const bool changed_doubleton = run_repeated_doubleton();
    changed_any = changed_doubleton || changed_any;
    if (changed_doubleton || changed_singleton) {
      changed_any = run_cleanup() || changed_any;
    }
    return changed_any;
  };

  // Follow bound dependencies to a bounded fixed point.  Bound-only rounds
  // stay speculative and are committed only if the same plan either performs
  // a structural reduction or enables a row reduction in the activity check.
  // This prevents deeper propagation from perturbing the solver when it does
  // not reduce the model.
  auto run_primal_propagation_with_activity_closure = [&]() -> bool {
    if (!params.enable_primal_propagation) {
      return false;
    }

    bool committed_any = false;
    bool saw_candidate_change = false;
    bool post_activity_changed = false;
    bool discarded_bound_only_candidate = false;
    const int max_rounds = std::max(1, params.primal_propagation_max_rounds);
    const int configured_max_bound_only_rounds =
        std::max(1, params.primal_propagation_max_bound_only_rounds);

    // Exceptionally wide models with a modest row count can require one
    // confirmation round after the generic nnz-work cap.  Keep this exception
    // narrow instead of disabling the cap for every large instance.
    const std::int64_t propagation_rows =
        static_cast<std::int64_t>(current.lp.A.rows);
    const std::int64_t propagation_cols =
        static_cast<std::int64_t>(current.lp.A.cols);
    const std::int64_t propagation_nnz =
        static_cast<std::int64_t>(current.lp.A.nnz);
    const bool allow_wide_dense_confirmation_round =
        propagation_rows >= 512 && propagation_rows <= 2048 &&
        propagation_cols >= 128 * propagation_rows &&
        propagation_cols <= 256 * propagation_rows &&
        propagation_nnz >= 2048 * propagation_rows &&
        propagation_nnz <= 8192 * propagation_rows;

    auto effective_max_bound_only_rounds = [&]() -> int {
      int effective = configured_max_bound_only_rounds;
      const std::int64_t budget =
          params.primal_propagation_bound_only_nnz_round_budget;
      if (budget <= 0 || current.lp.A.nnz <= 0) {
        return effective;
      }
      const std::int64_t nnz = static_cast<std::int64_t>(current.lp.A.nnz);
      const std::int64_t budget_rounds =
          std::max<std::int64_t>(1, budget / nnz);
      if (budget_rounds < static_cast<std::int64_t>(effective)) {
        effective = static_cast<int>(budget_rounds);
      }
      if (allow_wide_dense_confirmation_round) {
        effective =
            std::max(effective, std::min(2, configured_max_bound_only_rounds));
      }
      return effective;
    };

    int rounds = 0;
    int bound_only_rounds = 0;
    int bound_only_round_limit = configured_max_bound_only_rounds;
    bool stopped_on_bound_only_work_cap = false;
    const auto closure_start = std::chrono::steady_clock::now();

    while (rounds < max_rounds && !terminal && !time_exceeded()) {
      bound_only_round_limit = effective_max_bound_only_rounds();

      PresolvePlanGpu plan;
      PresolveStatsGpu stats;
      auto plan_guard = make_scope_exit([&]() { free_plan(plan); });
      auto stats_guard = make_scope_exit([&]() { free_stats(stats); });
      allocate_plan(plan, current.lp);
      allocate_stats(stats,
                     current.lp.A.rows,
                     current.lp.A.cols,
                     true,
                     false);
      recompute_row_stats_from_csr(stats, current.lp);

      bool group_changed = false;
      bool structural_change = false;
      while (rounds < max_rounds && !terminal && !time_exceeded()) {
        plan.has_change = false;
        apply_rule_primal_propagation(plan, current.lp, stats, params);
        ++rounds;
        summary.has_infeasible = plan.has_infeasible;
        summary.has_unbounded = plan.has_unbounded;
        terminal = plan.has_infeasible || plan.has_unbounded;
        if (terminal || !plan.has_change) {
          break;
        }

        group_changed = true;
        saw_candidate_change = true;
        structural_change =
            plan.has_new_A || plan.has_row_action || plan.has_col_action;
        if (structural_change) {
          bound_only_rounds = 0;
          break;
        }

        ++bound_only_rounds;
        if (bound_only_rounds >= bound_only_round_limit) {
          stopped_on_bound_only_work_cap =
              bound_only_round_limit < configured_max_bound_only_rounds;
          break;
        }
      }

      bool commit_group = structural_change;
      if (group_changed && !terminal && !structural_change &&
          !time_exceeded()) {
        apply_rule_activity_checks(plan, current.lp, stats, params);
        summary.has_infeasible = plan.has_infeasible;
        summary.has_unbounded = plan.has_unbounded;
        terminal = plan.has_infeasible || plan.has_unbounded;
        post_activity_changed = !terminal && plan.has_row_action;
        commit_group = post_activity_changed;
        discarded_bound_only_candidate = !terminal && !commit_group;
      }

      if (group_changed && !terminal && commit_group) {
        apply_plan_to_working_lp(current, plan, &summary.record);
        invalidate_singleton_col_clean();
        committed_any = true;
        if (post_activity_changed && params.post_propagation_phase_barrier) {
          post_propagation_phase_barrier_hit = true;
        }
      }

      free_stats(stats);
      stats_guard.release();
      free_plan(plan);
      plan_guard.release();

      if (terminal || !group_changed || !structural_change ||
          bound_only_rounds >= bound_only_round_limit) {
        break;
      }
    }

    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(),
                          "profile primal propagation closure synchronize");
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - closure_start;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=primal_closure"
                << " candidate_changed="
                << (saw_candidate_change ? "true" : "false")
                << " committed=" << (committed_any ? "true" : "false")
                << " post_activity_changed="
                << (post_activity_changed ? "true" : "false")
                << " discarded_bound_only="
                << (discarded_bound_only_candidate ? "true" : "false")
                << " rounds=" << rounds
                << " bound_only_limit=" << bound_only_round_limit
                << " work_cap_hit="
                << (stopped_on_bound_only_work_cap ? "true" : "false")
                << " confirmation_round_guard="
                << (allow_wide_dense_confirmation_round ? "true" : "false")
                << " time=" << elapsed.count() << "s\n";
    }

    if (profile_rules && post_propagation_phase_barrier_hit) {
      std::cerr << ">>> [GPU Presolve C++ profile] phase=primal_closure_barrier"
                << " post_activity_changed=true\n";
    }
    return committed_any;
  };

  auto run_medium_phase = [&]() -> bool {
    bool changed_any = false;
    bool changed_prop = false;
    changed_prop = run_row_phase(false, false, true, false, false) || changed_prop;
    changed_prop =
        run_primal_propagation_with_activity_closure() || changed_prop;
    changed_any = changed_prop || changed_any;
    if (post_propagation_phase_barrier_hit) {
      return changed_any;
    }
    if (changed_prop) {
      changed_any = run_cleanup() || changed_any;
    }
    const bool changed_parallel_rows = run_row_phase(false, false, false, false, true);
    changed_any = changed_parallel_rows || changed_any;
    const bool changed_parallel_cols = run_col_phase(false, false, false, false, false, false, false, true, false);
    changed_any = changed_parallel_cols || changed_any;
    if (changed_parallel_rows || changed_parallel_cols) {
      changed_any = run_cleanup() || changed_any;
    }
    return changed_any;
  };

  auto run_fixed_row_sequence = [&]() -> bool {
    bool changed_any = false;
    changed_any = run_row_phase(true, false, false, false, false) || changed_any;
    changed_any = run_row_phase(false, true, false, false, false) || changed_any;
    changed_any = run_row_phase(false, false, true, false, false) || changed_any;
    const bool changed_primal = run_primal_propagation_with_activity_closure();
    changed_any = changed_primal || changed_any;
    if (post_propagation_phase_barrier_hit) {
      return changed_any;
    }
    if (changed_primal) {
      changed_any = run_trivial_cleanup_recirculation() || changed_any;
    }
    changed_any = run_row_phase(false, false, false, false, true) || changed_any;
    return changed_any;
  };

  auto run_cleanup_trigger_col_rule = [&](bool close_bounds,
                                          bool singleton_cols_dual,
                                          bool singleton_cols_eq,
                                          bool doubleton_eq,
                                          bool structural_l1,
                                          bool dual_fix,
                                          bool parallel_cols) -> bool {
    const bool changed = run_col_phase(close_bounds,
                                       false,
                                       singleton_cols_dual,
                                       singleton_cols_eq,
                                       doubleton_eq,
                                       structural_l1,
                                       dual_fix,
                                       parallel_cols,
                                       false);
    if (changed) {
      (void)run_trivial_cleanup_recirculation();
    }
    return changed;
  };

  auto run_fixed_col_sequence = [&]() -> bool {
    bool changed_any = false;
    changed_any = run_cleanup_trigger_col_rule(true, false, false, false, false, false, false) || changed_any;
    changed_any = run_cleanup_trigger_col_rule(false, false, false, false, true, false, false) || changed_any;
    changed_any = run_col_phase(false, true, false, false, false, false, false, false, false) || changed_any;

    bool changed_rule = false;
    do {
      changed_rule = run_cleanup_trigger_col_rule(false, true, false, false, false, false, false);
      changed_any = changed_rule || changed_any;
    } while (changed_rule && !terminal && !time_exceeded());

    do {
      changed_rule = run_cleanup_trigger_col_rule(false, false, true, false, false, false, false);
      changed_any = changed_rule || changed_any;
    } while (changed_rule && !terminal && !time_exceeded());

    do {
      changed_rule = run_cleanup_trigger_col_rule(false, false, false, true, false, false, false);
      changed_any = changed_rule || changed_any;
    } while (changed_rule && !params.doubleton_eq_single_batch_per_iter && !terminal && !time_exceeded());

    changed_any = run_cleanup_trigger_col_rule(false, false, false, false, false, true, false) || changed_any;
    changed_any = run_cleanup_trigger_col_rule(false, false, false, false, false, false, true) || changed_any;
    return changed_any;
  };

  if (!params.use_tiered_scheduler) {
    for (int iter = 0; iter < params.max_iters && !terminal && !time_exceeded(); ++iter) {
      bool changed_iter = false;
      changed_iter = run_fixed_row_sequence() || changed_iter;
      if (post_propagation_phase_barrier_hit) {
        ++summary.iterations;
        break;
      }
      changed_iter = run_fixed_col_sequence() || changed_iter;
      ++summary.iterations;
      if (!changed_iter) {
        break;
      }
    }
  } else {
    bool changed_bootstrap = false;
    if (params.enable_tiered_bootstrap && params.max_iters > 0 && !time_exceeded()) {
      const bool changed_structural_before =
          run_col_phase(false, false, false, false, false, true, false, false, false);
      changed_bootstrap = changed_structural_before || changed_bootstrap;
      if (changed_structural_before) {
        changed_bootstrap = run_cleanup() || changed_bootstrap;
      }
      bool changed_generic_bootstrap = false;
      changed_generic_bootstrap = run_row_phase(true, false, false, false, false) || changed_generic_bootstrap;
      changed_generic_bootstrap = run_row_phase(false, true, false, false, false) || changed_generic_bootstrap;
      changed_generic_bootstrap = run_row_phase(false, false, true, false, false) || changed_generic_bootstrap;
      changed_generic_bootstrap = run_row_phase(false, false, false, false, true) || changed_generic_bootstrap;

      bool changed_rule = run_col_phase(true, true, false, false, false, false, false, false, false);
      changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
      if (changed_rule) {
        changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
      }

      changed_generic_bootstrap = run_col_phase(false, true, false, false, false, false, false, false, false) ||
                                  changed_generic_bootstrap;

      do {
        changed_rule = run_col_phase(false, false, true, false, false, false, false, false, false);
        changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
        if (changed_rule) {
          changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
        }
      } while (changed_rule && !terminal && !time_exceeded());

      do {
        changed_rule = run_col_phase(false, false, false, true, false, false, false, false, false);
        changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
        if (changed_rule) {
          changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
        }
      } while (changed_rule && !terminal && !time_exceeded());

      changed_rule = run_col_phase(false, false, false, false, false, false, true, false, false);
      changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
      if (changed_rule) {
        changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
      }

      changed_rule = run_col_phase(false, false, false, false, false, false, false, true, false);
      changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
      if (changed_rule) {
        changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
      }
      changed_bootstrap = changed_generic_bootstrap || changed_bootstrap;
      if (changed_generic_bootstrap) {
        const bool changed_structural_after =
            run_col_phase(false, false, false, false, false, true, false, false, false);
        changed_bootstrap = changed_structural_after || changed_bootstrap;
        if (changed_structural_after) {
          changed_bootstrap = run_cleanup() || changed_bootstrap;
        }
      }
      ++summary.iterations;
    }

    // Folding GPUPresolver falls through to the staged tiered cycles when the
    // bootstrap pass makes no change; a changed bootstrap is handled like the
    // next tiered iteration here because the C++ port keeps bootstrap outside
    // the main loop.
    const bool run_tiered_cycles = true;
    if (run_tiered_cycles) {
      bool fast_phase = true;
      std::int32_t cycle_nnz_before = current.lp.A.nnz;
      constexpr double progress_ratio = 0.95;
      for (int iter = summary.iterations;
           iter < params.max_iters && !terminal && !time_exceeded();
           ++iter) {
        bool changed_iter = false;
        changed_iter = run_cleanup() || changed_iter;
        const std::int32_t nnz_before_phase = current.lp.A.nnz;
        if (fast_phase) {
          changed_iter = run_fast_phase() || changed_iter;
          const bool productive = _has_good_nnz_progress(nnz_before_phase, current.lp.A.nnz, progress_ratio);
          fast_phase = productive;
        } else {
          changed_iter = run_medium_phase() || changed_iter;
          if (post_propagation_phase_barrier_hit) {
            ++summary.iterations;
            break;
          }
          const bool productive = _has_good_nnz_progress(cycle_nnz_before, current.lp.A.nnz, progress_ratio);
          if (!changed_iter || !productive) {
            ++summary.iterations;
            break;
          }
          cycle_nnz_before = current.lp.A.nnz;
          fast_phase = true;
        }
        ++summary.iterations;
        if (!changed_iter && fast_phase) {
          break;
        }
      }
    }
  }

  if (!terminal && !time_exceeded() && params.enable_redundant_bounds) {
    (void)run_col_phase(false, false, false, false, false, false, false, false, true);
  }

  throw_if_cuda_error(cudaDeviceSynchronize(), "run_gpu_presolve_impl synchronize");
  summary.reduced_rows = current.lp.A.rows;
  summary.reduced_cols = current.lp.A.cols;
  summary.reduced_nnz = current.lp.A.nnz;
  summary.obj_constant_delta = current.lp.obj_constant - lp.obj_constant;
  summary.record.m1 = summary.reduced_rows;
  summary.record.n1 = summary.reduced_cols;
  summary.record.obj_constant_new = current.lp.obj_constant;
  rebuild_org2red(summary.record.row_org2red, summary.record.row_red2org, summary.record.m0);
  rebuild_org2red(summary.record.col_org2red, summary.record.col_red2org, summary.record.n0);
  const std::chrono::duration<double> presolve_elapsed =
      std::chrono::steady_clock::now() - presolve_start;
  summary.elapsed_seconds = presolve_elapsed.count();
  if (keep_reduced_lp) {
    summary.reduced_lp = current.lp;
    summary.owns_reduced_lp = current.owns;
    current.owns = false;
  } else {
    current.free_owned();
  }
  summary_guard.release();
  return summary;
}

}  // namespace

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, false);
}

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, true);
}

void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary) {
  if (!summary.owns_reduced_lp) {
    summary.reduced_lp = LPInfoGpu{};
    return;
  }
  cudaFree(summary.reduced_lp.A.rowPtr);
  cudaFree(summary.reduced_lp.A.colVal);
  cudaFree(summary.reduced_lp.A.nzVal);
  cudaFree(summary.reduced_lp.AT.rowPtr);
  cudaFree(summary.reduced_lp.AT.colVal);
  cudaFree(summary.reduced_lp.AT.nzVal);
  cudaFree(summary.reduced_lp.c);
  cudaFree(summary.reduced_lp.AL);
  cudaFree(summary.reduced_lp.AU);
  cudaFree(summary.reduced_lp.l);
  cudaFree(summary.reduced_lp.u);
  summary.reduced_lp = LPInfoGpu{};
  summary.owns_reduced_lp = false;
}

void free_gpu_presolve_record_resources(PresolveRecordGpu& record) {
  free_postsolve_tape_gpu(record.tape_gpu);
}

}  // namespace gpu_presolver::presolve
