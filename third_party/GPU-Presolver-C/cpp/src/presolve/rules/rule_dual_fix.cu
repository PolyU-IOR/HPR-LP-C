#include "gpu_presolver/presolve/rules/rule_dual_fix.hpp"

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

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  throw_if_cuda_error(cudaMalloc(&temp_storage, temp_bytes), context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  cudaFree(temp_storage);
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

void append_finite_fixed_col_tape_from_device(PostsolveTape& tape,
                                              const std::uint8_t* fixed_mask,
                                              const std::uint8_t* infinite_fix_mask,
                                              const double* fixed_val,
                                              const std::uint8_t* keep_row,
                                              const double* c,
                                              const DeviceCsrMatrix& AT,
                                              std::int32_t m,
                                              std::int32_t n,
                                              const char* context) {
  std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_infinite_fix_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_keep_row(static_cast<std::size_t>(m));
  std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
  std::vector<double> host_c(static_cast<std::size_t>(n));
  std::vector<std::int32_t> host_at_row_ptr(static_cast<std::size_t>(n + 1));
  std::vector<std::int32_t> host_at_col_val(static_cast<std::size_t>(AT.nnz));
  std::vector<double> host_at_nz_val(static_cast<std::size_t>(AT.nnz));

  throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask, static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_infinite_fix_mask.data(), infinite_fix_mask,
                                 static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (m > 0) {
    throw_if_cuda_error(cudaMemcpy(host_keep_row.data(), keep_row, static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                 sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_c.data(), c, sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_at_row_ptr.data(), AT.rowPtr,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(n + 1),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (AT.nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_at_col_val.data(), AT.colVal,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(AT.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_at_nz_val.data(), AT.nzVal,
                                   sizeof(double) * static_cast<std::size_t>(AT.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }

  for (std::int32_t col = 0; col < n; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    if (host_fixed_mask[col_idx] == std::uint8_t{0} ||
        host_infinite_fix_mask[col_idx] != std::uint8_t{0}) {
      continue;
    }
    std::vector<std::int32_t> indices{col};
    std::vector<double> vals{host_fixed_val[col_idx], host_c[col_idx]};
    for (std::int32_t p = host_at_row_ptr[col_idx]; p < host_at_row_ptr[col_idx + 1]; ++p) {
      const std::int32_t row = host_at_col_val[static_cast<std::size_t>(p)];
      if (host_keep_row[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
        continue;
      }
      indices.push_back(row);
      vals.push_back(host_at_nz_val[static_cast<std::size_t>(p)]);
    }
    append_postsolve_record(
        tape, PostsolveReductionType::FixedCol, indices, vals, PostsolveDualMode::Minimal);
  }
}

__global__ void _kernel_dual_fix_finite_tape_counts(std::int32_t* selected_scan,
                                                    std::int32_t* active_nnz_scan,
                                                    const std::uint8_t* fixed_mask,
                                                    const std::uint8_t* infinite_fix_mask,
                                                    const std::uint8_t* keep_row,
                                                    const std::int32_t* at_row_ptr,
                                                    const std::int32_t* at_col_val,
                                                    std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    const bool selected =
        fixed_mask[col] != std::uint8_t{0} && infinite_fix_mask[col] == std::uint8_t{0};
    selected_scan[col] = selected ? 1 : 0;
    std::int32_t active_nnz = 0;
    if (selected) {
      for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
        const std::int32_t row = at_col_val[p];
        if (keep_row[row] != std::uint8_t{0}) {
          ++active_nnz;
        }
      }
    }
    active_nnz_scan[col] = active_nnz;
  }
}

__global__ void _kernel_dual_fix_pack_finite_tape(std::int32_t* packed_cols,
                                                  double* packed_fixed_vals,
                                                  double* packed_c,
                                                  std::int32_t* packed_nnz_starts,
                                                  std::int32_t* packed_rows,
                                                  double* packed_coeffs,
                                                  const std::int32_t* selected_scan,
                                                  const std::int32_t* active_nnz_scan,
                                                  const std::uint8_t* fixed_mask,
                                                  const std::uint8_t* infinite_fix_mask,
                                                  const double* fixed_val,
                                                  const std::uint8_t* keep_row,
                                                  const double* c,
                                                  const std::int32_t* at_row_ptr,
                                                  const std::int32_t* at_col_val,
                                                  const double* at_nz_val,
                                                  std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n ||
      fixed_mask[col] == std::uint8_t{0} ||
      infinite_fix_mask[col] != std::uint8_t{0}) {
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
    if (out < nnz_stop) {
      packed_rows[out] = row;
      packed_coeffs[out] = at_nz_val[p];
      ++out;
    }
  }
}

void append_finite_fixed_col_tape_compacted_from_device(PostsolveTape& tape,
                                                        const std::uint8_t* fixed_mask,
                                                        const std::uint8_t* infinite_fix_mask,
                                                        const double* fixed_val,
                                                        const std::uint8_t* keep_row,
                                                        const double* c,
                                                        const DeviceCsrMatrix& AT,
                                                        std::int32_t n,
                                                        const char* context) {
  if (n <= 0) {
    return;
  }

  std::int32_t* selected_scan = nullptr;
  std::int32_t* active_nnz_scan = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  throw_if_cuda_error(cudaMalloc(&active_nnz_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_dual_fix_finite_tape_counts<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      selected_scan,
      active_nnz_scan,
      fixed_mask,
      infinite_fix_mask,
      keep_row,
      AT.rowPtr,
      AT.colVal,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_finite_tape_counts");
  inclusive_scan_i32(selected_scan, n, context);
  inclusive_scan_i32(active_nnz_scan, n, context);

  std::int32_t counts[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(&counts[0], selected_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(&counts[1], active_nnz_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  const std::int32_t finite_count = counts[0];
  const std::int32_t active_nnz = counts[1];
  if (finite_count <= 0) {
    cudaFree(selected_scan);
    cudaFree(active_nnz_scan);
    return;
  }

  std::int32_t* packed_cols = nullptr;
  double* packed_fixed_vals = nullptr;
  double* packed_c = nullptr;
  std::int32_t* packed_nnz_starts = nullptr;
  std::int32_t* packed_rows = nullptr;
  double* packed_coeffs = nullptr;
  throw_if_cuda_error(cudaMalloc(&packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(finite_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_fixed_vals,
                                 sizeof(double) * static_cast<std::size_t>(finite_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_c,
                                 sizeof(double) * static_cast<std::size_t>(finite_count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_nnz_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(finite_count + 1)),
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

  _kernel_dual_fix_pack_finite_tape<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      packed_cols,
      packed_fixed_vals,
      packed_c,
      packed_nnz_starts,
      packed_rows,
      packed_coeffs,
      selected_scan,
      active_nnz_scan,
      fixed_mask,
      infinite_fix_mask,
      fixed_val,
      keep_row,
      c,
      AT.rowPtr,
      AT.colVal,
      AT.nzVal,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_pack_finite_tape");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(finite_count));
  std::vector<double> host_fixed_vals(static_cast<std::size_t>(finite_count));
  std::vector<double> host_c(static_cast<std::size_t>(finite_count));
  std::vector<std::int32_t> host_nnz_starts(static_cast<std::size_t>(finite_count + 1));
  std::vector<std::int32_t> host_rows(static_cast<std::size_t>(active_nnz));
  std::vector<double> host_coeffs(static_cast<std::size_t>(active_nnz));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(finite_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_fixed_vals.data(), packed_fixed_vals,
                                 sizeof(double) * static_cast<std::size_t>(finite_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_c.data(), packed_c,
                                 sizeof(double) * static_cast<std::size_t>(finite_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_nnz_starts.data(), packed_nnz_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(finite_count + 1),
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

  for (std::int32_t rec = 0; rec < finite_count; ++rec) {
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
        tape, PostsolveReductionType::FixedCol, indices, vals, PostsolveDualMode::Minimal);
  }

  cudaFree(selected_scan);
  cudaFree(active_nnz_scan);
  cudaFree(packed_cols);
  cudaFree(packed_fixed_vals);
  cudaFree(packed_c);
  cudaFree(packed_nnz_starts);
  if (packed_rows != nullptr) {
    cudaFree(packed_rows);
  }
  if (packed_coeffs != nullptr) {
    cudaFree(packed_coeffs);
  }
}

void append_infinite_fixed_col_tape_from_device(PostsolveTape& tape,
                                                const std::uint8_t* fixed_mask,
                                                const std::uint8_t* infinite_fix_mask,
                                                const double* fixed_val,
                                                const std::uint8_t* keep_row,
                                                const double* l,
                                                const double* u,
                                                const double* AL,
                                                const double* AU,
                                                const DeviceCsrMatrix& A,
                                                const DeviceCsrMatrix& AT,
                                                std::int32_t m,
                                                std::int32_t n,
                                                const char* context) {
  std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_infinite_fix_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_keep_row(static_cast<std::size_t>(m));

  throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask, static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_infinite_fix_mask.data(), infinite_fix_mask,
                                 static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  bool has_infinite_fixed_col = false;
  for (std::int32_t col = 0; col < n; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    if (host_fixed_mask[col_idx] != std::uint8_t{0} &&
        host_infinite_fix_mask[col_idx] != std::uint8_t{0}) {
      has_infinite_fixed_col = true;
      break;
    }
  }
  if (!has_infinite_fixed_col) {
    return;
  }

  std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
  std::vector<double> host_l(static_cast<std::size_t>(n));
  std::vector<double> host_u(static_cast<std::size_t>(n));
  std::vector<double> host_AL(static_cast<std::size_t>(m));
  std::vector<double> host_AU(static_cast<std::size_t>(m));
  std::vector<std::int32_t> host_a_row_ptr(static_cast<std::size_t>(m + 1));
  std::vector<std::int32_t> host_a_col_val(static_cast<std::size_t>(A.nnz));
  std::vector<double> host_a_nz_val(static_cast<std::size_t>(A.nnz));
  std::vector<std::int32_t> host_at_row_ptr(static_cast<std::size_t>(n + 1));
  std::vector<std::int32_t> host_at_col_val(static_cast<std::size_t>(AT.nnz));

  throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                 sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_l.data(), l, sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_u.data(), u, sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (m > 0) {
    throw_if_cuda_error(cudaMemcpy(host_keep_row.data(), keep_row, static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_AL.data(), AL, sizeof(double) * static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_AU.data(), AU, sizeof(double) * static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_a_row_ptr.data(), A.rowPtr,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(m + 1),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  if (A.nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_a_col_val.data(), A.colVal,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(A.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_a_nz_val.data(), A.nzVal,
                                   sizeof(double) * static_cast<std::size_t>(A.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  throw_if_cuda_error(cudaMemcpy(host_at_row_ptr.data(), AT.rowPtr,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(n + 1),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (AT.nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_at_col_val.data(), AT.colVal,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(AT.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }

  for (std::int32_t col = 0; col < n; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    if (host_fixed_mask[col_idx] == std::uint8_t{0} ||
        host_infinite_fix_mask[col_idx] == std::uint8_t{0}) {
      continue;
    }
    const bool fix_to_pos_inf = host_fixed_val[col_idx] > 0.0;
    std::vector<std::int32_t> indices{fix_to_pos_inf ? 1 : -1, col};
    std::vector<double> vals{0.0, fix_to_pos_inf ? host_l[col_idx] : host_u[col_idx]};
    std::int32_t active_rows = 0;
    for (std::int32_t p = host_at_row_ptr[col_idx]; p < host_at_row_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
      const std::int32_t row = host_at_col_val[static_cast<std::size_t>(p)];
      if (host_keep_row[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
        continue;
      }
      ++active_rows;
      const std::int32_t row_start = host_a_row_ptr[static_cast<std::size_t>(row)];
      const std::int32_t row_stop = host_a_row_ptr[static_cast<std::size_t>(row + 1)];
      const std::int32_t row_len = row_stop - row_start;
      indices.push_back(row_len);
      vals.push_back(std::isfinite(host_AL[static_cast<std::size_t>(row)])
                         ? host_AL[static_cast<std::size_t>(row)]
                         : host_AU[static_cast<std::size_t>(row)]);
      for (std::int32_t q = row_start; q < row_stop; ++q) {
        indices.push_back(host_a_col_val[static_cast<std::size_t>(q)]);
        vals.push_back(host_a_nz_val[static_cast<std::size_t>(q)]);
      }
    }
    vals[0] = static_cast<double>(active_rows);
    append_postsolve_record(
        tape, PostsolveReductionType::FixedColInf, indices, vals, PostsolveDualMode::Exact);
  }
}

__global__ void _kernel_dual_fix_candidates(std::int32_t* status_flag,
                                            std::uint8_t* fixed_mask,
                                            std::uint8_t* infinite_fix_mask,
                                            double* fixed_val,
                                            double* obj_contrib,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const double* c,
                                            const double* l,
                                            const double* u,
                                            const double* AL,
                                            const double* AU,
                                            const std::int32_t* at_row_ptr,
                                            const std::int32_t* at_col_val,
                                            const double* at_nz_val,
                                            double zero_tol,
                                            double bound_tol,
                                            std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    fixed_mask[j] = std::uint8_t{0};
    infinite_fix_mask[j] = std::uint8_t{0};
    fixed_val[j] = 0.0;
    obj_contrib[j] = 0.0;
    if (keep_col[j] == std::uint8_t{0}) {
      return;
    }

    const double lj = l[j];
    const double uj = u[j];
    if (lj > uj + bound_tol) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    bool has_down_lock = false;
    bool has_up_lock = false;
    const std::int32_t p_start = at_row_ptr[j];
    const std::int32_t p_stop = at_row_ptr[j + 1];
    for (std::int32_t p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] == std::uint8_t{0}) {
        continue;
      }

      const double aij = at_nz_val[p];
      if (aij > zero_tol) {
        has_down_lock = has_down_lock || isfinite(AL[row]);
        has_up_lock = has_up_lock || isfinite(AU[row]);
      } else if (aij < -zero_tol) {
        has_down_lock = has_down_lock || isfinite(AU[row]);
        has_up_lock = has_up_lock || isfinite(AL[row]);
      }

      if (has_down_lock && has_up_lock) {
        break;
      }
    }

    const double cj = c[j];
    bool fixed = false;
    bool infinite_fix = false;
    double vj = 0.0;

    if (cj > zero_tol) {
      if (!has_down_lock) {
        if (isfinite(lj)) {
          vj = lj;
          fixed = true;
        } else {
          atomicMax(&status_flag[0], 2);
          return;
        }
      }
    } else if (cj < -zero_tol) {
      if (!has_up_lock) {
        if (isfinite(uj)) {
          vj = uj;
          fixed = true;
        } else {
          atomicMax(&status_flag[0], 2);
          return;
        }
      }
    } else if (fabs(cj) <= zero_tol) {
      if (!has_down_lock) {
        fixed = true;
        if (isfinite(lj)) {
          vj = lj;
        } else {
          vj = -INFINITY;
          infinite_fix = true;
        }
      } else if (!has_up_lock) {
        fixed = true;
        if (isfinite(uj)) {
          vj = uj;
        } else {
          vj = INFINITY;
          infinite_fix = true;
        }
      }
    }

    if (fixed) {
      fixed_mask[j] = std::uint8_t{1};
      infinite_fix_mask[j] = infinite_fix ? std::uint8_t{1} : std::uint8_t{0};
      fixed_val[j] = vj;
      obj_contrib[j] = infinite_fix ? 0.0 : (cj * vj);
      atomicMax(&status_flag[1], 1);
    }
  }
}

__global__ void _kernel_dual_fix_row_shift(double* row_shift,
                                           const std::uint8_t* fixed_mask,
                                           const std::uint8_t* infinite_fix_mask,
                                           const double* fixed_val,
                                           const std::uint8_t* keep_row,
                                           const std::int32_t* at_row_ptr,
                                           const std::int32_t* at_col_val,
                                           const double* at_nz_val,
                                           std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0} && infinite_fix_mask[j] == std::uint8_t{0}) {
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

__global__ void _kernel_dual_fix_row_remove(std::uint8_t* row_remove,
                                            std::int32_t* row_action_flag,
                                            const std::uint8_t* fixed_mask,
                                            const std::uint8_t* infinite_fix_mask,
                                            const std::uint8_t* keep_row,
                                            const std::int32_t* at_row_ptr,
                                            const std::int32_t* at_col_val,
                                            std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0} && infinite_fix_mask[j] != std::uint8_t{0}) {
    const std::int32_t p_start = at_row_ptr[j];
    const std::int32_t p_stop = at_row_ptr[j + 1];
    for (std::int32_t p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] != std::uint8_t{0}) {
        row_remove[row] = std::uint8_t{1};
        atomicMax(row_action_flag, 1);
      }
    }
  }
}

__global__ void _kernel_dual_fix_keep_row_after(std::uint8_t* keep_row_after,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* row_remove,
                                                std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m) {
    keep_row_after[row] =
        (keep_row[row] != std::uint8_t{0} && row_remove[row] == std::uint8_t{0})
            ? std::uint8_t{1}
            : std::uint8_t{0};
  }
}

__global__ void _kernel_dual_fix_apply(std::uint8_t* keep_col,
                                       std::uint8_t* keep_row,
                                       double* AL,
                                       double* AU,
                                       double* new_l,
                                       double* new_u,
                                       const std::uint8_t* fixed_mask,
                                       const std::uint8_t* row_remove,
                                       const double* row_shift,
                                       const double* fixed_val,
                                       std::int32_t m,
                                       std::int32_t n) {
  const std::int32_t k = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n && fixed_mask[k] != std::uint8_t{0}) {
    new_l[k] = fixed_val[k];
    new_u[k] = fixed_val[k];
    keep_col[k] = std::uint8_t{0};
  }
  if (k < m) {
    AL[k] -= row_shift[k];
    AU[k] -= row_shift[k];
    if (row_remove[k] != std::uint8_t{0}) {
      keep_row[k] = std::uint8_t{0};
    }
  }
}

}  // namespace

void apply_rule_dual_fix(PresolvePlanGpu& plan,
                         const LPInfoGpu& lp,
                         const PresolveStatsGpu& stats,
                         const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t n = lp.A.cols;
  const std::int32_t m = lp.A.rows;
  if (n == 0) {
    return;
  }

  std::int32_t* status_flag = nullptr;
  std::uint8_t* fixed_mask = nullptr;
  std::uint8_t* infinite_fix_mask = nullptr;
  double* fixed_val = nullptr;
  double* obj_contrib = nullptr;
  double* row_shift = nullptr;
  std::uint8_t* row_remove = nullptr;
  std::int32_t* row_action_flag = nullptr;
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t) * 2), "cudaMalloc dual_fix status_flag");
  throw_if_cuda_error(cudaMalloc(&fixed_mask, static_cast<std::size_t>(n)), "cudaMalloc dual_fix fixed_mask");
  throw_if_cuda_error(cudaMalloc(&infinite_fix_mask, static_cast<std::size_t>(n)), "cudaMalloc dual_fix infinite_fix_mask");
  throw_if_cuda_error(cudaMalloc(&fixed_val, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc dual_fix fixed_val");
  throw_if_cuda_error(cudaMalloc(&obj_contrib, sizeof(double) * static_cast<std::size_t>(n)), "cudaMalloc dual_fix obj_contrib");
  throw_if_cuda_error(cudaMalloc(&row_shift, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc dual_fix row_shift");
  throw_if_cuda_error(cudaMalloc(&row_remove, static_cast<std::size_t>(m)), "cudaMalloc dual_fix row_remove");
  throw_if_cuda_error(cudaMalloc(&row_action_flag, sizeof(std::int32_t)), "cudaMalloc dual_fix row_action_flag");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 2), "cudaMemset dual_fix status_flag");
  throw_if_cuda_error(cudaMemset(row_action_flag, 0, sizeof(std::int32_t)), "cudaMemset dual_fix row_action_flag");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset dual_fix row_shift");
  throw_if_cuda_error(cudaMemset(row_remove, 0, static_cast<std::size_t>(m)), "cudaMemset dual_fix row_remove");

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_dual_fix_candidates<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      fixed_mask,
      infinite_fix_mask,
      fixed_val,
      obj_contrib,
      plan.keep_col_mask,
      plan.keep_row_mask,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      plan.new_AL,
      plan.new_AU,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      pparams.bound_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_candidates");

  std::int32_t status[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy dual_fix status");
  if (status[0] == 1) {
    plan.has_infeasible = true;
  } else if (status[0] == 2) {
    plan.has_unbounded = true;
  } else if (status[1] != 0) {
    _kernel_dual_fix_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        row_shift,
        fixed_mask,
        infinite_fix_mask,
        fixed_val,
        plan.keep_row_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_row_shift");
    _kernel_dual_fix_row_remove<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        row_remove,
        row_action_flag,
        fixed_mask,
        infinite_fix_mask,
        plan.keep_row_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_row_remove");
    std::int32_t row_action = 0;
    throw_if_cuda_error(cudaMemcpy(&row_action, row_action_flag, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy dual_fix row_action_flag");
    const bool row_changed = row_action != 0;
    if (pparams.record_postsolve_tape) {
      append_infinite_fixed_col_tape_from_device(plan.tape,
                                                 fixed_mask,
                                                 infinite_fix_mask,
                                                 fixed_val,
                                                 plan.keep_row_mask,
                                                 plan.new_l,
                                                 plan.new_u,
                                                 plan.new_AL,
                                                 plan.new_AU,
                                                 lp.A,
                                                 lp.AT,
                                                 m,
                                                 n,
                                                 "cudaMemcpy dual_fix fixed-col-inf tape");
      std::uint8_t* keep_row_after_for_tape = plan.keep_row_mask;
      if (m > 0) {
        throw_if_cuda_error(cudaMalloc(&keep_row_after_for_tape, static_cast<std::size_t>(m)),
                            "cudaMalloc dual_fix keep_row_after tape");
        const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
        _kernel_dual_fix_keep_row_after<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
            keep_row_after_for_tape, plan.keep_row_mask, row_remove, m);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_keep_row_after");
      }
      if (pparams.record_postsolve_tape_cpu) {
        append_finite_fixed_col_tape_compacted_from_device(plan.tape,
                                                           fixed_mask,
                                                           infinite_fix_mask,
                                                           fixed_val,
                                                           keep_row_after_for_tape,
                                                           plan.new_c,
                                                           lp.AT,
                                                           n,
                                                           "cudaMemcpy dual_fix fixed-col compact tape");
      }
      if (m > 0) {
        cudaFree(keep_row_after_for_tape);
      }
    }
    const std::int32_t max_mn = m > n ? m : n;
    const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_dual_fix_apply<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        fixed_mask,
        row_remove,
        row_shift,
        fixed_val,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_dual_fix synchronize");
    const double obj_delta = sum_device_double(obj_contrib, n, "cudaMemcpy dual_fix objective delta");
    plan.obj_constant_delta += obj_delta;
    plan.has_row_action = plan.has_row_action || row_changed;
    plan.has_col_action = true;
    plan.has_change = true;
  }

  cudaFree(status_flag);
  cudaFree(fixed_mask);
  cudaFree(infinite_fix_mask);
  cudaFree(fixed_val);
  cudaFree(obj_contrib);
  cudaFree(row_shift);
  cudaFree(row_remove);
  cudaFree(row_action_flag);
}

}  // namespace gpu_presolver::presolve
