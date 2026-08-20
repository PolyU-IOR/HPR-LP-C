#include "folding_internal.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::folding {
namespace {

constexpr int kFoldingThreads = 256;

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

template <class T>
T* copy_to_device(const std::vector<T>& values, const char* context) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  throw_if_cuda_error(cudaMalloc(&device, sizeof(T) * values.size()), context);
  throw_if_cuda_error(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
                      context);
  return device;
}

__global__ void kernel_count_color(std::int32_t* color_count,
                                   const std::int32_t* color_id,
                                   std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    atomicAdd(color_count + (color_id[i] - 1), 1);
  }
}

__global__ void kernel_write_scale_from_count(double* scale,
                                              const std::int32_t* color_id,
                                              const std::int32_t* color_count,
                                              std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    scale[i] = 1.0 / static_cast<double>(color_count[color_id[i] - 1]);
  }
}

__global__ void kernel_unfold_cols(double* x,
                                   double* z,
                                   const std::int32_t* col_color_id,
                                   const double* col_scale,
                                   const double* x_red,
                                   const double* z_red,
                                   std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    const std::int32_t color = col_color_id[j] - 1;
    if (x != nullptr && x_red != nullptr) {
      x[j] = x_red[color];
    }
    if (z != nullptr && z_red != nullptr) {
      z[j] = z_red[color] * col_scale[j];
    }
  }
}

__global__ void kernel_unfold_rows(double* y,
                                   const std::int32_t* row_color_id,
                                   const double* row_scale,
                                   const double* y_red,
                                   std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n && y != nullptr && y_red != nullptr) {
    y[i] = y_red[row_color_id[i] - 1] * row_scale[i];
  }
}

}  // namespace

FoldingMapDevice build_device_map(const FoldingWorkspace& workspace,
                                  FoldingMapProfile* profile) {
  const auto total_start = std::chrono::steady_clock::now();
  FoldingMapDevice map;
  map.original_rows = workspace.num_row;
  map.original_cols = workspace.num_col;
  map.reduced_rows = workspace.num_row_color;
  map.reduced_cols = workspace.num_col_color;
  map.owns_buffers = true;

  std::int32_t* row_count = nullptr;
  std::int32_t* col_count = nullptr;
  try {
    const auto copy_start = std::chrono::steady_clock::now();
    if (workspace.num_row > 0) {
      throw_if_cuda_error(
          cudaMalloc(&map.row_color_id, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_row)),
          "cudaMalloc map.row_color_id");
      throw_if_cuda_error(
          cudaMalloc(&map.row_scale, sizeof(double) * static_cast<std::size_t>(workspace.num_row)),
          "cudaMalloc map.row_scale");
      throw_if_cuda_error(cudaMemcpy(map.row_color_id,
                                     workspace.row_color_id,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_row),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy row_color_id D2D");
    }
    if (workspace.num_col > 0) {
      throw_if_cuda_error(
          cudaMalloc(&map.col_color_id, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_col)),
          "cudaMalloc map.col_color_id");
      throw_if_cuda_error(
          cudaMalloc(&map.col_scale, sizeof(double) * static_cast<std::size_t>(workspace.num_col)),
          "cudaMalloc map.col_scale");
      throw_if_cuda_error(cudaMemcpy(map.col_color_id,
                                     workspace.col_color_id,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_col),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy col_color_id D2D");
    }
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize build_device_map copy");
    if (profile != nullptr) {
      profile->copy_color_ids_seconds +=
          seconds_since(copy_start, std::chrono::steady_clock::now());
    }

    const auto count_start = std::chrono::steady_clock::now();
    if (workspace.num_row_color > 0) {
      throw_if_cuda_error(
          cudaMalloc(&row_count, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_row_color)),
          "cudaMalloc row_count");
      throw_if_cuda_error(
          cudaMemset(row_count, 0, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_row_color)),
          "cudaMemset row_count");
    }
    if (workspace.num_col_color > 0) {
      throw_if_cuda_error(
          cudaMalloc(&col_count, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_col_color)),
          "cudaMalloc col_count");
      throw_if_cuda_error(
          cudaMemset(col_count, 0, sizeof(std::int32_t) * static_cast<std::size_t>(workspace.num_col_color)),
          "cudaMemset col_count");
    }
    if (workspace.num_row > 0) {
      const int row_blocks = (workspace.num_row + kFoldingThreads - 1) / kFoldingThreads;
      kernel_count_color<<<row_blocks, kFoldingThreads>>>(row_count, map.row_color_id, workspace.num_row);
    }
    if (workspace.num_col > 0) {
      const int col_blocks = (workspace.num_col + kFoldingThreads - 1) / kFoldingThreads;
      kernel_count_color<<<col_blocks, kFoldingThreads>>>(col_count, map.col_color_id, workspace.num_col);
    }
    throw_if_cuda_error(cudaGetLastError(), "kernel_count_color build_device_map");
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize build_device_map count");
    if (profile != nullptr) {
      profile->count_color_seconds +=
          seconds_since(count_start, std::chrono::steady_clock::now());
    }

    const auto scale_start = std::chrono::steady_clock::now();
    if (workspace.num_row > 0) {
      const int row_blocks = (workspace.num_row + kFoldingThreads - 1) / kFoldingThreads;
      kernel_write_scale_from_count<<<row_blocks, kFoldingThreads>>>(
          map.row_scale, map.row_color_id, row_count, workspace.num_row);
    }
    if (workspace.num_col > 0) {
      const int col_blocks = (workspace.num_col + kFoldingThreads - 1) / kFoldingThreads;
      kernel_write_scale_from_count<<<col_blocks, kFoldingThreads>>>(
          map.col_scale, map.col_color_id, col_count, workspace.num_col);
    }
    throw_if_cuda_error(cudaGetLastError(), "kernel_write_scale_from_count");
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize build_device_map scale");
    if (profile != nullptr) {
      profile->scale_seconds +=
          seconds_since(scale_start, std::chrono::steady_clock::now());
      profile->total_seconds =
          seconds_since(total_start, std::chrono::steady_clock::now());
    }
  } catch (...) {
    cudaFree(row_count);
    cudaFree(col_count);
    free_folding_map_device(map);
    throw;
  }

  cudaFree(row_count);
  cudaFree(col_count);
  return map;
}

FoldingMapHost copy_map_to_host(const FoldingMapDevice& map) {
  FoldingMapHost host;
  host.original_rows = map.original_rows;
  host.original_cols = map.original_cols;
  host.reduced_rows = map.reduced_rows;
  host.reduced_cols = map.reduced_cols;
  host.row_color_id = copy_to_host(map.row_color_id,
                                   static_cast<std::size_t>(map.original_rows),
                                   "cudaMemcpy map.row_color_id");
  host.col_color_id = copy_to_host(map.col_color_id,
                                   static_cast<std::size_t>(map.original_cols),
                                   "cudaMemcpy map.col_color_id");
  host.row_scale = copy_to_host(map.row_scale,
                                static_cast<std::size_t>(map.original_rows),
                                "cudaMemcpy map.row_scale");
  host.col_scale = copy_to_host(map.col_scale,
                                static_cast<std::size_t>(map.original_cols),
                                "cudaMemcpy map.col_scale");
  return host;
}

void free_folding_map_device(FoldingMapDevice& map) {
  if (!map.owns_buffers) {
    map = FoldingMapDevice{};
    return;
  }
  cudaFree(map.row_color_id);
  cudaFree(map.col_color_id);
  cudaFree(map.row_scale);
  cudaFree(map.col_scale);
  map = FoldingMapDevice{};
}

std::vector<double> fold_initial_vector(const std::vector<double>* value,
                                        const std::vector<std::int32_t>& color_index,
                                        std::int32_t reduced_length) {
  if (value == nullptr) {
    return {};
  }
  std::vector<double> folded(static_cast<std::size_t>(reduced_length), 0.0);
  std::vector<bool> seen(static_cast<std::size_t>(reduced_length), false);
  for (std::size_t idx = 0; idx < color_index.size(); ++idx) {
    const std::int32_t color = color_index[idx] - 1;
    if (!seen[static_cast<std::size_t>(color)]) {
      folded[static_cast<std::size_t>(color)] = (*value)[idx];
      seen[static_cast<std::size_t>(color)] = true;
    }
  }
  return folded;
}

void unfold_solution(const FoldingMapHost& map,
                     const std::vector<double>& x_red,
                     const std::vector<double>& y_red,
                     const std::vector<double>& z_red,
                     std::vector<double>* x,
                     std::vector<double>* y,
                     std::vector<double>* z) {
  x->assign(static_cast<std::size_t>(map.original_cols), 0.0);
  y->assign(static_cast<std::size_t>(map.original_rows), 0.0);
  z->assign(static_cast<std::size_t>(map.original_cols), 0.0);
  for (std::int32_t j = 0; j < map.original_cols; ++j) {
    const std::int32_t color = map.col_color_id[static_cast<std::size_t>(j)] - 1;
    (*x)[static_cast<std::size_t>(j)] = x_red[static_cast<std::size_t>(color)];
    (*z)[static_cast<std::size_t>(j)] =
        z_red[static_cast<std::size_t>(color)] * map.col_scale[static_cast<std::size_t>(j)];
  }
  for (std::int32_t i = 0; i < map.original_rows; ++i) {
    const std::int32_t color = map.row_color_id[static_cast<std::size_t>(i)] - 1;
    (*y)[static_cast<std::size_t>(i)] =
        y_red[static_cast<std::size_t>(color)] * map.row_scale[static_cast<std::size_t>(i)];
  }
}

UnfoldedSolutionHost unfold_solution_device_to_host(const FoldingMapDevice& map,
                                                    const double* x_red_device,
                                                    const double* y_red_device,
                                                    const double* z_red_device) {
  UnfoldedSolutionHost unfolded;
  double* x_device = nullptr;
  double* y_device = nullptr;
  double* z_device = nullptr;
  try {
    if (map.original_cols > 0 && x_red_device != nullptr) {
      throw_if_cuda_error(cudaMalloc(&x_device, sizeof(double) * static_cast<std::size_t>(map.original_cols)),
                          "cudaMalloc unfolded x");
    }
    if (map.original_rows > 0 && y_red_device != nullptr) {
      throw_if_cuda_error(cudaMalloc(&y_device, sizeof(double) * static_cast<std::size_t>(map.original_rows)),
                          "cudaMalloc unfolded y");
    }
    if (map.original_cols > 0 && z_red_device != nullptr) {
      throw_if_cuda_error(cudaMalloc(&z_device, sizeof(double) * static_cast<std::size_t>(map.original_cols)),
                          "cudaMalloc unfolded z");
    }

    if (map.original_cols > 0 && (x_device != nullptr || z_device != nullptr)) {
      const int col_blocks = (map.original_cols + kFoldingThreads - 1) / kFoldingThreads;
      kernel_unfold_cols<<<col_blocks, kFoldingThreads>>>(x_device,
                                                          z_device,
                                                          map.col_color_id,
                                                          map.col_scale,
                                                          x_red_device,
                                                          z_red_device,
                                                          map.original_cols);
    }
    if (map.original_rows > 0 && y_device != nullptr) {
      const int row_blocks = (map.original_rows + kFoldingThreads - 1) / kFoldingThreads;
      kernel_unfold_rows<<<row_blocks, kFoldingThreads>>>(y_device,
                                                          map.row_color_id,
                                                          map.row_scale,
                                                          y_red_device,
                                                          map.original_rows);
    }
    throw_if_cuda_error(cudaGetLastError(), "kernel_unfold_solution");
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize unfold_solution");

    unfolded.x = copy_to_host(x_device,
                              static_cast<std::size_t>(map.original_cols),
                              "cudaMemcpy unfolded x");
    unfolded.y = copy_to_host(y_device,
                              static_cast<std::size_t>(map.original_rows),
                              "cudaMemcpy unfolded y");
    unfolded.z = copy_to_host(z_device,
                              static_cast<std::size_t>(map.original_cols),
                              "cudaMemcpy unfolded z");
  } catch (...) {
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(z_device);
    throw;
  }
  cudaFree(x_device);
  cudaFree(y_device);
  cudaFree(z_device);
  return unfolded;
}

UnfoldedSolutionHost unfold_solution_to_host(const FoldingMapDevice& map,
                                             const std::vector<double>& x_red,
                                             const std::vector<double>& y_red,
                                             const std::vector<double>& z_red) {
  if (static_cast<std::int32_t>(x_red.size()) != map.reduced_cols ||
      static_cast<std::int32_t>(z_red.size()) != map.reduced_cols ||
      static_cast<std::int32_t>(y_red.size()) != map.reduced_rows) {
    throw std::invalid_argument("unfold_solution_to_host reduced dimensions do not match folding map");
  }

  double* x_red_device = nullptr;
  double* y_red_device = nullptr;
  double* z_red_device = nullptr;
  try {
    x_red_device = copy_to_device(x_red, "cudaMalloc x_red_device");
    y_red_device = copy_to_device(y_red, "cudaMalloc y_red_device");
    z_red_device = copy_to_device(z_red, "cudaMalloc z_red_device");
    UnfoldedSolutionHost unfolded =
        unfold_solution_device_to_host(map, x_red_device, y_red_device, z_red_device);
    cudaFree(x_red_device);
    cudaFree(y_red_device);
    cudaFree(z_red_device);
    return unfolded;
  } catch (...) {
    cudaFree(x_red_device);
    cudaFree(y_red_device);
    cudaFree(z_red_device);
    throw;
  }
}

void free_lp_device(presolve::LPInfoGpu& lp) {
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
  lp = presolve::LPInfoGpu{};
}

}  // namespace gpu_presolver::folding
