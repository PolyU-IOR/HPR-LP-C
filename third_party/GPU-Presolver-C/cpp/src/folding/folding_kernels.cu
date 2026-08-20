#include "folding_internal.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::folding {
namespace {

constexpr int kFoldingThreads = 256;
constexpr double kTwoPi = 6.283185307179586476925286766559;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

std::uint64_t mix64_host(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ std::uint64_t mix64_device(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ double unit_from_mix(std::uint64_t x) {
  const std::uint64_t mixed = mix64_device(x);
  const double unit = static_cast<double>(mixed >> 11) * 0x1.0p-53;
  return unit > 0.0 ? unit : 0x1.0p-53;
}

std::uint64_t color_seed(std::int32_t round, std::uint64_t side) {
  return mix64_host(static_cast<std::uint64_t>(round) ^ (side * 0x9e3779b97f4a7c15ULL));
}

struct ColorPairLess {
  __host__ __device__ bool operator()(const ColorPair& lhs, const ColorPair& rhs) const {
    if (lhs.color_id != rhs.color_id) {
      return lhs.color_id < rhs.color_id;
    }
    return lhs.color_sig < rhs.color_sig;
  }
};

__global__ void kernel_pack_color(ColorPair* color,
                                  const std::int32_t* color_id,
                                  const double* color_sig,
                                  std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    color[i].color_id = color_id[i];
    color[i].color_sig = static_cast<float>(color_sig[i]);
  }
}

__global__ void kernel_mark_color_start(std::int32_t* color_start,
                                        const ColorPair* color,
                                        double tol,
                                        std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    if (i == 0) {
      color_start[i] = 1;
    } else {
      const bool changed_id = color[i].color_id != color[i - 1].color_id;
      const bool changed_sig = static_cast<double>(color[i].color_sig) >
                               static_cast<double>(color[i - 1].color_sig) + tol;
      color_start[i] = (changed_id || changed_sig) ? 1 : 0;
    }
  }
}

__global__ void kernel_write_color_id(std::int32_t* color_id,
                                      const std::int32_t* perm,
                                      const std::int32_t* prefix,
                                      std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    color_id[perm[i]] = prefix[i];
  }
}

__global__ void kernel_write_color_sig(double* color_sig,
                                       const std::int32_t* color_id,
                                       std::uint64_t seed,
                                       std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    const std::uint64_t color = static_cast<std::uint64_t>(color_id[i]);
    const double u1 = unit_from_mix(seed ^ (color * 0xbf58476d1ce4e5b9ULL));
    const double u2 = unit_from_mix((seed + 0x9e3779b97f4a7c15ULL) ^
                                    (color * 0x94d049bb133111ebULL));
    color_sig[i] = sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
  }
}

std::int32_t update_color_id(std::int32_t* color_id,
                             std::int32_t* color_start,
                             std::int32_t* perm,
                             ColorPair* color,
                             const double* color_sig,
                             double tol,
                             std::int32_t n) {
  if (n <= 0) {
    return 0;
  }
  const int blocks = (n + kFoldingThreads - 1) / kFoldingThreads;
  kernel_pack_color<<<blocks, kFoldingThreads>>>(color, color_id, color_sig, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_pack_color");

  thrust::device_ptr<ColorPair> color_ptr(color);
  thrust::device_ptr<std::int32_t> perm_ptr(perm);
  thrust::sequence(perm_ptr, perm_ptr + n);
  thrust::sort_by_key(color_ptr, color_ptr + n, perm_ptr, ColorPairLess{});

  kernel_mark_color_start<<<blocks, kFoldingThreads>>>(color_start, color, tol, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_mark_color_start");
  thrust::inclusive_scan(thrust::device_pointer_cast(color_start),
                         thrust::device_pointer_cast(color_start + n),
                         thrust::device_pointer_cast(color_start));

  std::int32_t new_color_num = 0;
  throw_if_cuda_error(
      cudaMemcpy(&new_color_num, color_start + (n - 1), sizeof(std::int32_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy update_color_id");
  kernel_write_color_id<<<blocks, kFoldingThreads>>>(color_id, perm, color_start, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_write_color_id");
  return new_color_num;
}

void write_color_sig(double* color_sig,
                     const std::int32_t* color_id,
                     std::int32_t n,
                     std::uint64_t seed) {
  if (n <= 0) {
    return;
  }
  const int blocks = (n + kFoldingThreads - 1) / kFoldingThreads;
  kernel_write_color_sig<<<blocks, kFoldingThreads>>>(color_sig, color_id, seed, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_write_color_sig");
}

}  // namespace

void init_workspace(FoldingWorkspace& workspace, std::int32_t num_row, std::int32_t num_col) {
  workspace.num_row = num_row;
  workspace.num_col = num_col;
  workspace.num_row_color = 1;
  workspace.num_col_color = 1;
  throw_if_cuda_error(cudaMalloc(&workspace.row_color, sizeof(ColorPair) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color, sizeof(ColorPair) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color");
  throw_if_cuda_error(cudaMalloc(&workspace.row_color_id, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color_id");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color_id, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color_id");
  throw_if_cuda_error(cudaMalloc(&workspace.row_color_sig, sizeof(double) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color_sig");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color_sig, sizeof(double) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color_sig");
  throw_if_cuda_error(cudaMalloc(&workspace.row_color_start, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color_start");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color_start, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color_start");
  throw_if_cuda_error(cudaMalloc(&workspace.row_perm, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_perm");
  throw_if_cuda_error(cudaMalloc(&workspace.col_perm, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_perm");
  if (num_row > 0) {
    throw_if_cuda_error(cudaMemset(workspace.row_color_id, 0, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                        "cudaMemset row_color_id");
    throw_if_cuda_error(cudaMemset(workspace.row_color_sig, 0, sizeof(double) * static_cast<std::size_t>(num_row)),
                        "cudaMemset row_color_sig");
  }
  if (num_col > 0) {
    throw_if_cuda_error(cudaMemset(workspace.col_color_id, 0, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                        "cudaMemset col_color_id");
    throw_if_cuda_error(cudaMemset(workspace.col_color_sig, 0, sizeof(double) * static_cast<std::size_t>(num_col)),
                        "cudaMemset col_color_sig");
  }
}

void free_workspace(FoldingWorkspace& workspace) {
  cudaFree(workspace.row_color);
  cudaFree(workspace.col_color);
  cudaFree(workspace.row_color_id);
  cudaFree(workspace.col_color_id);
  cudaFree(workspace.row_color_sig);
  cudaFree(workspace.col_color_sig);
  cudaFree(workspace.row_color_start);
  cudaFree(workspace.col_color_start);
  cudaFree(workspace.row_perm);
  cudaFree(workspace.col_perm);
  workspace = FoldingWorkspace{};
}

std::int32_t folding_update_row_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol) {
  return update_color_id(workspace.row_color_id,
                         workspace.row_color_start,
                         workspace.row_perm,
                         workspace.row_color,
                         color_sig,
                         tol,
                         workspace.num_row);
}

std::int32_t folding_update_col_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol) {
  return update_color_id(workspace.col_color_id,
                         workspace.col_color_start,
                         workspace.col_perm,
                         workspace.col_color,
                         color_sig,
                         tol,
                         workspace.num_col);
}

void folding_write_row_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side) {
  write_color_sig(workspace.row_color_sig,
                  workspace.row_color_id,
                  workspace.num_row,
                  color_seed(round, side));
}

void folding_write_col_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side) {
  write_color_sig(workspace.col_color_sig,
                  workspace.col_color_id,
                  workspace.num_col,
                  color_seed(round, side));
}

}  // namespace gpu_presolver::folding
