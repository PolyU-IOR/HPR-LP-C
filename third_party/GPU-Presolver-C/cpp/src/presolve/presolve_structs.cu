#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cuda_runtime.h>

#include <utility>

namespace gpu_presolver::presolve {

PostsolveTapeGpu::~PostsolveTapeGpu() {
  reset();
}

PostsolveTapeGpu::PostsolveTapeGpu(PostsolveTapeGpu&& other) noexcept
    : types(std::exchange(other.types, nullptr)),
      index_starts(std::exchange(other.index_starts, nullptr)),
      value_starts(std::exchange(other.value_starts, nullptr)),
      dual_modes(std::exchange(other.dual_modes, nullptr)),
      indices(std::exchange(other.indices, nullptr)),
      vals(std::exchange(other.vals, nullptr)),
      record_count(std::exchange(other.record_count, 0)),
      index_count(std::exchange(other.index_count, 0)),
      value_count(std::exchange(other.value_count, 0)),
      record_capacity(std::exchange(other.record_capacity, 0)),
      index_capacity(std::exchange(other.index_capacity, 0)),
      value_capacity(std::exchange(other.value_capacity, 0)) {}

PostsolveTapeGpu& PostsolveTapeGpu::operator=(PostsolveTapeGpu&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  types = std::exchange(other.types, nullptr);
  index_starts = std::exchange(other.index_starts, nullptr);
  value_starts = std::exchange(other.value_starts, nullptr);
  dual_modes = std::exchange(other.dual_modes, nullptr);
  indices = std::exchange(other.indices, nullptr);
  vals = std::exchange(other.vals, nullptr);
  record_count = std::exchange(other.record_count, 0);
  index_count = std::exchange(other.index_count, 0);
  value_count = std::exchange(other.value_count, 0);
  record_capacity = std::exchange(other.record_capacity, 0);
  index_capacity = std::exchange(other.index_capacity, 0);
  value_capacity = std::exchange(other.value_capacity, 0);
  return *this;
}

void PostsolveTapeGpu::reset() noexcept {
  if (types != nullptr) cudaFreeAsync(types, nullptr);
  if (index_starts != nullptr) cudaFreeAsync(index_starts, nullptr);
  if (value_starts != nullptr) cudaFreeAsync(value_starts, nullptr);
  if (dual_modes != nullptr) cudaFreeAsync(dual_modes, nullptr);
  if (indices != nullptr) cudaFreeAsync(indices, nullptr);
  if (vals != nullptr) cudaFreeAsync(vals, nullptr);
  types = nullptr;
  index_starts = nullptr;
  value_starts = nullptr;
  dual_modes = nullptr;
  indices = nullptr;
  vals = nullptr;
  record_count = 0;
  index_count = 0;
  value_count = 0;
  record_capacity = 0;
  index_capacity = 0;
  value_capacity = 0;
}

static_assert(sizeof(std::uint8_t) == 1, "Julia UInt8 masks must map to one-byte C++ masks");
static_assert(sizeof(std::int32_t) == 4, "Julia Int32 indices must map to four-byte C++ indices");
static_assert(sizeof(double) == 8, "Julia Float64 values must map to eight-byte C++ doubles");

}  // namespace gpu_presolver::presolve
