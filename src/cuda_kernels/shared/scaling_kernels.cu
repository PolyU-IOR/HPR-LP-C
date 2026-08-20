#include "cuda_kernels/shared/scaling_kernels.cuh"
#include <cmath>

namespace {

constexpr int kScalingCooperativeRowThreshold = 256;
constexpr int kScalingThreadsPerBlock = 256;
constexpr unsigned kScalingFullWarpMask = 0xffffffffu;

} // namespace


__global__ void CSR_A_row_norm_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *result, int norm) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int warp_base = threadIdx.x - lane;
    const int start = row < m ? rowPtr[row] : 0;
    const int end = row < m ? rowPtr[row + 1] : 0;
    const bool cooperative = row < m &&
        end - start > kScalingCooperativeRowThreshold;

    // The launch contract uses numThreads == 256.  Each warp owns its
    // corresponding 32-value slice of this shared tile.
    __shared__ HPRLP_FLOAT ordered_values[kScalingThreadsPerBlock];
    unsigned cooperative_rows =
        __ballot_sync(kScalingFullWarpMask, cooperative);
    while (cooperative_rows != 0u) {
        const int owner_lane = __ffs(cooperative_rows) - 1;
        const int cooperative_row =
            __shfl_sync(kScalingFullWarpMask, row, owner_lane);
        const int cooperative_start =
            __shfl_sync(kScalingFullWarpMask, start, owner_lane);
        const int cooperative_end =
            __shfl_sync(kScalingFullWarpMask, end, owner_lane);

        if (norm == 99) {
            HPRLP_FLOAT maximum = 0.0;
            for (int j = cooperative_start + lane;
                 j < cooperative_end; j += warpSize) {
                maximum = fmax(maximum, fabs(value[j]));
            }
            for (int offset = warpSize / 2;
                 offset > 0; offset /= 2) {
                maximum = fmax(
                    maximum,
                    __shfl_down_sync(
                        kScalingFullWarpMask, maximum, offset));
            }
            if (lane == 0) {
                result[cooperative_row] = sqrt(maximum);
                if (result[cooperative_row] < 1e-15) {
                    result[cooperative_row] = 1.0;
                }
            }
        } else if (norm == 1) {
            HPRLP_FLOAT sum = 0.0;
            for (int base = cooperative_start;
                 base < cooperative_end; base += warpSize) {
                const int j = base + lane;
                ordered_values[threadIdx.x] =
                    j < cooperative_end ? fabs(value[j]) : 0.0;
                __syncwarp(kScalingFullWarpMask);

                if (lane == 0) {
                    const int tile_count =
                        cooperative_end - base < warpSize
                            ? cooperative_end - base
                            : warpSize;
                    for (int offset = 0;
                         offset < tile_count; ++offset) {
                        sum += ordered_values[warp_base + offset];
                    }
                }
                __syncwarp(kScalingFullWarpMask);
            }
            if (lane == 0) {
                result[cooperative_row] = sqrt(sum);
                if (result[cooperative_row] < 1e-15) {
                    result[cooperative_row] = 1.0;
                }
            }
        }
        cooperative_rows &= cooperative_rows - 1u;
    }

    if (row >= m || cooperative) {
        return;
    }

    // Rows up to the threshold retain the original instruction order and
    // one-thread ownership exactly.
    if (norm == 99) {
        result[row] = 0.0;
        for (int j = start; j < end; ++j) {
            if (result[row] < std::fabs(value[j])) {
                result[row] = std::fabs(value[j]);
            }
        }
        result[row] = std::sqrt(result[row]);
        if (result[row] < 1e-15) {
            result[row] = 1.0;
        }
    } else if (norm == 1) {
        result[row] = 0.0;
        for (int j = start; j < end; ++j) {
            result[row] += std::fabs(value[j]);
        }
        result[row] = std::sqrt(result[row]);
        if (result[row] < 1e-15) {
            result[row] = 1.0;
        }
    }
}

__global__ void mul_CSR_A_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int start = row < m ? rowPtr[row] : 0;
    const int end = row < m ? rowPtr[row + 1] : 0;
    const bool cooperative = row < m &&
        end - start > kScalingCooperativeRowThreshold;

    unsigned cooperative_rows =
        __ballot_sync(kScalingFullWarpMask, cooperative);
    while (cooperative_rows != 0u) {
        const int owner_lane = __ffs(cooperative_rows) - 1;
        const int cooperative_row =
            __shfl_sync(kScalingFullWarpMask, row, owner_lane);
        const int cooperative_start =
            __shfl_sync(kScalingFullWarpMask, start, owner_lane);
        const int cooperative_end =
            __shfl_sync(kScalingFullWarpMask, end, owner_lane);

        if (divide) {
            for (int j = cooperative_start + lane;
                 j < cooperative_end; j += warpSize) {
                value[j] /= x[cooperative_row];
            }
        } else {
            for (int j = cooperative_start + lane;
                 j < cooperative_end; j += warpSize) {
                value[j] *= x[cooperative_row];
            }
        }
        __syncwarp(kScalingFullWarpMask);
        cooperative_rows &= cooperative_rows - 1u;
    }

    if (row >= m || cooperative) {
        return;
    }
    if (divide) {
        for (int j = start; j < end; ++j) {
            value[j] /= x[row];
        }
    } else {
        for (int j = start; j < end; ++j) {
            value[j] *= x[row];
        }
    }
}


__global__ void mul_CSR_AT_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int start = row < m ? rowPtr[row] : 0;
    const int end = row < m ? rowPtr[row + 1] : 0;
    const bool cooperative = row < m &&
        end - start > kScalingCooperativeRowThreshold;

    unsigned cooperative_rows =
        __ballot_sync(kScalingFullWarpMask, cooperative);
    while (cooperative_rows != 0u) {
        const int owner_lane = __ffs(cooperative_rows) - 1;
        const int cooperative_start =
            __shfl_sync(kScalingFullWarpMask, start, owner_lane);
        const int cooperative_end =
            __shfl_sync(kScalingFullWarpMask, end, owner_lane);

        if (divide) {
            for (int j = cooperative_start + lane;
                 j < cooperative_end; j += warpSize) {
                value[j] /= x[colIndex[j]];
            }
        } else {
            for (int j = cooperative_start + lane;
                 j < cooperative_end; j += warpSize) {
                value[j] *= x[colIndex[j]];
            }
        }
        __syncwarp(kScalingFullWarpMask);
        cooperative_rows &= cooperative_rows - 1u;
    }

    if (row >= m || cooperative) {
        return;
    }
    if (divide) {
        for (int j = start; j < end; ++j) {
            value[j] /= x[colIndex[j]];
        }
    } else {
        for (int j = start; j < end; ++j) {
            value[j] *= x[colIndex[j]];
        }
    }
}
