#ifndef HPRLP_WIDE_FUSED_KERNELS_CUH
#define HPRLP_WIDE_FUSED_KERNELS_CUH

#include "api/structs.h"
#include "cuda_kernels/backends/detail/update_device_helpers.cuh"

#include <cstdint>

// One cooperative block per row. These kernels deliberately use 64-bit CSR
// positions while retaining int32 row/column dimensions. They do not depend
// on any int32 entry-position metadata or on the packed operator formats.
template <typename RowOffset, typename ColumnIndex>
__global__ void hprlp_wide_fused_x_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *cost, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const RowOffset *row_ptr,
    const ColumnIndex *col_index, const HPRLP_FLOAT *value,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *inverse_col_norm, int value_mode,
    int uniform_sign, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    HPRLP_FLOAT partial = 0.0;
    const std::int64_t begin = row_ptr[row];
    const std::int64_t end = row_ptr[row + 1];
    for (std::int64_t entry = begin + threadIdx.x;
         entry < end; entry += blockDim.x) {
        if (value_mode == 0) {
            partial = fma(value[entry], y[col_index[entry]], partial);
        } else {
            HPRLP_FLOAT term =
                y[col_index[entry]] * inverse_row_norm[col_index[entry]];
            if (value_mode == 2 && value[entry] < 0.0) term = -term;
            partial += term;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    __shared__ HPRLP_FLOAT warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp != 0) return;
    const int warp_count = blockDim.x >> 5;
    partial = lane < warp_count ? warp_sums[lane] : 0.0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    if (lane != 0) return;

    if (value_mode != 0) {
        partial *= inverse_col_norm[row];
        if (value_mode == 1 && uniform_sign < 0) partial = -partial;
    }

    const HPRLP_FLOAT xi = x[row];
    const HPRLP_FLOAT trial = fma(
        sigma_params[0], partial - cost[row], xi);
    const HPRLP_FLOAT projected =
        hprlp::cuda_kernels::detail::project_x_with_bounds(
            trial, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - xi;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
}

template <typename RowOffset, typename ColumnIndex>
__global__ void hprlp_wide_fused_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *x_hat,
    const RowOffset *row_ptr, const ColumnIndex *col_index,
    const HPRLP_FLOAT *value, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *row_shift, int value_mode,
    int uniform_sign, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    HPRLP_FLOAT partial = 0.0;
    const std::int64_t begin = row_ptr[row];
    const std::int64_t end = row_ptr[row + 1];
    for (std::int64_t entry = begin + threadIdx.x;
         entry < end; entry += blockDim.x) {
        if (value_mode == 0) {
            partial = fma(value[entry], x_hat[col_index[entry]], partial);
        } else {
            HPRLP_FLOAT term =
                x_hat[col_index[entry]] * inverse_col_norm[col_index[entry]];
            if (value_mode == 2 && value[entry] < 0.0) term = -term;
            partial += term;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    __shared__ HPRLP_FLOAT warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp != 0) return;
    const int warp_count = blockDim.x >> 5;
    partial = lane < warp_count ? warp_sums[lane] : 0.0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    if (lane != 0) return;

    if (value_mode != 0) {
        partial *= inverse_row_norm[row];
        if (value_mode == 1 && uniform_sign < 0) partial = -partial;
    }
    if (row_shift != nullptr) partial += row_shift[row];

    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT trial = fma(-sigma_params[1], yi, partial);
    const HPRLP_FLOAT delta =
        hprlp::cuda_kernels::detail::project_y_delta(
            trial, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT projected = sigma_params[2] * delta;
    const HPRLP_FLOAT reflected = 2.0 * projected - yi;
    y[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_y[row]);
}

#endif
