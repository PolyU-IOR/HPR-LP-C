#include "cuda_kernels/backends/generic/generic_fused_kernels.cuh"
#include "gpu/preprocessing/policies/row_bucket_policy.h"
#include "cuda_kernels/backends/detail/update_device_helpers.cuh"

using hprlp::cuda_kernels::detail::project_x_with_bounds;
using hprlp::cuda_kernels::detail::project_y_delta;

__global__ void fused_update_x_z_rows_short_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                                   const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                                   const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                                   const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                                   const HPRLP_FLOAT *halpern_factors,
                                                   const int *row_ids, int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrows) {
        int row = row_ids[tid];
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(AT_value[idx], y[AT_colIndex[idx]], acc);
        }

        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[row];
        HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
        HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
        HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
    }
}

__global__ void fused_update_x_z_all_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const int *AT_rowPtr,
    const int *AT_colIndex, const HPRLP_FLOAT *AT_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(AT_value[idx], y[AT_colIndex[idx]], acc);
        }

        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[row];
        HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
        HPRLP_FLOAT x_bar = project_x_with_bounds(
            z_temp, l[row], u[row], x_bound_type[row]);
        HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
    }
}
__global__ void fused_update_x_z_direct_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const int *AT_rowPtr,
    const int *AT_colIndex, const HPRLP_FLOAT *AT_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }
    const int start = AT_rowPtr[row];
    const int end = AT_rowPtr[row + 1];
    if (end - start > HPRLP_SCALAR_ROW_MAX_NNZ) {
        return;
    }

    HPRLP_FLOAT acc = 0.0;
    for (int index = start; index < end; ++index) {
        acc = fma(AT_value[index], y[AT_colIndex[index]], acc);
    }
    const HPRLP_FLOAT sigma = sigma_params[0];
    const HPRLP_FLOAT fact1 = halpern_factors[0];
    const HPRLP_FLOAT fact2 = halpern_factors[1];
    const HPRLP_FLOAT xi = x[row];
    const HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
    const HPRLP_FLOAT x_bar = project_x_with_bounds(
        z_temp, l[row], u[row], x_bound_type[row]);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
    x_hat[row] = x_hat_value;
}

__global__ void fused_update_x_z_rows_warp_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                                  const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                                  const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                                  const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                                  const HPRLP_FLOAT *halpern_factors,
                                                  const int *row_ids, int nrows) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_idx = blockIdx.x * warps_per_block + warp_in_block;

    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        HPRLP_FLOAT sigma = sigma_params[0];
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        HPRLP_FLOAT acc = 0.0;
        for (int idx = start + lane; idx < end; idx += 32) {
            acc = fma(AT_value[idx], y[AT_colIndex[idx]], acc);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            HPRLP_FLOAT fact1 = halpern_factors[0];
            HPRLP_FLOAT fact2 = halpern_factors[1];
            HPRLP_FLOAT xi = x[row];
            HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
            HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
            HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
            x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
            x_hat[row] = x_hat_value;
        }
    }
}

__global__ void fused_update_x_z_rows_block_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                                   const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                                   const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                                   const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                                   const HPRLP_FLOAT *halpern_factors,
                                                   const int *row_ids, int nrows) {
    int row_idx = blockIdx.x;
    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        int num_warps = blockDim.x >> 5;
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
            acc = fma(AT_value[idx], y[AT_colIndex[idx]], acc);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        __shared__ HPRLP_FLOAT warp_sums[32];
        if (lane == 0) {
            warp_sums[warp] = acc;
        }
        __syncthreads();

        if (warp == 0) {
            acc = (lane < num_warps) ? warp_sums[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }
            if (lane == 0) {
                HPRLP_FLOAT sigma = sigma_params[0];
                HPRLP_FLOAT fact1 = halpern_factors[0];
                HPRLP_FLOAT fact2 = halpern_factors[1];
                HPRLP_FLOAT xi = x[row];
                HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
                HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
                HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
                x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
                x_hat[row] = x_hat_value;
            }
        }
    }
}
__global__ void fused_update_y_rows_short_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                                 const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                                 const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                                 const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                                 const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrows) {
        int row = row_ids[tid];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT acc = 0.0;
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }

        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT yi = y[row];
        HPRLP_FLOAT v = fma(-fact1, yi, acc);
        HPRLP_FLOAT d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
        HPRLP_FLOAT y_bar = fact2 * d;
        HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
        y[row] = fma(halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
    }
}

__global__ void fused_update_y_all_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const int *A_rowPtr,
    const int *A_colIndex, const HPRLP_FLOAT *A_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT acc = 0.0;
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }

        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT yi = y[row];
        HPRLP_FLOAT v = fma(-fact1, yi, acc);
        HPRLP_FLOAT d = project_y_delta(
            v, AL[row], AU[row], y_bound_type[row]);
        HPRLP_FLOAT y_bar = fact2 * d;
        HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
        y[row] = fma(halpern_fact2, y_hat,
                     halpern_fact1 * last_y[row]);
    }
}
__global__ void fused_update_y_all_short_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const int *A_rowPtr,
    const uint16_t *A_colIndex, const HPRLP_FLOAT *A_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT acc = 0.0;
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }

        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT yi = y[row];
        HPRLP_FLOAT v = fma(-fact1, yi, acc);
        HPRLP_FLOAT d = project_y_delta(
            v, AL[row], AU[row], y_bound_type[row]);
        HPRLP_FLOAT y_bar = fact2 * d;
        HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
        y[row] = fma(halpern_fact2, y_hat,
                     halpern_fact1 * last_y[row]);
    }
}

__global__ void fused_update_y_direct_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const int *A_rowPtr,
    const int *A_colIndex, const HPRLP_FLOAT *A_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int start = A_rowPtr[row];
    const int end = A_rowPtr[row + 1];
    if (end - start > HPRLP_SCALAR_ROW_MAX_NNZ) {
        return;
    }

    HPRLP_FLOAT acc = 0.0;
    for (int index = start; index < end; ++index) {
        acc = fma(A_value[index], x_hat[A_colIndex[index]], acc);
    }
    const HPRLP_FLOAT fact1 = sigma_params[1];
    const HPRLP_FLOAT fact2 = sigma_params[2];
    const HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
    const HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT v = fma(-fact1, yi, acc);
    const HPRLP_FLOAT d = project_y_delta(
        v, AL[row], AU[row], y_bound_type[row]);
    const HPRLP_FLOAT y_bar = fact2 * d;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    y[row] = fma(halpern_fact2, y_hat,
                 halpern_fact1 * last_y[row]);
}

__global__ void fused_update_y_rows_warp_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                                const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                                const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                                const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                                const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_idx = blockIdx.x * warps_per_block + warp_in_block;

    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        HPRLP_FLOAT acc = 0.0;
        for (int idx = start + lane; idx < end; idx += 32) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
            HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
            HPRLP_FLOAT yi = y[row];
            HPRLP_FLOAT v = fma(-fact1, yi, acc);
            HPRLP_FLOAT d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
            HPRLP_FLOAT y_bar = fact2 * d;
            HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
            y[row] = fma(halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
        }
    }
}

__global__ void fused_update_y_rows_block_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                                 const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                                 const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                                 const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                                 const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    int row_idx = blockIdx.x;
    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        int num_warps = blockDim.x >> 5;
        HPRLP_FLOAT acc = 0.0;
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        for (int idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        __shared__ HPRLP_FLOAT warp_sums[32];
        if (lane == 0) {
            warp_sums[warp] = acc;
        }
        __syncthreads();

        if (warp == 0) {
            acc = (lane < num_warps) ? warp_sums[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }
            if (lane == 0) {
                HPRLP_FLOAT fact1 = sigma_params[1];
                HPRLP_FLOAT fact2 = sigma_params[2];
                HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
                HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
                HPRLP_FLOAT yi = y[row];
                HPRLP_FLOAT v = fma(-fact1, yi, acc);
                HPRLP_FLOAT d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
                HPRLP_FLOAT y_bar = fact2 * d;
                HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
                y[row] = fma(halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
            }
        }
    }
}

__global__ void segmented_update_y_partial_kernel(
    HPRLP_FLOAT *partials, const HPRLP_FLOAT *x_hat,
    const int *A_colIndex, const HPRLP_FLOAT *A_value,
    const int *tile_begin, const int *tile_end, int tile_count) {
    const int tile = blockIdx.x;
    if (tile >= tile_count) return;

    HPRLP_FLOAT acc = 0.0;
    for (int index = tile_begin[tile] + threadIdx.x;
         index < tile_end[tile]; index += blockDim.x) {
        acc = fma(A_value[index], x_hat[A_colIndex[index]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ HPRLP_FLOAT warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        acc = lane < num_warps ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) partials[tile] = acc;
    }
}

__global__ void segmented_update_y_finalize_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *partials, const int *row_ids,
    const int *row_tile_ptr, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row_index = blockIdx.x;
    if (row_index >= row_count) return;

    HPRLP_FLOAT acc = 0.0;
    for (int tile = row_tile_ptr[row_index] + threadIdx.x;
         tile < row_tile_ptr[row_index + 1]; tile += blockDim.x) {
        acc += partials[tile];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ HPRLP_FLOAT warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        acc = lane < num_warps ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const int row = row_ids[row_index];
            const HPRLP_FLOAT fact1 = sigma_params[1];
            const HPRLP_FLOAT fact2 = sigma_params[2];
            const HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
            const HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
            const HPRLP_FLOAT yi = y[row];
            const HPRLP_FLOAT value = fma(-fact1, yi, acc);
            const HPRLP_FLOAT delta = project_y_delta(
                value, AL[row], AU[row], y_bound_type[row]);
            const HPRLP_FLOAT y_bar = fact2 * delta;
            const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
            y[row] = fma(halpern_fact2, y_hat,
                         halpern_fact1 * last_y[row]);
        }
    }
}

__global__ void segmented_update_x_finalize_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *partials, const int *row_ids,
    const int *row_tile_ptr, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row_index = blockIdx.x;
    if (row_index >= row_count) return;
    HPRLP_FLOAT acc = 0.0;
    for (int tile = row_tile_ptr[row_index] + threadIdx.x;
         tile < row_tile_ptr[row_index + 1]; tile += blockDim.x) {
        acc += partials[tile];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    __shared__ HPRLP_FLOAT warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const int row = row_ids[row_index];
            const HPRLP_FLOAT xi = x[row];
            const HPRLP_FLOAT z_temp = fma(
                sigma_params[0], acc - c[row], xi);
            const HPRLP_FLOAT x_bar = project_x_with_bounds(
                z_temp, l[row], u[row], x_bound_type[row]);
            const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
            x[row] = fma(halpern_factors[1], x_hat_value,
                         halpern_factors[0] * last_x[row]);
            x_hat[row] = x_hat_value;
        }
    }
}
