#include "cuda_kernels/backends/unit/unit_kernels.cuh"
#include "cuda_kernels/backends/detail/update_device_helpers.cuh"

using hprlp::cuda_kernels::detail::project_x_with_bounds;
using hprlp::cuda_kernels::detail::project_y_delta;

__global__ void update_y_normal_unit_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU,
                                            HPRLP_FLOAT *Ax, const HPRLP_FLOAT *inverse_row_norm,
                                            HPRLP_FLOAT *last_y, const HPRLP_FLOAT *sigma_params,
                                            const HPRLP_FLOAT *halpern_factors,
                                            int uniform_unit_sign, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT yi = y[i];
        HPRLP_FLOAT ai = Ax[i] * inverse_row_norm[i];
        if (uniform_unit_sign < 0) {
            ai = -ai;
        }
        HPRLP_FLOAT li = AL[i];
        HPRLP_FLOAT ui = AU[i];
        HPRLP_FLOAT y0i = last_y[i];
        HPRLP_FLOAT v = ai - fact1 * yi;
        HPRLP_FLOAT d = fmax(li - v, fmin(ui - v, 0.0));
        HPRLP_FLOAT y_bar_val = fact2 * d;
        HPRLP_FLOAT y_hat_val = 2 * y_bar_val - yi;
        HPRLP_FLOAT y_new_val = halpern_fact2 * y_hat_val + halpern_fact1 * y0i;
        y[i] = y_new_val;
    }
}

__global__ void unit_coltile_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x, const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const int *row_tile_offsets, const uint16_t *local_cols,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int tile_count, int tile_cols,
    int uniform_unit_sign, int m) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    constexpr int kWarpSize = 32;
    constexpr int kMaxWarpsPerBlock = 32;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const long long row_base =
        static_cast<long long>(row) * tile_count;

    HPRLP_FLOAT acc = 0.0;
    for (int tile = warp; tile < tile_count; tile += warps_per_block) {
        const int begin = row_tile_offsets[row_base + tile];
        const int end = row_tile_offsets[row_base + tile + 1];
        const int column_base = tile * tile_cols;
        for (int index = begin + lane; index < end; index += kWarpSize) {
            const int column =
                column_base + static_cast<int>(local_cols[index]);
            acc += scaled_x[column];
        }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ HPRLP_FLOAT warp_sums[kMaxWarpsPerBlock];
    if (lane == 0) {
        warp_sums[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        acc = lane < warps_per_block ? warp_sums[lane] : 0.0;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            acc *= inverse_row_norm[row];
            if (uniform_unit_sign < 0) {
                acc = -acc;
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
            const HPRLP_FLOAT y_new = fma(
                halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
            y[row] = y_new;
            if (scaled_y_output != nullptr) {
                scaled_y_output[row] = y_new * inverse_row_norm[row];
            }
        }
    }
}

__global__ void unit_coltile_update_y_zero_bitset_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x,
    const std::uint32_t *positive_zero_bits,
    const HPRLP_FLOAT *inverse_row_norm,
    const int *row_tile_offsets, const uint16_t *local_cols,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int tile_count, int tile_cols,
    int uniform_unit_sign, int m) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    constexpr int kWarpSize = 32;
    constexpr int kMaxWarpsPerBlock = 32;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const long long row_base =
        static_cast<long long>(row) * tile_count;

    HPRLP_FLOAT acc = 0.0;
    for (int tile = warp; tile < tile_count; tile += warps_per_block) {
        const int begin = row_tile_offsets[row_base + tile];
        const int end = row_tile_offsets[row_base + tile + 1];
        const int column_base = tile * tile_cols;
        for (int index = begin + lane; index < end; index += kWarpSize) {
            const int column =
                column_base + static_cast<int>(local_cols[index]);
            const std::uint32_t zero_word =
                positive_zero_bits[column >> 5];
            if (((zero_word >> (column & 31)) & 1u) == 0u) {
                acc += scaled_x[column];
            }
        }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ HPRLP_FLOAT warp_sums[kMaxWarpsPerBlock];
    if (lane == 0) {
        warp_sums[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        acc = lane < warps_per_block ? warp_sums[lane] : 0.0;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            acc *= inverse_row_norm[row];
            if (uniform_unit_sign < 0) {
                acc = -acc;
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
    }
}
__global__ void fused_update_x_z_all_short_unit_nonnegative_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr,
    const uint16_t *AT_colIndex, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int uniform_unit_sign, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc += scaled_y[AT_colIndex[idx]];
        }
        HPRLP_FLOAT aty = acc * inverse_col_norm[row];
        if (uniform_unit_sign < 0) {
            aty = -aty;
        }
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[row];
        HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
        HPRLP_FLOAT x_bar = fmax(z_temp, 0.0);
        HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
        if (scaled_x_hat_output != nullptr) {
            scaled_x_hat_output[row] =
                x_hat_value * inverse_col_norm[row];
        }
    }
}

__global__ void fused_update_x_z_all_scalar_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int uniform_unit_sign, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        HPRLP_FLOAT acc = 0.0;
        const int start = AT_rowPtr[row];
        const int end = AT_rowPtr[row + 1];
        for (int index = start; index < end; ++index) {
            acc += scaled_y[AT_colIndex[index]];
        }
        HPRLP_FLOAT aty = acc * inverse_col_norm[row];
        if (uniform_unit_sign < 0) {
            aty = -aty;
        }
        const HPRLP_FLOAT sigma = sigma_params[0];
        const HPRLP_FLOAT fact1 = halpern_factors[0];
        const HPRLP_FLOAT fact2 = halpern_factors[1];
        const HPRLP_FLOAT xi = x[row];
        const HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
        const HPRLP_FLOAT x_bar = project_x_with_bounds(
            z_temp, l[row], u[row], x_bound_type[row]);
        const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
        if (scaled_x_hat_output != nullptr) {
            scaled_x_hat_output[row] =
                x_hat_value * inverse_col_norm[row];
        }
    }
}

__global__ void fused_update_x_z_all_scalar_unit_active_scatter_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *Ax, const int *AT_rowPtr,
    const uint16_t *AT_colIndex, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int uniform_unit_sign, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    HPRLP_FLOAT acc = 0.0;
    const int start = AT_rowPtr[row];
    const int end = AT_rowPtr[row + 1];
    for (int index = start; index < end; ++index) {
        acc += scaled_y[AT_colIndex[index]];
    }
    const HPRLP_FLOAT inverse_norm = inverse_col_norm[row];
    HPRLP_FLOAT aty = acc * inverse_norm;
    if (uniform_unit_sign < 0) {
        aty = -aty;
    }
    const HPRLP_FLOAT xi = x[row];
    const HPRLP_FLOAT z_temp = fma(
        sigma_params[0], aty - c[row], xi);
    const HPRLP_FLOAT x_bar = project_x_with_bounds(
        z_temp, l[row], u[row], x_bound_type[row]);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[row] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[row]);
    x_hat[row] = x_hat_value;

    // Preserve the incumbent's positive-zero convention.  Uniform matrix
    // sign is still applied once in update_y_normal_unit_kernel.
    if (__double_as_longlong(x_hat_value) != 0ll) {
        const HPRLP_FLOAT scaled_x_hat = x_hat_value * inverse_norm;
        for (int index = start; index < end; ++index) {
            atomicAdd(&Ax[AT_colIndex[index]], scaled_x_hat);
        }
    }
}

__global__ void fused_update_x_z_rows_short_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign, const int *row_ids, int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrows) {
        int row = row_ids[tid];
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc += scaled_y[AT_colIndex[idx]];
        }
        HPRLP_FLOAT aty = acc * inverse_col_norm[row];
        if (uniform_unit_sign < 0) {
            aty = -aty;
        }
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[row];
        HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
        HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
        HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
        if (scaled_x_hat_output != nullptr) {
            scaled_x_hat_output[row] =
                x_hat_value * inverse_col_norm[row];
        }
    }
}

__global__ void fused_update_x_z_rows_warp_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign, const int *row_ids, int nrows) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_idx = blockIdx.x * warps_per_block + warp_in_block;
    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        HPRLP_FLOAT acc = 0.0;
        for (int idx = start + lane; idx < end; idx += 32) {
            acc += scaled_y[AT_colIndex[idx]];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            HPRLP_FLOAT aty = acc * inverse_col_norm[row];
            if (uniform_unit_sign < 0) {
                aty = -aty;
            }
            HPRLP_FLOAT sigma = sigma_params[0];
            HPRLP_FLOAT fact1 = halpern_factors[0];
            HPRLP_FLOAT fact2 = halpern_factors[1];
            HPRLP_FLOAT xi = x[row];
            HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
            HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
            HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
            x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
            x_hat[row] = x_hat_value;
            if (scaled_x_hat_output != nullptr) {
                scaled_x_hat_output[row] =
                    x_hat_value * inverse_col_norm[row];
            }
        }
    }
}

__global__ void fused_update_x_z_rows_block_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign, const int *row_ids, int nrows) {
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
            acc += scaled_y[AT_colIndex[idx]];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        __shared__ HPRLP_FLOAT warp_sums[32];
        if (lane == 0) warp_sums[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            acc = (lane < num_warps) ? warp_sums[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }
            if (lane == 0) {
                HPRLP_FLOAT aty = acc * inverse_col_norm[row];
                if (uniform_unit_sign < 0) {
                    aty = -aty;
                }
                HPRLP_FLOAT sigma = sigma_params[0];
                HPRLP_FLOAT fact1 = halpern_factors[0];
                HPRLP_FLOAT fact2 = halpern_factors[1];
                HPRLP_FLOAT xi = x[row];
                HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
                HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
                HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
                x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
                x_hat[row] = x_hat_value;
                if (scaled_x_hat_output != nullptr) {
                    scaled_x_hat_output[row] =
                        x_hat_value * inverse_col_norm[row];
                }
            }
        }
    }
}
