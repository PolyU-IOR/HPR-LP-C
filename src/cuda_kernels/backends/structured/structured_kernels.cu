#include "cuda_kernels/backends/structured/structured_kernels.cuh"
#include "cuda_kernels/backends/detail/update_device_helpers.cuh"
#include "gpu/preprocessing/operators/structured/row_template_operator.h"

using hprlp::cuda_kernels::detail::decode_biased_u16;
using hprlp::cuda_kernels::detail::project_x_with_bounds;
using hprlp::cuda_kernels::detail::project_y_delta;

__device__ __forceinline__ void row_template_finish_x(
    int row, HPRLP_FLOAT original_acc,
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
    const HPRLP_FLOAT aty = original_acc * inverse_col_scale;
    const HPRLP_FLOAT xi = x[row];
    const HPRLP_FLOAT z_temp = fma(
        sigma_params[0], aty - c[row], xi);
    const HPRLP_FLOAT x_bar = project_x_with_bounds(
        z_temp, l[row], u[row], x_bound_type[row]);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[row] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[row]);
    scaled_x_hat[row] = x_hat_value * inverse_col_scale;
}

__global__ void row_template_update_x_all_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const uint8_t *row_template_ids, const int *row_bases,
    const int *template_ptr, const int *template_offsets,
    const HPRLP_FLOAT *template_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    HPRLP_FLOAT acc = 0.0;
    const uint8_t template_id = row_template_ids[row];
    if (template_id != HPRLP_ROW_TEMPLATE_FALLBACK) {
        const int begin = template_ptr[template_id];
        const int end = template_ptr[template_id + 1];
        const int base = row_bases[row];
        for (int entry = begin; entry < end; ++entry) {
            acc = fma(template_values[entry],
                      scaled_y[base + template_offsets[entry]], acc);
        }
    } else {
        return;
    }

    row_template_finish_x(
        row, acc, x, l, u, x_bound_type, c, last_x,
        inverse_col_norm, scaled_x_hat, sigma_params, halpern_factors);
}

__global__ void affine_block_update_x_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *block_row_begin, const int *block_row_count,
    const int *block_entry_ptr, const int *entry_base_columns,
    const int *entry_column_strides, const HPRLP_FLOAT *entry_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int block = blockIdx.x;
    const int local = threadIdx.x;
    if (local >= block_row_count[block]) return;
    const int row = block_row_begin[block] + local;
    const int begin = block_entry_ptr[block];
    const int end = block_entry_ptr[block + 1];
    HPRLP_FLOAT acc = 0.0;
    for (int entry = begin; entry < end; ++entry) {
        const int column = entry_base_columns[entry] +
                           local * entry_column_strides[entry];
        acc = fma(entry_values[entry], scaled_y[column], acc);
    }
    row_template_finish_x(
        row, acc, x, l, u, x_bound_type, c, last_x,
        inverse_col_norm, scaled_x_hat, sigma_params, halpern_factors);
}

__global__ void row_template_update_x_fallback_short_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        acc = fma(values[entry], scaled_y[col_indices[entry]], acc);
    }
    row_template_finish_x(
        row, acc, x, l, u, x_bound_type, c, last_x,
        inverse_col_norm, scaled_x_hat, sigma_params, halpern_factors);
}

__global__ void row_template_update_x_fallback_warp_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp >= nrows) return;
    const int row = row_ids[warp];
    HPRLP_FLOAT acc = 0.0;
    for (int entry = row_ptr[row] + lane;
         entry < row_ptr[row + 1]; entry += 32) {
        acc = fma(values[entry], scaled_y[col_indices[entry]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        row_template_finish_x(
            row, acc, x, l, u, x_bound_type, c, last_x,
            inverse_col_norm, scaled_x_hat, sigma_params,
            halpern_factors);
    }
}

__global__ void row_template_update_x_fallback_block_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT acc = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        acc = fma(values[entry], scaled_y[col_indices[entry]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    __shared__ HPRLP_FLOAT warp_sums[32];
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            row_template_finish_x(
                row, acc, x, l, u, x_bound_type, c, last_x,
                inverse_col_norm, scaled_x_hat, sigma_params,
                halpern_factors);
        }
    }
}
__global__ void structured_update_x_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *col_norm,
    const uint16_t *output_cols, const int *row_ptr,
    const uint16_t *rows, const uint16_t *values_u16,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int coefficient_bias, bool coefficient_has_escape,
    int coefficient_escape_value, int output_count) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index < output_count) {
        const int col = output_cols[output_index];
        HPRLP_FLOAT acc = 0.0;
        for (int index = row_ptr[output_index];
             index < row_ptr[output_index + 1]; ++index) {
            const HPRLP_FLOAT coefficient = decode_biased_u16(
                values_u16[index], coefficient_bias,
                coefficient_has_escape, coefficient_escape_value);
            acc = fma(coefficient, scaled_y[rows[index]], acc);
        }
        const HPRLP_FLOAT aty = acc * col_norm[col];
        const HPRLP_FLOAT sigma = sigma_params[0];
        const HPRLP_FLOAT fact1 = halpern_factors[0];
        const HPRLP_FLOAT fact2 = halpern_factors[1];
        const HPRLP_FLOAT xi = x[col];
        const HPRLP_FLOAT z_temp = fma(sigma, aty - c[col], xi);
        const HPRLP_FLOAT x_bar = project_x_with_bounds(
            z_temp, l[col], u[col], x_bound_type[col]);
        const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[col] = fma(fact2, x_hat_value, fact1 * last_x[col]);
        x_hat[col] = x_hat_value;
    }
}

__global__ void structured_update_x_dense_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *col_norm,
    const uint16_t *dense_cols, const uint16_t *dense_rows,
    const uint16_t *values_u16,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int coefficient_bias, bool coefficient_has_escape,
    int coefficient_escape_value, int dense_row_count,
    int dense_col_count) {
    const int dense_col_index = blockIdx.x;
    if (dense_col_index < dense_col_count) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;
        HPRLP_FLOAT acc = 0.0;
        const int code_offset = dense_col_index * dense_row_count;
        for (int row_index = threadIdx.x; row_index < dense_row_count;
             row_index += blockDim.x) {
            const HPRLP_FLOAT coefficient = decode_biased_u16(
                values_u16[code_offset + row_index], coefficient_bias,
                coefficient_has_escape, coefficient_escape_value);
            acc = fma(coefficient, scaled_y[dense_rows[row_index]], acc);
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
            acc = lane < num_warps ? warp_sums[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }
            if (lane == 0) {
                const int col = dense_cols[dense_col_index];
                const HPRLP_FLOAT aty = acc * col_norm[col];
                const HPRLP_FLOAT sigma = sigma_params[0];
                const HPRLP_FLOAT fact1 = halpern_factors[0];
                const HPRLP_FLOAT fact2 = halpern_factors[1];
                const HPRLP_FLOAT xi = x[col];
                const HPRLP_FLOAT z_temp = fma(sigma, aty - c[col], xi);
                const HPRLP_FLOAT x_bar = project_x_with_bounds(
                    z_temp, l[col], u[col], x_bound_type[col]);
                const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
                x[col] = fma(fact2, x_hat_value, fact1 * last_x[col]);
                x_hat[col] = x_hat_value;
            }
        }
    }
}

__global__ void structured_update_y_sparse_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x, const HPRLP_FLOAT *row_norm,
    const uint16_t *rows, const uint16_t *col0, const uint16_t *col1,
    const int8_t *second_sign, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index < row_count) {
        const int row = rows[row_index];
        HPRLP_FLOAT acc = fma(1.0, scaled_x[col0[row_index]], 0.0);
        acc = fma(static_cast<HPRLP_FLOAT>(second_sign[row_index]),
                  scaled_x[col1[row_index]], acc);
        acc *= row_norm[row];
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
        y[row] = fma(halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
    }
}

__global__ void structured_update_y_dense_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x, const HPRLP_FLOAT *row_norm,
    const uint16_t *dense_rows, const uint16_t *dense_cols,
    const uint16_t *dense_local_cols, const uint16_t *values_u16,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int dense_row_count,
    int dense_col_count, int coefficient_bias,
    bool coefficient_has_escape, int coefficient_escape_value) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int dense_row_index =
        blockIdx.x * warps_per_block + warp_in_block;
    if (dense_row_index < dense_row_count) {
        HPRLP_FLOAT acc = 0.0;
        const int row_width = dense_col_count + 1;
        const int code_offset = dense_row_index * row_width;
        for (int position = lane; position < row_width; position += 32) {
            const int col = position < dense_col_count
                ? dense_cols[position]
                : dense_local_cols[dense_row_index];
            const HPRLP_FLOAT coefficient = decode_biased_u16(
                values_u16[code_offset + position], coefficient_bias,
                coefficient_has_escape, coefficient_escape_value);
            acc = fma(coefficient, scaled_x[col], acc);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const int row = dense_rows[dense_row_index];
            acc *= row_norm[row];
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
__device__ __forceinline__ void row_template_finish_y(
    int row, HPRLP_FLOAT original_acc,
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    const HPRLP_FLOAT ax = original_acc * inverse_row_scale;
    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT v = fma(-sigma_params[1], yi, ax);
    const HPRLP_FLOAT d = project_y_delta(
        v, AL[row], AU[row], y_bound_type[row]);
    const HPRLP_FLOAT y_bar = sigma_params[2] * d;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    const HPRLP_FLOAT y_new = fma(
        halpern_factors[1], y_hat,
        halpern_factors[0] * last_y[row]);
    y[row] = y_new;
    scaled_y[row] = y_new * inverse_row_scale;
}

__global__ void row_template_update_y_all_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const uint8_t *row_template_ids, const int *row_bases,
    const int *template_ptr, const int *template_offsets,
    const HPRLP_FLOAT *template_values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    HPRLP_FLOAT acc = 0.0;
    const uint8_t template_id = row_template_ids[row];
    if (template_id != HPRLP_ROW_TEMPLATE_FALLBACK) {
        const int begin = template_ptr[template_id];
        const int end = template_ptr[template_id + 1];
        const int base = row_bases[row];
        for (int entry = begin; entry < end; ++entry) {
            acc = fma(template_values[entry],
                      scaled_x_hat[base + template_offsets[entry]], acc);
        }
    } else {
        return;
    }

    row_template_finish_y(
        row, acc, y, AL, AU, y_bound_type, last_y,
        inverse_row_norm, scaled_y, sigma_params, halpern_factors);
}

__global__ void affine_block_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *block_row_begin, const int *block_row_count,
    const int *block_entry_ptr, const int *entry_base_columns,
    const int *entry_column_strides, const HPRLP_FLOAT *entry_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int block = blockIdx.x;
    const int local = threadIdx.x;
    if (local >= block_row_count[block]) return;
    const int row = block_row_begin[block] + local;
    const int begin = block_entry_ptr[block];
    const int end = block_entry_ptr[block + 1];
    HPRLP_FLOAT acc = 0.0;
    for (int entry = begin; entry < end; ++entry) {
        const int column = entry_base_columns[entry] +
                           local * entry_column_strides[entry];
        acc = fma(entry_values[entry], scaled_x_hat[column], acc);
    }
    row_template_finish_y(
        row, acc, y, AL, AU, y_bound_type, last_y,
        inverse_row_norm, scaled_y, sigma_params, halpern_factors);
}

__global__ void row_template_update_y_fallback_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        acc = fma(values[entry], scaled_x_hat[col_indices[entry]], acc);
    }
    row_template_finish_y(
        row, acc, y, AL, AU, y_bound_type, last_y,
        inverse_row_norm, scaled_y, sigma_params, halpern_factors);
}

__global__ void row_template_update_y_fallback_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp >= nrows) return;
    const int row = row_ids[warp];
    HPRLP_FLOAT acc = 0.0;
    for (int entry = row_ptr[row] + lane;
         entry < row_ptr[row + 1]; entry += 32) {
        acc = fma(values[entry], scaled_x_hat[col_indices[entry]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        row_template_finish_y(
            row, acc, y, AL, AU, y_bound_type, last_y,
            inverse_row_norm, scaled_y, sigma_params, halpern_factors);
    }
}

__global__ void row_template_update_y_fallback_block_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT acc = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        acc = fma(values[entry], scaled_x_hat[col_indices[entry]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    __shared__ HPRLP_FLOAT warp_sums[32];
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            row_template_finish_y(
                row, acc, y, AL, AU, y_bound_type, last_y,
                inverse_row_norm, scaled_y, sigma_params,
                halpern_factors);
        }
    }
}
