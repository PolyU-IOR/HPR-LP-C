#include "cuda_kernels/backends/dictionary/dictionary_kernels.cuh"
#include "cuda_kernels/backends/detail/update_device_helpers.cuh"

using hprlp::cuda_kernels::detail::project_x_with_bounds;
using hprlp::cuda_kernels::detail::project_y_delta;

__global__ void fused_update_x_z_rows_short_dictionary_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *dictionary, const uint8_t *value_codes,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrows) {
        int row = row_ids[tid];
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(dictionary[value_codes[idx]],
                      scaled_y[AT_colIndex[idx]], acc);
        }
        HPRLP_FLOAT aty = acc * inverse_col_norm[row];
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[row];
        HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
        HPRLP_FLOAT x_bar = project_x_with_bounds(
            z_temp, l[row], u[row], x_bound_type[row]);
        HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
    }
}

__global__ void fused_update_x_z_rows_warp_dictionary_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *dictionary, const uint8_t *value_codes,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
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
            acc = fma(dictionary[value_codes[idx]],
                      scaled_y[AT_colIndex[idx]], acc);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            HPRLP_FLOAT aty = acc * inverse_col_norm[row];
            HPRLP_FLOAT sigma = sigma_params[0];
            HPRLP_FLOAT fact1 = halpern_factors[0];
            HPRLP_FLOAT fact2 = halpern_factors[1];
            HPRLP_FLOAT xi = x[row];
            HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
            HPRLP_FLOAT x_bar = project_x_with_bounds(
                z_temp, l[row], u[row], x_bound_type[row]);
            HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
            x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
            x_hat[row] = x_hat_value;
        }
    }
}

__global__ void fused_update_x_z_rows_block_dictionary_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *dictionary, const uint8_t *value_codes,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
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
            acc = fma(dictionary[value_codes[idx]],
                      scaled_y[AT_colIndex[idx]], acc);
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
                HPRLP_FLOAT aty = acc * inverse_col_norm[row];
                HPRLP_FLOAT sigma = sigma_params[0];
                HPRLP_FLOAT fact1 = halpern_factors[0];
                HPRLP_FLOAT fact2 = halpern_factors[1];
                HPRLP_FLOAT xi = x[row];
                HPRLP_FLOAT z_temp = fma(sigma, aty - c[row], xi);
                HPRLP_FLOAT x_bar = project_x_with_bounds(
                    z_temp, l[row], u[row], x_bound_type[row]);
                HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
                x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
                x_hat[row] = x_hat_value;
            }
        }
    }
}

__device__ __forceinline__ bool hprlp_index_in_state_run(
    int index, int begin, int count) {
    return static_cast<unsigned>(index - begin) <
           static_cast<unsigned>(count);
}

__device__ __forceinline__ HPRLP_FLOAT
hprlp_packed_fixed_degree_dot(
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries_soa,
    const HPRLP_FLOAT *input, int item, int row_count, int degree,
    unsigned code_bits) {
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
#define HPRLP_FIXED_DEGREE_ACCUMULATE(OFFSET)                              \
    do {                                                                   \
        const uint32_t entry =                                              \
            packed_entries_soa[(OFFSET) * row_count + item];               \
        acc = fma(dictionary[entry & code_mask],                           \
                  input[entry >> code_bits], acc);                         \
    } while (false)
    switch (degree) {
        case 1:
            HPRLP_FIXED_DEGREE_ACCUMULATE(0);
            break;
        case 2:
            HPRLP_FIXED_DEGREE_ACCUMULATE(0);
            HPRLP_FIXED_DEGREE_ACCUMULATE(1);
            break;
        case 3:
            HPRLP_FIXED_DEGREE_ACCUMULATE(0);
            HPRLP_FIXED_DEGREE_ACCUMULATE(1);
            HPRLP_FIXED_DEGREE_ACCUMULATE(2);
            break;
        case 4:
            HPRLP_FIXED_DEGREE_ACCUMULATE(0);
            HPRLP_FIXED_DEGREE_ACCUMULATE(1);
            HPRLP_FIXED_DEGREE_ACCUMULATE(2);
            HPRLP_FIXED_DEGREE_ACCUMULATE(3);
            break;
        default:
            for (int offset = 0; offset < degree; ++offset) {
                HPRLP_FIXED_DEGREE_ACCUMULATE(offset);
            }
            break;
    }
#undef HPRLP_FIXED_DEGREE_ACCUMULATE
    return acc;
}

__device__ __forceinline__ void finish_packed_dictionary_x(
    int row, HPRLP_FLOAT aty, HPRLP_FLOAT inverse_col_scale,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT xi = x[row];
    const HPRLP_FLOAT objective = hprlp_index_in_state_run(
        row, state_plan.x_objective_zero_begin,
        state_plan.x_objective_zero_count) ? 0.0 : c[row];
    const HPRLP_FLOAT z_temp = fma(
        sigma_params[0], aty - objective, xi);
    const HPRLP_FLOAT x_bar = hprlp_index_in_state_run(
        row, state_plan.x_zero_lower_boxed_begin,
        state_plan.x_zero_lower_boxed_count)
        ? fmin(fmax(z_temp, 0.0), u[row])
        : project_x_with_bounds(
              z_temp, l[row], u[row], x_bound_type[row]);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[row] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[row]);
    if (x_hat != nullptr) {
        x_hat[row] = x_hat_value;
    }
    if (scaled_x_hat_output != nullptr) {
        scaled_x_hat_output[row] = x_hat_value * inverse_col_scale;
    }
}

__global__ void packed_dictionary_update_x_rows_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row]; index < AT_rowPtr[row + 1]; ++index) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_y[entry >> code_bits], acc);
    }
    const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
    finish_packed_dictionary_x(
        row, acc * inverse_col_scale, inverse_col_scale,
        scaled_x_hat_output, state_plan, x, x_hat, l, u, x_bound_type,
        c, last_x, sigma_params, halpern_factors);
}

__global__ void packed_dictionary_update_x_rows_warp_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int item = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row] + lane; index < AT_rowPtr[row + 1];
         index += 32) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_y[entry >> code_bits], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
        finish_packed_dictionary_x(
            row, acc * inverse_col_scale, inverse_col_scale,
            scaled_x_hat_output, state_plan, x, x_hat, l, u,
            x_bound_type, c, last_x, sigma_params, halpern_factors);
    }
}

__global__ void packed_dictionary_update_x_combined_short_warp_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *short_row_ids, int short_row_count,
    const int *medium_row_ids, int medium_row_count) {
    const int short_block_count =
        (short_row_count + blockDim.x - 1) / blockDim.x;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    if (static_cast<int>(blockIdx.x) < short_block_count) {
        const int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= short_row_count) return;
        const int row = short_row_ids[item];
        HPRLP_FLOAT acc = 0.0;
        for (int index = AT_rowPtr[row]; index < AT_rowPtr[row + 1];
             ++index) {
            const uint32_t entry = packed_entries[index];
            acc = fma(dictionary[entry & code_mask],
                      scaled_y[entry >> code_bits], acc);
        }
        const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
        finish_packed_dictionary_x(
            row, acc * inverse_col_scale, inverse_col_scale,
            scaled_x_hat_output, state_plan, x, x_hat, l, u,
            x_bound_type, c, last_x, sigma_params, halpern_factors);
        return;
    }

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int local_block =
        static_cast<int>(blockIdx.x) - short_block_count;
    const int item = local_block * warps_per_block + warp_in_block;
    if (item >= medium_row_count) return;
    const int row = medium_row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row] + lane; index < AT_rowPtr[row + 1];
         index += 32) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_y[entry >> code_bits], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane != 0) return;
    const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
    finish_packed_dictionary_x(
        row, acc * inverse_col_scale, inverse_col_scale,
        scaled_x_hat_output, state_plan, x, x_hat, l, u, x_bound_type,
        c, last_x, sigma_params, halpern_factors);
}

__global__ void packed_dictionary_update_x_rows_block_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row] + threadIdx.x;
         index < AT_rowPtr[row + 1]; index += blockDim.x) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_y[entry >> code_bits], acc);
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
            const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
            finish_packed_dictionary_x(
                row, acc * inverse_col_scale, inverse_col_scale,
                scaled_x_hat_output, state_plan, x, x_hat, l, u,
                x_bound_type, c, last_x, sigma_params, halpern_factors);
        }
    }
}

__global__ void packed_dictionary_update_x_fixed_degree_run_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries_soa,
    unsigned code_bits, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_begin, int row_count,
    int degree) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int row = row_begin + item;
    const HPRLP_FLOAT acc = hprlp_packed_fixed_degree_dot(
        dictionary, packed_entries_soa, scaled_y, item, row_count,
        degree, code_bits);
    const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
    finish_packed_dictionary_x(
        row, acc * inverse_col_scale, inverse_col_scale,
        scaled_x_hat_output, state_plan, x, x_hat, l, u, x_bound_type,
        c, last_x, sigma_params, halpern_factors);
}

__global__ void packed_dictionary_segmented_x_finalize_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan, const HPRLP_FLOAT *partials,
    const int *row_ids, const int *row_tile_ptr,
    const HPRLP_FLOAT *sigma_params,
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
            const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
            finish_packed_dictionary_x(
                row, acc * inverse_col_scale, inverse_col_scale,
                scaled_x_hat_output, state_plan, x, x_hat, l, u,
                x_bound_type, c, last_x, sigma_params,
                halpern_factors);
        }
    }
}

__global__ void u32_u16_dictionary_update_x_rows_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *AT_rowPtr,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row]; index < AT_rowPtr[row + 1]; ++index) {
        acc = fma(dictionary[value_codes[index]],
                  scaled_y[input_indices[index]], acc);
    }
    const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
    finish_packed_dictionary_x(
        row, acc * inverse_col_scale, inverse_col_scale,
        scaled_x_hat_output, state_plan, x, x_hat, l, u, x_bound_type,
        c, last_x, sigma_params, halpern_factors);
}

__global__ void u32_u16_dictionary_update_x_rows_warp_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *AT_rowPtr,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= nrows) return;
    const int row = row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row] + lane; index < AT_rowPtr[row + 1];
         index += 32) {
        acc = fma(dictionary[value_codes[index]],
                  scaled_y[input_indices[index]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
        finish_packed_dictionary_x(
            row, acc * inverse_col_scale, inverse_col_scale,
            scaled_x_hat_output, state_plan, x, x_hat, l, u,
            x_bound_type, c, last_x, sigma_params, halpern_factors);
    }
}

__global__ void u32_u16_dictionary_update_x_rows_block_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *AT_rowPtr,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT acc = 0.0;
    for (int index = AT_rowPtr[row] + threadIdx.x;
         index < AT_rowPtr[row + 1]; index += blockDim.x) {
        acc = fma(dictionary[value_codes[index]],
                  scaled_y[input_indices[index]], acc);
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
            const HPRLP_FLOAT inverse_col_scale = inverse_col_norm[row];
            finish_packed_dictionary_x(
                row, acc * inverse_col_scale, inverse_col_scale,
                scaled_x_hat_output, state_plan, x, x_hat, l, u,
                x_bound_type, c, last_x, sigma_params, halpern_factors);
        }
    }
}

__device__ __forceinline__ void finish_packed_dictionary_y(
    int row, HPRLP_FLOAT original_acc, HPRLP_FLOAT inverse_row_scale,
    HPRLP_FLOAT *scaled_y_output, HPRLPPackedStatePlan state_plan,
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT ax = original_acc * inverse_row_scale;
    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT v = fma(-sigma_params[1], yi, ax);
    HPRLP_FLOAT d;
    if (hprlp_index_in_state_run(
            row, state_plan.y_equality_begin,
            state_plan.y_equality_count)) {
        d = AL[row] - v;
    } else if (hprlp_index_in_state_run(
                   row, state_plan.y_lower_only_begin,
                   state_plan.y_lower_only_count)) {
        d = fmax(AL[row] - v, 0.0);
    } else if (hprlp_index_in_state_run(
                   row, state_plan.y_upper_only_begin,
                   state_plan.y_upper_only_count)) {
        d = fmin(AU[row] - v, 0.0);
    } else {
        d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
    }
    const HPRLP_FLOAT y_bar = sigma_params[2] * d;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    const HPRLP_FLOAT y_new = fma(
        halpern_factors[1], y_hat,
        halpern_factors[0] * last_y[row]);
    y[row] = y_new;
    if (scaled_y_output != nullptr) {
        scaled_y_output[row] = y_new * inverse_row_scale;
    }
}

__device__ __forceinline__ void finish_packed_dictionary_y_shifted(
    int row, HPRLP_FLOAT original_acc, HPRLP_FLOAT inverse_row_scale,
    HPRLP_FLOAT activity_shift, HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan, HPRLP_FLOAT *y,
    const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT ax =
        original_acc * inverse_row_scale + activity_shift;
    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT v = fma(-sigma_params[1], yi, ax);
    HPRLP_FLOAT d;
    if (hprlp_index_in_state_run(
            row, state_plan.y_equality_begin,
            state_plan.y_equality_count)) {
        d = AL[row] - v;
    } else if (hprlp_index_in_state_run(
                   row, state_plan.y_lower_only_begin,
                   state_plan.y_lower_only_count)) {
        d = fmax(AL[row] - v, 0.0);
    } else if (hprlp_index_in_state_run(
                   row, state_plan.y_upper_only_begin,
                   state_plan.y_upper_only_count)) {
        d = fmin(AU[row] - v, 0.0);
    } else {
        d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
    }
    const HPRLP_FLOAT y_bar = sigma_params[2] * d;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    const HPRLP_FLOAT y_new = fma(
        halpern_factors[1], y_hat,
        halpern_factors[0] * last_y[row]);
    y[row] = y_new;
    if (scaled_y_output != nullptr) {
        scaled_y_output[row] = y_new * inverse_row_scale;
    }
}

__device__ __forceinline__ HPRLP_FLOAT packed_dictionary_delta_shift(
    int row, HPRLP_FLOAT shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values) {
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        const int column = delta_columns[entry];
        const HPRLP_FLOAT fixed =
            delta_old_mask[column] == HPRLP_XBAR_AT_LOWER
                ? delta_lower[column]
                : (delta_old_mask[column] == HPRLP_XBAR_AT_UPPER
                       ? delta_upper[column]
                       : 0.0);
        shift = fma(delta_values[entry],
                    delta_x_hat[column] - fixed, shift);
    }
    return shift;
}

__global__ void packed_dictionary_update_y_rows_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row]; index < A_rowPtr[row + 1]; ++index) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    finish_packed_dictionary_y(
        row, acc, inverse_row_scale, scaled_y_output, state_plan, y, AL,
        AU, y_bound_type, last_y, sigma_params, halpern_factors);
}

__global__ void packed_dictionary_update_y_rows_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * (blockDim.x >> 5) +
                     (threadIdx.x >> 5);
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + lane; index < A_rowPtr[row + 1];
         index += 32) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
        finish_packed_dictionary_y(
            row, acc, inverse_row_scale, scaled_y_output, state_plan, y,
            AL, AU, y_bound_type, last_y, sigma_params, halpern_factors);
    }
}

__global__ void packed_dictionary_update_y_rows_block_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + threadIdx.x;
         index < A_rowPtr[row + 1]; index += blockDim.x) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
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
            const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
            finish_packed_dictionary_y(
                row, acc, inverse_row_scale, scaled_y_output, state_plan,
                y, AL, AU, y_bound_type, last_y, sigma_params,
                halpern_factors);
        }
    }
}

__global__ void packed_dictionary_update_y_rows_short_shifted_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row]; index < A_rowPtr[row + 1]; ++index) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    finish_packed_dictionary_y_shifted(
        row, acc, inverse_row_scale, activity_shift[row],
        scaled_y_output, state_plan, y, AL, AU, y_bound_type, last_y,
        sigma_params, halpern_factors);
}

__global__ void packed_dictionary_update_y_rows_warp_shifted_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * (blockDim.x >> 5) +
                     (threadIdx.x >> 5);
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + lane; index < A_rowPtr[row + 1];
         index += 32) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
        finish_packed_dictionary_y_shifted(
            row, acc, inverse_row_scale, activity_shift[row],
            scaled_y_output, state_plan, y, AL, AU, y_bound_type,
            last_y, sigma_params, halpern_factors);
    }
}

__global__ void packed_dictionary_update_y_rows_block_shifted_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + threadIdx.x;
         index < A_rowPtr[row + 1]; index += blockDim.x) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
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
            const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
            finish_packed_dictionary_y_shifted(
                row, acc, inverse_row_scale, activity_shift[row],
                scaled_y_output, state_plan, y, AL, AU, y_bound_type,
                last_y, sigma_params, halpern_factors);
        }
    }
}

__global__ void packed_dictionary_update_y_rows_short_shifted_delta_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const uint8_t *delta_old_mask,
    const int *delta_row_ptr, const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row]; index < A_rowPtr[row + 1]; ++index) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    const HPRLP_FLOAT shift = packed_dictionary_delta_shift(
        row, activity_shift[row], delta_x_hat, delta_lower, delta_upper,
        delta_old_mask, delta_row_ptr, delta_columns, delta_values);
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    finish_packed_dictionary_y_shifted(
        row, acc, inverse_row_scale, shift, scaled_y_output, state_plan,
        y, AL, AU, y_bound_type, last_y, sigma_params, halpern_factors);
}

__global__ void packed_dictionary_update_y_rows_warp_shifted_delta_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const uint8_t *delta_old_mask,
    const int *delta_row_ptr, const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * (blockDim.x >> 5) +
                     (threadIdx.x >> 5);
    if (item >= nrows) return;
    const int row = row_ids[item];
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + lane; index < A_rowPtr[row + 1];
         index += 32) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const HPRLP_FLOAT shift = packed_dictionary_delta_shift(
            row, activity_shift[row], delta_x_hat, delta_lower,
            delta_upper, delta_old_mask, delta_row_ptr, delta_columns,
            delta_values);
        const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
        finish_packed_dictionary_y_shifted(
            row, acc, inverse_row_scale, shift, scaled_y_output,
            state_plan, y, AL, AU, y_bound_type, last_y, sigma_params,
            halpern_factors);
    }
}

__global__ void packed_dictionary_update_y_rows_block_shifted_delta_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const uint8_t *delta_old_mask,
    const int *delta_row_ptr, const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + threadIdx.x;
         index < A_rowPtr[row + 1]; index += blockDim.x) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
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
            const HPRLP_FLOAT shift = packed_dictionary_delta_shift(
                row, activity_shift[row], delta_x_hat, delta_lower,
                delta_upper, delta_old_mask, delta_row_ptr,
                delta_columns, delta_values);
            const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
            finish_packed_dictionary_y_shifted(
                row, acc, inverse_row_scale, shift, scaled_y_output,
                state_plan, y, AL, AU, y_bound_type, last_y,
                sigma_params, halpern_factors);
        }
    }
}

__global__ void packed_dictionary_update_y_combined_short_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const uint8_t *delta_old_mask,
    const int *delta_row_ptr, const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *short_row_ids, int short_row_count,
    const int *medium_row_ids, int medium_row_count) {
    const int short_block_count =
        (short_row_count + blockDim.x - 1) / blockDim.x;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    if (static_cast<int>(blockIdx.x) < short_block_count) {
        const int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= short_row_count) return;
        const int row = short_row_ids[item];
        HPRLP_FLOAT acc = 0.0;
        for (int index = A_rowPtr[row]; index < A_rowPtr[row + 1];
             ++index) {
            const uint32_t entry = packed_entries[index];
            acc = fma(dictionary[entry & code_mask],
                      scaled_x_hat[entry >> code_bits], acc);
        }
        const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
        if (activity_shift != nullptr) {
            HPRLP_FLOAT shift = activity_shift[row];
            if (delta_row_ptr != nullptr) {
                shift = packed_dictionary_delta_shift(
                    row, shift, delta_x_hat, delta_lower, delta_upper,
                    delta_old_mask, delta_row_ptr, delta_columns,
                    delta_values);
            }
            finish_packed_dictionary_y_shifted(
                row, acc, inverse_row_scale, shift, scaled_y_output,
                state_plan, y, AL, AU, y_bound_type, last_y,
                sigma_params, halpern_factors);
        } else {
            finish_packed_dictionary_y(
                row, acc, inverse_row_scale, scaled_y_output, state_plan,
                y, AL, AU, y_bound_type, last_y, sigma_params,
                halpern_factors);
        }
        return;
    }

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int local_block =
        static_cast<int>(blockIdx.x) - short_block_count;
    const int item = local_block * warps_per_block + warp_in_block;
    if (item >= medium_row_count) return;
    const int row = medium_row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + lane; index < A_rowPtr[row + 1];
         index += 32) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane != 0) return;
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    if (activity_shift != nullptr) {
        HPRLP_FLOAT shift = activity_shift[row];
        if (delta_row_ptr != nullptr) {
            shift = packed_dictionary_delta_shift(
                row, shift, delta_x_hat, delta_lower, delta_upper,
                delta_old_mask, delta_row_ptr, delta_columns,
                delta_values);
        }
        finish_packed_dictionary_y_shifted(
            row, acc, inverse_row_scale, shift, scaled_y_output,
            state_plan, y, AL, AU, y_bound_type, last_y, sigma_params,
            halpern_factors);
    } else {
        finish_packed_dictionary_y(
            row, acc, inverse_row_scale, scaled_y_output, state_plan, y,
            AL, AU, y_bound_type, last_y, sigma_params,
            halpern_factors);
    }
}

__global__ void packed_dictionary_update_y_fixed_degree_run_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries_soa,
    unsigned code_bits, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_begin, int row_count,
    int degree) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int row = row_begin + item;
    const HPRLP_FLOAT acc = hprlp_packed_fixed_degree_dot(
        dictionary, packed_entries_soa, scaled_x_hat, item, row_count,
        degree, code_bits);
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    finish_packed_dictionary_y(
        row, acc, inverse_row_scale, scaled_y_output, state_plan, y, AL,
        AU, y_bound_type, last_y, sigma_params, halpern_factors);
}

__global__ void packed_dictionary_segmented_y_partial_kernel(
    HPRLP_FLOAT *partials, const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *tile_begin, const int *tile_end, unsigned code_bits,
    int tile_count) {
    const int tile = blockIdx.x;
    if (tile >= tile_count) return;
    const uint32_t code_mask = code_bits == 0
        ? 0u
        : ((UINT32_C(1) << code_bits) - 1u);
    HPRLP_FLOAT acc = 0.0;
    for (int index = tile_begin[tile] + threadIdx.x;
         index < tile_end[tile]; index += blockDim.x) {
        const uint32_t entry = packed_entries[index];
        acc = fma(dictionary[entry & code_mask],
                  scaled_x_hat[entry >> code_bits], acc);
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
        if (lane == 0) partials[tile] = acc;
    }
}

__global__ void packed_dictionary_segmented_y_finalize_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output, HPRLPPackedStatePlan state_plan,
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
            finish_packed_dictionary_y(
                row, acc, inverse_row_norm[row], scaled_y_output,
                state_plan, y, AL, AU, y_bound_type, last_y,
                sigma_params, halpern_factors);
        }
    }
}

__global__ void u32_u16_dictionary_update_y_rows_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *A_rowPtr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row]; index < A_rowPtr[row + 1]; ++index) {
        acc = fma(dictionary[value_codes[index]],
                  scaled_x_hat[input_indices[index]], acc);
    }
    const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
    finish_packed_dictionary_y(
        row, acc, inverse_row_scale, scaled_y_output, state_plan, y, AL,
        AU, y_bound_type, last_y, sigma_params, halpern_factors);
}

__global__ void u32_u16_dictionary_update_y_rows_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *A_rowPtr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * (blockDim.x >> 5) +
                     (threadIdx.x >> 5);
    if (item >= nrows) return;
    const int row = row_ids[item];
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + lane; index < A_rowPtr[row + 1];
         index += 32) {
        acc = fma(dictionary[value_codes[index]],
                  scaled_x_hat[input_indices[index]], acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
        finish_packed_dictionary_y(
            row, acc, inverse_row_scale, scaled_y_output, state_plan, y,
            AL, AU, y_bound_type, last_y, sigma_params, halpern_factors);
    }
}

__global__ void u32_u16_dictionary_update_y_rows_block_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *A_rowPtr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    const int item = blockIdx.x;
    if (item >= nrows) return;
    const int row = row_ids[item];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT acc = 0.0;
    for (int index = A_rowPtr[row] + threadIdx.x;
         index < A_rowPtr[row + 1]; index += blockDim.x) {
        acc = fma(dictionary[value_codes[index]],
                  scaled_x_hat[input_indices[index]], acc);
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
            const HPRLP_FLOAT inverse_row_scale = inverse_row_norm[row];
            finish_packed_dictionary_y(
                row, acc, inverse_row_scale, scaled_y_output, state_plan,
                y, AL, AU, y_bound_type, last_y, sigma_params,
                halpern_factors);
        }
    }
}
