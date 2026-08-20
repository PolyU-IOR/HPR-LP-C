#include "cuda_kernels/backends/structured/factorized_stencil_kernels.cuh"

namespace {

__device__ __forceinline__ HPRLP_FLOAT factorized_project_x(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t bound_type) {
    if (bound_type == 0) return value;
    if (bound_type == 1) return fmax(value, lower);
    if (bound_type == 2) return fmin(value, upper);
    return fmin(fmax(value, lower), upper);
}

__device__ __forceinline__ HPRLP_FLOAT factorized_project_y_delta(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t bound_type) {
    if (bound_type == 0) return 0.0;
    if (bound_type == 1) return fmax(lower - value, 0.0);
    if (bound_type == 2) return fmin(upper - value, 0.0);
    return fmax(lower - value, fmin(upper - value, 0.0));
}

__device__ __forceinline__ void factorized_finish_x_static(
    int col, HPRLP_FLOAT aty, HPRLP_FLOAT *x,
    const HPRLP_FLOAT *last_x, const std::uint8_t *static_codes,
    const HPRLP_factorized_x_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    HPRLP_FLOAT inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_out, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    HPRLP_factorized_x_static_record record;
    const std::uint8_t code = static_codes[col];
    if (code == static_cast<std::uint8_t>(255)) {
        record.lower = lower[col];
        record.upper = upper[col];
        record.objective = objective[col];
        record.bound_type = bound_type[col];
    } else {
        record = static_records[code];
    }
    const HPRLP_FLOAT xi = x[col];
    const HPRLP_FLOAT z_temp = fma(
        sigma_params[0], aty - record.objective, xi);
    const HPRLP_FLOAT x_bar = factorized_project_x(
        z_temp, record.lower, record.upper, record.bound_type);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[col] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[col]);
    scaled_x_hat_out[col] = x_hat_value * inverse_col_norm;
}

__device__ __forceinline__ void factorized_finish_y_static(
    int row, HPRLP_FLOAT ax, HPRLP_FLOAT *y,
    const HPRLP_FLOAT *last_y, const std::uint8_t *static_codes,
    const HPRLP_factorized_y_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    HPRLP_FLOAT inverse_row_norm, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    HPRLP_factorized_y_static_record record;
    const std::uint8_t code = static_codes[row];
    if (code == static_cast<std::uint8_t>(255)) {
        record.lower = lower[row];
        record.upper = upper[row];
        record.bound_type = bound_type[row];
    } else {
        record = static_records[code];
    }
    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT v = fma(-sigma_params[1], yi, ax);
    const HPRLP_FLOAT delta = factorized_project_y_delta(
        v, record.lower, record.upper, record.bound_type);
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    const HPRLP_FLOAT y_new = fma(
        halpern_factors[1], y_hat, halpern_factors[0] * last_y[row]);
    y[row] = y_new;
    scaled_y_out[row] = y_new * inverse_row_norm;
}

}  // namespace

namespace {

__device__ __forceinline__ HPRLP_FLOAT windowed_stencil_diagonal_device(
    const HPRLPWindowedStencilShape &shape, int gy, int gx) {
    const bool boundary_y =
        gy == 0 || gy == shape.grid_height - 1;
    if (boundary_y) {
        return gx == 0 ? shape.diagonal_boundary_left
                       : shape.diagonal_boundary;
    }
    return gx == 0 ? shape.diagonal_interior_left
                   : shape.diagonal_interior;
}

__device__ __forceinline__ void windowed_finish_x_boxed_zero(
    int col, HPRLP_FLOAT aty, HPRLP_FLOAT *x,
    HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    const HPRLP_FLOAT *last_x,
    HPRLP_FLOAT inverse_col_norm, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT xi = x[col];
    const HPRLP_FLOAT z_temp = fma(sigma_params[0], aty, xi);
    const HPRLP_FLOAT x_bar = fmin(fmax(z_temp, lower), upper);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[col] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[col]);
    scaled_x_hat_out[col] = x_hat_value * inverse_col_norm;
}

__device__ __forceinline__ void windowed_x_box_bounds(
    int col, const std::uint8_t *static_codes,
    const HPRLP_factorized_x_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    HPRLP_FLOAT *lower_out, HPRLP_FLOAT *upper_out) {
    const std::uint8_t code = static_codes[col];
    if (code == static_cast<std::uint8_t>(255)) {
        *lower_out = lower[col];
        *upper_out = upper[col];
    } else {
        *lower_out = static_records[code].lower;
        *upper_out = static_records[code].upper;
    }
}

__device__ __forceinline__ HPRLP_FLOAT windowed_y_bound(
    int row, std::uint8_t bound_type,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_y_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper) {
    const std::uint8_t code = static_codes[row];
    if (code == static_cast<std::uint8_t>(255)) {
        return bound_type == 2 ? upper[row] : lower[row];
    }
    return bound_type == 2 ? static_records[code].upper
                           : static_records[code].lower;
}

__device__ __forceinline__ void windowed_finish_x_exception(
    int col, HPRLP_FLOAT aty, HPRLP_FLOAT *x,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x, HPRLP_FLOAT inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_out, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT xi = x[col];
    const HPRLP_FLOAT z_temp = fma(
        sigma_params[0], aty - objective[col], xi);
    const HPRLP_FLOAT x_bar = factorized_project_x(
        z_temp, lower[col], upper[col], bound_type[col]);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[col] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[col]);
    scaled_x_hat_out[col] = x_hat_value * inverse_col_norm;
}

__device__ __forceinline__ void windowed_finish_y_state(
    int row, HPRLP_FLOAT ax, HPRLP_FLOAT *y,
    const HPRLP_FLOAT *last_y, HPRLP_FLOAT bound,
    std::uint8_t bound_type, HPRLP_FLOAT inverse_row_norm,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT v = fma(-sigma_params[1], yi, ax);
    HPRLP_FLOAT delta;
    if (bound_type == 1) {
        delta = fmax(bound - v, 0.0);
    } else if (bound_type == 2) {
        delta = fmin(bound - v, 0.0);
    } else {
        // The host gate requires bitwise-identical lower/upper equality
        // bounds.  The canonical clamp therefore returns this same delta.
        delta = bound - v;
    }
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    const HPRLP_FLOAT y_new = fma(
        halpern_factors[1], y_hat, halpern_factors[0] * last_y[row]);
    y[row] = y_new;
    scaled_y_out[row] = y_new * inverse_row_norm;
}

}  // namespace

__global__ void windowed_stencil_update_x_grid_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *last_x,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_x_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_out, HPRLP_FLOAT *dense_partials,
    unsigned int *dense_counter,
    HPRLPWindowedStencilShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int gy = blockIdx.x;
    const int gx = threadIdx.x;
    if (gy >= shape.grid_height) return;

    if (gx < shape.grid_width) {
        const int col =
            shape.grid_col0 + shape.grid_width * gy + gx;
        const HPRLP_FLOAT inverse_norm = inverse_col_norm[col];
        HPRLP_FLOAT acc = 0.0;
        const bool observed =
            gy >= shape.observation_y0 &&
            gy < shape.observation_y0 + shape.observation_height &&
            gx >= shape.observation_x0 &&
            gx < shape.observation_x0 + shape.observation_width;
        if (observed) {
            const int observation = shape.observation_width *
                (gy - shape.observation_y0) +
                gx - shape.observation_x0;
            acc = fma(shape.observation_coefficient,
                      scaled_y[observation], acc);
            acc = fma(shape.observation_coefficient,
                      scaled_y[shape.observation_count + observation],
                      acc);
        }
        if (gy > 0 && gx < shape.equation_width) {
            const int row = shape.stencil_row0 +
                shape.equation_width * (gy - 1) + gx;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        if (gx > 0) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx - 1;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        if (gx < shape.equation_width) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx;
            acc = fma(windowed_stencil_diagonal_device(shape, gy, gx),
                      scaled_y[row], acc);
        }
        if (gx < shape.equation_width - 1) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx + 1;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        if (gy < shape.grid_height - 1 &&
            gx < shape.equation_width) {
            const int row = shape.stencil_row0 +
                shape.equation_width * (gy + 1) + gx;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        factorized_finish_x_static(
            col, acc * inverse_norm, x, last_x, static_codes,
            static_records, lower, upper, bound_type, objective,
            inverse_norm, scaled_x_hat_out, sigma_params,
            halpern_factors);
    }

    if (gy >= shape.observation_y0 &&
        gy < shape.observation_y0 + shape.observation_height) {
        HPRLP_FLOAT dense_term = 0.0;
        if (gx < shape.observation_width) {
            const int observation = shape.observation_width *
                (gy - shape.observation_y0) + gx;
            dense_term = fma(
                shape.dense_negative, scaled_y[observation],
                shape.dense_positive *
                    scaled_y[shape.observation_count + observation]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dense_term +=
                __shfl_down_sync(0xffffffff, dense_term, offset);
        }
        __shared__ HPRLP_FLOAT warp_sums[32];
        __shared__ int is_last_observation_block;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warp_count = blockDim.x >> 5;
        if (lane == 0) warp_sums[warp] = dense_term;
        __syncthreads();
        if (warp == 0) {
            dense_term = lane < warp_count ? warp_sums[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                dense_term +=
                    __shfl_down_sync(0xffffffff, dense_term, offset);
            }
            if (lane == 0) {
                dense_partials[gy - shape.observation_y0] = dense_term;
                __threadfence();
                const unsigned int ticket =
                    atomicAdd(dense_counter, 1u);
                is_last_observation_block =
                    ticket == static_cast<unsigned int>(
                                  shape.observation_height - 1);
            }
        }
        __syncthreads();

        if (!is_last_observation_block) return;

        HPRLP_FLOAT acc = 0.0;
        for (int index = gx; index < shape.observation_height;
             index += blockDim.x) {
            acc += dense_partials[index];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) warp_sums[warp] = acc;
        __syncthreads();
        if (warp != 0) return;

        acc = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const int col = shape.dense_col;
            const HPRLP_FLOAT inverse_norm = inverse_col_norm[col];
            factorized_finish_x_static(
                col, acc * inverse_norm, x, last_x, static_codes,
                static_records, lower, upper, bound_type, objective,
                inverse_norm, scaled_x_hat_out, sigma_params,
                halpern_factors);
            *dense_counter = 0u;
        }
    }
}

__global__ void windowed_stencil_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *last_y,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_y_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *scaled_x_hat, const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_out, HPRLPWindowedStencilShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    if (blockIdx.x < shape.observation_height) {
        const int packed = threadIdx.x;
        if (packed >= 2 * shape.observation_width) return;
        const int half = packed >= shape.observation_width;
        const int ox = packed - half * shape.observation_width;
        const int oy = blockIdx.x;
        const int observation = shape.observation_width * oy + ox;
        const int row = half * shape.observation_count + observation;
        const int col = shape.grid_col0 +
            shape.grid_width * (shape.observation_y0 + oy) +
            shape.observation_x0 + ox;
        const HPRLP_FLOAT inverse_norm = inverse_row_norm[row];
        HPRLP_FLOAT acc = fma(
            half == 0 ? shape.dense_negative : shape.dense_positive,
            scaled_x_hat[shape.dense_col], 0.0);
        acc = fma(shape.observation_coefficient,
                  scaled_x_hat[col], acc);
        factorized_finish_y_static(
            row, acc * inverse_norm, y, last_y, static_codes,
            static_records, lower, upper, bound_type, inverse_norm,
            scaled_y_out, sigma_params, halpern_factors);
        return;
    }

    const int gy = blockIdx.x - shape.observation_height;
    const int gx = threadIdx.x;
    if (gy >= shape.grid_height || gx >= shape.equation_width) return;
    const int row = shape.stencil_row0 +
        shape.equation_width * gy + gx;
    const int center =
        shape.grid_col0 + shape.grid_width * gy + gx;
    const HPRLP_FLOAT inverse_norm = inverse_row_norm[row];
    HPRLP_FLOAT acc = 0.0;
    if (gy > 0) {
        acc = fma(shape.off_diagonal,
                  scaled_x_hat[center - shape.grid_width], acc);
    }
    if (gx > 0) {
        acc = fma(shape.off_diagonal, scaled_x_hat[center - 1], acc);
    }
    acc = fma(windowed_stencil_diagonal_device(shape, gy, gx),
              scaled_x_hat[center], acc);
    acc = fma(shape.off_diagonal, scaled_x_hat[center + 1], acc);
    if (gy < shape.grid_height - 1) {
        acc = fma(shape.off_diagonal,
                  scaled_x_hat[center + shape.grid_width], acc);
    }
    factorized_finish_y_static(
        row, acc * inverse_norm, y, last_y, static_codes,
        static_records, lower, upper, bound_type, inverse_norm,
        scaled_y_out, sigma_params, halpern_factors);
}

__global__ void windowed_stencil_state_update_x_grid_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *last_x,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_x_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_out, HPRLP_FLOAT *dense_partials,
    unsigned int *dense_counter, HPRLPWindowedStencilShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int gy = blockIdx.x;
    const int gx = threadIdx.x;
    if (gy >= shape.grid_height) return;

    if (gx < shape.grid_width) {
        const int col =
            shape.grid_col0 + shape.grid_width * gy + gx;
        const HPRLP_FLOAT inverse_norm = inverse_col_norm[col];
        HPRLP_FLOAT acc = 0.0;
        const bool observed =
            gy >= shape.observation_y0 &&
            gy < shape.observation_y0 + shape.observation_height &&
            gx >= shape.observation_x0 &&
            gx < shape.observation_x0 + shape.observation_width;
        if (observed) {
            const int observation = shape.observation_width *
                (gy - shape.observation_y0) +
                gx - shape.observation_x0;
            acc = fma(shape.observation_coefficient,
                      scaled_y[observation], acc);
            acc = fma(shape.observation_coefficient,
                      scaled_y[shape.observation_count + observation],
                      acc);
        }
        if (gy > 0 && gx < shape.equation_width) {
            const int row = shape.stencil_row0 +
                shape.equation_width * (gy - 1) + gx;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        if (gx > 0) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx - 1;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        if (gx < shape.equation_width) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx;
            acc = fma(windowed_stencil_diagonal_device(shape, gy, gx),
                      scaled_y[row], acc);
        }
        if (gx < shape.equation_width - 1) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx + 1;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        if (gy < shape.grid_height - 1 &&
            gx < shape.equation_width) {
            const int row = shape.stencil_row0 +
                shape.equation_width * (gy + 1) + gx;
            acc = fma(shape.off_diagonal, scaled_y[row], acc);
        }
        HPRLP_FLOAT lower_bound;
        HPRLP_FLOAT upper_bound;
        windowed_x_box_bounds(
            col, static_codes, static_records, lower, upper,
            &lower_bound, &upper_bound);
        windowed_finish_x_boxed_zero(
            col, acc * inverse_norm, x, lower_bound, upper_bound,
            last_x, inverse_norm, scaled_x_hat_out, sigma_params,
            halpern_factors);
    }

    if (gy >= shape.observation_y0 &&
        gy < shape.observation_y0 + shape.observation_height) {
        HPRLP_FLOAT dense_term = 0.0;
        if (gx < shape.observation_width) {
            const int observation = shape.observation_width *
                (gy - shape.observation_y0) + gx;
            dense_term = fma(
                shape.dense_negative, scaled_y[observation],
                shape.dense_positive *
                    scaled_y[shape.observation_count + observation]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dense_term +=
                __shfl_down_sync(0xffffffff, dense_term, offset);
        }
        __shared__ HPRLP_FLOAT warp_sums[32];
        __shared__ int is_last_observation_block;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warp_count = blockDim.x >> 5;
        if (lane == 0) warp_sums[warp] = dense_term;
        __syncthreads();
        if (warp == 0) {
            dense_term = lane < warp_count ? warp_sums[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                dense_term +=
                    __shfl_down_sync(0xffffffff, dense_term, offset);
            }
            if (lane == 0) {
                dense_partials[gy - shape.observation_y0] = dense_term;
                __threadfence();
                const unsigned int ticket = atomicAdd(dense_counter, 1u);
                is_last_observation_block =
                    ticket == static_cast<unsigned int>(
                                  shape.observation_height - 1);
            }
        }
        __syncthreads();
        if (!is_last_observation_block) return;

        HPRLP_FLOAT acc = 0.0;
        for (int index = gx; index < shape.observation_height;
             index += blockDim.x) {
            acc += dense_partials[index];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) warp_sums[warp] = acc;
        __syncthreads();
        if (warp != 0) return;
        acc = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const int col = shape.dense_col;
            const HPRLP_FLOAT inverse_norm = inverse_col_norm[col];
            windowed_finish_x_exception(
                col, acc * inverse_norm, x, lower, upper, bound_type,
                objective, last_x, inverse_norm, scaled_x_hat_out,
                sigma_params, halpern_factors);
            *dense_counter = 0u;
        }
    }
}

__global__ void windowed_stencil_state_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *last_y,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_y_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *scaled_x_hat, const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_out, HPRLPWindowedStencilShape shape,
    std::uint8_t observation_first_bound_type,
    std::uint8_t observation_second_bound_type,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    if (blockIdx.x < shape.observation_height) {
        const int packed = threadIdx.x;
        if (packed >= 2 * shape.observation_width) return;
        const int half = packed >= shape.observation_width;
        const int ox = packed - half * shape.observation_width;
        const int oy = blockIdx.x;
        const int observation = shape.observation_width * oy + ox;
        const int row = half * shape.observation_count + observation;
        const int col = shape.grid_col0 +
            shape.grid_width * (shape.observation_y0 + oy) +
            shape.observation_x0 + ox;
        const HPRLP_FLOAT inverse_norm = inverse_row_norm[row];
        HPRLP_FLOAT acc = fma(
            half == 0 ? shape.dense_negative : shape.dense_positive,
            scaled_x_hat[shape.dense_col], 0.0);
        acc = fma(shape.observation_coefficient,
                  scaled_x_hat[col], acc);
        const std::uint8_t row_type =
            half == 0 ? observation_first_bound_type
                      : observation_second_bound_type;
        const HPRLP_FLOAT bound = windowed_y_bound(
            row, row_type, static_codes, static_records, lower, upper);
        windowed_finish_y_state(
            row, acc * inverse_norm, y, last_y, bound, row_type,
            inverse_norm, scaled_y_out, sigma_params, halpern_factors);
        return;
    }

    const int gy = blockIdx.x - shape.observation_height;
    const int gx = threadIdx.x;
    if (gy >= shape.grid_height || gx >= shape.equation_width) return;
    const int row = shape.stencil_row0 +
        shape.equation_width * gy + gx;
    const int center = shape.grid_col0 + shape.grid_width * gy + gx;
    const HPRLP_FLOAT inverse_norm = inverse_row_norm[row];
    HPRLP_FLOAT acc = 0.0;
    if (gy > 0) {
        acc = fma(shape.off_diagonal,
                  scaled_x_hat[center - shape.grid_width], acc);
    }
    if (gx > 0) {
        acc = fma(shape.off_diagonal, scaled_x_hat[center - 1], acc);
    }
    acc = fma(windowed_stencil_diagonal_device(shape, gy, gx),
              scaled_x_hat[center], acc);
    acc = fma(shape.off_diagonal, scaled_x_hat[center + 1], acc);
    if (gy < shape.grid_height - 1) {
        acc = fma(shape.off_diagonal,
                  scaled_x_hat[center + shape.grid_width], acc);
    }
    windowed_finish_y_state(
        row, acc * inverse_norm, y, last_y,
        windowed_y_bound(
            row, 3, static_codes, static_records, lower, upper), 3,
        inverse_norm, scaled_y_out, sigma_params, halpern_factors);
}
