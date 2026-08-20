#include "cuda_kernels/backends/unit/signed_unit_kernels.cuh"

namespace {

template <typename Entry>
struct SignedEntryTraits;

template <>
struct SignedEntryTraits<std::uint16_t> {
    __device__ __forceinline__ static int index(std::uint16_t entry) {
        return static_cast<int>(entry & 0x7fffu);
    }
    __device__ __forceinline__ static bool negative(std::uint16_t entry) {
        return (entry & 0x8000u) != 0;
    }
};

template <>
struct SignedEntryTraits<std::uint32_t> {
    __device__ __forceinline__ static int index(std::uint32_t entry) {
        return static_cast<int>(entry & 0x7fffffffu);
    }
    __device__ __forceinline__ static bool negative(std::uint32_t entry) {
        return (entry & 0x80000000u) != 0;
    }
};

template <typename Entry>
__device__ __forceinline__ HPRLP_FLOAT signed_input(
    Entry entry, const HPRLP_FLOAT *input) {
    const HPRLP_FLOAT value = input[SignedEntryTraits<Entry>::index(entry)];
    return SignedEntryTraits<Entry>::negative(entry) ? -value : value;
}

__device__ __forceinline__ HPRLP_FLOAT signed_delta_input(
    std::uint32_t entry, const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *scaled_fixed) {
    const int index = SignedEntryTraits<std::uint32_t>::index(entry);
    const HPRLP_FLOAT value =
        scaled_x_hat[index] - scaled_fixed[index];
    return SignedEntryTraits<std::uint32_t>::negative(entry)
        ? -value : value;
}

template <typename Entry>
__device__ __forceinline__ HPRLP_FLOAT signed_input_skip_positive_zero(
    Entry entry, const HPRLP_FLOAT *input,
    const std::uint8_t *input_nonzero) {
    const int index = SignedEntryTraits<Entry>::index(entry);
    if (input_nonzero[index] == 0) {
        return 0.0;
    }
    const HPRLP_FLOAT value = input[index];
    return SignedEntryTraits<Entry>::negative(entry) ? -value : value;
}

__device__ __forceinline__ HPRLP_FLOAT signed_project_x(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t bound_type) {
    if (bound_type == 0) {
        return value;
    }
    if (bound_type == 1) {
        return fmax(value, lower);
    }
    if (bound_type == 2) {
        return fmin(value, upper);
    }
    return fmin(fmax(value, lower), upper);
}

__device__ __forceinline__ HPRLP_FLOAT signed_project_y_delta(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t bound_type) {
    if (bound_type == 0) {
        return 0.0;
    }
    if (bound_type == 1) {
        return fmax(lower - value, 0.0);
    }
    if (bound_type == 2) {
        return fmin(upper - value, 0.0);
    }
    return fmax(lower - value, fmin(upper - value, 0.0));
}

__device__ __forceinline__ void signed_finish_x(
    int row, HPRLP_FLOAT sum, HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat,
    HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT aty = sum * inverse_col_norm[row];
    const HPRLP_FLOAT current_x = x[row];
    const HPRLP_FLOAT projected = signed_project_x(
        fma(sigma_params[0], aty - objective[row], current_x),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current_x;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    // A null output selects the signed/signed single-state producer.  The
    // transformed cache below is the normal Y input; canonical check
    // iterations still pass a real x_hat pointer and reconstruct it exactly.
    if (x_hat != nullptr) {
        x_hat[row] = reflected;
    }
    const HPRLP_FLOAT scaled_reflected =
        reflected * inverse_col_norm[row];
    scaled_x_hat_out[row] = scaled_reflected;
    if (scaled_x_hat_nonzero_out != nullptr) {
        scaled_x_hat_nonzero_out[row] =
            __double_as_longlong(scaled_reflected) == 0ll ? 0 : 1;
    }
}

__device__ __forceinline__ void signed_finish_y(
    int row, HPRLP_FLOAT sum, HPRLP_FLOAT *y,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT ax = sum * inverse_row_norm[row];
    const HPRLP_FLOAT current_y = y[row];
    const HPRLP_FLOAT value = fma(-sigma_params[1], current_y, ax);
    const HPRLP_FLOAT delta = signed_project_y_delta(
        value, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT projected = sigma_params[2] * delta;
    const HPRLP_FLOAT reflected = 2.0 * projected - current_y;
    const HPRLP_FLOAT next_y = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_y[row]);
    y[row] = next_y;
    scaled_y_out[row] = next_y * inverse_row_norm[row];
}

__device__ __forceinline__ void signed_finish_y_shifted(
    int row, HPRLP_FLOAT sum, HPRLP_FLOAT *y,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT ax =
        sum * inverse_row_norm[row] + activity_shift[row];
    const HPRLP_FLOAT current_y = y[row];
    const HPRLP_FLOAT value = fma(-sigma_params[1], current_y, ax);
    const HPRLP_FLOAT delta = signed_project_y_delta(
        value, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT projected = sigma_params[2] * delta;
    const HPRLP_FLOAT reflected = 2.0 * projected - current_y;
    const HPRLP_FLOAT next_y = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_y[row]);
    y[row] = next_y;
    scaled_y_out[row] = next_y * inverse_row_norm[row];
}

__device__ __forceinline__ void signed_finish_x_state_specialized(
    int row, HPRLP_FLOAT sum, HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat,
    HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *upper, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int zero_objective_state_run_begin,
    int zero_objective_state_run_count) {
    const HPRLP_FLOAT aty = sum * inverse_col_norm[row];
    const unsigned int zero_objective_state_offset =
        static_cast<unsigned int>(row - zero_objective_state_run_begin);
    const HPRLP_FLOAT objective_value =
        zero_objective_state_offset <
                static_cast<unsigned int>(zero_objective_state_run_count)
            ? 0.0
            : objective[row];
    const HPRLP_FLOAT current_x = x[row];
    const HPRLP_FLOAT projected = fmin(
        fmax(fma(sigma_params[0], aty - objective_value, current_x), 0.0),
        upper[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current_x;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    if (x_hat != nullptr) {
        x_hat[row] = reflected;
    }
    const HPRLP_FLOAT scaled_reflected =
        reflected * inverse_col_norm[row];
    scaled_x_hat_out[row] = scaled_reflected;
    if (scaled_x_hat_nonzero_out != nullptr) {
        scaled_x_hat_nonzero_out[row] =
            __double_as_longlong(scaled_reflected) == 0ll ? 0 : 1;
    }
}

__device__ __forceinline__ void signed_finish_y_state_specialized(
    int row, HPRLP_FLOAT sum, HPRLP_FLOAT *y,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int upper_zero_state_run_begin,
    int upper_zero_state_run_count) {
    const HPRLP_FLOAT ax = sum * inverse_row_norm[row];
    const HPRLP_FLOAT current_y = y[row];
    const HPRLP_FLOAT value = fma(-sigma_params[1], current_y, ax);
    const unsigned int upper_zero_state_offset =
        static_cast<unsigned int>(row - upper_zero_state_run_begin);
    const HPRLP_FLOAT delta =
        upper_zero_state_offset <
                static_cast<unsigned int>(upper_zero_state_run_count)
        ? fmin(0.0 - value, 0.0)
        : signed_project_y_delta(
              value, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT projected = sigma_params[2] * delta;
    const HPRLP_FLOAT reflected = 2.0 * projected - current_y;
    const HPRLP_FLOAT next_y = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_y[row]);
    y[row] = next_y;
    scaled_y_out[row] = next_y * inverse_row_norm[row];
}

template <typename Entry>
__device__ __forceinline__ void signed_x_rows_short(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count) {
    const int row_position = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_position >= row_count) {
        return;
    }
    const int row = row_ids[row_position];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum += signed_input(entries[entry], scaled_y);
    }
    signed_finish_x(row, sum, x, x_hat, scaled_x_hat_out,
                    scaled_x_hat_nonzero_out,
                    lower, upper, bound_type,
                    objective, last_x, inverse_col_norm, sigma_params,
                    halpern_factors);
}

template <typename Entry>
__device__ __forceinline__ void signed_x_all_scalar(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum += signed_input(entries[entry], scaled_y);
    }
    signed_finish_x(row, sum, x, x_hat, scaled_x_hat_out,
                    scaled_x_hat_nonzero_out,
                    lower, upper, bound_type,
                    objective, last_x, inverse_col_norm, sigma_params,
                    halpern_factors);
}

__device__ __forceinline__ void signed_x_all_scalar_split_u16(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *indices, const std::uint8_t *negative,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        const HPRLP_FLOAT value = scaled_y[indices[entry]];
        sum += negative[entry] != 0 ? -value : value;
    }
    signed_finish_x(row, sum, x, x_hat, scaled_x_hat_out,
                    scaled_x_hat_nonzero_out, lower, upper, bound_type,
                    objective, last_x, inverse_col_norm, sigma_params,
                    halpern_factors);
}

__device__ __forceinline__ void signed_x_all_scalar_degree3_run_u32(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    const unsigned int run_offset =
        static_cast<unsigned int>(row - run_row_begin);
    if (run_offset < static_cast<unsigned int>(run_row_count)) {
        const int entry =
            run_entry_begin + 3 * static_cast<int>(run_offset);
        sum += signed_input(entries[entry], scaled_y);
        sum += signed_input(entries[entry + 1], scaled_y);
        sum += signed_input(entries[entry + 2], scaled_y);
    } else {
        for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
            sum += signed_input(entries[entry], scaled_y);
        }
    }
    signed_finish_x(row, sum, x, x_hat, scaled_x_hat_out,
                    scaled_x_hat_nonzero_out, lower, upper, bound_type,
                    objective, last_x, inverse_col_norm, sigma_params,
                    halpern_factors);
}

__device__ __forceinline__ void
signed_x_all_scalar_degree3_run_state_specialized_u32(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int zero_objective_state_run_begin,
    int zero_objective_state_run_count, int run_row_begin,
    int run_row_count, int run_entry_begin) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    const unsigned int run_offset =
        static_cast<unsigned int>(row - run_row_begin);
    if (run_offset < static_cast<unsigned int>(run_row_count)) {
        const int entry =
            run_entry_begin + 3 * static_cast<int>(run_offset);
        sum += signed_input(entries[entry], scaled_y);
        sum += signed_input(entries[entry + 1], scaled_y);
        sum += signed_input(entries[entry + 2], scaled_y);
    } else {
        for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
            sum += signed_input(entries[entry], scaled_y);
        }
    }
    signed_finish_x_state_specialized(
        row, sum, x, x_hat, scaled_x_hat_out,
        scaled_x_hat_nonzero_out, upper, objective, last_x,
        inverse_col_norm, sigma_params, halpern_factors,
        zero_objective_state_run_begin,
        zero_objective_state_run_count);
}

template <typename Entry>
__device__ __forceinline__ void signed_x_rows_warp(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row_position = blockIdx.x * warps_per_block + warp_in_block;
    if (row_position >= row_count) {
        return;
    }
    const int row = row_ids[row_position];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + lane; entry < row_ptr[row + 1];
         entry += 32) {
        sum += signed_input(entries[entry], scaled_y);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        signed_finish_x(row, sum, x, x_hat, scaled_x_hat_out,
                        scaled_x_hat_nonzero_out,
                        lower, upper, bound_type,
                        objective, last_x, inverse_col_norm, sigma_params,
                        halpern_factors);
    }
}

template <typename Entry>
__device__ __forceinline__ void signed_x_rows_block(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count, HPRLP_FLOAT *warp_sums) {
    const int row_position = blockIdx.x;
    if (row_position >= row_count) {
        return;
    }
    const int row = row_ids[row_position];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        sum += signed_input(entries[entry], scaled_y);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            signed_finish_x(row, sum, x, x_hat, scaled_x_hat_out,
                            scaled_x_hat_nonzero_out,
                            lower, upper,
                            bound_type, objective, last_x,
                            inverse_col_norm, sigma_params,
                            halpern_factors);
        }
    }
}

template <typename Entry, bool SkipPositiveZero, bool AddActivityShift>
__device__ __forceinline__ void signed_y_rows_short(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count) {
    const int row_position = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_position >= row_count) {
        return;
    }
    const int row = row_ids[row_position];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
    }
    if (AddActivityShift) {
        signed_finish_y_shifted(
            row, sum, y, scaled_y_out, lower, upper, bound_type, last_y,
            inverse_row_norm, activity_shift, sigma_params,
            halpern_factors);
    } else {
        signed_finish_y(row, sum, y, scaled_y_out,
                        lower, upper, bound_type, last_y,
                        inverse_row_norm, sigma_params, halpern_factors);
    }
}

template <bool SkipPositiveZero>
__device__ __forceinline__ void
signed_y_direct_short_degree2_run_state_specialized_u32(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int upper_zero_state_run_begin, int upper_zero_state_run_count,
    int run_row_begin,
    int run_row_count, int run_entry_begin) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    const unsigned int run_offset =
        static_cast<unsigned int>(row - run_row_begin);
    if (run_offset < static_cast<unsigned int>(run_row_count)) {
        const int entry =
            run_entry_begin + 2 * static_cast<int>(run_offset);
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry + 1], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry + 1], scaled_x);
    } else {
        const int begin = row_ptr[row];
        const int end = row_ptr[row + 1];
        if (end - begin > 16) {
            return;
        }
        for (int entry = begin; entry < end; ++entry) {
            sum += SkipPositiveZero
                ? signed_input_skip_positive_zero(
                      entries[entry], scaled_x, scaled_x_nonzero)
                : signed_input(entries[entry], scaled_x);
        }
    }
    signed_finish_y_state_specialized(
        row, sum, y, scaled_y_out, lower, upper, bound_type,
        last_y, inverse_row_norm, sigma_params, halpern_factors,
        upper_zero_state_run_begin, upper_zero_state_run_count);
}

template <typename Entry, bool SkipPositiveZero, bool AddActivityShift>
__device__ __forceinline__ void signed_y_all_scalar(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
    }
    if (AddActivityShift) {
        signed_finish_y_shifted(
            row, sum, y, scaled_y_out, lower, upper, bound_type, last_y,
            inverse_row_norm, activity_shift, sigma_params,
            halpern_factors);
    } else {
        signed_finish_y(row, sum, y, scaled_y_out,
                        lower, upper, bound_type, last_y,
                        inverse_row_norm, sigma_params, halpern_factors);
    }
}

template <typename Entry>
__device__ __forceinline__ void signed_y_all_scalar_delta(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum += signed_input(entries[entry], scaled_x);
    }
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        sum += signed_delta_input(
            delta_entries[entry], delta_scaled_x_hat,
            delta_scaled_fixed);
    }
    signed_finish_y_shifted(
        row, sum, y, scaled_y_out, lower, upper, bound_type, last_y,
        inverse_row_norm, activity_shift, sigma_params, halpern_factors);
}

template <typename Entry, bool SkipPositiveZero>
__device__ __forceinline__ void signed_y_direct_short(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];
    if (end - begin > 16) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    for (int entry = begin; entry < end; ++entry) {
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
    }
    signed_finish_y(row, sum, y, scaled_y_out,
                    lower, upper, bound_type, last_y,
                    inverse_row_norm, sigma_params, halpern_factors);
}

template <bool SkipPositiveZero>
__device__ __forceinline__ void signed_y_direct_short_degree2_run_u32(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }
    HPRLP_FLOAT sum = 0.0;
    const unsigned int run_offset =
        static_cast<unsigned int>(row - run_row_begin);
    if (run_offset < static_cast<unsigned int>(run_row_count)) {
        const int entry =
            run_entry_begin + 2 * static_cast<int>(run_offset);
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry + 1], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry + 1], scaled_x);
    } else {
        const int begin = row_ptr[row];
        const int end = row_ptr[row + 1];
        if (end - begin > 16) {
            return;
        }
        for (int entry = begin; entry < end; ++entry) {
            sum += SkipPositiveZero
                ? signed_input_skip_positive_zero(
                      entries[entry], scaled_x, scaled_x_nonzero)
                : signed_input(entries[entry], scaled_x);
        }
    }
    signed_finish_y(row, sum, y, scaled_y_out,
                    lower, upper, bound_type, last_y,
                    inverse_row_norm, sigma_params, halpern_factors);
}

template <typename Entry, bool SkipPositiveZero, bool AddActivityShift>
__device__ __forceinline__ void signed_y_rows_warp(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row_position = blockIdx.x * warps_per_block + warp_in_block;
    if (row_position >= row_count) {
        return;
    }
    const int row = row_ids[row_position];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + lane; entry < row_ptr[row + 1];
         entry += 32) {
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        if (AddActivityShift) {
            signed_finish_y_shifted(
                row, sum, y, scaled_y_out, lower, upper, bound_type,
                last_y, inverse_row_norm, activity_shift, sigma_params,
                halpern_factors);
        } else {
            signed_finish_y(row, sum, y, scaled_y_out,
                            lower, upper, bound_type, last_y,
                            inverse_row_norm, sigma_params,
                            halpern_factors);
        }
    }
}

template <typename Entry, bool SkipPositiveZero, bool AddActivityShift>
__device__ __forceinline__ void signed_y_rows_block(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count, HPRLP_FLOAT *warp_sums) {
    const int row_position = blockIdx.x;
    if (row_position >= row_count) {
        return;
    }
    const int row = row_ids[row_position];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            if (AddActivityShift) {
                signed_finish_y_shifted(
                    row, sum, y, scaled_y_out, lower, upper, bound_type,
                    last_y, inverse_row_norm, activity_shift,
                    sigma_params, halpern_factors);
            } else {
                signed_finish_y(row, sum, y, scaled_y_out,
                                lower, upper, bound_type,
                                last_y, inverse_row_norm, sigma_params,
                                halpern_factors);
            }
        }
    }
}

template <typename Entry>
__device__ __forceinline__ void signed_y_rows_short_delta(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count) {
    const int row_position = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_position >= row_count) return;
    const int row = row_ids[row_position];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum += signed_input(entries[entry], scaled_x);
    }
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        sum += signed_delta_input(
            delta_entries[entry], delta_scaled_x_hat,
            delta_scaled_fixed);
    }
    signed_finish_y_shifted(
        row, sum, y, scaled_y_out, lower, upper, bound_type, last_y,
        inverse_row_norm, activity_shift, sigma_params, halpern_factors);
}

template <typename Entry>
__device__ __forceinline__ void signed_y_rows_warp_delta(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row_position = blockIdx.x * warps_per_block + warp_in_block;
    if (row_position >= row_count) return;
    const int row = row_ids[row_position];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + lane; entry < row_ptr[row + 1];
         entry += 32) {
        sum += signed_input(entries[entry], scaled_x);
    }
    for (int entry = delta_row_ptr[row] + lane;
         entry < delta_row_ptr[row + 1]; entry += 32) {
        sum += signed_delta_input(
            delta_entries[entry], delta_scaled_x_hat,
            delta_scaled_fixed);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        signed_finish_y_shifted(
            row, sum, y, scaled_y_out, lower, upper, bound_type, last_y,
            inverse_row_norm, activity_shift, sigma_params,
            halpern_factors);
    }
}

template <typename Entry>
__device__ __forceinline__ void signed_y_rows_block_delta(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids,
    int row_count, HPRLP_FLOAT *warp_sums) {
    const int row_position = blockIdx.x;
    if (row_position >= row_count) return;
    const int row = row_ids[row_position];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        sum += signed_input(entries[entry], scaled_x);
    }
    for (int entry = delta_row_ptr[row] + threadIdx.x;
         entry < delta_row_ptr[row + 1]; entry += blockDim.x) {
        sum += signed_delta_input(
            delta_entries[entry], delta_scaled_x_hat,
            delta_scaled_fixed);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            signed_finish_y_shifted(
                row, sum, y, scaled_y_out, lower, upper, bound_type,
                last_y, inverse_row_norm, activity_shift, sigma_params,
                halpern_factors);
        }
    }
}

template <bool AddActivityShift, bool AddDelta>
__device__ __forceinline__ void signed_finish_y_combined(
    int row, HPRLP_FLOAT sum, HPRLP_FLOAT *y,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const std::uint32_t *delta_nonempty_words,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    HPRLP_FLOAT ax = sum * inverse_row_norm[row];
    if (AddActivityShift) {
        ax += activity_shift[row];
    }
    if (AddDelta &&
        (delta_nonempty_words == nullptr ||
         (delta_nonempty_words[row >> 5] & (1u << (row & 31))) != 0)) {
        HPRLP_FLOAT delta_sum = 0.0;
        for (int entry = delta_row_ptr[row];
             entry < delta_row_ptr[row + 1]; ++entry) {
            delta_sum += signed_delta_input(
                delta_entries[entry], delta_scaled_x_hat,
                delta_scaled_fixed);
        }
        ax += delta_sum * inverse_row_norm[row];
    }
    const HPRLP_FLOAT current_y = y[row];
    const HPRLP_FLOAT value = fma(-sigma_params[1], current_y, ax);
    const HPRLP_FLOAT delta = signed_project_y_delta(
        value, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT projected = sigma_params[2] * delta;
    const HPRLP_FLOAT reflected = 2.0 * projected - current_y;
    const HPRLP_FLOAT next_y = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_y[row]);
    y[row] = next_y;
    scaled_y_out[row] = next_y * inverse_row_norm[row];
}

// One launch covers the same three disjoint row buckets as the split signed
// path.  Every row retains its original CSR traversal and reduction tree, so
// this changes scheduling only, not floating-point parenthesization.
template <typename Entry, bool SkipPositiveZero, bool AddActivityShift,
          bool AddDelta>
__device__ __forceinline__ void signed_y_combined(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const std::uint32_t *delta_nonempty_words,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const Entry *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int direct_row_count,
    const int *medium_row_ids, int medium_row_count,
    const int *long_row_ids, int long_row_count,
    HPRLP_FLOAT *warp_sums) {
    const int warps_per_block = blockDim.x >> 5;
    const int direct_block_count =
        (direct_row_count + blockDim.x - 1) / blockDim.x;
    const int medium_block_count =
        (medium_row_count + warps_per_block - 1) / warps_per_block;

    if (static_cast<int>(blockIdx.x) < direct_block_count) {
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= direct_row_count) {
            return;
        }
        const int begin = row_ptr[row];
        const int end = row_ptr[row + 1];
        if (end - begin > 16) {
            return;
        }
        HPRLP_FLOAT sum = 0.0;
        for (int entry = begin; entry < end; ++entry) {
            sum += SkipPositiveZero
                ? signed_input_skip_positive_zero(
                      entries[entry], scaled_x, scaled_x_nonzero)
                : signed_input(entries[entry], scaled_x);
        }
        signed_finish_y_combined<AddActivityShift, AddDelta>(
            row, sum, y, scaled_y_out, lower, upper, bound_type, last_y,
            inverse_row_norm, activity_shift, delta_scaled_x_hat,
            delta_scaled_fixed, delta_row_ptr, delta_entries,
            delta_nonempty_words, sigma_params, halpern_factors);
        return;
    }

    const int local_block =
        static_cast<int>(blockIdx.x) - direct_block_count;
    if (local_block < medium_block_count) {
        const int lane = threadIdx.x & 31;
        const int warp_in_block = threadIdx.x >> 5;
        const int row_position =
            local_block * warps_per_block + warp_in_block;
        if (row_position >= medium_row_count) {
            return;
        }
        const int row = medium_row_ids[row_position];
        HPRLP_FLOAT sum = 0.0;
        for (int entry = row_ptr[row] + lane; entry < row_ptr[row + 1];
             entry += 32) {
            sum += SkipPositiveZero
                ? signed_input_skip_positive_zero(
                      entries[entry], scaled_x, scaled_x_nonzero)
                : signed_input(entries[entry], scaled_x);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            signed_finish_y_combined<AddActivityShift, AddDelta>(
                row, sum, y, scaled_y_out, lower, upper, bound_type,
                last_y, inverse_row_norm, activity_shift,
                delta_scaled_x_hat, delta_scaled_fixed, delta_row_ptr,
                delta_entries, delta_nonempty_words, sigma_params,
                halpern_factors);
        }
        return;
    }

    const int row_position = local_block - medium_block_count;
    if (row_position >= long_row_count) {
        return;
    }
    const int row = long_row_ids[row_position];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        sum += SkipPositiveZero
            ? signed_input_skip_positive_zero(
                  entries[entry], scaled_x, scaled_x_nonzero)
            : signed_input(entries[entry], scaled_x);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            signed_finish_y_combined<AddActivityShift, AddDelta>(
                row, sum, y, scaled_y_out, lower, upper, bound_type,
                last_y, inverse_row_norm, activity_shift,
                delta_scaled_x_hat, delta_scaled_fixed, delta_row_ptr,
                delta_entries, delta_nonempty_words, sigma_params,
                halpern_factors);
        }
    }
}

}  // namespace

#define HPRLP_DEFINE_SIGNED_X_KERNELS(SUFFIX, ENTRY_TYPE)                 \
    __global__ void signed_unit_update_x_rows_short_##SUFFIX##_kernel(    \
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,     \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_x_rows_short(x, x_hat, lower, upper, bound_type,           \
                            objective, last_x, scaled_y,                  \
                            scaled_x_hat_out, nullptr,                    \
                            inverse_col_norm, row_ptr, entries,           \
                            sigma_params, halpern_factors, row_ids,       \
                            row_count);                                    \
    }                                                                      \
    __global__ void signed_unit_update_x_rows_warp_##SUFFIX##_kernel(    \
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,     \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_x_rows_warp(x, x_hat, lower, upper, bound_type,            \
                           objective, last_x, scaled_y,                   \
                           scaled_x_hat_out, nullptr,                     \
                           inverse_col_norm, row_ptr, entries,            \
                           sigma_params, halpern_factors, row_ids,        \
                           row_count);                                     \
    }                                                                      \
    __global__ void signed_unit_update_x_rows_block_##SUFFIX##_kernel(   \
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,     \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        __shared__ HPRLP_FLOAT warp_sums[32];                             \
        signed_x_rows_block(x, x_hat, lower, upper, bound_type,           \
                            objective, last_x, scaled_y,                  \
                            scaled_x_hat_out, nullptr,                    \
                            inverse_col_norm, row_ptr, entries,           \
                            sigma_params, halpern_factors, row_ids,       \
                            row_count, warp_sums);                         \
    }

#define HPRLP_DEFINE_SIGNED_Y_KERNELS(SUFFIX, ENTRY_TYPE)                 \
    __global__ void signed_unit_update_y_rows_short_##SUFFIX##_kernel(    \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out,                                        \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_short<ENTRY_TYPE, false, false>(                    \
                            y, lower, upper, bound_type, last_y,          \
                            scaled_x, nullptr, scaled_y_out,              \
                            nullptr, inverse_row_norm,                    \
                            row_ptr,                                      \
                            entries, sigma_params, halpern_factors,       \
                            row_ids, row_count);                           \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_warp_##SUFFIX##_kernel(    \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out,                                        \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_warp<ENTRY_TYPE, false, false>(                     \
                           y, lower, upper, bound_type, last_y,           \
                           scaled_x, nullptr, scaled_y_out,               \
                           nullptr, inverse_row_norm,                     \
                           row_ptr,                                       \
                           entries, sigma_params, halpern_factors,        \
                           row_ids, row_count);                            \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_block_##SUFFIX##_kernel(   \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out,                                        \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        __shared__ HPRLP_FLOAT warp_sums[32];                             \
        signed_y_rows_block<ENTRY_TYPE, false, false>(                    \
                            y, lower, upper, bound_type, last_y,          \
                            scaled_x, nullptr, scaled_y_out,              \
                            nullptr, inverse_row_norm,                    \
                            row_ptr,                                      \
                            entries, sigma_params, halpern_factors,       \
                            row_ids, row_count, warp_sums);                \
    }

#define HPRLP_DEFINE_SIGNED_X_FLAG_KERNELS(SUFFIX, ENTRY_TYPE)            \
    __global__ void signed_unit_update_x_rows_short_flagged_##SUFFIX##_kernel(\
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,     \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        std::uint8_t *scaled_x_hat_nonzero_out,                           \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_x_rows_short(x, x_hat, lower, upper, bound_type,           \
                            objective, last_x, scaled_y,                  \
                            scaled_x_hat_out,                             \
                            scaled_x_hat_nonzero_out,                     \
                            inverse_col_norm, row_ptr, entries,           \
                            sigma_params, halpern_factors, row_ids,       \
                            row_count);                                    \
    }                                                                      \
    __global__ void signed_unit_update_x_rows_warp_flagged_##SUFFIX##_kernel(\
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,     \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        std::uint8_t *scaled_x_hat_nonzero_out,                           \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_x_rows_warp(x, x_hat, lower, upper, bound_type,            \
                           objective, last_x, scaled_y,                   \
                           scaled_x_hat_out,                              \
                           scaled_x_hat_nonzero_out,                      \
                           inverse_col_norm, row_ptr, entries,            \
                           sigma_params, halpern_factors, row_ids,        \
                           row_count);                                     \
    }                                                                      \
    __global__ void signed_unit_update_x_rows_block_flagged_##SUFFIX##_kernel(\
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,     \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        std::uint8_t *scaled_x_hat_nonzero_out,                           \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        __shared__ HPRLP_FLOAT warp_sums[32];                             \
        signed_x_rows_block(x, x_hat, lower, upper, bound_type,           \
                            objective, last_x, scaled_y,                  \
                            scaled_x_hat_out,                             \
                            scaled_x_hat_nonzero_out,                     \
                            inverse_col_norm, row_ptr, entries,           \
                            sigma_params, halpern_factors, row_ids,       \
                            row_count, warp_sums);                         \
    }

#define HPRLP_DEFINE_SIGNED_Y_SKIP_KERNELS(SUFFIX, ENTRY_TYPE)            \
    __global__ void signed_unit_update_y_rows_short_skip_zero_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,  \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_short<ENTRY_TYPE, true, false>(                     \
            y, lower, upper, bound_type, last_y, scaled_x,                \
            scaled_x_nonzero, scaled_y_out, nullptr, inverse_row_norm,   \
            row_ptr,                                                      \
            entries, sigma_params, halpern_factors, row_ids, row_count); \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_warp_skip_zero_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,  \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_warp<ENTRY_TYPE, true, false>(                      \
            y, lower, upper, bound_type, last_y, scaled_x,                \
            scaled_x_nonzero, scaled_y_out, nullptr, inverse_row_norm,   \
            row_ptr,                                                      \
            entries, sigma_params, halpern_factors, row_ids, row_count); \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_block_skip_zero_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,  \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        __shared__ HPRLP_FLOAT warp_sums[32];                             \
        signed_y_rows_block<ENTRY_TYPE, true, false>(                     \
            y, lower, upper, bound_type, last_y, scaled_x,                \
            scaled_x_nonzero, scaled_y_out, nullptr, inverse_row_norm,   \
            row_ptr,                                                      \
            entries, sigma_params, halpern_factors, row_ids, row_count,  \
            warp_sums);                                                    \
    }

#define HPRLP_DEFINE_SIGNED_Y_SHIFTED_KERNELS(SUFFIX, ENTRY_TYPE)         \
    __global__ void signed_unit_update_y_rows_short_shifted_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_short<ENTRY_TYPE, false, true>(                     \
            y, lower, upper, bound_type, last_y, scaled_x, nullptr,       \
            scaled_y_out, activity_shift, inverse_row_norm, row_ptr,     \
            entries, sigma_params, halpern_factors, row_ids, row_count); \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_warp_shifted_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_warp<ENTRY_TYPE, false, true>(                      \
            y, lower, upper, bound_type, last_y, scaled_x, nullptr,       \
            scaled_y_out, activity_shift, inverse_row_norm, row_ptr,     \
            entries, sigma_params, halpern_factors, row_ids, row_count); \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_block_shifted_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        __shared__ HPRLP_FLOAT warp_sums[32];                             \
        signed_y_rows_block<ENTRY_TYPE, false, true>(                     \
            y, lower, upper, bound_type, last_y, scaled_x, nullptr,       \
            scaled_y_out, activity_shift, inverse_row_norm, row_ptr,     \
            entries, sigma_params, halpern_factors, row_ids, row_count,  \
            warp_sums);                                                    \
    }

#define HPRLP_DEFINE_SIGNED_Y_DELTA_KERNELS(SUFFIX, ENTRY_TYPE)           \
    __global__ void signed_unit_update_y_rows_short_delta_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *delta_scaled_x_hat,                            \
        const HPRLP_FLOAT *delta_scaled_fixed,                            \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const int *delta_row_ptr,              \
        const std::uint32_t *delta_entries,                               \
        const HPRLP_FLOAT *sigma_params,                                  \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_short_delta<ENTRY_TYPE>(                            \
            y, lower, upper, bound_type, last_y, scaled_x, scaled_y_out, \
            activity_shift, delta_scaled_x_hat, delta_scaled_fixed,       \
            inverse_row_norm, row_ptr, entries, delta_row_ptr,            \
            delta_entries, sigma_params, halpern_factors, row_ids,        \
            row_count);                                                    \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_warp_delta_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *delta_scaled_x_hat,                            \
        const HPRLP_FLOAT *delta_scaled_fixed,                            \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const int *delta_row_ptr,              \
        const std::uint32_t *delta_entries,                               \
        const HPRLP_FLOAT *sigma_params,                                  \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        signed_y_rows_warp_delta<ENTRY_TYPE>(                             \
            y, lower, upper, bound_type, last_y, scaled_x, scaled_y_out, \
            activity_shift, delta_scaled_x_hat, delta_scaled_fixed,       \
            inverse_row_norm, row_ptr, entries, delta_row_ptr,            \
            delta_entries, sigma_params, halpern_factors, row_ids,        \
            row_count);                                                    \
    }                                                                      \
    __global__ void signed_unit_update_y_rows_block_delta_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *delta_scaled_x_hat,                            \
        const HPRLP_FLOAT *delta_scaled_fixed,                            \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const int *delta_row_ptr,              \
        const std::uint32_t *delta_entries,                               \
        const HPRLP_FLOAT *sigma_params,                                  \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count) {                                                   \
        __shared__ HPRLP_FLOAT warp_sums[32];                             \
        signed_y_rows_block_delta<ENTRY_TYPE>(                            \
            y, lower, upper, bound_type, last_y, scaled_x, scaled_y_out, \
            activity_shift, delta_scaled_x_hat, delta_scaled_fixed,       \
            inverse_row_norm, row_ptr, entries, delta_row_ptr,            \
            delta_entries, sigma_params, halpern_factors, row_ids,        \
            row_count, warp_sums);                                         \
    }

HPRLP_DEFINE_SIGNED_X_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_X_KERNELS(u32, std::uint32_t)
HPRLP_DEFINE_SIGNED_Y_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_Y_KERNELS(u32, std::uint32_t)
HPRLP_DEFINE_SIGNED_X_FLAG_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_X_FLAG_KERNELS(u32, std::uint32_t)
HPRLP_DEFINE_SIGNED_Y_SKIP_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_Y_SKIP_KERNELS(u32, std::uint32_t)
HPRLP_DEFINE_SIGNED_Y_SHIFTED_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_Y_SHIFTED_KERNELS(u32, std::uint32_t)
HPRLP_DEFINE_SIGNED_Y_DELTA_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_Y_DELTA_KERNELS(u32, std::uint32_t)

#define HPRLP_DEFINE_SIGNED_Y_COMBINED_KERNELS(SUFFIX, ENTRY_TYPE)         \
    __global__ void signed_unit_update_y_combined_##SUFFIX##_kernel(       \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                          \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,          \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,            \
        HPRLP_FLOAT *scaled_y_out,                                         \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,           \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,        \
        const HPRLP_FLOAT *halpern_factors, int direct_row_count,          \
        const int *medium_row_ids, int medium_row_count,                   \
        const int *long_row_ids, int long_row_count) {                     \
        __shared__ HPRLP_FLOAT warp_sums[32];                              \
        signed_y_combined<ENTRY_TYPE, false, false, false>(                \
            y, lower, upper, bound_type, last_y, scaled_x, nullptr,        \
            scaled_y_out, nullptr, nullptr, nullptr, nullptr, nullptr,      \
            nullptr, inverse_row_norm, row_ptr, entries,                    \
            sigma_params, halpern_factors, direct_row_count,               \
            medium_row_ids, medium_row_count, long_row_ids,                \
            long_row_count, warp_sums);                                    \
    }                                                                      \
    __global__ void                                                        \
    signed_unit_update_y_combined_skip_zero_##SUFFIX##_kernel(             \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                          \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,          \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,            \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,   \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,           \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,        \
        const HPRLP_FLOAT *halpern_factors, int direct_row_count,          \
        const int *medium_row_ids, int medium_row_count,                   \
        const int *long_row_ids, int long_row_count) {                     \
        __shared__ HPRLP_FLOAT warp_sums[32];                              \
        signed_y_combined<ENTRY_TYPE, true, false, false>(                 \
            y, lower, upper, bound_type, last_y, scaled_x,                 \
            scaled_x_nonzero, scaled_y_out, nullptr, nullptr, nullptr,      \
            nullptr, nullptr, nullptr, inverse_row_norm, row_ptr,           \
            entries, sigma_params, halpern_factors, direct_row_count,      \
            medium_row_ids, medium_row_count, long_row_ids,                \
            long_row_count, warp_sums);                                    \
    }                                                                      \
    __global__ void                                                        \
    signed_unit_update_y_combined_shifted_##SUFFIX##_kernel(               \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                          \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,          \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,            \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,      \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,           \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,        \
        const HPRLP_FLOAT *halpern_factors, int direct_row_count,          \
        const int *medium_row_ids, int medium_row_count,                   \
        const int *long_row_ids, int long_row_count) {                     \
        __shared__ HPRLP_FLOAT warp_sums[32];                              \
        signed_y_combined<ENTRY_TYPE, false, true, false>(                 \
            y, lower, upper, bound_type, last_y, scaled_x, nullptr,        \
            scaled_y_out, activity_shift, nullptr, nullptr, nullptr,       \
            nullptr, nullptr, inverse_row_norm, row_ptr,                    \
            entries, sigma_params, halpern_factors, direct_row_count,      \
            medium_row_ids, medium_row_count, long_row_ids,                \
            long_row_count, warp_sums);                                    \
    }                                                                      \
    __global__ void                                                        \
    signed_unit_update_y_combined_shifted_delta_##SUFFIX##_kernel(         \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                          \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,          \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,            \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,      \
        const HPRLP_FLOAT *delta_scaled_x_hat,                             \
        const HPRLP_FLOAT *delta_scaled_fixed,                             \
        const int *delta_row_ptr,                                          \
        const std::uint32_t *delta_entries,                                \
        const std::uint32_t *delta_nonempty_words,                         \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,           \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,        \
        const HPRLP_FLOAT *halpern_factors, int direct_row_count,          \
        const int *medium_row_ids, int medium_row_count,                   \
        const int *long_row_ids, int long_row_count) {                     \
        __shared__ HPRLP_FLOAT warp_sums[32];                              \
        signed_y_combined<ENTRY_TYPE, false, true, true>(                  \
            y, lower, upper, bound_type, last_y, scaled_x, nullptr,        \
            scaled_y_out, activity_shift, delta_scaled_x_hat,              \
            delta_scaled_fixed, delta_row_ptr, delta_entries,              \
            delta_nonempty_words, inverse_row_norm, row_ptr, entries,      \
            sigma_params, halpern_factors, direct_row_count,               \
            medium_row_ids, medium_row_count, long_row_ids,                \
            long_row_count, warp_sums);                                    \
    }

HPRLP_DEFINE_SIGNED_Y_COMBINED_KERNELS(u16, std::uint16_t)
HPRLP_DEFINE_SIGNED_Y_COMBINED_KERNELS(u32, std::uint32_t)

__global__ void signed_unit_update_x_all_scalar_u16_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_x_all_scalar(x, x_hat, lower, upper, bound_type, objective,
                        last_x, scaled_y, scaled_x_hat_out, nullptr,
                        inverse_col_norm, row_ptr,
                        entries, sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_x_all_scalar_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_x_all_scalar(x, x_hat, lower, upper, bound_type, objective,
                        last_x, scaled_y, scaled_x_hat_out, nullptr,
                        inverse_col_norm, row_ptr,
                        entries, sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar<std::uint16_t, false, false>(
                        y, lower, upper, bound_type, last_y, scaled_x,
                        nullptr, scaled_y_out, nullptr, inverse_row_norm,
                        row_ptr, entries,
                        sigma_params,
                        halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar<std::uint32_t, false, false>(
                        y, lower, upper, bound_type, last_y, scaled_x,
                        nullptr, scaled_y_out, nullptr, inverse_row_norm,
                        row_ptr, entries,
                        sigma_params,
                        halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_shifted_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar<std::uint16_t, false, true>(
        y, lower, upper, bound_type, last_y, scaled_x, nullptr,
        scaled_y_out, activity_shift, inverse_row_norm, row_ptr, entries,
        sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_shifted_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar<std::uint32_t, false, true>(
        y, lower, upper, bound_type, last_y, scaled_x, nullptr,
        scaled_y_out, activity_shift, inverse_row_norm, row_ptr, entries,
        sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_shifted_delta_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar_delta<std::uint16_t>(
        y, lower, upper, bound_type, last_y, scaled_x, scaled_y_out,
        activity_shift, delta_scaled_x_hat, delta_scaled_fixed,
        inverse_row_norm, row_ptr, entries, delta_row_ptr, delta_entries,
        sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_shifted_delta_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar_delta<std::uint32_t>(
        y, lower, upper, bound_type, last_y, scaled_x, scaled_y_out,
        activity_shift, delta_scaled_x_hat, delta_scaled_fixed,
        inverse_row_norm, row_ptr, entries, delta_row_ptr, delta_entries,
        sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_direct_short_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_direct_short<std::uint16_t, false>(
                          y, lower, upper, bound_type, last_y, scaled_x,
                          nullptr, scaled_y_out, inverse_row_norm, row_ptr, entries,
                          sigma_params,
                          halpern_factors, row_count);
}

__global__ void signed_unit_update_y_direct_short_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_direct_short<std::uint32_t, false>(
                          y, lower, upper, bound_type, last_y, scaled_x,
                          nullptr, scaled_y_out, inverse_row_norm, row_ptr, entries,
                          sigma_params,
                          halpern_factors, row_count);
}

__global__ void signed_unit_update_x_all_scalar_flagged_u16_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_x_all_scalar(x, x_hat, lower, upper, bound_type, objective,
                        last_x, scaled_y, scaled_x_hat_out,
                        scaled_x_hat_nonzero_out, inverse_col_norm, row_ptr,
                        entries, sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_x_all_scalar_flagged_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_x_all_scalar(x, x_hat, lower, upper, bound_type, objective,
                        last_x, scaled_y, scaled_x_hat_out,
                        scaled_x_hat_nonzero_out, inverse_col_norm, row_ptr,
                        entries, sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_x_all_scalar_split_u16_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *indices, const std::uint8_t *negative,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_x_all_scalar_split_u16(
        x, x_hat, lower, upper, bound_type, objective, last_x, scaled_y,
        scaled_x_hat_out, nullptr, inverse_col_norm, row_ptr, indices,
        negative, sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_x_all_scalar_split_u16_flagged_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *indices, const std::uint8_t *negative,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_x_all_scalar_split_u16(
        x, x_hat, lower, upper, bound_type, objective, last_x, scaled_y,
        scaled_x_hat_out, scaled_x_hat_nonzero_out, inverse_col_norm,
        row_ptr, indices, negative, sigma_params, halpern_factors,
        row_count);
}

__global__ void signed_unit_update_x_all_scalar_degree3_run_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin) {
    signed_x_all_scalar_degree3_run_u32(
        x, x_hat, lower, upper, bound_type, objective, last_x, scaled_y,
        scaled_x_hat_out, nullptr, inverse_col_norm, row_ptr, entries,
        sigma_params, halpern_factors, row_count, run_row_begin,
        run_row_count, run_entry_begin);
}

__global__ void
signed_unit_update_x_all_scalar_degree3_run_flagged_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin) {
    signed_x_all_scalar_degree3_run_u32(
        x, x_hat, lower, upper, bound_type, objective, last_x, scaled_y,
        scaled_x_hat_out, scaled_x_hat_nonzero_out, inverse_col_norm,
        row_ptr, entries, sigma_params, halpern_factors, row_count,
        run_row_begin, run_row_count, run_entry_begin);
}

__global__ void
signed_unit_update_x_all_scalar_degree3_run_state_specialized_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int zero_objective_state_run_begin,
    int zero_objective_state_run_count, int run_row_begin,
    int run_row_count, int run_entry_begin) {
    signed_x_all_scalar_degree3_run_state_specialized_u32(
        x, x_hat, upper, objective, last_x, scaled_y,
        scaled_x_hat_out, nullptr, inverse_col_norm, row_ptr, entries,
        sigma_params, halpern_factors, row_count,
        zero_objective_state_run_begin, zero_objective_state_run_count,
        run_row_begin, run_row_count,
        run_entry_begin);
}

__global__ void
signed_unit_update_x_all_scalar_degree3_run_state_specialized_flagged_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int zero_objective_state_run_begin,
    int zero_objective_state_run_count, int run_row_begin,
    int run_row_count, int run_entry_begin) {
    signed_x_all_scalar_degree3_run_state_specialized_u32(
        x, x_hat, upper, objective, last_x, scaled_y,
        scaled_x_hat_out, scaled_x_hat_nonzero_out, inverse_col_norm,
        row_ptr, entries, sigma_params, halpern_factors, row_count,
        zero_objective_state_run_begin, zero_objective_state_run_count,
        run_row_begin, run_row_count,
        run_entry_begin);
}

__global__ void signed_unit_update_y_all_scalar_skip_zero_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar<std::uint16_t, true, false>(
        y, lower, upper, bound_type, last_y, scaled_x, scaled_x_nonzero,
        scaled_y_out, nullptr, inverse_row_norm, row_ptr, entries,
        sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_all_scalar_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_all_scalar<std::uint32_t, true, false>(
        y, lower, upper, bound_type, last_y, scaled_x, scaled_x_nonzero,
        scaled_y_out, nullptr, inverse_row_norm, row_ptr, entries,
        sigma_params, halpern_factors, row_count);
}

__global__ void signed_unit_update_y_direct_short_skip_zero_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_direct_short<std::uint16_t, true>(
        y, lower, upper, bound_type, last_y, scaled_x, scaled_x_nonzero,
        scaled_y_out, inverse_row_norm, row_ptr, entries, sigma_params,
        halpern_factors, row_count);
}

__global__ void signed_unit_update_y_direct_short_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count) {
    signed_y_direct_short<std::uint32_t, true>(
        y, lower, upper, bound_type, last_y, scaled_x, scaled_x_nonzero,
        scaled_y_out, inverse_row_norm, row_ptr, entries, sigma_params,
        halpern_factors, row_count);
}

__global__ void signed_unit_update_y_direct_short_degree2_run_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin) {
    signed_y_direct_short_degree2_run_u32<false>(
        y, lower, upper, bound_type, last_y, scaled_x, nullptr,
        scaled_y_out, inverse_row_norm, row_ptr, entries, sigma_params,
        halpern_factors, row_count, run_row_begin, run_row_count,
        run_entry_begin);
}

__global__ void
signed_unit_update_y_direct_short_degree2_run_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin) {
    signed_y_direct_short_degree2_run_u32<true>(
        y, lower, upper, bound_type, last_y, scaled_x,
        scaled_x_nonzero, scaled_y_out, inverse_row_norm, row_ptr,
        entries, sigma_params, halpern_factors, row_count, run_row_begin,
        run_row_count, run_entry_begin);
}

__global__ void
signed_unit_update_y_direct_short_degree2_run_state_specialized_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int upper_zero_state_run_begin, int upper_zero_state_run_count,
    int run_row_begin,
    int run_row_count, int run_entry_begin) {
    signed_y_direct_short_degree2_run_state_specialized_u32<false>(
        y, lower, upper, bound_type, last_y, scaled_x, nullptr,
        scaled_y_out, inverse_row_norm, row_ptr, entries, sigma_params,
        halpern_factors, row_count, upper_zero_state_run_begin,
        upper_zero_state_run_count,
        run_row_begin, run_row_count, run_entry_begin);
}

__global__ void
signed_unit_update_y_direct_short_degree2_run_state_specialized_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int upper_zero_state_run_begin, int upper_zero_state_run_count,
    int run_row_begin,
    int run_row_count, int run_entry_begin) {
    signed_y_direct_short_degree2_run_state_specialized_u32<true>(
        y, lower, upper, bound_type, last_y, scaled_x,
        scaled_x_nonzero, scaled_y_out, inverse_row_norm, row_ptr,
        entries, sigma_params, halpern_factors, row_count,
        upper_zero_state_run_begin, upper_zero_state_run_count,
        run_row_begin, run_row_count,
        run_entry_begin);
}

#undef HPRLP_DEFINE_SIGNED_X_KERNELS
#undef HPRLP_DEFINE_SIGNED_Y_KERNELS
#undef HPRLP_DEFINE_SIGNED_X_FLAG_KERNELS
#undef HPRLP_DEFINE_SIGNED_Y_SKIP_KERNELS
#undef HPRLP_DEFINE_SIGNED_Y_SHIFTED_KERNELS
#undef HPRLP_DEFINE_SIGNED_Y_DELTA_KERNELS
#undef HPRLP_DEFINE_SIGNED_Y_COMBINED_KERNELS
