#ifndef HPRLP_PACKED_DICTIONARY_LAUNCHER_CUH
#define HPRLP_PACKED_DICTIONARY_LAUNCHER_CUH

#include "api/structs.h"
#include "cuda_kernels/backends/dictionary/dictionary_kernels.cuh"

#include <cstdint>

struct HPRLP_packed_dictionary_x_view_gpu {
    HPRLP_FLOAT *x;
    HPRLP_FLOAT *x_hat;
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    const std::uint8_t *bound_type;
    const HPRLP_FLOAT *objective;
    const HPRLP_FLOAT *last_x;
    const HPRLP_FLOAT *scaled_y;
    const HPRLP_FLOAT *inverse_col_norm;
    HPRLP_FLOAT *scaled_x_hat_output;
    HPRLPPackedStatePlan state_plan;
    const HPRLP_FLOAT *dictionary;
    const std::uint32_t *packed_entries;
    const int *row_ptr;
    unsigned code_bits;
    const int *short_row_ids;
    int short_row_count;
    const int *medium_row_ids;
    int medium_row_count;
    const int *long_row_ids;
    int long_row_count;
    bool combine_short_medium;
};

inline void hprlp_enqueue_packed_dictionary_x(
    const HPRLP_packed_dictionary_x_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    const int warps_per_block = threads_per_block / 32;
    const int short_blocks =
        (view.short_row_count + threads_per_block - 1) /
        threads_per_block;
    const int medium_blocks =
        (view.medium_row_count + warps_per_block - 1) /
        warps_per_block;
    if (view.combine_short_medium && view.short_row_count > 0 &&
        view.medium_row_count > 0 && view.long_row_count == 0) {
        packed_dictionary_update_x_combined_short_warp_kernel<<<
            short_blocks + medium_blocks, threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper, view.bound_type,
            view.objective, view.last_x, view.scaled_y,
            view.inverse_col_norm, view.scaled_x_hat_output,
            view.state_plan, view.dictionary, view.packed_entries,
            view.row_ptr, view.code_bits, sigma_params, halpern_factors,
            view.short_row_ids, view.short_row_count,
            view.medium_row_ids, view.medium_row_count);
        return;
    }
    if (view.short_row_count > 0) {
        packed_dictionary_update_x_rows_short_kernel<<<
            short_blocks,
            threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper, view.bound_type,
            view.objective, view.last_x, view.scaled_y,
            view.inverse_col_norm, view.scaled_x_hat_output,
            view.state_plan, view.dictionary, view.packed_entries,
            view.row_ptr, view.code_bits, sigma_params, halpern_factors,
            view.short_row_ids, view.short_row_count);
    }
    if (view.medium_row_count > 0) {
        packed_dictionary_update_x_rows_warp_kernel<<<
            medium_blocks,
            threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper, view.bound_type,
            view.objective, view.last_x, view.scaled_y,
            view.inverse_col_norm, view.scaled_x_hat_output,
            view.state_plan, view.dictionary, view.packed_entries,
            view.row_ptr, view.code_bits, sigma_params, halpern_factors,
            view.medium_row_ids, view.medium_row_count);
    }
    if (view.long_row_count > 0) {
        packed_dictionary_update_x_rows_block_kernel<<<
            view.long_row_count, threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper, view.bound_type,
            view.objective, view.last_x, view.scaled_y,
            view.inverse_col_norm, view.scaled_x_hat_output,
            view.state_plan, view.dictionary, view.packed_entries,
            view.row_ptr, view.code_bits, sigma_params, halpern_factors,
            view.long_row_ids, view.long_row_count);
    }
}

struct HPRLP_packed_dictionary_y_view_gpu {
    HPRLP_FLOAT *y;
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    const std::uint8_t *bound_type;
    const HPRLP_FLOAT *last_y;
    const HPRLP_FLOAT *scaled_x_hat;
    const HPRLP_FLOAT *inverse_row_norm;
    HPRLP_FLOAT *scaled_y_output;
    const HPRLP_FLOAT *activity_shift;
    HPRLPPackedStatePlan state_plan;
    const HPRLP_FLOAT *dictionary;
    const std::uint32_t *packed_entries;
    const int *row_ptr;
    unsigned code_bits;
    const int *short_row_ids;
    int short_row_count;
    const int *medium_row_ids;
    int medium_row_count;
    const int *long_row_ids;
    int long_row_count;
    const HPRLP_FLOAT *delta_x_hat;
    const HPRLP_FLOAT *delta_lower;
    const HPRLP_FLOAT *delta_upper;
    const std::uint8_t *delta_old_mask;
    const int *delta_row_ptr;
    const int *delta_columns;
    const HPRLP_FLOAT *delta_values;
    bool combine_short_medium;
};

inline void hprlp_enqueue_packed_dictionary_y(
    const HPRLP_packed_dictionary_y_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    const int warps_per_block = threads_per_block / 32;
    const int short_blocks =
        (view.short_row_count + threads_per_block - 1) /
        threads_per_block;
    const int medium_blocks =
        (view.medium_row_count + warps_per_block - 1) /
        warps_per_block;
    if (view.combine_short_medium && view.short_row_count > 0 &&
        view.medium_row_count > 0 && view.long_row_count == 0) {
        packed_dictionary_update_y_combined_short_warp_kernel<<<
            short_blocks + medium_blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x_hat, view.inverse_row_norm,
            view.scaled_y_output, view.activity_shift,
            view.delta_x_hat, view.delta_lower, view.delta_upper,
            view.delta_old_mask, view.delta_row_ptr,
            view.delta_columns, view.delta_values, view.state_plan,
            view.dictionary, view.packed_entries, view.row_ptr,
            view.code_bits, sigma_params, halpern_factors,
            view.short_row_ids, view.short_row_count,
            view.medium_row_ids, view.medium_row_count);
        return;
    }
    if (view.delta_row_ptr != nullptr) {
        if (view.short_row_count > 0) {
            packed_dictionary_update_y_rows_short_shifted_delta_kernel<<<
                short_blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x_hat, view.inverse_row_norm,
                view.scaled_y_output, view.activity_shift,
                view.delta_x_hat, view.delta_lower, view.delta_upper,
                view.delta_old_mask, view.delta_row_ptr,
                view.delta_columns, view.delta_values, view.state_plan,
                view.dictionary, view.packed_entries, view.row_ptr,
                view.code_bits, sigma_params, halpern_factors,
                view.short_row_ids, view.short_row_count);
        }
        if (view.medium_row_count > 0) {
            packed_dictionary_update_y_rows_warp_shifted_delta_kernel<<<
                medium_blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x_hat, view.inverse_row_norm,
                view.scaled_y_output, view.activity_shift,
                view.delta_x_hat, view.delta_lower, view.delta_upper,
                view.delta_old_mask, view.delta_row_ptr,
                view.delta_columns, view.delta_values, view.state_plan,
                view.dictionary, view.packed_entries, view.row_ptr,
                view.code_bits, sigma_params, halpern_factors,
                view.medium_row_ids, view.medium_row_count);
        }
        if (view.long_row_count > 0) {
            packed_dictionary_update_y_rows_block_shifted_delta_kernel<<<
                view.long_row_count, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x_hat, view.inverse_row_norm,
                view.scaled_y_output, view.activity_shift,
                view.delta_x_hat, view.delta_lower, view.delta_upper,
                view.delta_old_mask, view.delta_row_ptr,
                view.delta_columns, view.delta_values, view.state_plan,
                view.dictionary, view.packed_entries, view.row_ptr,
                view.code_bits, sigma_params, halpern_factors,
                view.long_row_ids, view.long_row_count);
        }
        return;
    }
    if (view.activity_shift != nullptr) {
        if (view.short_row_count > 0) {
            packed_dictionary_update_y_rows_short_shifted_kernel<<<
                short_blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x_hat, view.inverse_row_norm,
                view.scaled_y_output, view.activity_shift,
                view.state_plan, view.dictionary, view.packed_entries,
                view.row_ptr, view.code_bits, sigma_params,
                halpern_factors, view.short_row_ids,
                view.short_row_count);
        }
        if (view.medium_row_count > 0) {
            packed_dictionary_update_y_rows_warp_shifted_kernel<<<
                medium_blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x_hat, view.inverse_row_norm,
                view.scaled_y_output, view.activity_shift,
                view.state_plan, view.dictionary, view.packed_entries,
                view.row_ptr, view.code_bits, sigma_params,
                halpern_factors, view.medium_row_ids,
                view.medium_row_count);
        }
        if (view.long_row_count > 0) {
            packed_dictionary_update_y_rows_block_shifted_kernel<<<
                view.long_row_count, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x_hat, view.inverse_row_norm,
                view.scaled_y_output, view.activity_shift,
                view.state_plan, view.dictionary, view.packed_entries,
                view.row_ptr, view.code_bits, sigma_params,
                halpern_factors, view.long_row_ids,
                view.long_row_count);
        }
        return;
    }
    if (view.short_row_count > 0) {
        packed_dictionary_update_y_rows_short_kernel<<<
            short_blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x_hat, view.inverse_row_norm,
            view.scaled_y_output, view.state_plan, view.dictionary,
            view.packed_entries, view.row_ptr, view.code_bits,
            sigma_params, halpern_factors, view.short_row_ids,
            view.short_row_count);
    }
    if (view.medium_row_count > 0) {
        packed_dictionary_update_y_rows_warp_kernel<<<
            medium_blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x_hat, view.inverse_row_norm,
            view.scaled_y_output, view.state_plan, view.dictionary,
            view.packed_entries, view.row_ptr, view.code_bits,
            sigma_params, halpern_factors, view.medium_row_ids,
            view.medium_row_count);
    }
    if (view.long_row_count > 0) {
        packed_dictionary_update_y_rows_block_kernel<<<
            view.long_row_count, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x_hat, view.inverse_row_norm,
            view.scaled_y_output, view.state_plan, view.dictionary,
            view.packed_entries, view.row_ptr, view.code_bits,
            sigma_params, halpern_factors, view.long_row_ids,
            view.long_row_count);
    }
}

#endif
