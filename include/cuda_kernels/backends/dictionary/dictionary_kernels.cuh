#ifndef HPRLP_DICTIONARY_KERNELS_CUH
#define HPRLP_DICTIONARY_KERNELS_CUH

#include "api/structs.h"
#include <cstdint>

__global__
void fused_update_x_z_rows_short_dictionary_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *dictionary, const uint8_t *value_codes,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void fused_update_x_z_rows_warp_dictionary_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *dictionary, const uint8_t *value_codes,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void fused_update_x_z_rows_block_dictionary_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    const HPRLP_FLOAT *dictionary, const uint8_t *value_codes,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void packed_dictionary_update_x_rows_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void packed_dictionary_update_x_rows_warp_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void packed_dictionary_update_x_combined_short_warp_kernel(
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
    const int *medium_row_ids, int medium_row_count);

__global__
void packed_dictionary_update_x_rows_block_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *AT_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void packed_dictionary_update_x_fixed_degree_run_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries_soa,
    unsigned code_bits, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_begin, int row_count,
    int degree);

__global__
void packed_dictionary_segmented_x_finalize_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan, const HPRLP_FLOAT *partials,
    const int *row_ids, const int *row_tile_ptr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__
void u32_u16_dictionary_update_x_rows_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *AT_rowPtr,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void u32_u16_dictionary_update_x_rows_warp_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *AT_rowPtr,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void u32_u16_dictionary_update_x_rows_block_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *AT_rowPtr,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_block_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *A_rowPtr, unsigned code_bits,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_short_shifted_kernel(
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
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_warp_shifted_kernel(
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
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_block_shifted_kernel(
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
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_short_shifted_delta_kernel(
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
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_warp_shifted_delta_kernel(
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
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_rows_block_shifted_delta_kernel(
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
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void packed_dictionary_update_y_combined_short_warp_kernel(
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
    const int *medium_row_ids, int medium_row_count);

__global__
void packed_dictionary_update_y_fixed_degree_run_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries_soa,
    unsigned code_bits, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_begin, int row_count,
    int degree);

__global__
void packed_dictionary_segmented_y_partial_kernel(
    HPRLP_FLOAT *partials, const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *dictionary, const uint32_t *packed_entries,
    const int *tile_begin, const int *tile_end, unsigned code_bits,
    int tile_count);

__global__
void packed_dictionary_segmented_y_finalize_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output, HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *partials, const int *row_ids,
    const int *row_tile_ptr, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__
void u32_u16_dictionary_update_y_rows_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *A_rowPtr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void u32_u16_dictionary_update_y_rows_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *A_rowPtr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void u32_u16_dictionary_update_y_rows_block_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    HPRLPPackedStatePlan state_plan,
    const HPRLP_FLOAT *dictionary, const uint32_t *input_indices,
    const uint16_t *value_codes, const int *A_rowPtr,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

#endif
