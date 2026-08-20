#ifndef HPRLP_STRUCTURED_KERNELS_CUH
#define HPRLP_STRUCTURED_KERNELS_CUH

#include "api/structs.h"
#include <cstdint>

__global__
void row_template_update_x_all_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const uint8_t *row_template_ids, const int *row_bases,
    const int *template_ptr, const int *template_offsets,
    const HPRLP_FLOAT *template_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n);

__global__
void affine_block_update_x_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *block_row_begin, const int *block_row_count,
    const int *block_entry_ptr, const int *entry_base_columns,
    const int *entry_column_strides, const HPRLP_FLOAT *entry_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors);

__global__
void row_template_update_x_fallback_short_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void row_template_update_x_fallback_warp_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void row_template_update_x_fallback_block_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);
__global__
void structured_update_x_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *col_norm,
    const uint16_t *output_cols, const int *row_ptr,
    const uint16_t *rows, const uint16_t *values_u16,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int coefficient_bias, bool coefficient_has_escape,
    int coefficient_escape_value, int output_count);

__global__
void structured_update_x_dense_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *col_norm,
    const uint16_t *dense_cols, const uint16_t *dense_rows,
    const uint16_t *values_u16,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int coefficient_bias, bool coefficient_has_escape,
    int coefficient_escape_value, int dense_row_count, int dense_col_count);

__global__
void structured_update_y_sparse_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x, const HPRLP_FLOAT *row_norm,
    const uint16_t *rows, const uint16_t *col0, const uint16_t *col1,
    const int8_t *second_sign, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__
void structured_update_y_dense_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x, const HPRLP_FLOAT *row_norm,
    const uint16_t *dense_rows, const uint16_t *dense_cols,
    const uint16_t *dense_local_cols, const uint16_t *values_u16,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int dense_row_count,
    int dense_col_count, int coefficient_bias,
    bool coefficient_has_escape, int coefficient_escape_value);
__global__
void row_template_update_y_all_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const uint8_t *row_template_ids, const int *row_bases,
    const int *template_ptr, const int *template_offsets,
    const HPRLP_FLOAT *template_values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m);

__global__
void affine_block_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *block_row_begin, const int *block_row_count,
    const int *block_entry_ptr, const int *entry_base_columns,
    const int *entry_column_strides, const HPRLP_FLOAT *entry_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors);

__global__
void row_template_update_y_fallback_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void row_template_update_y_fallback_warp_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void row_template_update_y_fallback_block_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm, HPRLP_FLOAT *scaled_y,
    const int *row_ptr, const int *col_indices,
    const HPRLP_FLOAT *values, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

#endif
