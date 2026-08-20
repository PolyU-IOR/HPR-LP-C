#ifndef HPRLP_UNIT_KERNELS_CUH
#define HPRLP_UNIT_KERNELS_CUH

#include "api/structs.h"
#include <cstdint>

__global__
void update_y_normal_unit_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU,
                                 HPRLP_FLOAT *Ax, const HPRLP_FLOAT *inverse_row_norm,
                                 HPRLP_FLOAT *last_y, const HPRLP_FLOAT *sigma_params,
                                 const HPRLP_FLOAT *halpern_factors,
                                 int uniform_unit_sign, int m);

__global__
void unit_coltile_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x, const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_output,
    const int *row_tile_offsets, const uint16_t *local_cols,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int tile_count, int tile_cols,
    int uniform_unit_sign, int m);

__global__
void unit_coltile_update_y_zero_bitset_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x,
    const std::uint32_t *positive_zero_bits,
    const HPRLP_FLOAT *inverse_row_norm,
    const int *row_tile_offsets, const uint16_t *local_cols,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int tile_count, int tile_cols,
    int uniform_unit_sign, int m);
__global__
void fused_update_x_z_all_short_unit_nonnegative_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *c,
    const HPRLP_FLOAT *last_x, const HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_col_norm, HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr,
    const uint16_t *AT_colIndex, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int uniform_unit_sign, int n);

__global__
void fused_update_x_z_all_scalar_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int uniform_unit_sign, int n);

// Sparse-state pair: preserve the unit-factor X arithmetic and
// simultaneously form A*x_hat by scattering only nonzero scaled x_hat values.
// Atomic row accumulation changes the sum order, so iteration regression is a
// mandatory admission gate for this backend.
__global__
void fused_update_x_z_all_scalar_unit_active_scatter_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *Ax, const int *AT_rowPtr,
    const uint16_t *AT_colIndex, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int uniform_unit_sign, int n);

__global__
void fused_update_x_z_rows_short_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign, const int *row_ids, int nrows);

__global__
void fused_update_x_z_rows_warp_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign, const int *row_ids, int nrows);

__global__
void fused_update_x_z_rows_block_unit_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_output,
    const int *AT_rowPtr, const uint16_t *AT_colIndex,
    const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign, const int *row_ids, int nrows);


#endif
