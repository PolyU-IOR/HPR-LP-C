#ifndef HPRLP_GENERIC_FUSED_KERNELS_CUH
#define HPRLP_GENERIC_FUSED_KERNELS_CUH

#include "api/structs.h"
#include <cstdint>

__global__
void fused_update_x_z_rows_short_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                        const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                        const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                        const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                        const HPRLP_FLOAT *halpern_factors,
                                        const int *row_ids, int nrows);

__global__
void fused_update_x_z_all_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const int *AT_rowPtr,
    const int *AT_colIndex, const HPRLP_FLOAT *AT_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n);
__global__
void fused_update_x_z_direct_short_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const int *AT_rowPtr,
    const int *AT_colIndex, const HPRLP_FLOAT *AT_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n);

__global__
void fused_update_x_z_rows_warp_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                       const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                       const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                       const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                       const HPRLP_FLOAT *halpern_factors,
                                       const int *row_ids, int nrows);

__global__
void fused_update_x_z_rows_block_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                        const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                        const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                        const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                        const HPRLP_FLOAT *halpern_factors,
                                        const int *row_ids, int nrows);
__global__
void fused_update_y_rows_short_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                      const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                      const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                      const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                         const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void fused_update_y_all_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const int *A_rowPtr,
    const int *A_colIndex, const HPRLP_FLOAT *A_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m);
__global__
void fused_update_y_all_short_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const int *A_rowPtr,
    const uint16_t *A_colIndex, const HPRLP_FLOAT *A_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m);

__global__
void fused_update_y_direct_short_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const int *A_rowPtr,
    const int *A_colIndex, const HPRLP_FLOAT *A_value,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m);

__global__
void fused_update_y_rows_warp_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                     const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                     const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                     const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                     const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void fused_update_y_rows_block_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                      const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                      const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                      const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                      const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows);

__global__
void segmented_update_y_partial_kernel(
    HPRLP_FLOAT *partials, const HPRLP_FLOAT *x_hat,
    const int *A_colIndex, const HPRLP_FLOAT *A_value,
    const int *tile_begin, const int *tile_end, int tile_count);

__global__
void segmented_update_y_finalize_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *partials, const int *row_ids,
    const int *row_tile_ptr, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__
void segmented_update_x_finalize_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const uint8_t *x_bound_type,
    const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *partials, const int *row_ids,
    const int *row_tile_ptr, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);


#endif
