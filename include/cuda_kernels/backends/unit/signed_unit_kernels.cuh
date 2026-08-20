#ifndef HPRLP_SIGNED_UNIT_KERNELS_CUH
#define HPRLP_SIGNED_UNIT_KERNELS_CUH

#include "api/structs.h"

#include <cstdint>

#define HPRLP_DECLARE_SIGNED_X_KERNELS(SUFFIX, ENTRY_TYPE)                 \
    __global__ void signed_unit_update_x_rows_short_##SUFFIX##_kernel(     \
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,         \
        const HPRLP_FLOAT *u, const std::uint8_t *bound_type,             \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_x_rows_warp_##SUFFIX##_kernel(     \
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,         \
        const HPRLP_FLOAT *u, const std::uint8_t *bound_type,             \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_x_rows_block_##SUFFIX##_kernel(    \
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,         \
        const HPRLP_FLOAT *u, const std::uint8_t *bound_type,             \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count)

#define HPRLP_DECLARE_SIGNED_Y_KERNELS(SUFFIX, ENTRY_TYPE)                 \
    __global__ void signed_unit_update_y_rows_short_##SUFFIX##_kernel(     \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out,                                        \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_y_rows_warp_##SUFFIX##_kernel(     \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out,                                        \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_y_rows_block_##SUFFIX##_kernel(    \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out,                                        \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count)

HPRLP_DECLARE_SIGNED_X_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_X_KERNELS(u32, std::uint32_t);
HPRLP_DECLARE_SIGNED_Y_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_KERNELS(u32, std::uint32_t);

#define HPRLP_DECLARE_SIGNED_X_FLAG_KERNELS(SUFFIX, ENTRY_TYPE)           \
    __global__ void signed_unit_update_x_rows_short_flagged_##SUFFIX##_kernel(\
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,         \
        const HPRLP_FLOAT *u, const std::uint8_t *bound_type,             \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        std::uint8_t *scaled_x_hat_nonzero_out,                           \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_x_rows_warp_flagged_##SUFFIX##_kernel(\
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,         \
        const HPRLP_FLOAT *u, const std::uint8_t *bound_type,             \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        std::uint8_t *scaled_x_hat_nonzero_out,                           \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_x_rows_block_flagged_##SUFFIX##_kernel(\
        HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,         \
        const HPRLP_FLOAT *u, const std::uint8_t *bound_type,             \
        const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,          \
        const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,       \
        std::uint8_t *scaled_x_hat_nonzero_out,                           \
        const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count)

#define HPRLP_DECLARE_SIGNED_Y_SKIP_KERNELS(SUFFIX, ENTRY_TYPE)           \
    __global__ void signed_unit_update_y_rows_short_skip_zero_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,  \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_y_rows_warp_skip_zero_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,  \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_y_rows_block_skip_zero_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,  \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count)

#define HPRLP_DECLARE_SIGNED_Y_SHIFTED_KERNELS(SUFFIX, ENTRY_TYPE)        \
    __global__ void signed_unit_update_y_rows_short_shifted_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_y_rows_warp_shifted_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count);                                                    \
    __global__ void signed_unit_update_y_rows_block_shifted_##SUFFIX##_kernel(\
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                         \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,         \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,           \
        HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,     \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,          \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,       \
        const HPRLP_FLOAT *halpern_factors, const int *row_ids,           \
        int row_count)

HPRLP_DECLARE_SIGNED_X_FLAG_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_X_FLAG_KERNELS(u32, std::uint32_t);
HPRLP_DECLARE_SIGNED_Y_SKIP_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_SKIP_KERNELS(u32, std::uint32_t);
HPRLP_DECLARE_SIGNED_Y_SHIFTED_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_SHIFTED_KERNELS(u32, std::uint32_t);

#define HPRLP_DECLARE_SIGNED_Y_DELTA_KERNELS(SUFFIX, ENTRY_TYPE)          \
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
        int row_count);                                                    \
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
        int row_count);                                                    \
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
        int row_count)

HPRLP_DECLARE_SIGNED_Y_DELTA_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_DELTA_KERNELS(u32, std::uint32_t);

#define HPRLP_DECLARE_SIGNED_Y_COMBINED_KERNELS(SUFFIX, ENTRY_TYPE)        \
    __global__ void signed_unit_update_y_combined_##SUFFIX##_kernel(       \
        HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,                          \
        const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,          \
        const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,            \
        HPRLP_FLOAT *scaled_y_out,                                         \
        const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,           \
        const ENTRY_TYPE *entries, const HPRLP_FLOAT *sigma_params,        \
        const HPRLP_FLOAT *halpern_factors, int direct_row_count,          \
        const int *medium_row_ids, int medium_row_count,                   \
        const int *long_row_ids, int long_row_count);                      \
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
        const int *long_row_ids, int long_row_count)

#define HPRLP_DECLARE_SIGNED_Y_COMBINED_SHIFTED_KERNELS(SUFFIX, ENTRY_TYPE)\
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
        const int *long_row_ids, int long_row_count)

#define HPRLP_DECLARE_SIGNED_Y_COMBINED_DELTA_KERNELS(SUFFIX, ENTRY_TYPE)  \
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
        const int *long_row_ids, int long_row_count)

HPRLP_DECLARE_SIGNED_Y_COMBINED_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_COMBINED_KERNELS(u32, std::uint32_t);
HPRLP_DECLARE_SIGNED_Y_COMBINED_SHIFTED_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_COMBINED_SHIFTED_KERNELS(u32, std::uint32_t);
HPRLP_DECLARE_SIGNED_Y_COMBINED_DELTA_KERNELS(u16, std::uint16_t);
HPRLP_DECLARE_SIGNED_Y_COMBINED_DELTA_KERNELS(u32, std::uint32_t);

__global__ void signed_unit_update_x_all_scalar_u16_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_x_all_scalar_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_x_all_scalar_flagged_u16_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_x_all_scalar_flagged_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_x_all_scalar_split_u16_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *indices, const std::uint8_t *negative,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_x_all_scalar_split_u16_flagged_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint16_t *indices, const std::uint8_t *negative,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_x_all_scalar_degree3_run_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin);

__global__ void
signed_unit_update_x_all_scalar_degree3_run_flagged_u32_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l,
    const HPRLP_FLOAT *u, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *scaled_y, HPRLP_FLOAT *scaled_x_hat_out,
    std::uint8_t *scaled_x_hat_nonzero_out,
    const HPRLP_FLOAT *inverse_col_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin);

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
    int run_row_count, int run_entry_begin);

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
    int run_row_count, int run_entry_begin);

__global__ void signed_unit_update_y_all_scalar_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_all_scalar_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_all_scalar_shifted_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_all_scalar_shifted_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out, const HPRLP_FLOAT *activity_shift,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

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
    const HPRLP_FLOAT *halpern_factors, int row_count);

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
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_all_scalar_skip_zero_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_all_scalar_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_direct_short_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_direct_short_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_direct_short_skip_zero_u16_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint16_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_direct_short_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count);

__global__ void signed_unit_update_y_direct_short_degree2_run_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin);

__global__ void
signed_unit_update_y_direct_short_degree2_run_skip_zero_u32_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y, const HPRLP_FLOAT *scaled_x,
    const std::uint8_t *scaled_x_nonzero, HPRLP_FLOAT *scaled_y_out,
    const HPRLP_FLOAT *inverse_row_norm, const int *row_ptr,
    const std::uint32_t *entries, const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int row_count,
    int run_row_begin, int run_row_count, int run_entry_begin);

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
    int run_row_count, int run_entry_begin);

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
    int run_row_count, int run_entry_begin);

#undef HPRLP_DECLARE_SIGNED_X_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_KERNELS
#undef HPRLP_DECLARE_SIGNED_X_FLAG_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_SKIP_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_SHIFTED_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_DELTA_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_COMBINED_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_COMBINED_SHIFTED_KERNELS
#undef HPRLP_DECLARE_SIGNED_Y_COMBINED_DELTA_KERNELS

#endif
