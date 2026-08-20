#ifndef HPRLP_SIGNED_UNIT_LAUNCHER_CUH
#define HPRLP_SIGNED_UNIT_LAUNCHER_CUH

#include "api/structs.h"
#include "cuda_kernels/backends/unit/signed_unit_kernels.cuh"

#include <cstdint>

struct HPRLP_signed_unit_x_split_u16_view_gpu {
    int columns;
    HPRLP_FLOAT *x;
    HPRLP_FLOAT *x_hat;
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    const std::uint8_t *bound_type;
    const HPRLP_FLOAT *objective;
    const HPRLP_FLOAT *last_x;
    const HPRLP_FLOAT *scaled_y;
    HPRLP_FLOAT *scaled_x_hat_output;
    std::uint8_t *scaled_x_hat_nonzero_output;
    const HPRLP_FLOAT *inverse_col_norm;
    const int *AT_row_ptr;
    const std::uint16_t *AT_constraint_index;
    const std::uint8_t *AT_negative;
};

struct HPRLP_signed_unit_x_packed_view_gpu {
    int columns;
    HPRLP_FLOAT *x;
    HPRLP_FLOAT *x_hat;
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    const std::uint8_t *bound_type;
    const HPRLP_FLOAT *objective;
    const HPRLP_FLOAT *last_x;
    const HPRLP_FLOAT *scaled_y;
    HPRLP_FLOAT *scaled_x_hat_output;
    std::uint8_t *scaled_x_hat_nonzero_output;
    const HPRLP_FLOAT *inverse_col_norm;
    const int *AT_row_ptr;
    const std::uint16_t *AT_entries_u16;
    const std::uint32_t *AT_entries_u32;
};

inline void hprlp_enqueue_signed_unit_x_packed_scalar(
    const HPRLP_signed_unit_x_packed_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    if (view.columns <= 0) return;
    const int blocks =
        (view.columns + threads_per_block - 1) / threads_per_block;
    if (view.AT_entries_u16 != nullptr) {
        if (view.scaled_x_hat_nonzero_output != nullptr) {
            signed_unit_update_x_all_scalar_flagged_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.x, view.x_hat, view.lower, view.upper,
                view.bound_type, view.objective, view.last_x,
                view.scaled_y, view.scaled_x_hat_output,
                view.scaled_x_hat_nonzero_output, view.inverse_col_norm,
                view.AT_row_ptr, view.AT_entries_u16, sigma_params,
                halpern_factors, view.columns);
        } else {
            signed_unit_update_x_all_scalar_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.x, view.x_hat, view.lower, view.upper,
                view.bound_type, view.objective, view.last_x,
                view.scaled_y, view.scaled_x_hat_output,
                view.inverse_col_norm, view.AT_row_ptr,
                view.AT_entries_u16, sigma_params, halpern_factors,
                view.columns);
        }
        return;
    }
    if (view.scaled_x_hat_nonzero_output != nullptr) {
        signed_unit_update_x_all_scalar_flagged_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper,
            view.bound_type, view.objective, view.last_x, view.scaled_y,
            view.scaled_x_hat_output,
            view.scaled_x_hat_nonzero_output, view.inverse_col_norm,
            view.AT_row_ptr, view.AT_entries_u32, sigma_params,
            halpern_factors, view.columns);
    } else {
        signed_unit_update_x_all_scalar_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper,
            view.bound_type, view.objective, view.last_x, view.scaled_y,
            view.scaled_x_hat_output, view.inverse_col_norm,
            view.AT_row_ptr, view.AT_entries_u32, sigma_params,
            halpern_factors, view.columns);
    }
}

struct HPRLP_signed_unit_x_packed_bucket_view_gpu {
    HPRLP_signed_unit_x_packed_view_gpu base;
    const int *short_row_ids;
    int short_row_count;
    const int *medium_row_ids;
    int medium_row_count;
    const int *long_row_ids;
    int long_row_count;
};

inline void hprlp_enqueue_signed_unit_x_packed_bucketed(
    const HPRLP_signed_unit_x_packed_bucket_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int scalar_threads,
    int cooperative_threads,
    cudaStream_t stream) {
    const HPRLP_signed_unit_x_packed_view_gpu &base = view.base;
    const int warps_per_block = cooperative_threads / 32;

#define HPRLP_LAUNCH_SIGNED_X_PACKED_BUCKETS(SUFFIX, ENTRIES)             \
    do {                                                                  \
        if (view.short_row_count > 0) {                                   \
            const int blocks =                                           \
                (view.short_row_count + scalar_threads - 1) /             \
                scalar_threads;                                          \
            if (base.scaled_x_hat_nonzero_output != nullptr) {            \
                signed_unit_update_x_rows_short_flagged_##SUFFIX##_kernel<<<\
                    blocks, scalar_threads, 0, stream>>>(                 \
                    base.x, base.x_hat, base.lower, base.upper,           \
                    base.bound_type, base.objective, base.last_x,         \
                    base.scaled_y, base.scaled_x_hat_output,              \
                    base.scaled_x_hat_nonzero_output,                     \
                    base.inverse_col_norm, base.AT_row_ptr, ENTRIES,      \
                    sigma_params, halpern_factors, view.short_row_ids,    \
                    view.short_row_count);                                \
            } else {                                                      \
                signed_unit_update_x_rows_short_##SUFFIX##_kernel<<<      \
                    blocks, scalar_threads, 0, stream>>>(                 \
                    base.x, base.x_hat, base.lower, base.upper,           \
                    base.bound_type, base.objective, base.last_x,         \
                    base.scaled_y, base.scaled_x_hat_output,              \
                    base.inverse_col_norm, base.AT_row_ptr, ENTRIES,      \
                    sigma_params, halpern_factors, view.short_row_ids,    \
                    view.short_row_count);                                \
            }                                                             \
        }                                                                 \
        if (view.medium_row_count > 0) {                                  \
            const int blocks =                                           \
                (view.medium_row_count + warps_per_block - 1) /           \
                warps_per_block;                                         \
            if (base.scaled_x_hat_nonzero_output != nullptr) {            \
                signed_unit_update_x_rows_warp_flagged_##SUFFIX##_kernel<<<\
                    blocks, cooperative_threads, 0, stream>>>(            \
                    base.x, base.x_hat, base.lower, base.upper,           \
                    base.bound_type, base.objective, base.last_x,         \
                    base.scaled_y, base.scaled_x_hat_output,              \
                    base.scaled_x_hat_nonzero_output,                     \
                    base.inverse_col_norm, base.AT_row_ptr, ENTRIES,      \
                    sigma_params, halpern_factors, view.medium_row_ids,   \
                    view.medium_row_count);                               \
            } else {                                                      \
                signed_unit_update_x_rows_warp_##SUFFIX##_kernel<<<       \
                    blocks, cooperative_threads, 0, stream>>>(            \
                    base.x, base.x_hat, base.lower, base.upper,           \
                    base.bound_type, base.objective, base.last_x,         \
                    base.scaled_y, base.scaled_x_hat_output,              \
                    base.inverse_col_norm, base.AT_row_ptr, ENTRIES,      \
                    sigma_params, halpern_factors, view.medium_row_ids,   \
                    view.medium_row_count);                               \
            }                                                             \
        }                                                                 \
        if (view.long_row_count > 0) {                                    \
            if (base.scaled_x_hat_nonzero_output != nullptr) {            \
                signed_unit_update_x_rows_block_flagged_##SUFFIX##_kernel<<<\
                    view.long_row_count, cooperative_threads, 0, stream>>>(\
                    base.x, base.x_hat, base.lower, base.upper,           \
                    base.bound_type, base.objective, base.last_x,         \
                    base.scaled_y, base.scaled_x_hat_output,              \
                    base.scaled_x_hat_nonzero_output,                     \
                    base.inverse_col_norm, base.AT_row_ptr, ENTRIES,      \
                    sigma_params, halpern_factors, view.long_row_ids,     \
                    view.long_row_count);                                 \
            } else {                                                      \
                signed_unit_update_x_rows_block_##SUFFIX##_kernel<<<      \
                    view.long_row_count, cooperative_threads, 0, stream>>>(\
                    base.x, base.x_hat, base.lower, base.upper,           \
                    base.bound_type, base.objective, base.last_x,         \
                    base.scaled_y, base.scaled_x_hat_output,              \
                    base.inverse_col_norm, base.AT_row_ptr, ENTRIES,      \
                    sigma_params, halpern_factors, view.long_row_ids,     \
                    view.long_row_count);                                 \
            }                                                             \
        }                                                                 \
    } while (false)

    if (base.AT_entries_u16 != nullptr) {
        HPRLP_LAUNCH_SIGNED_X_PACKED_BUCKETS(
            u16, base.AT_entries_u16);
    } else {
        HPRLP_LAUNCH_SIGNED_X_PACKED_BUCKETS(
            u32, base.AT_entries_u32);
    }

#undef HPRLP_LAUNCH_SIGNED_X_PACKED_BUCKETS
}

inline void hprlp_enqueue_signed_unit_x_split_u16_scalar(
    const HPRLP_signed_unit_x_split_u16_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    if (view.columns <= 0) return;
    const int blocks =
        (view.columns + threads_per_block - 1) / threads_per_block;
    if (view.scaled_x_hat_nonzero_output != nullptr) {
        signed_unit_update_x_all_scalar_split_u16_flagged_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper, view.bound_type,
            view.objective, view.last_x, view.scaled_y,
            view.scaled_x_hat_output,
            view.scaled_x_hat_nonzero_output, view.inverse_col_norm,
            view.AT_row_ptr, view.AT_constraint_index, view.AT_negative,
            sigma_params, halpern_factors, view.columns);
        return;
    }
    signed_unit_update_x_all_scalar_split_u16_kernel<<<
        blocks, threads_per_block, 0, stream>>>(
        view.x, view.x_hat, view.lower, view.upper, view.bound_type,
        view.objective, view.last_x, view.scaled_y,
        view.scaled_x_hat_output, view.inverse_col_norm,
        view.AT_row_ptr, view.AT_constraint_index, view.AT_negative,
        sigma_params, halpern_factors, view.columns);
}

struct HPRLP_signed_unit_y_scalar_view_gpu {
    int rows;
    HPRLP_FLOAT *y;
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    const std::uint8_t *bound_type;
    const HPRLP_FLOAT *last_y;
    const HPRLP_FLOAT *scaled_x;
    const std::uint8_t *scaled_x_nonzero;
    HPRLP_FLOAT *scaled_y_output;
    const HPRLP_FLOAT *activity_shift;
    const HPRLP_FLOAT *inverse_row_norm;
    const int *A_row_ptr;
    const std::uint16_t *A_entries_u16;
    const std::uint32_t *A_entries_u32;
};

inline void hprlp_enqueue_signed_unit_y_scalar(
    const HPRLP_signed_unit_y_scalar_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    if (view.rows <= 0) return;
    const int blocks = (view.rows + threads_per_block - 1) /
        threads_per_block;
    if (view.A_entries_u16 != nullptr) {
        if (view.activity_shift != nullptr) {
            signed_unit_update_y_all_scalar_shifted_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_y_output,
                view.activity_shift, view.inverse_row_norm,
                view.A_row_ptr, view.A_entries_u16, sigma_params,
                halpern_factors, view.rows);
        } else if (view.scaled_x_nonzero != nullptr) {
            signed_unit_update_y_all_scalar_skip_zero_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_x_nonzero,
                view.scaled_y_output, view.inverse_row_norm,
                view.A_row_ptr, view.A_entries_u16, sigma_params,
                halpern_factors, view.rows);
        } else {
            signed_unit_update_y_all_scalar_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_y_output,
                view.inverse_row_norm, view.A_row_ptr,
                view.A_entries_u16, sigma_params, halpern_factors,
                view.rows);
        }
        return;
    }
    if (view.activity_shift != nullptr) {
        signed_unit_update_y_all_scalar_shifted_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_y_output,
            view.activity_shift, view.inverse_row_norm, view.A_row_ptr,
            view.A_entries_u32, sigma_params, halpern_factors,
            view.rows);
    } else if (view.scaled_x_nonzero != nullptr) {
        signed_unit_update_y_all_scalar_skip_zero_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_x_nonzero,
            view.scaled_y_output, view.inverse_row_norm, view.A_row_ptr,
            view.A_entries_u32, sigma_params, halpern_factors,
            view.rows);
    } else {
        signed_unit_update_y_all_scalar_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_y_output,
            view.inverse_row_norm, view.A_row_ptr, view.A_entries_u32,
            sigma_params, halpern_factors, view.rows);
    }
}

struct HPRLP_signed_unit_y_scalar_delta_view_gpu {
    HPRLP_signed_unit_y_scalar_view_gpu base;
    const HPRLP_FLOAT *delta_scaled_x_hat;
    const HPRLP_FLOAT *delta_scaled_fixed;
    const int *delta_row_ptr;
    const std::uint32_t *delta_entries;
};

inline void hprlp_enqueue_signed_unit_y_scalar_delta(
    const HPRLP_signed_unit_y_scalar_delta_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    if (view.base.rows <= 0) return;
    const int blocks = (view.base.rows + threads_per_block - 1) /
        threads_per_block;
    if (view.base.A_entries_u16 != nullptr) {
        signed_unit_update_y_all_scalar_shifted_delta_u16_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.base.y, view.base.lower, view.base.upper,
            view.base.bound_type, view.base.last_y, view.base.scaled_x,
            view.base.scaled_y_output, view.base.activity_shift,
            view.delta_scaled_x_hat, view.delta_scaled_fixed,
            view.base.inverse_row_norm, view.base.A_row_ptr,
            view.base.A_entries_u16, view.delta_row_ptr,
            view.delta_entries, sigma_params, halpern_factors,
            view.base.rows);
        return;
    }
    signed_unit_update_y_all_scalar_shifted_delta_u32_kernel<<<
        blocks, threads_per_block, 0, stream>>>(
        view.base.y, view.base.lower, view.base.upper,
        view.base.bound_type, view.base.last_y, view.base.scaled_x,
        view.base.scaled_y_output, view.base.activity_shift,
        view.delta_scaled_x_hat, view.delta_scaled_fixed,
        view.base.inverse_row_norm, view.base.A_row_ptr,
        view.base.A_entries_u32, view.delta_row_ptr,
        view.delta_entries, sigma_params, halpern_factors,
        view.base.rows);
}

struct HPRLP_signed_unit_y_bucket_view_gpu {
    HPRLP_signed_unit_y_scalar_view_gpu base;
    const int *short_row_ids;
    int short_row_count;
    const int *medium_row_ids;
    int medium_row_count;
    const int *long_row_ids;
    int long_row_count;
    const HPRLP_FLOAT *delta_scaled_x_hat;
    const HPRLP_FLOAT *delta_scaled_fixed;
    const int *delta_row_ptr;
    const std::uint32_t *delta_entries;
};

inline void hprlp_enqueue_signed_unit_y_bucketed(
    const HPRLP_signed_unit_y_bucket_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int scalar_threads,
    int cooperative_threads,
    cudaStream_t stream) {
    const HPRLP_signed_unit_y_scalar_view_gpu &base = view.base;
    const int warps_per_block = cooperative_threads / 32;

#define HPRLP_LAUNCH_SIGNED_Y_PACKED_BUCKETS(SUFFIX, ENTRIES)             \
    do {                                                                  \
        if (view.short_row_count > 0) {                                   \
            const int blocks =                                           \
                (view.short_row_count + scalar_threads - 1) /             \
                scalar_threads;                                          \
            if (view.delta_row_ptr != nullptr) {                          \
                signed_unit_update_y_rows_short_delta_##SUFFIX##_kernel<<<\
                    blocks, scalar_threads, 0, stream>>>(                 \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.activity_shift, view.delta_scaled_x_hat,         \
                    view.delta_scaled_fixed, base.inverse_row_norm,       \
                    base.A_row_ptr, ENTRIES, view.delta_row_ptr,          \
                    view.delta_entries, sigma_params, halpern_factors,    \
                    view.short_row_ids, view.short_row_count);            \
            } else if (base.activity_shift != nullptr) {                  \
                signed_unit_update_y_rows_short_shifted_##SUFFIX##_kernel<<<\
                    blocks, scalar_threads, 0, stream>>>(                 \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.activity_shift, base.inverse_row_norm,           \
                    base.A_row_ptr, ENTRIES, sigma_params,                \
                    halpern_factors, view.short_row_ids,                  \
                    view.short_row_count);                                \
            } else if (base.scaled_x_nonzero != nullptr) {                \
                signed_unit_update_y_rows_short_skip_zero_##SUFFIX##_kernel<<<\
                    blocks, scalar_threads, 0, stream>>>(                 \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_x_nonzero,    \
                    base.scaled_y_output, base.inverse_row_norm,          \
                    base.A_row_ptr, ENTRIES, sigma_params,                \
                    halpern_factors, view.short_row_ids,                  \
                    view.short_row_count);                                \
            } else {                                                      \
                signed_unit_update_y_rows_short_##SUFFIX##_kernel<<<      \
                    blocks, scalar_threads, 0, stream>>>(                 \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.inverse_row_norm, base.A_row_ptr, ENTRIES,       \
                    sigma_params, halpern_factors, view.short_row_ids,    \
                    view.short_row_count);                                \
            }                                                             \
        }                                                                 \
        if (view.medium_row_count > 0) {                                  \
            const int blocks =                                           \
                (view.medium_row_count + warps_per_block - 1) /           \
                warps_per_block;                                         \
            if (view.delta_row_ptr != nullptr) {                          \
                signed_unit_update_y_rows_warp_delta_##SUFFIX##_kernel<<< \
                    blocks, cooperative_threads, 0, stream>>>(            \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.activity_shift, view.delta_scaled_x_hat,         \
                    view.delta_scaled_fixed, base.inverse_row_norm,       \
                    base.A_row_ptr, ENTRIES, view.delta_row_ptr,          \
                    view.delta_entries, sigma_params, halpern_factors,    \
                    view.medium_row_ids, view.medium_row_count);          \
            } else if (base.activity_shift != nullptr) {                  \
                signed_unit_update_y_rows_warp_shifted_##SUFFIX##_kernel<<<\
                    blocks, cooperative_threads, 0, stream>>>(            \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.activity_shift, base.inverse_row_norm,           \
                    base.A_row_ptr, ENTRIES, sigma_params,                \
                    halpern_factors, view.medium_row_ids,                 \
                    view.medium_row_count);                               \
            } else if (base.scaled_x_nonzero != nullptr) {                \
                signed_unit_update_y_rows_warp_skip_zero_##SUFFIX##_kernel<<<\
                    blocks, cooperative_threads, 0, stream>>>(            \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_x_nonzero,    \
                    base.scaled_y_output, base.inverse_row_norm,          \
                    base.A_row_ptr, ENTRIES, sigma_params,                \
                    halpern_factors, view.medium_row_ids,                 \
                    view.medium_row_count);                               \
            } else {                                                      \
                signed_unit_update_y_rows_warp_##SUFFIX##_kernel<<<       \
                    blocks, cooperative_threads, 0, stream>>>(            \
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.inverse_row_norm, base.A_row_ptr, ENTRIES,       \
                    sigma_params, halpern_factors, view.medium_row_ids,   \
                    view.medium_row_count);                               \
            }                                                             \
        }                                                                 \
        if (view.long_row_count > 0) {                                    \
            if (view.delta_row_ptr != nullptr) {                          \
                signed_unit_update_y_rows_block_delta_##SUFFIX##_kernel<<<\
                    view.long_row_count, cooperative_threads, 0, stream>>>(\
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.activity_shift, view.delta_scaled_x_hat,         \
                    view.delta_scaled_fixed, base.inverse_row_norm,       \
                    base.A_row_ptr, ENTRIES, view.delta_row_ptr,          \
                    view.delta_entries, sigma_params, halpern_factors,    \
                    view.long_row_ids, view.long_row_count);              \
            } else if (base.activity_shift != nullptr) {                  \
                signed_unit_update_y_rows_block_shifted_##SUFFIX##_kernel<<<\
                    view.long_row_count, cooperative_threads, 0, stream>>>(\
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.activity_shift, base.inverse_row_norm,           \
                    base.A_row_ptr, ENTRIES, sigma_params,                \
                    halpern_factors, view.long_row_ids,                   \
                    view.long_row_count);                                 \
            } else if (base.scaled_x_nonzero != nullptr) {                \
                signed_unit_update_y_rows_block_skip_zero_##SUFFIX##_kernel<<<\
                    view.long_row_count, cooperative_threads, 0, stream>>>(\
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_x_nonzero,    \
                    base.scaled_y_output, base.inverse_row_norm,          \
                    base.A_row_ptr, ENTRIES, sigma_params,                \
                    halpern_factors, view.long_row_ids,                   \
                    view.long_row_count);                                 \
            } else {                                                      \
                signed_unit_update_y_rows_block_##SUFFIX##_kernel<<<      \
                    view.long_row_count, cooperative_threads, 0, stream>>>(\
                    base.y, base.lower, base.upper, base.bound_type,      \
                    base.last_y, base.scaled_x, base.scaled_y_output,     \
                    base.inverse_row_norm, base.A_row_ptr, ENTRIES,       \
                    sigma_params, halpern_factors, view.long_row_ids,     \
                    view.long_row_count);                                 \
            }                                                             \
        }                                                                 \
    } while (false)

    if (base.A_entries_u16 != nullptr) {
        HPRLP_LAUNCH_SIGNED_Y_PACKED_BUCKETS(
            u16, base.A_entries_u16);
    } else {
        HPRLP_LAUNCH_SIGNED_Y_PACKED_BUCKETS(
            u32, base.A_entries_u32);
    }

#undef HPRLP_LAUNCH_SIGNED_Y_PACKED_BUCKETS
}

struct HPRLP_signed_unit_y_combined_view_gpu {
    int rows;
    HPRLP_FLOAT *y;
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    const std::uint8_t *bound_type;
    const HPRLP_FLOAT *last_y;
    const HPRLP_FLOAT *scaled_x;
    const std::uint8_t *scaled_x_nonzero;
    HPRLP_FLOAT *scaled_y_output;
    const HPRLP_FLOAT *activity_shift;
    const HPRLP_FLOAT *inverse_row_norm;
    const int *A_row_ptr;
    const std::uint16_t *A_entries_u16;
    const std::uint32_t *A_entries_u32;
    const int *medium_row_ids;
    int medium_row_count;
    const int *long_row_ids;
    int long_row_count;
    // Optional signed reduced-matrix delta correction. The delta input and
    // fixed value are pre-scaled by the original column norm; packed entries
    // carry only the delta-column index and sign.
    const HPRLP_FLOAT *delta_scaled_x_hat;
    const HPRLP_FLOAT *delta_scaled_fixed;
    const int *delta_row_ptr;
    const std::uint32_t *delta_entries;
    const std::uint32_t *delta_nonempty_words;
};

inline void hprlp_enqueue_signed_unit_y_combined(
    const HPRLP_signed_unit_y_combined_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int threads_per_block,
    cudaStream_t stream) {
    if (view.rows <= 0) return;
    const int warps_per_block = threads_per_block / 32;
    const int direct_blocks =
        (view.rows + threads_per_block - 1) / threads_per_block;
    const int medium_blocks =
        (view.medium_row_count + warps_per_block - 1) /
        warps_per_block;
    const int blocks =
        direct_blocks + medium_blocks + view.long_row_count;
    if (view.A_entries_u16 != nullptr) {
        if (view.delta_row_ptr != nullptr) {
            signed_unit_update_y_combined_shifted_delta_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_y_output,
                view.activity_shift, view.delta_scaled_x_hat,
                view.delta_scaled_fixed, view.delta_row_ptr,
                view.delta_entries, view.delta_nonempty_words,
                view.inverse_row_norm, view.A_row_ptr,
                view.A_entries_u16, sigma_params, halpern_factors,
                view.rows, view.medium_row_ids, view.medium_row_count,
                view.long_row_ids, view.long_row_count);
        } else if (view.activity_shift != nullptr) {
            signed_unit_update_y_combined_shifted_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_y_output,
                view.activity_shift, view.inverse_row_norm, view.A_row_ptr,
                view.A_entries_u16, sigma_params, halpern_factors,
                view.rows, view.medium_row_ids, view.medium_row_count,
                view.long_row_ids, view.long_row_count);
        } else if (view.scaled_x_nonzero != nullptr) {
            signed_unit_update_y_combined_skip_zero_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_x_nonzero,
                view.scaled_y_output, view.inverse_row_norm, view.A_row_ptr,
                view.A_entries_u16, sigma_params, halpern_factors,
                view.rows, view.medium_row_ids, view.medium_row_count,
                view.long_row_ids, view.long_row_count);
        } else {
            signed_unit_update_y_combined_u16_kernel<<<
                blocks, threads_per_block, 0, stream>>>(
                view.y, view.lower, view.upper, view.bound_type,
                view.last_y, view.scaled_x, view.scaled_y_output,
                view.inverse_row_norm, view.A_row_ptr, view.A_entries_u16,
                sigma_params, halpern_factors, view.rows,
                view.medium_row_ids, view.medium_row_count,
                view.long_row_ids, view.long_row_count);
        }
        return;
    }
    if (view.delta_row_ptr != nullptr) {
        signed_unit_update_y_combined_shifted_delta_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_y_output,
            view.activity_shift, view.delta_scaled_x_hat,
            view.delta_scaled_fixed, view.delta_row_ptr,
            view.delta_entries, view.delta_nonempty_words,
            view.inverse_row_norm, view.A_row_ptr, view.A_entries_u32,
            sigma_params, halpern_factors, view.rows,
            view.medium_row_ids, view.medium_row_count,
            view.long_row_ids, view.long_row_count);
    } else if (view.activity_shift != nullptr) {
        signed_unit_update_y_combined_shifted_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_y_output,
            view.activity_shift, view.inverse_row_norm, view.A_row_ptr,
            view.A_entries_u32, sigma_params, halpern_factors, view.rows,
            view.medium_row_ids, view.medium_row_count,
            view.long_row_ids, view.long_row_count);
    } else if (view.scaled_x_nonzero != nullptr) {
        signed_unit_update_y_combined_skip_zero_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_x_nonzero,
            view.scaled_y_output, view.inverse_row_norm, view.A_row_ptr,
            view.A_entries_u32, sigma_params, halpern_factors, view.rows,
            view.medium_row_ids, view.medium_row_count,
            view.long_row_ids, view.long_row_count);
    } else {
        signed_unit_update_y_combined_u32_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.y, view.lower, view.upper, view.bound_type,
            view.last_y, view.scaled_x, view.scaled_y_output,
            view.inverse_row_norm, view.A_row_ptr, view.A_entries_u32,
            sigma_params, halpern_factors, view.rows,
            view.medium_row_ids, view.medium_row_count,
            view.long_row_ids, view.long_row_count);
    }
}

#endif
