#ifndef HPRLP_UNIT_FACTORIZED_LAUNCHER_CUH
#define HPRLP_UNIT_FACTORIZED_LAUNCHER_CUH

#include "api/structs.h"
#include "cuda_kernels/backends/unit/unit_kernels.cuh"
#include "cuda_kernels/cuda_check.h"

#include <cstddef>
#include <cstdint>

struct HPRLP_unit_factorized_x_view_gpu {
    int rows;
    int columns;
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
    HPRLP_FLOAT *scattered_activity_output;
    const int *AT_row_ptr;
    const std::uint16_t *AT_constraint_index;
};

inline void hprlp_enqueue_unit_factorized_x_scalar(
    const HPRLP_unit_factorized_x_view_gpu &view,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign,
    int threads_per_block,
    cudaStream_t stream) {
    if (view.columns <= 0) return;
    const int blocks =
        (view.columns + threads_per_block - 1) / threads_per_block;
    if (view.scattered_activity_output != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(
            view.scattered_activity_output, 0,
            static_cast<std::size_t>(view.rows) * sizeof(HPRLP_FLOAT),
            stream));
        fused_update_x_z_all_scalar_unit_active_scatter_kernel<<<
            blocks, threads_per_block, 0, stream>>>(
            view.x, view.x_hat, view.lower, view.upper, view.bound_type,
            view.objective, view.last_x, view.scaled_y,
            view.inverse_col_norm, view.scattered_activity_output,
            view.AT_row_ptr, view.AT_constraint_index, sigma_params,
            halpern_factors, uniform_unit_sign, view.columns);
        return;
    }
    fused_update_x_z_all_scalar_unit_kernel<<<
        blocks, threads_per_block, 0, stream>>>(
        view.x, view.x_hat, view.lower, view.upper, view.bound_type,
        view.objective, view.last_x, view.scaled_y,
        view.inverse_col_norm, view.scaled_x_hat_output, view.AT_row_ptr,
        view.AT_constraint_index, sigma_params, halpern_factors,
        uniform_unit_sign, view.columns);
}

#endif
