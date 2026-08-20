#ifndef HPRLP_GRID_SLACK_LAPLACIAN_KERNELS_CUH
#define HPRLP_GRID_SLACK_LAPLACIAN_KERNELS_CUH

#include "api/structs.h"

__global__ void grid_slack_laplacian_update_x_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const int *AT_row_ptr,
    const HPRLP_FLOAT *AT_value, HPRLPGridSlackLaplacianShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n);

__global__ void grid_slack_laplacian_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *A_value,
    HPRLPGridSlackLaplacianShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m);

#endif
