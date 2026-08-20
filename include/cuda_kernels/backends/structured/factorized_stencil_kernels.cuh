#ifndef HPRLP_FACTORIZED_STENCIL_KERNELS_CUH
#define HPRLP_FACTORIZED_STENCIL_KERNELS_CUH

#include "api/structs.h"

// Parameterized structural-family kernels. The detector infers every shape
// and coefficient field from CSR/CSR^T; no model identity or fixed dimension
// participates in eligibility.
__global__ void windowed_stencil_update_x_grid_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *last_x,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_x_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_out, HPRLP_FLOAT *dense_partials,
    unsigned int *dense_counter,
    HPRLPWindowedStencilShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors);

__global__ void windowed_stencil_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *last_y,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_y_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *scaled_x_hat, const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_out, HPRLPWindowedStencilShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors);

// State-specialized variants admitted only after the host certifies that all
// grid variables are zero-objective boxed variables, the two
// observation row blocks are uniform one-sided constraints, and the stencil
// rows are equalities.  The matrix shape is still inferred and verified from
// CSR/CSR^T; no model identity is used.
__global__ void windowed_stencil_state_update_x_grid_kernel(
    HPRLP_FLOAT *x, const HPRLP_FLOAT *last_x,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_x_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *scaled_y, const HPRLP_FLOAT *inverse_col_norm,
    HPRLP_FLOAT *scaled_x_hat_out, HPRLP_FLOAT *dense_partials,
    unsigned int *dense_counter, HPRLPWindowedStencilShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors);

__global__ void windowed_stencil_state_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *last_y,
    const std::uint8_t *static_codes,
    const HPRLP_factorized_y_static_record *static_records,
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *scaled_x_hat, const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y_out, HPRLPWindowedStencilShape shape,
    std::uint8_t observation_first_bound_type,
    std::uint8_t observation_second_bound_type,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors);

#endif
