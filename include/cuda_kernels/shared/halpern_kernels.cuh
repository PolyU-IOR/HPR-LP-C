#ifndef HPRLP_HALPERN_KERNELS_CUH
#define HPRLP_HALPERN_KERNELS_CUH

#include "api/structs.h"

__global__
void advance_halpern_factors_kernel(int *halpern_inner, HPRLP_FLOAT *halpern_factors);

__global__
void prepare_halpern_factor_batch_kernel(
    int *halpern_inner, HPRLP_FLOAT *halpern_factors,
    HPRLP_FLOAT *halpern_factor_batch, int batch_size);


#endif
