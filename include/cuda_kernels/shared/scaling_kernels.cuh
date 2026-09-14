#ifndef HPRLP_SCALING_KERNELS_CUH
#define HPRLP_SCALING_KERNELS_CUH

#include "api/structs.h"

__global__
void CSR_A_row_norm_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *result, int norm = 1);


__global__
void mul_CSR_A_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide = false);


__global__
void mul_CSR_AT_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide = false);

#endif
