#ifndef HPRLP_RESIDUAL_KERNELS_CUH
#define HPRLP_RESIDUAL_KERNELS_CUH

#include "api/structs.h"

__global__
void residual_compute_Rp_kernel(HPRLP_FLOAT *row_norm, HPRLP_FLOAT *Rp, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, int m);


__global__
void residual_compute_lu_kernel(HPRLP_FLOAT *col_norm, HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *l, HPRLP_FLOAT *u, int n);


__global__
void residual_compute_Rd_kernel(HPRLP_FLOAT *col_norm, HPRLP_FLOAT *ATy, HPRLP_FLOAT *z, HPRLP_FLOAT *c, HPRLP_FLOAT *Rd, int n);

#endif
