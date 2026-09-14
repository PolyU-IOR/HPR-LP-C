#ifndef HPRLP_SIMPLE_UPDATE_KERNELS_CUH
#define HPRLP_SIMPLE_UPDATE_KERNELS_CUH

#include "api/structs.h"

__global__
void update_zx_check_kernel(HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x, HPRLP_FLOAT *z_bar, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u,
                        HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, HPRLP_FLOAT *last_x,
                        const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors, int n);

__global__
void update_zx_normal_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, HPRLP_FLOAT *ATy,
                            HPRLP_FLOAT *c, HPRLP_FLOAT *last_x,
                            const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors, int n);

// Y update kernels
__global__
void update_y_check_kernel(HPRLP_FLOAT *y_temp, HPRLP_FLOAT *y_bar, HPRLP_FLOAT *y, HPRLP_FLOAT *y_obj, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax,
                        HPRLP_FLOAT *last_y, const HPRLP_FLOAT *sigma_params,
                        const HPRLP_FLOAT *halpern_factors, int m);

__global__
void update_y_normal_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax,
                            HPRLP_FLOAT *last_y, const HPRLP_FLOAT *sigma_params,
                            const HPRLP_FLOAT *halpern_factors, int m);

#endif
