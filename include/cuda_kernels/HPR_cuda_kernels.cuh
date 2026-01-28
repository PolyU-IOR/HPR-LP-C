#ifndef HPRLP_CUDA_KERNELS_H
#define HPRLP_CUDA_KERNELS_H

#include "../structs.h"
#include <iostream>
#include <cmath>
#include <cfloat>
#include <stdio.h>

#define numThreads 256
// define the numBlocks function
#define numBlocks(n) (n + numThreads - 1) / numThreads

__global__
void set_vector_value_device_kernel(HPRLP_FLOAT *x, int n, HPRLP_FLOAT value);


__global__ 
void conceptual_b_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int m);


__global__ 
void axpy_kernel(HPRLP_FLOAT a, const HPRLP_FLOAT* x, const HPRLP_FLOAT* y, HPRLP_FLOAT* z, int len);


__global__ 
void axpby_kernel(HPRLP_FLOAT a, const HPRLP_FLOAT *x, HPRLP_FLOAT b, const HPRLP_FLOAT *y, HPRLP_FLOAT *z, int len);


__global__ 
void vector_dot_product_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n, bool divide = false);


__global__ 
void CSR_A_row_norm_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *result, int norm = 1);


__global__ 
void mul_CSR_A_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide = false);


__global__ 
void mul_CSR_AT_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide = false);





__global__
void residual_compute_Rp_kernel(HPRLP_FLOAT *row_norm, HPRLP_FLOAT *Rp, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, int m);


__global__
void residual_compute_lu_kernel(HPRLP_FLOAT *col_norm, HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *l, HPRLP_FLOAT *u, int n);


__global__
void residual_compute_Rd_kernel(HPRLP_FLOAT *col_norm, HPRLP_FLOAT *ATy, HPRLP_FLOAT *z, HPRLP_FLOAT *c, HPRLP_FLOAT *Rd, int n);


// X update kernels
__global__ 
void update_zx_check_kernel(HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x, HPRLP_FLOAT *z_bar, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, 
                        HPRLP_FLOAT sigma, HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, HPRLP_FLOAT *last_x, HPRLP_FLOAT *Halpern_params, int n);

__global__
void update_zx_normal_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, HPRLP_FLOAT sigma, HPRLP_FLOAT *ATy,
                            HPRLP_FLOAT *c, HPRLP_FLOAT *last_x, HPRLP_FLOAT *Halpern_params, int n);

// Y update kernels
__global__ 
void update_y_check_kernel(HPRLP_FLOAT *y_temp, HPRLP_FLOAT *y_bar, HPRLP_FLOAT *y, HPRLP_FLOAT *y_obj, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2,
                        HPRLP_FLOAT *last_y, HPRLP_FLOAT *Halpern_params, int m);

__global__ 
void update_y_normal_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2, HPRLP_FLOAT *last_y, HPRLP_FLOAT *Halpern_params, int m);


#endif