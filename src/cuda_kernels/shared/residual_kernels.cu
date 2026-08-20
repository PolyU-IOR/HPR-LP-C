#include "cuda_kernels/shared/residual_kernels.cuh"
#include <cmath>


__global__ void residual_compute_Rp_kernel(HPRLP_FLOAT *row_norm, HPRLP_FLOAT *Rp, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, int m){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds to prevent out-of-bounds access
    if(i < m) {
        HPRLP_FLOAT v = Ax[i];
        HPRLP_FLOAT low = AL[i];
        HPRLP_FLOAT high = AU[i];
        HPRLP_FLOAT row_normi = row_norm[i];
        HPRLP_FLOAT Rpi = fmax(fmin(high - v, 0.0), low - v);
        Rp[i] = Rpi * row_normi;
    }
}

__global__ void residual_compute_lu_kernel(HPRLP_FLOAT *col_norm, HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *l, HPRLP_FLOAT *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HPRLP_FLOAT temp = (x_bar[i] < l[i]) ? (l[i] - x_bar[i]) : ((x_bar[i] > u[i]) ? (x_bar[i] - u[i]) : 0.0);
        x_temp[i] = temp / col_norm[i];
    }
}


__global__ void residual_compute_Rd_kernel(HPRLP_FLOAT *col_norm, HPRLP_FLOAT *ATy, HPRLP_FLOAT *z, HPRLP_FLOAT *c, HPRLP_FLOAT *Rd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HPRLP_FLOAT rdi = c[i] - ATy[i] - z[i];
        Rd[i] = rdi * col_norm[i];
    }
}
