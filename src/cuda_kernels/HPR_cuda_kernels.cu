#include "HPR_cuda_kernels.cuh"

namespace {

__device__ __forceinline__ HPRLP_FLOAT project_x_with_bounds(HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper, uint8_t bound_type) {
    if (bound_type == 0) {
        return value;
    }
    if (bound_type == 1) {
        return fmax(value, lower);
    }
    if (bound_type == 2) {
        return fmin(value, upper);
    }
    return fmin(fmax(value, lower), upper);
}

__device__ __forceinline__ HPRLP_FLOAT project_y_delta(HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper, uint8_t bound_type) {
    if (bound_type == 0) {
        return 0.0;
    }
    if (bound_type == 1) {
        return fmax(lower - value, 0.0);
    }
    if (bound_type == 2) {
        return fmin(upper - value, 0.0);
    }
    return fmax(lower - value, fmin(upper - value, 0.0));
}

}


__global__ void conceptual_b_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        HPRLP_FLOAT x_val = x[idx];
        HPRLP_FLOAT y_val = y[idx];
        x_val = (std::isinf(x_val)) ? 0.0 : x_val;
        y_val = (std::isinf(y_val)) ? 0.0 : y_val;
        result[idx] = max(std::abs(x_val), std::abs(y_val));
    }
}


__global__ void axpy_kernel(HPRLP_FLOAT a, const HPRLP_FLOAT* x, const HPRLP_FLOAT* y, HPRLP_FLOAT* z, int len){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) z[i] = y[i] + a * x[i];
}


__global__ void axpby_kernel(HPRLP_FLOAT a, const HPRLP_FLOAT* x, HPRLP_FLOAT b, const HPRLP_FLOAT* y, HPRLP_FLOAT* z, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) z[i] = a * x[i] + b * y[i];
}


__global__ void set_vector_value_device_kernel(HPRLP_FLOAT *x, int len, HPRLP_FLOAT value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        x[i] = value;
    }
}



__global__ void set_vector_value_device_kernel(int *x, int len, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        x[i] = value;
    }
}


__global__ void vector_dot_product_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n, bool divide) {
    if (divide) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            result[i] = x[i] / y[i];
        }
    } else {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            result[i] = x[i] * y[i];
        }
    }
}



__global__ void CSR_A_row_norm_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *result, int norm) {
    if (norm == 99) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m) {
            result[i] = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                if (result[i] < std::fabs(value[j])) {
                    result[i] = std::fabs(value[j]);
                }
            }
            result[i] = std::sqrt(result[i]);
            if (result[i] < 1e-15){
                result[i] = 1.0;
            }
        }
    } 
    else if (norm == 1) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m) {
            result[i] = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                result[i] += std::fabs(value[j]);
            }
            result[i] = std::sqrt(result[i]);
            if (result[i] < 1e-15){
                result[i] = 1.0;
            }
        }
    } 
}



__global__ void mul_CSR_A_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide) {
    if (divide) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m) {
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                value[j] /= x[i];
            }
        }
    } else {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m) {
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                value[j] *= x[i];
            }
        }
    }
}


__global__ void mul_CSR_AT_row_kernel(int m, int *rowPtr, int *colIndex, HPRLP_FLOAT *value, HPRLP_FLOAT *x, bool divide) {
    if (divide) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m) {
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                value[j] /= x[colIndex[j]];
            }
        }
    } else {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < m) {
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                value[j] *= x[colIndex[j]];
            }
        }
    }
}


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


__global__ void advance_halpern_factors_kernel(int *halpern_inner, HPRLP_FLOAT *halpern_factors) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int next_inner = halpern_inner[0] + 1;
        halpern_inner[0] = next_inner;
        HPRLP_FLOAT fact1 = 1.0 / (next_inner + 2.0);
        halpern_factors[0] = fact1;
        halpern_factors[1] = 1.0 - fact1;
    }
}


__global__ void update_zx_check_kernel(HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x, HPRLP_FLOAT *z_bar, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, 
                        HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, HPRLP_FLOAT *last_x,
                        const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[i];
        HPRLP_FLOAT ATy_ci = ATy[i] - c[i];
        HPRLP_FLOAT z_temp = xi + sigma * ATy_ci;
        HPRLP_FLOAT li = l[i];
        HPRLP_FLOAT ui = u[i];
        HPRLP_FLOAT x_bar_val = fmin(ui, fmax(li, z_temp));   
        HPRLP_FLOAT z_bar_val = (x_bar_val - z_temp) / sigma;
        HPRLP_FLOAT x_hat_val = 2 * x_bar_val - xi;
        HPRLP_FLOAT x_new_val = fact2 * x_hat_val + fact1 * last_x[i];
        x_temp[i] = x_bar_val - x_hat_val;
        z_bar[i] = z_bar_val;
        x_bar[i] = x_bar_val;
        x_hat[i] = x_hat_val;
        x[i] = x_new_val;
    }
}


__global__ void update_zx_normal_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, HPRLP_FLOAT *ATy, HPRLP_FLOAT *c,
                      HPRLP_FLOAT *last_x, const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        
        HPRLP_FLOAT xi = x[i];
        HPRLP_FLOAT li = l[i];
        HPRLP_FLOAT ui = u[i];
        HPRLP_FLOAT z_temp = xi + sigma * (ATy[i] - c[i]);
        HPRLP_FLOAT x_bar_val = fmin(ui, fmax(li, z_temp));            
        HPRLP_FLOAT x_hat_val = 2 * x_bar_val - xi;
        HPRLP_FLOAT x_new_val= fact2 * x_hat_val + fact1 * last_x[i];
        x_hat[i] = x_hat_val;
        x[i] = x_new_val;
    }
}

__global__ void update_y_check_kernel(HPRLP_FLOAT *y_temp, HPRLP_FLOAT *y_bar, HPRLP_FLOAT *y, HPRLP_FLOAT *y_obj, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax,
                        HPRLP_FLOAT *last_y, const HPRLP_FLOAT *sigma_params, const HPRLP_FLOAT *halpern_factors, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT yi = y[i];
        HPRLP_FLOAT ai = Ax[i];
        HPRLP_FLOAT li = AL[i];
        HPRLP_FLOAT ui = AU[i];
        HPRLP_FLOAT y0i = last_y[i];
        HPRLP_FLOAT v = ai - fact1 * yi;
        HPRLP_FLOAT d = fmax(li - v, fmin(ui - v, 0.0));
        HPRLP_FLOAT y_bar_val = fact2 * d;
        HPRLP_FLOAT y_hat_val = 2 * y_bar_val - yi;
        HPRLP_FLOAT y_new_val = halpern_fact2 * y_hat_val + halpern_fact1 * y0i;
        y_temp[i] = y_bar_val - y_hat_val;
        y_bar[i] = y_bar_val;
        y_obj[i] = v + d;
        y[i] = y_new_val;
    }
}

__global__ void update_y_normal_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax,
                                       HPRLP_FLOAT *last_y, const HPRLP_FLOAT *sigma_params,
                                       const HPRLP_FLOAT *halpern_factors, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT yi = y[i];
        HPRLP_FLOAT ai = Ax[i];
        HPRLP_FLOAT li = AL[i];
        HPRLP_FLOAT ui = AU[i];
        HPRLP_FLOAT y0i = last_y[i];
        HPRLP_FLOAT v = ai - fact1 * yi;
        HPRLP_FLOAT d = fmax(li - v, fmin(ui - v, 0.0));
        HPRLP_FLOAT y_bar_val = fact2 * d;
        HPRLP_FLOAT y_hat_val = 2 * y_bar_val - yi;
        HPRLP_FLOAT y_new_val = halpern_fact2 * y_hat_val + halpern_fact1 * y0i;
        y[i] = y_new_val;
    }
}

__global__ void fused_update_x_z_rows_short_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                                   const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                                   const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                                   const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                                   const HPRLP_FLOAT *halpern_factors,
                                                   const int *row_ids, int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrows) {
        int row = row_ids[tid];
        HPRLP_FLOAT sigma = sigma_params[0];
        HPRLP_FLOAT acc = 0.0;
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(AT_value[idx], y[AT_colIndex[idx]], acc);
        }

        HPRLP_FLOAT fact1 = halpern_factors[0];
        HPRLP_FLOAT fact2 = halpern_factors[1];
        HPRLP_FLOAT xi = x[row];
        HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
        HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
        HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
        x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
        x_hat[row] = x_hat_value;
    }
}

__global__ void fused_update_x_z_rows_warp_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                                  const uint8_t *x_bound_type, const HPRLP_FLOAT *c, const HPRLP_FLOAT *last_x,
                                                  const HPRLP_FLOAT *y, const int *AT_rowPtr, const int *AT_colIndex,
                                                  const HPRLP_FLOAT *AT_value, const HPRLP_FLOAT *sigma_params,
                                                  const HPRLP_FLOAT *halpern_factors,
                                                  const int *row_ids, int nrows) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_idx = blockIdx.x * warps_per_block + warp_in_block;

    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        HPRLP_FLOAT sigma = sigma_params[0];
        int start = AT_rowPtr[row];
        int end = AT_rowPtr[row + 1];
        HPRLP_FLOAT acc = 0.0;
        for (int idx = start + lane; idx < end; idx += 32) {
            acc = fma(AT_value[idx], y[AT_colIndex[idx]], acc);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            HPRLP_FLOAT fact1 = halpern_factors[0];
            HPRLP_FLOAT fact2 = halpern_factors[1];
            HPRLP_FLOAT xi = x[row];
            HPRLP_FLOAT z_temp = fma(sigma, acc - c[row], xi);
            HPRLP_FLOAT x_bar = project_x_with_bounds(z_temp, l[row], u[row], x_bound_type[row]);
            HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
            x[row] = fma(fact2, x_hat_value, fact1 * last_x[row]);
            x_hat[row] = x_hat_value;
        }
    }
}

__global__ void fused_update_y_rows_short_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                                 const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                                 const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                                 const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                                 const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nrows) {
        int row = row_ids[tid];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        HPRLP_FLOAT acc = 0.0;
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }

        HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
        HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
        HPRLP_FLOAT yi = y[row];
        HPRLP_FLOAT v = fma(-fact1, yi, acc);
        HPRLP_FLOAT d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
        HPRLP_FLOAT y_bar = fact2 * d;
        HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
        y[row] = fma(halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
    }
}

__global__ void fused_update_y_rows_warp_kernel(HPRLP_FLOAT *y, const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                                const uint8_t *y_bound_type, const HPRLP_FLOAT *last_y,
                                                const HPRLP_FLOAT *x_hat, const int *A_rowPtr, const int *A_colIndex,
                                                const HPRLP_FLOAT *A_value, const HPRLP_FLOAT *sigma_params,
                                                const HPRLP_FLOAT *halpern_factors, const int *row_ids, int nrows) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int row_idx = blockIdx.x * warps_per_block + warp_in_block;

    if (row_idx < nrows) {
        int row = row_ids[row_idx];
        HPRLP_FLOAT fact1 = sigma_params[1];
        HPRLP_FLOAT fact2 = sigma_params[2];
        int start = A_rowPtr[row];
        int end = A_rowPtr[row + 1];
        HPRLP_FLOAT acc = 0.0;
        for (int idx = start + lane; idx < end; idx += 32) {
            acc = fma(A_value[idx], x_hat[A_colIndex[idx]], acc);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            HPRLP_FLOAT halpern_fact1 = halpern_factors[0];
            HPRLP_FLOAT halpern_fact2 = halpern_factors[1];
            HPRLP_FLOAT yi = y[row];
            HPRLP_FLOAT v = fma(-fact1, yi, acc);
            HPRLP_FLOAT d = project_y_delta(v, AL[row], AU[row], y_bound_type[row]);
            HPRLP_FLOAT y_bar = fact2 * d;
            HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
            y[row] = fma(halpern_fact2, y_hat, halpern_fact1 * last_y[row]);
        }
    }
}
