#include "HPR_cuda_kernels.cuh"


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


<<<<<<< HEAD
__global__ void update_zx_check_kernel(HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x, HPRLP_FLOAT *z_bar, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, 
                        HPRLP_FLOAT sigma, HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, HPRLP_FLOAT *last_x, HPRLP_FLOAT *Halpern_params, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HPRLP_FLOAT fact1 = Halpern_params[0];  // Get Halpern iteration parameters
        HPRLP_FLOAT fact2 = Halpern_params[1];
=======
__global__ void compute_zx_kernel(HPRLP_FLOAT *x_temp, HPRLP_FLOAT *x, HPRLP_FLOAT *z_bar, HPRLP_FLOAT *x_bar, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, 
                        HPRLP_FLOAT sigma, HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, HPRLP_FLOAT *last_x, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
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


<<<<<<< HEAD
__global__ void update_zx_normal_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, HPRLP_FLOAT sigma, HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, 
                      HPRLP_FLOAT *last_x, HPRLP_FLOAT *Halpern_params, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HPRLP_FLOAT fact1 = Halpern_params[0];
        HPRLP_FLOAT fact2 = Halpern_params[1];
        
=======
__global__ void compute_x_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, HPRLP_FLOAT *l, HPRLP_FLOAT *u, HPRLP_FLOAT sigma, HPRLP_FLOAT *ATy, HPRLP_FLOAT *c, 
                      HPRLP_FLOAT *last_x, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
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

<<<<<<< HEAD
__global__ void update_y_check_kernel(HPRLP_FLOAT *y_temp, HPRLP_FLOAT *y_bar, HPRLP_FLOAT *y, HPRLP_FLOAT *y_obj, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2,
                        HPRLP_FLOAT *last_y, HPRLP_FLOAT *Halpern_params, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        HPRLP_FLOAT halpern_fact1 = Halpern_params[0];
        HPRLP_FLOAT halpern_fact2 = Halpern_params[1];
=======

__global__ void compute_y_1_kernel(HPRLP_FLOAT *y_temp, HPRLP_FLOAT *y_bar, HPRLP_FLOAT *y, HPRLP_FLOAT *y_obj, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2,
                        HPRLP_FLOAT *last_y, HPRLP_FLOAT halpern_fact1, HPRLP_FLOAT halpern_fact2, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
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

<<<<<<< HEAD
__global__ void update_y_normal_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2, HPRLP_FLOAT *last_y, HPRLP_FLOAT *Halpern_params, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        HPRLP_FLOAT halpern_fact1 = Halpern_params[0];
        HPRLP_FLOAT halpern_fact2 = Halpern_params[1];
=======
__global__ void compute_y_2_kernel(HPRLP_FLOAT *y, HPRLP_FLOAT *AL, HPRLP_FLOAT *AU, HPRLP_FLOAT *Ax, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2, HPRLP_FLOAT *last_y, HPRLP_FLOAT halpern_fact1, HPRLP_FLOAT halpern_fact2, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
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
<<<<<<< HEAD
}
=======
}
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
