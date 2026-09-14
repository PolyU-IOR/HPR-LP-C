#include "cuda_kernels/backends/simple/simple_update_kernels.cuh"

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
