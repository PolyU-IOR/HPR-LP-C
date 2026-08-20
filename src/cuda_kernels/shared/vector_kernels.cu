#include "cuda_kernels/shared/vector_kernels.cuh"
#include <cmath>


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

__global__ void vector_dot_product_zero_bitset_kernel(
    const HPRLP_FLOAT *x, const HPRLP_FLOAT *y, HPRLP_FLOAT *result,
    std::uint32_t *positive_zero_bits, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = index < n;
    HPRLP_FLOAT value = 0.0;
    if (valid) {
        value = x[index] * y[index];
        result[index] = value;
    }
    const bool is_positive_zero =
        valid && static_cast<unsigned long long>(
                     __double_as_longlong(value)) == 0ull;
    const unsigned int zero_mask = __ballot_sync(
        0xffffffffu, is_positive_zero);
    if ((threadIdx.x & 31) == 0 && valid) {
        positive_zero_bits[index >> 5] = zero_mask;
    }
}

__global__ void pack_positive_zero_bitset_count_kernel(
    const HPRLP_FLOAT *input, std::uint32_t *positive_zero_bits,
    unsigned long long *positive_zero_count, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = index < n;
    HPRLP_FLOAT value = 1.0;
    if (valid) value = input[index];
    const bool is_positive_zero =
        valid && static_cast<unsigned long long>(
                     __double_as_longlong(value)) == 0ull;
    const unsigned int zero_mask = __ballot_sync(
        0xffffffffu, is_positive_zero);
    if ((threadIdx.x & 31) == 0 && valid) {
        positive_zero_bits[index >> 5] = zero_mask;
        atomicAdd(positive_zero_count,
                  static_cast<unsigned long long>(__popc(zero_mask)));
    }
}

__global__ void reciprocal_vector_kernel(const HPRLP_FLOAT *input, HPRLP_FLOAT *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0 / input[i];
    }
}
