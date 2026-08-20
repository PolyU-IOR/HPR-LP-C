#ifndef HPRLP_VECTOR_KERNELS_CUH
#define HPRLP_VECTOR_KERNELS_CUH

#include "api/structs.h"
#include <cstdint>

__global__
void set_vector_value_device_kernel(HPRLP_FLOAT *x, int n, HPRLP_FLOAT value);

__global__
void set_vector_value_device_kernel(int *x, int n, int value);


__global__ 
void conceptual_b_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int m);


__global__ 
void axpy_kernel(HPRLP_FLOAT a, const HPRLP_FLOAT* x, const HPRLP_FLOAT* y, HPRLP_FLOAT* z, int len);


__global__ 
void axpby_kernel(HPRLP_FLOAT a, const HPRLP_FLOAT *x, HPRLP_FLOAT b, const HPRLP_FLOAT *y, HPRLP_FLOAT *z, int len);


__global__ 
void vector_dot_product_kernel(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n, bool divide = false);

__global__
void vector_dot_product_zero_bitset_kernel(
    const HPRLP_FLOAT *x, const HPRLP_FLOAT *y, HPRLP_FLOAT *result,
    std::uint32_t *positive_zero_bits, int n);

__global__
void pack_positive_zero_bitset_count_kernel(
    const HPRLP_FLOAT *input, std::uint32_t *positive_zero_bits,
    unsigned long long *positive_zero_count, int n);

__global__
void reciprocal_vector_kernel(const HPRLP_FLOAT *input, HPRLP_FLOAT *output, int n);

#endif
