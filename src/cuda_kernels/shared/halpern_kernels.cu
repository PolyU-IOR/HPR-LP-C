#include "cuda_kernels/shared/halpern_kernels.cuh"

__global__ void advance_halpern_factors_kernel(int *halpern_inner, HPRLP_FLOAT *halpern_factors) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int next_inner = halpern_inner[0] + 1;
        halpern_inner[0] = next_inner;
        HPRLP_FLOAT fact1 = 1.0 / (next_inner + 2.0);
        halpern_factors[0] = fact1;
        halpern_factors[1] = 1.0 - fact1;
    }
}

__global__ void prepare_halpern_factor_batch_kernel(
    int *halpern_inner, HPRLP_FLOAT *halpern_factors,
    HPRLP_FLOAT *halpern_factor_batch, int batch_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int next_inner = halpern_inner[0];
        HPRLP_FLOAT final_fact1 = halpern_factors[0];
        HPRLP_FLOAT final_fact2 = halpern_factors[1];
        for (int pair = 0; pair < batch_size; ++pair) {
            halpern_factor_batch[2 * pair] = final_fact1;
            halpern_factor_batch[2 * pair + 1] = final_fact2;
            ++next_inner;
            final_fact1 = 1.0 / (next_inner + 2.0);
            final_fact2 = 1.0 - final_fact1;
        }
        halpern_inner[0] = next_inner;
        halpern_factors[0] = final_fact1;
        halpern_factors[1] = final_fact2;
    }
}
