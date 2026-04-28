#include "scaling.h"

namespace {

__global__ void curtis_reid_log_update_kernel(int m,
                                              const int *rowPtr,
                                              const int *colIndex,
                                              const HPRLP_FLOAT *value,
                                              const HPRLP_FLOAT *other_log_scale,
                                              HPRLP_FLOAT *result) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }

    const int start = rowPtr[row];
    const int end = rowPtr[row + 1];
    const int count = end - start;
    if (count <= 0) {
        result[row] = 0.0;
        return;
    }

    HPRLP_FLOAT sum = 0.0;
    for (int idx = start; idx < end; ++idx) {
        const HPRLP_FLOAT abs_value = fmax(fabs(value[idx]), 1e-300);
        sum += -log(abs_value) - other_log_scale[colIndex[idx]];
    }

    result[row] = sum / static_cast<HPRLP_FLOAT>(count);
}

__global__ void exp_clamp_kernel(HPRLP_FLOAT *log_scale, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        log_scale[idx] = fmin(fmax(exp(log_scale[idx]), 1e-30), 1e30);
    }
}

void apply_curtis_reid_scaling(LP_info_gpu *lp_info_gpu,
                               HPRLP_FLOAT *rowNormA,
                               HPRLP_FLOAT *colNormA,
                               HPRLP_FLOAT *tempNorm1,
                               HPRLP_FLOAT *tempNorm2) {
    set_vector_value_device(tempNorm1, lp_info_gpu->m, 0.0);
    set_vector_value_device(tempNorm2, lp_info_gpu->n, 0.0);

    for (int i = 0; i < 20; ++i) {
        curtis_reid_log_update_kernel<<<numBlocks(lp_info_gpu->m), numThreads>>>(
            lp_info_gpu->A->row,
            lp_info_gpu->A->rowPtr,
            lp_info_gpu->A->colIndex,
            lp_info_gpu->A->value,
            tempNorm2,
            tempNorm1);

        curtis_reid_log_update_kernel<<<numBlocks(lp_info_gpu->n), numThreads>>>(
            lp_info_gpu->AT->row,
            lp_info_gpu->AT->rowPtr,
            lp_info_gpu->AT->colIndex,
            lp_info_gpu->AT->value,
            tempNorm1,
            tempNorm2);
    }

    exp_clamp_kernel<<<numBlocks(lp_info_gpu->m), numThreads>>>(tempNorm1, lp_info_gpu->m);
    exp_clamp_kernel<<<numBlocks(lp_info_gpu->n), numThreads>>>(tempNorm2, lp_info_gpu->n);

    vector_dot_product(rowNormA, tempNorm1, rowNormA, lp_info_gpu->m, true);
    vector_dot_product(colNormA, tempNorm2, colNormA, lp_info_gpu->n, true);

    mul_CSR_A_row(lp_info_gpu->A, tempNorm1, false);
    mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, false);

    mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, false);
    mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, false);

    vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, lp_info_gpu->m, false);
    vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, lp_info_gpu->m, false);
    vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, lp_info_gpu->n, false);
    vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, lp_info_gpu->n, true);
    vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, lp_info_gpu->n, true);
}

} // namespace


void scaling(LP_info_gpu *lp_info_gpu, Scaling_info* scaling_info, const HPRLP_parameters *param, cublasHandle_t cublasHandle) {
    int m = lp_info_gpu->m;
    int n = lp_info_gpu->n;

    create_zero_vector_device(scaling_info->row_norm, m);
    create_zero_vector_device(scaling_info->col_norm, n);
    create_zero_vector_device(scaling_info->l_org, n);
    create_zero_vector_device(scaling_info->u_org, n);

    HPRLP_FLOAT *rowNormA = scaling_info->row_norm;
    HPRLP_FLOAT *colNormA = scaling_info->col_norm;
    HPRLP_FLOAT *tempNorm1;
    HPRLP_FLOAT *tempNorm2;

    create_zero_vector_device(tempNorm1, m);
    create_zero_vector_device(tempNorm2, n);

    set_vector_value_device(rowNormA, m, 1.0);
    set_vector_value_device(colNormA, n, 1.0);

    vMemcpy_device(scaling_info->l_org, lp_info_gpu->l, n);
    vMemcpy_device(scaling_info->u_org, lp_info_gpu->u, n);

    HPRLP_FLOAT *b;
    CUDA_CHECK(cudaMalloc(&b, m * sizeof(HPRLP_FLOAT)));

    gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);
    
    scaling_info->norm_b_org = 1 + l2_norm(b, m, cublasHandle);
    scaling_info->norm_c_org = 1 + l2_norm(lp_info_gpu->c, n, cublasHandle);

    if (param->use_CR_scaling) {
        apply_curtis_reid_scaling(lp_info_gpu, rowNormA, colNormA, tempNorm1, tempNorm2);
    }

    if (param->use_GM_scaling) {
        for (int i = 0; i < 20; ++i) {
            CSR_A_row_geometric_mean(lp_info_gpu->A, tempNorm1);
            vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);

            vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
            vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

            CSR_A_row_geometric_mean(lp_info_gpu->AT, tempNorm2);
            vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

            mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
            mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

            mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
            mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

            vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

            vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
            vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
        }
    }

    if (param->use_Ruiz_scaling){

        for (int i = 0; i < 10; ++i) {
            // find the max value of each row of A
            CSR_A_row_norm(lp_info_gpu->A, tempNorm1, 99);
            vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);
            
            // AL = AL / tempNorm   ;   AU = AU / tempNorm
            vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
            vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

            // find the max value of each column of A which is equivalent to each row of AT
            CSR_A_row_norm(lp_info_gpu->AT, tempNorm2, 99);
            vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

            // A = A / tempNorm, also for AT
            mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
            mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

            // A = A / tempNorm, also for AT
            mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
            mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

            // c = c / tempNorm
            vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

            // l = l * tempNorm, u = u * tempNorm
            vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
            vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
        }
    }
    
    // Pock and Chambolle scaling
    // compute the sum of each row of A, in tempNorm
    if(param->use_Pock_Chambolle_scaling){
        CSR_A_row_norm(lp_info_gpu->A, tempNorm1, 1);
        vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);

        // AL = AL / tempNorm   ;   AU = AU / tempNorm
        vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
        vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

        // compute the sum of each column of A
        CSR_A_row_norm(lp_info_gpu->AT, tempNorm2, 1);
        vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

        // A = A / tempNorm, also for AT
        mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
        mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

        // A = A / tempNorm, also for AT
        mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
        mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

        // c = c / tempNorm
        vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

        // l = l * tempNorm, u = u * tempNorm
        vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
        vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
    }

    if (param->use_bc_scaling){

        gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

        scaling_info->b_scale = 1 + l2_norm(b, m, cublasHandle);
        scaling_info->c_scale = 1 + l2_norm(lp_info_gpu->c, n, cublasHandle);

        
        // b = b / b_scale, c = c / c_scale
        const HPRLP_FLOAT bs = 1.0 / scaling_info->b_scale;
        const HPRLP_FLOAT cs = 1.0 / scaling_info->c_scale;

        CUBLAS_CHECK(cublasDscal(cublasHandle, m, &bs, lp_info_gpu->AU, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, m, &bs, lp_info_gpu->AL, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, n, &bs, lp_info_gpu->l, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, n, &bs, lp_info_gpu->u, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, n, &cs, lp_info_gpu->c, 1));
    }
    else{
        scaling_info->b_scale = 1.0;
        scaling_info->c_scale = 1.0;
    }

    gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

    scaling_info->norm_b = l2_norm(b, m, cublasHandle);
    scaling_info->norm_c = l2_norm(lp_info_gpu->c, n, cublasHandle);

    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(tempNorm1));
    CUDA_CHECK(cudaFree(tempNorm2));
}


void free_scaling_info(Scaling_info *scaling_info) {
    /*
     * Free device memory allocated in scaling function.
     */
    if (!scaling_info) return;
    
    if (scaling_info->row_norm) cudaFree(scaling_info->row_norm);
    if (scaling_info->col_norm) cudaFree(scaling_info->col_norm);
    if (scaling_info->l_org) cudaFree(scaling_info->l_org);
    if (scaling_info->u_org) cudaFree(scaling_info->u_org);
}