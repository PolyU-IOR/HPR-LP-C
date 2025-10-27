#include "scaling.h"


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