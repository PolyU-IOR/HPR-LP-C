#include "power_iteration.h"
#include <iostream>
#include <iomanip>
HPRLP_FLOAT power_method_cusparse(HPRLP_workspace_gpu *workspace, int max_iter, HPRLP_FLOAT tol) {
    int m = workspace->m;
    int n = workspace->n;

    srand(12345);
    HPRLP_FLOAT *q;
    cudaMalloc(&q, m * sizeof(HPRLP_FLOAT));
    HPRLP_FLOAT *ATq;
    cudaMalloc(&ATq, n * sizeof(HPRLP_FLOAT));
    HPRLP_FLOAT *z;
    cudaMalloc(&z, m * sizeof(HPRLP_FLOAT));

    // Initialize random vector on host using std::normal_distribution
    std::mt19937_64 rng(12345); // Mersenne Twister with seed 12345
    std::normal_distribution<HPRLP_FLOAT> dist(0.0, 1.0); // Mean 0, stddev 1
    HPRLP_FLOAT *z_host = new HPRLP_FLOAT[m];
    for (int i = 0; i < m; ++i) {
        z_host[i] = dist(rng) + 1e-8; // Normal distribution + small offset
    }
    cudaMemcpy(z, z_host, m * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice);
    
// normalize b
    HPRLP_FLOAT norm;
    int i = 0;
    HPRLP_FLOAT lambda_new;

/// CUSPARSE APIs
    cusparseDnVecDescr_t q_desc, ATq_desc, z_desc;
    cusparseCreateDnVec(&q_desc, m, q, CUDA_R_64F);
    cusparseCreateDnVec(&ATq_desc, n, ATq, CUDA_R_64F);
    cusparseCreateDnVec(&z_desc, m, z, CUDA_R_64F);

    HPRLP_FLOAT lambda;
    HPRLP_FLOAT alpha = 1.0;
    HPRLP_FLOAT beta = 0.0;
    for (i = 0; i < max_iter; ++i) {

        HPRLP_FLOAT scale = 1.0/l2_norm(z, m, workspace->cublasHandle);
        ax(scale, z, q, m, workspace->cublasHandle);

        cusparseSpMV(workspace->spmv_AT->cusparseHandle, workspace->spmv_AT->_operator,
                    &workspace->spmv_AT->alpha, workspace->spmv_AT->AT_cusparseDescr, q_desc, 
                    &workspace->spmv_AT->beta, ATq_desc, workspace->spmv_AT->computeType,
                    workspace->spmv_AT->alg, workspace->spmv_AT->buffer);

        cusparseSpMV(workspace->spmv_A->cusparseHandle, workspace->spmv_A->_operator,
                    &workspace->spmv_A->alpha, workspace->spmv_A->A_cusparseDescr, ATq_desc, 
                    &workspace->spmv_A->beta, z_desc, workspace->spmv_A->computeType,
                    workspace->spmv_A->alg, workspace->spmv_A->buffer);

        lambda = inner_product(q, z, m, workspace->cublasHandle);

        axpby(-lambda, q, 1.0, z, q, m);
        if (l2_norm(q, m, workspace->cublasHandle) < tol) {
            break;
        }
    }

    /// CUSPARSE APIs
    cusparseDestroyDnVec(q_desc);
    cusparseDestroyDnVec(ATq_desc);
    cusparseDestroyDnVec(z_desc);


    cudaFree(q);
    cudaFree(ATq_desc);
    cudaFree(z);
    delete[] z_host;

    if (i == max_iter) {
        std::cout << "Power method did not converge in " << max_iter << " iterations for specified tolerance " 
                  << std::scientific << std::setprecision(2) << tol << " \n" << std::defaultfloat;
        return lambda;
    } else {
        std::cout << "The estimated largest eigenvalue of AAT = " 
                  << std::scientific << std::setprecision(2) << lambda << "\n" << std::defaultfloat;
        return lambda;
    }
}