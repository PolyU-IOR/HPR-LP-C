#include "power_iteration.h"
#include <iostream>
#include <iomanip>
<<<<<<< HEAD
#include <cmath>
#include <limits>
#include <curand.h>

// Helper kernel: z = z + eps
// Corresponds to Julia: @. z = z + 1e-8
__global__ void add_epsilon_kernel(HPRLP_FLOAT* z, int n, HPRLP_FLOAT eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] += eps;
    }
}

// Interface remains unchanged: (workspace, max_iter, tol)
HPRLP_FLOAT power_method_cusparse(HPRLP_workspace_gpu *workspace, int max_iter, HPRLP_FLOAT tol) {
    // --- Internal Constants (Matched to Julia defaults) ---
    const int check_every = 10;
    const unsigned long long seed = 1; 

    int m = workspace->m;
    int n = workspace->n;

    // 1. Variable Mapping
    // q maps to workspace->y (length m) - used as temp buffer
    // z maps to workspace->Ax (length m) - used as current vector
    // ATq maps to workspace->ATy (length n)
    HPRLP_FLOAT *q = workspace->y;
    HPRLP_FLOAT *z = workspace->Ax;
    HPRLP_FLOAT *ATq = workspace->ATy;
    
    // 2. Local Backup for y (Since 'dy' is missing in struct)
    HPRLP_FLOAT *y_backup = NULL;
    cudaMalloc((void**)&y_backup, m * sizeof(HPRLP_FLOAT));
    cudaMemcpy(y_backup, workspace->y, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice);

    HPRLP_FLOAT lambda = 1.0;
    HPRLP_FLOAT error = INFINITY;
    bool converged = false;

    // Resource handles
    cusparseDnVecDescr_t q_desc = NULL, ATq_desc = NULL, z_desc = NULL;
    curandGenerator_t gen = NULL;

    do {
        // Create Descriptors
        // NOTE: If HPRLP_FLOAT is float, replace CUDA_R_64F with CUDA_R_32F
        cusparseCreateDnVec(&q_desc, m, q, CUDA_R_64F);
        cusparseCreateDnVec(&ATq_desc, n, ATq, CUDA_R_64F);
        cusparseCreateDnVec(&z_desc, m, z, CUDA_R_64F);

        // 3. Initialization: GPU Random Normal (Julia: CUDA.randn!(z))
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        
        // Generate standard normal (mean=0, std=1)
        // Switch to curandGenerateNormal if HPRLP_FLOAT is float
        curandGenerateNormalDouble(gen, (double*)z, m, 0.0, 1.0);

        // z = z + 1e-8
        int threads = 256;
        int blocks = (m + threads - 1) / threads;
        add_epsilon_kernel<<<blocks, threads>>>(z, m, 1e-8);

        // --- Main Loop ---
        for (int i = 1; i <= max_iter; ++i) {
            
            // Step 1: Normalize q = z / ||z||
            HPRLP_FLOAT z2 = inner_product(z, z, m, workspace->cublasHandle);
            HPRLP_FLOAT invn = 1.0 / sqrt(z2 + std::numeric_limits<HPRLP_FLOAT>::epsilon());
            
            // q = invn * z (Use copy + scal to strictly overwrite)
            cudaMemcpy(q, z, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice);
            HPRLP_FLOAT alpha_scale = invn;
            cublasDscal(workspace->cublasHandle, m, &alpha_scale, q, 1); 

            // Step 2: SpMV ATq = A^T * q
            cusparseSpMV(workspace->spmv_AT->cusparseHandle, workspace->spmv_AT->_operator, 
                         &workspace->spmv_AT->alpha, workspace->spmv_AT->AT_cusparseDescr, q_desc, 
                         &workspace->spmv_AT->beta, ATq_desc, 
                         workspace->spmv_AT->computeType, workspace->spmv_AT->alg, workspace->spmv_AT->buffer);

            // Step 3: SpMV z = A * ATq
            cusparseSpMV(workspace->spmv_A->cusparseHandle, workspace->spmv_A->_operator, 
                         &workspace->spmv_A->alpha, workspace->spmv_A->A_cusparseDescr, ATq_desc, 
                         &workspace->spmv_A->beta, z_desc, 
                         workspace->spmv_A->computeType, workspace->spmv_A->alg, workspace->spmv_A->buffer);

            // Step 4: Check Convergence (Every 'check_every' iterations)
            if (i % check_every == 0) {
                // lambda = q' * z
                lambda = inner_product(q, z, m, workspace->cublasHandle);
                
                // q = z - lambda * q
                // Logic: q = (-lambda)*q + (1.0)*z
                axpby(-lambda, q, 1.0, z, q, m);

                error = l2_norm(q, m, workspace->cublasHandle);

                if (error < tol) {
                    converged = true;
                    break;
                }
            }
        }

    } while (0);

    // 4. Restore workspace->y from local backup (Critical)
    cudaMemcpy(workspace->y, y_backup, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice);
    
    // Free local resources
    cudaFree(y_backup);
    if (q_desc) cusparseDestroyDnVec(q_desc);
    if (ATq_desc) cusparseDestroyDnVec(ATq_desc);
    if (z_desc) cusparseDestroyDnVec(z_desc);
    if (gen) curandDestroyGenerator(gen);

    // Logging matches original behavior
    if (!converged) {
        std::cout << "Power iteration did not converge within the specified tolerance.\n"
                  << "Max iter: " << max_iter << ", Error: " << std::scientific << std::setprecision(2) << error << "\n";
    } else {
        std::cout << "Estimated largest eigenvalue of AAT = " 
                  << std::scientific << std::setprecision(2) << lambda << "\n";
    }
    
    std::cout << std::defaultfloat;
    return lambda;
=======
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

    // Initialize random vector on host
    std::mt19937_64 rng(4); // Mersenne Twister with seed 4
    std::uniform_real_distribution<HPRLP_FLOAT> dist(0.0, 1.0);
    HPRLP_FLOAT *z_host = new HPRLP_FLOAT[m];
    for (int i = 0; i < m; ++i) {
        z_host[i] = dist(rng) + 1e-8;
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
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
}