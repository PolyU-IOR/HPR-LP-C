#include "power_iteration.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <curand.h>

// Helper kernel: z = z + eps
// Corresponds to Julia: @. z = z + 1e-8
__global__ void add_epsilon_kernel(HPRLP_FLOAT *z, int n, HPRLP_FLOAT eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        z[idx] += eps;
    }
}

// Interface remains unchanged: (workspace, max_iter, tol)
HPRLP_FLOAT power_method_cusparse(HPRLP_workspace_gpu *workspace, int max_iter, HPRLP_FLOAT tol)
{
    // --- Internal Constants (Matched to Julia defaults) ---
    const int check_every = 10;
    const unsigned long long seed = 1;

    int m = workspace->m;
    int n = workspace->n;

    // q maps to workspace->y (length m) - used as temp buffer
    // z maps to workspace->Ax (length m) - used as current vector
    // ATq maps to workspace->ATy (length n)
    HPRLP_FLOAT *q = workspace->y;
    HPRLP_FLOAT *z = workspace->Ax;
    HPRLP_FLOAT *ATq = workspace->ATy;

    // 2. Backup for y
    cudaMemcpy(workspace->y_temp, workspace->y, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice);

    HPRLP_FLOAT lambda = 1.0;
    HPRLP_FLOAT error = INFINITY;
    bool converged = false;

    // Resource handles
    curandGenerator_t gen = NULL;

    // Initialization: GPU Random Normal
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    // Generate standard normal (mean=0, std=1)
    // Switch to curandGenerateNormal if HPRLP_FLOAT is float
    curandGenerateNormalDouble(gen, (double *)z, m, 0.0, 1.0);

    // z = z + 1e-8
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    add_epsilon_kernel<<<blocks, threads>>>(z, m, 1e-8);

    // --- Main Loop ---
    for (int i = 1; i <= max_iter; ++i)
    {

        // Normalize q = z / ||z||
        HPRLP_FLOAT z2 = inner_product(z, z, m, workspace->cublasHandle);
        HPRLP_FLOAT invn = 1.0 / sqrt(z2 + std::numeric_limits<HPRLP_FLOAT>::epsilon());

        // q = invn * z (Use copy + scal to strictly overwrite)
        cudaMemcpy(q, z, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice);
        HPRLP_FLOAT alpha_scale = invn;
        cublasDscal(workspace->cublasHandle, m, &alpha_scale, q, 1);

        // SpMV ATq = A^T * q
        cusparseSpMV(workspace->spmv_AT->cusparseHandle, workspace->spmv_AT->_operator,
                     &workspace->spmv_AT->alpha, workspace->spmv_AT->AT_cusparseDescr, workspace->spmv_AT->y_cusparseDescr,
                     &workspace->spmv_AT->beta, workspace->spmv_AT->ATy_cusparseDescr,
                     workspace->spmv_AT->computeType, workspace->spmv_AT->alg, workspace->spmv_AT->buffer);

        // SpMV z = A * ATq
        cusparseSpMV(workspace->spmv_A->cusparseHandle, workspace->spmv_A->_operator,
                     &workspace->spmv_A->alpha, workspace->spmv_A->A_cusparseDescr, workspace->spmv_AT->ATy_cusparseDescr,
                     &workspace->spmv_A->beta, workspace->spmv_A->Ax_cusparseDescr,
                     workspace->spmv_A->computeType, workspace->spmv_A->alg, workspace->spmv_A->buffer);

        // Check Convergence (Every 'check_every' iterations)
        if (i % check_every == 0)
        {
            // lambda = q' * z
            lambda = inner_product(q, z, m, workspace->cublasHandle);

            // q = z - lambda * q
            axpby(-lambda, q, 1.0, z, q, m);

            error = l2_norm(q, m, workspace->cublasHandle);

            if (error < tol)
            {
                converged = true;
                break;
            }
        }
    }

    // Restore workspace->y from local backup (Critical)
    cudaMemcpy(workspace->y, workspace->y_temp, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice);

    // Free local resources
    if (gen)
        curandDestroyGenerator(gen);

    // Logging matches original behavior
    if (!converged)
    {
        std::cout << "Power iteration did not converge within the specified tolerance.\n"
                  << "Max iter: " << max_iter << ", Error: " << std::scientific << std::setprecision(2) << error << "\n";
    }

    std::cout << std::defaultfloat;
    return lambda;
}