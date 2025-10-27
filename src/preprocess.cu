#include "preprocess.h"

void copy_lpinfo_to_device(const LP_info_cpu *lp_info_cpu, LP_info_gpu *lp_info_gpu) {
    int m = lp_info_cpu->m;
    int n = lp_info_cpu->n;

    lp_info_gpu->m = m;
    lp_info_gpu->n = n;
    lp_info_gpu->obj_constant = lp_info_cpu->obj_constant;

    // Copy A to GPU
    lp_info_gpu->A = new sparseMatrix;
    transfer_CSR_matrix(lp_info_cpu->A, lp_info_gpu->A);
    
    // Generate AT on CPU first, then transfer to GPU
    lp_info_gpu->AT = new sparseMatrix;
    sparseMatrix AT_host;
    CSR_transpose_host(*(lp_info_cpu->A), &AT_host);
    transfer_CSR_matrix(&AT_host, lp_info_gpu->AT);
    
    // Free the temporary host AT
    free(AT_host.value);
    free(AT_host.colIndex);
    free(AT_host.rowPtr);

    CUDA_CHECK(cudaMalloc(&lp_info_gpu->AL, m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->AL, lp_info_cpu->AL, m * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&lp_info_gpu->AU, m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->AU, lp_info_cpu->AU, m * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&lp_info_gpu->l, n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->l, lp_info_cpu->l, n * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&lp_info_gpu->u, n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->u, lp_info_cpu->u, n * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&lp_info_gpu->c, n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->c, lp_info_cpu->c, n * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
}


void prepare_spmv(HPRLP_workspace_gpu *workspace) {
    int n = workspace->n;
    int m = workspace->m;
    workspace->spmv_A = new CUSPARSE_spmv_A;
    workspace->spmv_AT = new CUSPARSE_spmv_AT;
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    workspace->spmv_A->cusparseHandle = cusparseHandle;
    workspace->spmv_AT->cusparseHandle = cusparseHandle;
    workspace->spmv_A->alpha = 1.0;
    workspace->spmv_A->beta = 0.0;
    workspace->spmv_AT->alpha = 1.0;
    workspace->spmv_AT->beta = 0.0;
    workspace->spmv_A->_operator = CUSPARSE_OPERATION_NON_TRANSPOSE;
    workspace->spmv_A->computeType = CUDA_R_64F;
    workspace->spmv_A->alg = CUSPARSE_SPMV_CSR_ALG2;
    workspace->spmv_AT->_operator = CUSPARSE_OPERATION_NON_TRANSPOSE;
    workspace->spmv_AT->computeType = CUDA_R_64F;
    workspace->spmv_AT->alg = CUSPARSE_SPMV_CSR_ALG2;
    cusparseCreateDnVec(&workspace->spmv_A->x_bar_cusparseDescr, n, workspace->x_bar, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_A->x_hat_cusparseDescr, n, workspace->x_hat, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_A->x_temp_cusparseDescr, n, workspace->x_temp, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_AT->y_bar_cusparseDescr, m, workspace->y_bar, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_AT->y_cusparseDescr, m, workspace->y, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_AT->ATy_cusparseDescr, n, workspace->ATy, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_A->Ax_cusparseDescr, m, workspace->Ax, CUDA_R_64F);

    // CSR Sparse Matrix Descriptor
    cusparseCreateCsr(&workspace->spmv_A->A_cusparseDescr, workspace->m, workspace->n, workspace->A->numElements,
                workspace->A->rowPtr, workspace->A->colIndex, workspace->A->value,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&workspace->spmv_AT->AT_cusparseDescr, workspace->n, workspace->m, workspace->AT->numElements,
                workspace->AT->rowPtr, workspace->AT->colIndex, workspace->AT->value,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    
    cusparseSpMV_bufferSize(cusparseHandle,workspace->spmv_A->_operator,
                            &workspace->spmv_A->alpha, workspace->spmv_A->A_cusparseDescr, workspace->spmv_A->x_bar_cusparseDescr,
                            &workspace->spmv_A->beta, workspace->spmv_A->Ax_cusparseDescr, workspace->spmv_A->computeType,
                            workspace->spmv_A->alg, &workspace->spmv_A->buffersize);

    cudaMalloc(&workspace->spmv_A->buffer, workspace->spmv_A->buffersize);

    cusparseSpMV_preprocess(cusparseHandle,workspace->spmv_A->_operator,
                            &workspace->spmv_A->alpha, workspace->spmv_A->A_cusparseDescr, workspace->spmv_A->x_bar_cusparseDescr,
                            &workspace->spmv_A->beta, workspace->spmv_A->Ax_cusparseDescr, workspace->spmv_A->computeType,
                            workspace->spmv_A->alg, workspace->spmv_A->buffer);

    cusparseSpMV(cusparseHandle,workspace->spmv_A->_operator,
                        &workspace->spmv_A->alpha, workspace->spmv_A->A_cusparseDescr, workspace->spmv_A->x_bar_cusparseDescr,
                        &workspace->spmv_A->beta, workspace->spmv_A->Ax_cusparseDescr, workspace->spmv_A->computeType,
                        workspace->spmv_A->alg, workspace->spmv_A->buffer);

    cusparseSpMV_bufferSize(cusparseHandle,workspace->spmv_AT->_operator,
                        &workspace->spmv_AT->alpha, workspace->spmv_AT->AT_cusparseDescr, workspace->spmv_AT->y_bar_cusparseDescr,
                        &workspace->spmv_AT->beta, workspace->spmv_AT->ATy_cusparseDescr, workspace->spmv_AT->computeType,
                        workspace->spmv_AT->alg, &workspace->spmv_AT->buffersize);

    cudaMalloc(&workspace->spmv_AT->buffer, workspace->spmv_AT->buffersize);

    cusparseSpMV_preprocess(cusparseHandle,workspace->spmv_AT->_operator,
                        &workspace->spmv_AT->alpha, workspace->spmv_AT->AT_cusparseDescr, workspace->spmv_AT->y_bar_cusparseDescr,
                        &workspace->spmv_AT->beta, workspace->spmv_AT->ATy_cusparseDescr, workspace->spmv_AT->computeType,
                        workspace->spmv_AT->alg, workspace->spmv_AT->buffer);
}


void allocate_memory(HPRLP_workspace_gpu *workspace, LP_info_gpu *lp_info_gpu) {
    // allocate memory for the workspace
    int m = workspace->m;
    int n = workspace->n;


    create_zero_vector_device(workspace->x, n);
    create_zero_vector_device(workspace->last_x, n);
    create_zero_vector_device(workspace->x_temp, n);
    create_zero_vector_device(workspace->x_hat, n);
    create_zero_vector_device(workspace->x_bar, n);
    create_zero_vector_device(workspace->y, m);
    create_zero_vector_device(workspace->last_y, m);
    create_zero_vector_device(workspace->y_temp, m);
    create_zero_vector_device(workspace->y_bar, m);
    create_zero_vector_device(workspace->y_hat, m);
    create_zero_vector_device(workspace->y_obj, m);
    create_zero_vector_device(workspace->z_bar, n);

    workspace->A = lp_info_gpu->A;
    workspace->AT = lp_info_gpu->AT;
    workspace->AL = lp_info_gpu->AL;
    workspace->AU = lp_info_gpu->AU;
    workspace->c = lp_info_gpu->c;
    workspace->l = lp_info_gpu->l;
    workspace->u = lp_info_gpu->u;

    create_zero_vector_device(workspace->Rd, n);
    create_zero_vector_device(workspace->Rp, m);
    create_zero_vector_device(workspace->ATy, n);
    create_zero_vector_device(workspace->Ax, m);

    workspace->check = false;

    cublasCreate(&workspace->cublasHandle);
    
    prepare_spmv(workspace);
}


void free_workspace(HPRLP_workspace_gpu *workspace) {
    /*
     * Free all GPU memory allocated in allocate_memory and prepare_spmv.
     * This prevents memory leaks. Note: When called from Python ctypes, the
     * process may still segfault during Python interpreter shutdown due to
     * CUDA/ctypes interaction, but this is harmless (happens after results returned).
     */
    if (!workspace) return;
    
    // Destroy cuBLAS handle FIRST (before freeing vectors it might reference)
    if (workspace->cublasHandle) {
        cublasDestroy(workspace->cublasHandle);
        workspace->cublasHandle = nullptr;
    }
    
    // Destroy CUSPARSE descriptors BEFORE freeing the underlying memory
    // Free CUSPARSE resources for AT matrix operations
    if (workspace->spmv_AT) {
        if (workspace->spmv_AT->y_bar_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_AT->y_bar_cusparseDescr);
        if (workspace->spmv_AT->y_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_AT->y_cusparseDescr);
        if (workspace->spmv_AT->ATy_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_AT->ATy_cusparseDescr);
        if (workspace->spmv_AT->AT_cusparseDescr) cusparseDestroySpMat(workspace->spmv_AT->AT_cusparseDescr);
        if (workspace->spmv_AT->buffer) cudaFree(workspace->spmv_AT->buffer);
        // Destroy shared cusparse handle (only once, shared between spmv_A and spmv_AT)
        if (workspace->spmv_AT->cusparseHandle) {
            cusparseDestroy(workspace->spmv_AT->cusparseHandle);
        }
        delete workspace->spmv_AT;
        workspace->spmv_AT = nullptr;
    }
    
    // Free CUSPARSE resources for A matrix operations
    if (workspace->spmv_A) {
        if (workspace->spmv_A->x_bar_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->x_bar_cusparseDescr);
        if (workspace->spmv_A->x_hat_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->x_hat_cusparseDescr);
        if (workspace->spmv_A->x_temp_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->x_temp_cusparseDescr);
        if (workspace->spmv_A->Ax_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->Ax_cusparseDescr);
        if (workspace->spmv_A->A_cusparseDescr) cusparseDestroySpMat(workspace->spmv_A->A_cusparseDescr);
        if (workspace->spmv_A->buffer) cudaFree(workspace->spmv_A->buffer);
        // Note: cusparseHandle already destroyed above with spmv_AT
        delete workspace->spmv_A;
        workspace->spmv_A = nullptr;
    }
    
    // NOW free device vectors (after descriptors are destroyed)
    if (workspace->x) cudaFree(workspace->x);
    if (workspace->last_x) cudaFree(workspace->last_x);
    if (workspace->x_temp) cudaFree(workspace->x_temp);
    if (workspace->x_hat) cudaFree(workspace->x_hat);
    if (workspace->x_bar) cudaFree(workspace->x_bar);
    if (workspace->y) cudaFree(workspace->y);
    if (workspace->last_y) cudaFree(workspace->last_y);
    if (workspace->y_temp) cudaFree(workspace->y_temp);
    if (workspace->y_bar) cudaFree(workspace->y_bar);
    if (workspace->y_hat) cudaFree(workspace->y_hat);
    if (workspace->y_obj) cudaFree(workspace->y_obj);
    if (workspace->z_bar) cudaFree(workspace->z_bar);
    if (workspace->Rd) cudaFree(workspace->Rd);
    if (workspace->Rp) cudaFree(workspace->Rp);
    if (workspace->ATy) cudaFree(workspace->ATy);
    if (workspace->Ax) cudaFree(workspace->Ax);
    
    // Note: A, AT, AL, AU, c, l, u are just pointers to lp_info_gpu data.
    // They should NOT be freed here - they will be freed in free_lp_info().
}


void free_lp_info(LP_info_gpu *lp_info) {
    /*
     * Free GPU memory allocated in copy_lpinfo_to_device.
     */
    if (!lp_info) return;
    
    // Free sparse matrices A and AT
    if (lp_info->A) {
        if (lp_info->A->rowPtr) cudaFree(lp_info->A->rowPtr);
        if (lp_info->A->colIndex) cudaFree(lp_info->A->colIndex);
        if (lp_info->A->value) cudaFree(lp_info->A->value);
        delete lp_info->A;
    }
    
    if (lp_info->AT) {
        if (lp_info->AT->rowPtr) cudaFree(lp_info->AT->rowPtr);
        if (lp_info->AT->colIndex) cudaFree(lp_info->AT->colIndex);
        if (lp_info->AT->value) cudaFree(lp_info->AT->value);
        delete lp_info->AT;
    }
    
    // Free constraint and variable bound vectors
    if (lp_info->AL) cudaFree(lp_info->AL);
    if (lp_info->AU) cudaFree(lp_info->AU);
    if (lp_info->l) cudaFree(lp_info->l);
    if (lp_info->u) cudaFree(lp_info->u);
    if (lp_info->c) cudaFree(lp_info->c);
}

void free_lp_info_cpu(LP_info_cpu *lp_info) {
    /*
     * Free CPU memory allocated in build_model_from_arrays() or build_model_from_mps().
     * This is used to clean up LP_info_cpu structures after solving.
     * Note: AT is no longer stored in LP_info_cpu; it's generated on-the-fly
     */
    if (!lp_info) return;
    
    // Free sparse matrix A
    if (lp_info->A) {
        if (lp_info->A->rowPtr) free(lp_info->A->rowPtr);
        if (lp_info->A->colIndex) free(lp_info->A->colIndex);
        if (lp_info->A->value) free(lp_info->A->value);
        free(lp_info->A);
        lp_info->A = nullptr;
    }
    
    // Free constraint and variable bound vectors
    if (lp_info->AL) {
        free(lp_info->AL);
        lp_info->AL = nullptr;
    }
    if (lp_info->AU) {
        free(lp_info->AU);
        lp_info->AU = nullptr;
    }
    if (lp_info->l) {
        free(lp_info->l);
        lp_info->l = nullptr;
    }
    if (lp_info->u) {
        free(lp_info->u);
        lp_info->u = nullptr;
    }
    if (lp_info->c) {
        free(lp_info->c);
        lp_info->c = nullptr;
    }
}