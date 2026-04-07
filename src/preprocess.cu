#include "preprocess.h"

namespace {

constexpr HPRLP_FLOAT kInfiniteBoundThreshold = 1e90;
constexpr int kShortRowMaxNnz = 16;

void copy_int_vector_to_device(const std::vector<int> &host_values, int **device_values) {
    if (host_values.empty()) {
        *device_values = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(device_values, static_cast<int>(host_values.size()) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*device_values, host_values.data(), static_cast<int>(host_values.size()) * sizeof(int), cudaMemcpyHostToDevice));
}

void build_row_buckets(const sparseMatrix *matrix, int **rows_short, int *num_rows_short, int **rows_medium, int *num_rows_medium) {
    std::vector<int> row_ptr(matrix->row + 1);
    CUDA_CHECK(cudaMemcpy(row_ptr.data(), matrix->rowPtr, (matrix->row + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> short_rows;
    std::vector<int> medium_rows;
    short_rows.reserve(matrix->row);
    medium_rows.reserve(matrix->row / 2);

    for (int row = 0; row < matrix->row; ++row) {
        int nnz = row_ptr[row + 1] - row_ptr[row];
        if (nnz <= kShortRowMaxNnz) {
            short_rows.push_back(row);
        } else {
            medium_rows.push_back(row);
        }
    }

    *num_rows_short = static_cast<int>(short_rows.size());
    *num_rows_medium = static_cast<int>(medium_rows.size());
    copy_int_vector_to_device(short_rows, rows_short);
    copy_int_vector_to_device(medium_rows, rows_medium);
}

void build_bound_types(const HPRLP_FLOAT *lower_dev, const HPRLP_FLOAT *upper_dev, int len, uint8_t **bound_type_dev) {
    std::vector<HPRLP_FLOAT> lower(len);
    std::vector<HPRLP_FLOAT> upper(len);
    std::vector<uint8_t> bound_type(len);
    CUDA_CHECK(cudaMemcpy(lower.data(), lower_dev, len * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper.data(), upper_dev, len * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));

    for (int i = 0; i < len; ++i) {
        if (lower[i] <= -kInfiniteBoundThreshold && upper[i] >= kInfiniteBoundThreshold) {
            bound_type[i] = 0;
        } else if (upper[i] >= kInfiniteBoundThreshold) {
            bound_type[i] = 1;
        } else if (lower[i] <= -kInfiniteBoundThreshold) {
            bound_type[i] = 2;
        } else {
            bound_type[i] = 3;
        }
    }

    CUDA_CHECK(cudaMalloc(bound_type_dev, len * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(*bound_type_dev, bound_type.data(), len * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

}

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


void allocate_memory(HPRLP_workspace_gpu *workspace, LP_info_gpu *lp_info_gpu, const HPRLP_parameters *param) {
    // allocate memory for the workspace
    int m = workspace->m;
    int n = workspace->n;
    cudaStreamCreate(&workspace->stream);

    CUDA_CHECK(cudaMalloc((void**)&workspace->Halpern_params, 4 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemset(workspace->Halpern_params, 0, 4 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc((void**)&workspace->halpern_inner, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&workspace->halpern_factors, 2 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMallocHost(&workspace->iter_params_host, 4 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMallocHost(&workspace->halpern_inner_host, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&workspace->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT)));
    workspace->halpern_inner_host[0] = 0;
    workspace->halpern_factors_host[0] = 0.5;
    workspace->halpern_factors_host[1] = 0.5;
    memset(workspace->iter_params_host, 0, 4 * sizeof(HPRLP_FLOAT));
    CUDA_CHECK(cudaMemcpyAsync(workspace->halpern_inner, workspace->halpern_inner_host, sizeof(int),
                               cudaMemcpyHostToDevice, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace->halpern_factors, workspace->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, workspace->stream));

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

    workspace->graph = nullptr;  // Initialize CUDA Graph pointer
    workspace->graph_exec = nullptr;
    workspace->graph_initialized = false;
    workspace->use_custom_fused_x = false;
    workspace->use_custom_fused_y = false;

    cublasCreate(&workspace->cublasHandle);
    cublasSetStream(workspace->cublasHandle, workspace->stream);

    // device-mode CUBLAS handle for queued (async) reductions
    cublasCreate(&workspace->cublasHandle_device);
    cublasSetStream(workspace->cublasHandle_device, workspace->stream);
    cublasSetPointerMode(workspace->cublasHandle_device, CUBLAS_POINTER_MODE_DEVICE);

    // 10-slot reduction scalar staging buffers
    CUDA_CHECK(cudaMalloc(&workspace->reduction_scalars, 10 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemset(workspace->reduction_scalars, 0, 10 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMallocHost(&workspace->reduction_scalars_host, 10 * sizeof(HPRLP_FLOAT)));
    memset(workspace->reduction_scalars_host, 0, 10 * sizeof(HPRLP_FLOAT));

    prepare_spmv(workspace);
    cusparseSetStream(workspace->spmv_A->cusparseHandle, workspace->stream);
    if (workspace->spmv_AT->cusparseHandle != workspace->spmv_A->cusparseHandle) {
         cusparseSetStream(workspace->spmv_AT->cusparseHandle, workspace->stream);
    }

    if (!param->CUSPARSE_spmv) {
        build_bound_types(workspace->l, workspace->u, n, &workspace->x_bound_type);
        build_bound_types(workspace->AL, workspace->AU, m, &workspace->y_bound_type);
        build_row_buckets(workspace->A, &workspace->A_rows_short, &workspace->num_A_rows_short,
                          &workspace->A_rows_medium, &workspace->num_A_rows_medium);
        build_row_buckets(workspace->AT, &workspace->AT_rows_short, &workspace->num_AT_rows_short,
                          &workspace->AT_rows_medium, &workspace->num_AT_rows_medium);
    }
}


void free_workspace(HPRLP_workspace_gpu *workspace) {
    /*
     * Free all GPU memory allocated in allocate_memory and prepare_spmv.
     * This prevents memory leaks. Note: When called from Python ctypes, the
     * process may still segfault during Python interpreter shutdown due to
     * CUDA/ctypes interaction, but this is harmless (happens after results returned).
     */
    if (!workspace) return;
    
    // Destroy cuBLAS handles FIRST (before freeing vectors they might reference)
    if (workspace->cublasHandle_device) {
        cublasDestroy(workspace->cublasHandle_device);
        workspace->cublasHandle_device = nullptr;
    }
    if (workspace->cublasHandle) {
        cublasDestroy(workspace->cublasHandle);
        workspace->cublasHandle = nullptr;
    }

    // Free reduction scalar staging buffers
    if (workspace->reduction_scalars) {
        cudaFree(workspace->reduction_scalars);
        workspace->reduction_scalars = nullptr;
    }
    if (workspace->reduction_scalars_host) {
        cudaFreeHost(workspace->reduction_scalars_host);
        workspace->reduction_scalars_host = nullptr;
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

    // Destroy CUDA Graph resources
    if (workspace->graph_exec) {
        cudaGraphExecDestroy(workspace->graph_exec);
        workspace->graph_exec = nullptr;
    }
    if (workspace->graph) {
        cudaGraphDestroy(workspace->graph);
        workspace->graph = nullptr;
    }

    // Free Halpern runtime parameter buffers
    if (workspace->Halpern_params) {
        cudaFree(workspace->Halpern_params);
        workspace->Halpern_params = nullptr;
    }
    if (workspace->halpern_inner) {
        cudaFree(workspace->halpern_inner);
        workspace->halpern_inner = nullptr;
    }
    if (workspace->halpern_factors) {
        cudaFree(workspace->halpern_factors);
        workspace->halpern_factors = nullptr;
    }
    if (workspace->iter_params_host) {
        cudaFreeHost(workspace->iter_params_host);
        workspace->iter_params_host = nullptr;
    }
    if (workspace->halpern_inner_host) {
        cudaFreeHost(workspace->halpern_inner_host);
        workspace->halpern_inner_host = nullptr;
    }
    if (workspace->halpern_factors_host) {
        cudaFreeHost(workspace->halpern_factors_host);
        workspace->halpern_factors_host = nullptr;
    }

    if (workspace->x_bound_type) cudaFree(workspace->x_bound_type);
    if (workspace->y_bound_type) cudaFree(workspace->y_bound_type);
    if (workspace->A_rows_short) cudaFree(workspace->A_rows_short);
    if (workspace->A_rows_medium) cudaFree(workspace->A_rows_medium);
    if (workspace->AT_rows_short) cudaFree(workspace->AT_rows_short);
    if (workspace->AT_rows_medium) cudaFree(workspace->AT_rows_medium);
    
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