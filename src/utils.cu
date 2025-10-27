#include "utils.h"
#include "structs.h"

void set_vector_value_device(HPRLP_FLOAT *x, int n, HPRLP_FLOAT value){
    set_vector_value_device_kernel<<<numBlocks(n), numThreads>>>(x, n, value);
}

void create_zero_vector_device(HPRLP_FLOAT* &x, int n) {
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(HPRLP_FLOAT)));
    set_vector_value_device(x, n, 0.0);
}

void gen_conceptual_b(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n) {
    conceptual_b_kernel<<<numBlocks(n), numThreads>>>(x, y, result, n);
}

void vector_dot_product(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n, bool divide){
    vector_dot_product_kernel<<<numBlocks(n), numThreads>>>(x, y, result, n, divide);
}


void CSR_A_row_norm(const sparseMatrix *A, HPRLP_FLOAT *result, int norm) {
    CSR_A_row_norm_kernel<<<numBlocks(A->row), numThreads>>>(A->row, A->rowPtr, A->colIndex, A->value, result, norm);
}

void mul_CSR_A_row(sparseMatrix *A, HPRLP_FLOAT *x, bool divide) {
    mul_CSR_A_row_kernel<<<numBlocks(A->row), numThreads>>>(A->row, A->rowPtr, A->colIndex, A->value, x, divide);
}


void mul_CSR_AT_row(sparseMatrix *A, HPRLP_FLOAT *x, bool divide) {
    mul_CSR_AT_row_kernel<<<numBlocks(A->row), numThreads>>>(A->row, A->rowPtr, A->colIndex, A->value, x, divide);
}

void transfer_CSR_matrix(const sparseMatrix *A, sparseMatrix* d_A) {
    d_A->row = A->row;
    d_A->col = A->col;
    d_A->numElements = A->numElements;

    CUDA_CHECK(cudaMalloc(&d_A->value, A->numElements * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&d_A->colIndex, A->numElements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A->rowPtr, (A->row + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A->value, A->value, A->numElements * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A->colIndex, A->colIndex, A->numElements * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A->rowPtr, A->rowPtr, (A->row+1) * sizeof(int), cudaMemcpyHostToDevice));
}

void vMemcpy_device(HPRLP_FLOAT *dst, HPRLP_FLOAT *src, int n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice));
}

void ax(HPRLP_FLOAT a,  HPRLP_FLOAT *x, HPRLP_FLOAT *z, int n, cublasHandle_t cublasHandle) {
    // perform z = ax
    vMemcpy_device(z, x, n);                        // z = x
    CUBLAS_CHECK(cublasDscal(cublasHandle, n, &a, z, 1));  // z = az
}

void axpy(HPRLP_FLOAT a, const HPRLP_FLOAT* x, const HPRLP_FLOAT* y, HPRLP_FLOAT* z, int len){
    axpy_kernel<<<numBlocks(len), numThreads>>>(a, x, y, z, len);
}

HPRLP_FLOAT l2_norm(const HPRLP_FLOAT *x, int n, cublasHandle_t cublasHandle) {
    HPRLP_FLOAT result = 0.0;
    CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, x, 1, &result));
    return result;
}


HPRLP_FLOAT inner_product(const HPRLP_FLOAT *x, const HPRLP_FLOAT *y, int n, cublasHandle_t cublasHandle) {
    HPRLP_FLOAT result = 0.0;
    CUBLAS_CHECK(cublasDdot(cublasHandle, n, x, 1, y, 1, &result));
    return result;
}


void axpby(HPRLP_FLOAT a, const HPRLP_FLOAT *x, HPRLP_FLOAT b, const HPRLP_FLOAT *y, HPRLP_FLOAT *z, int len) {
    axpby_kernel<<<numBlocks(len), numThreads>>>(a, x, b, y, z, len);
}


int step(int iter) {
    return std::max(10, static_cast<int>(pow(10, floor(log10(iter))) / 10));
}


void record_to_file(HPRLP_FLOAT* deviceptr, int len, const std::string filename){
    /*
    Copy Device Data To Host and Record To Files for Comparing/Debugging etc.
    */
    std::vector<HPRLP_FLOAT> vec(len);
    CUDA_CHECK(cudaMemcpy(vec.data(), deviceptr, len * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    std::ofstream out(filename);
    for(auto element : vec) {
        out << std::setprecision(32) << element << std::endl;
    }
    out.close();
}

void showDevVec(HPRLP_FLOAT *devPtr, int len) {
    HPRLP_FLOAT *host = new HPRLP_FLOAT[len];
    CUDA_CHECK(cudaMemcpy(host, devPtr, sizeof(HPRLP_FLOAT) * len, cudaMemcpyDeviceToHost));
    for(int i = 0 ; i < len ; ++i) {
        std::cout << host[i] << " ";
    }
    std::cout << std::endl;
    delete[] host;
}


// inline
std::chrono::steady_clock::time_point time_now()
{
    return std::chrono::steady_clock::now();
}

// inline
HPRLP_FLOAT time_since(std::chrono::steady_clock::time_point clock_begin) {
    std::chrono::steady_clock::time_point clock_end = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration time_span = clock_end - clock_begin;
    return HPRLP_FLOAT(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
}


void collect_solution(HPRLP_workspace_gpu *workspace, Scaling_info *scaling_info, HPRLP_results *output) {
    /*
    Collect solution vectors (x and y) from GPU workspace to output structure.
    This copies the primal solution x and dual solution y from device to host.
    Applies inverse scaling: x = b_scale * (x_bar / col_norm), y = c_scale * (y_bar / row_norm)
    
    Args:
        workspace: GPU workspace containing the solution vectors
        scaling_info: Scaling information for inverse transformation
        output: HPRLP_results structure where the solution will be stored
    */
    int n = workspace->n;  // Number of primal variables
    int m = workspace->m;  // Number of dual variables (constraints)
    
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // Allocate memory for the solution vectors on host
    output->x = new HPRLP_FLOAT[n];
    output->y = new HPRLP_FLOAT[m];
    
    // Allocate temporary device memory for scaled solutions
    HPRLP_FLOAT *x_temp, *y_temp;
    cudaMalloc(&x_temp, n * sizeof(HPRLP_FLOAT));
    cudaMalloc(&y_temp, m * sizeof(HPRLP_FLOAT));
    
    // Apply inverse scaling on GPU:
    // Step 1: x_temp = x_bar / col_norm
    vector_dot_product(workspace->x_bar, scaling_info->col_norm, x_temp, n, true);  // divide=true
    
    // Step 2: x_temp = b_scale * x_temp (in-place)
    cublasDscal(workspace->cublasHandle, n, &(scaling_info->b_scale), x_temp, 1);
    
    // Step 3: y_temp = y_bar / row_norm
    vector_dot_product(workspace->y_bar, scaling_info->row_norm, y_temp, m, true);  // divide=true
    
    // Step 4: y_temp = c_scale * y_temp (in-place)
    cublasDscal(workspace->cublasHandle, m, &(scaling_info->c_scale), y_temp, 1);
    
    // Copy scaled solutions from device to host
    cudaMemcpy(output->x, x_temp, n * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost);
    cudaMemcpy(output->y, y_temp, m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost);
    
    // Free temporary device memory
    cudaFree(x_temp);
    cudaFree(y_temp);
}

/* CSR Matrix transpose (host utility) */
void CSR_transpose_host(sparseMatrix A, sparseMatrix *AT) {
    AT->row = A.col;
    AT->col = A.row;
    AT->numElements = A.numElements;
    AT->value = (HPRLP_FLOAT*)malloc(AT->numElements * sizeof(HPRLP_FLOAT));
    AT->colIndex = (int*)malloc(AT->numElements * sizeof(int));
    AT->rowPtr = (int*)malloc((AT->row + 2) * sizeof(int));

    for (int i = 0; i < AT->row + 2; i++) {
        AT->rowPtr[i] = 0;
    }

    for (int i = 0; i < A.numElements; i++) {
        AT->rowPtr[A.colIndex[i] + 2]++;
    }

    AT->rowPtr[0] = 0;
    for (int i = 2; i < AT->row + 2; i++) {
        AT->rowPtr[i] += AT->rowPtr[i - 1];
    }

    for (int i = 0; i < A.row; i++) {
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++) {
            int col = A.colIndex[j];
            int index = AT->rowPtr[col + 1]++;
            AT->value[index] = A.value[j];
            AT->colIndex[index] = i;
        }
    }
}

