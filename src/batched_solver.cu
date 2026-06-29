#include "batched_solver.h"
#include "preprocess.h"
#include "scaling.h"
#include "power_iteration.h"
#include "utils.h"
#include "cuda_kernels/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr HPRLP_FLOAT kInfReplacement = 1.0e100;
constexpr HPRLP_FLOAT kInfThreshold = 1.0e90;

struct BatchedScaling {
    HPRLP_FLOAT *row_norm = nullptr;
    HPRLP_FLOAT *col_norm = nullptr;
    std::vector<HPRLP_FLOAT> row_norm_host;
    std::vector<HPRLP_FLOAT> col_norm_host;
    std::vector<HPRLP_FLOAT> b_scale;
    std::vector<HPRLP_FLOAT> c_scale;
    std::vector<HPRLP_FLOAT> norm_b;
    std::vector<HPRLP_FLOAT> norm_c;
    std::vector<HPRLP_FLOAT> norm_b_org;
    std::vector<HPRLP_FLOAT> norm_c_org;
};

struct BatchedLPDevice {
    int m = 0;
    int n = 0;
    int B = 0;
    HPRLP_FLOAT *C = nullptr;
    HPRLP_FLOAT *AL = nullptr;
    HPRLP_FLOAT *AU = nullptr;
    HPRLP_FLOAT *L = nullptr;
    HPRLP_FLOAT *U = nullptr;
    std::vector<HPRLP_FLOAT> obj_constants;
};

struct BatchedSpMM {
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t A_descr = nullptr;
    cusparseSpMatDescr_t AT_descr = nullptr;
    cusparseDnMatDescr_t X_descr = nullptr;
    cusparseDnMatDescr_t X_hat_descr = nullptr;
    cusparseDnMatDescr_t X_bar_descr = nullptr;
    cusparseDnMatDescr_t DX_descr = nullptr;
    cusparseDnMatDescr_t AX_descr = nullptr;
    cusparseDnMatDescr_t Y_descr = nullptr;
    cusparseDnMatDescr_t Y_bar_descr = nullptr;
    cusparseDnMatDescr_t ATY_descr = nullptr;
    HPRLP_FLOAT alpha = 1.0;
    HPRLP_FLOAT beta = 0.0;
    void *buffer = nullptr;
    size_t buffer_size = 0;
};

struct BatchedWorkspace {
    int m = 0;
    int n = 0;
    int B = 0;
    HPRLP_FLOAT *X = nullptr;
    HPRLP_FLOAT *X_hat = nullptr;
    HPRLP_FLOAT *X_bar = nullptr;
    HPRLP_FLOAT *DX = nullptr;
    HPRLP_FLOAT *Y = nullptr;
    HPRLP_FLOAT *Y_hat = nullptr;
    HPRLP_FLOAT *Y_bar = nullptr;
    HPRLP_FLOAT *DY = nullptr;
    HPRLP_FLOAT *Y_obj = nullptr;
    HPRLP_FLOAT *Z_bar = nullptr;
    HPRLP_FLOAT *RP = nullptr;
    HPRLP_FLOAT *RD = nullptr;
    HPRLP_FLOAT *ATY = nullptr;
    HPRLP_FLOAT *AX = nullptr;
    HPRLP_FLOAT *last_X = nullptr;
    HPRLP_FLOAT *last_Y = nullptr;
    HPRLP_FLOAT *sigma = nullptr;
    HPRLP_FLOAT *halpern_fact1 = nullptr;
    HPRLP_FLOAT *halpern_fact2 = nullptr;
    unsigned char *active = nullptr;
    unsigned char *restart_flags = nullptr;
    HPRLP_FLOAT lambda_max = 1.0;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    BatchedSpMM spmm;
};

struct BatchedResidualHost {
    std::vector<HPRLP_FLOAT> primal_obj;
    std::vector<HPRLP_FLOAT> dual_obj;
    std::vector<HPRLP_FLOAT> err_Rp;
    std::vector<HPRLP_FLOAT> err_Rd;
    std::vector<HPRLP_FLOAT> rel_gap;
    std::vector<HPRLP_FLOAT> kkt_error;
};

struct BatchedRestartHost {
    std::vector<int> restart_flag;
    std::vector<unsigned char> first_restart;
    std::vector<HPRLP_FLOAT> last_gap;
    std::vector<HPRLP_FLOAT> current_gap;
    std::vector<HPRLP_FLOAT> save_gap;
    std::vector<HPRLP_FLOAT> best_gap;
    std::vector<HPRLP_FLOAT> best_sigma;
    std::vector<int> inner;
    std::vector<int> sufficient;
    std::vector<int> necessary;
    std::vector<int> long_restart;
    std::vector<int> times;
    std::vector<HPRLP_FLOAT> halpern_fact1;
    std::vector<HPRLP_FLOAT> halpern_fact2;
    std::vector<unsigned char> restart_flags;
    std::vector<HPRLP_FLOAT> sigma;
};

__global__ void update_x_z_check_batched_kernel(HPRLP_FLOAT *DX,
                                                HPRLP_FLOAT *X,
                                                HPRLP_FLOAT *Z_bar,
                                                HPRLP_FLOAT *X_bar,
                                                HPRLP_FLOAT *X_hat,
                                                const HPRLP_FLOAT *L,
                                                const HPRLP_FLOAT *U,
                                                const HPRLP_FLOAT *ATY,
                                                const HPRLP_FLOAT *C,
                                                const HPRLP_FLOAT *last_X,
                                                const HPRLP_FLOAT *sigma,
                                                const HPRLP_FLOAT *halpern_fact1,
                                                const HPRLP_FLOAT *halpern_fact2,
                                                const unsigned char *active,
                                                int n,
                                                int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int k = t / n;
    if (!active[k]) return;
    HPRLP_FLOAT sig = sigma[k];
    HPRLP_FLOAT xi = X[t];
    HPRLP_FLOAT z_trial = xi + sig * (ATY[t] - C[t]);
    HPRLP_FLOAT xbar = fmin(fmax(z_trial, L[t]), U[t]);
    HPRLP_FLOAT zbar = (xbar - z_trial) / sig;
    HPRLP_FLOAT xhat = 2.0 * xbar - xi;
    DX[t] = xbar - xhat;
    Z_bar[t] = zbar;
    X_bar[t] = xbar;
    X_hat[t] = xhat;
    X[t] = halpern_fact2[k] * xhat + halpern_fact1[k] * last_X[t];
}

__global__ void update_x_z_normal_batched_kernel(HPRLP_FLOAT *X,
                                                 HPRLP_FLOAT *X_hat,
                                                 const HPRLP_FLOAT *L,
                                                 const HPRLP_FLOAT *U,
                                                 const HPRLP_FLOAT *ATY,
                                                 const HPRLP_FLOAT *C,
                                                 const HPRLP_FLOAT *last_X,
                                                 const HPRLP_FLOAT *sigma,
                                                 const HPRLP_FLOAT *halpern_fact1,
                                                 const HPRLP_FLOAT *halpern_fact2,
                                                 const unsigned char *active,
                                                 int n,
                                                 int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int k = t / n;
    if (!active[k]) return;
    HPRLP_FLOAT xi = X[t];
    HPRLP_FLOAT z_trial = xi + sigma[k] * (ATY[t] - C[t]);
    HPRLP_FLOAT xbar = fmin(fmax(z_trial, L[t]), U[t]);
    HPRLP_FLOAT xhat = 2.0 * xbar - xi;
    X_hat[t] = xhat;
    X[t] = halpern_fact2[k] * xhat + halpern_fact1[k] * last_X[t];
}

__global__ void update_y_check_batched_kernel(HPRLP_FLOAT *DY,
                                              HPRLP_FLOAT *Y_bar,
                                              HPRLP_FLOAT *Y_hat,
                                              HPRLP_FLOAT *Y,
                                              HPRLP_FLOAT *Y_obj,
                                              const HPRLP_FLOAT *AL,
                                              const HPRLP_FLOAT *AU,
                                              const HPRLP_FLOAT *AX,
                                              const HPRLP_FLOAT *last_Y,
                                              const HPRLP_FLOAT *sigma,
                                              const HPRLP_FLOAT *halpern_fact1,
                                              const HPRLP_FLOAT *halpern_fact2,
                                              const unsigned char *active,
                                              HPRLP_FLOAT lambda_max,
                                              int m,
                                              int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int k = t / m;
    if (!active[k]) return;
    HPRLP_FLOAT fact1 = lambda_max * sigma[k];
    HPRLP_FLOAT yi = Y[t];
    HPRLP_FLOAT v = AX[t] - fact1 * yi;
    HPRLP_FLOAT d = fmax(AL[t] - v, fmin(AU[t] - v, 0.0));
    HPRLP_FLOAT yb = d / fact1;
    HPRLP_FLOAT yh = 2.0 * yb - yi;
    DY[t] = yb - yh;
    Y_bar[t] = yb;
    Y_hat[t] = yh;
    Y_obj[t] = v + d;
    Y[t] = halpern_fact2[k] * yh + halpern_fact1[k] * last_Y[t];
}

__global__ void update_y_normal_batched_kernel(HPRLP_FLOAT *Y,
                                               const HPRLP_FLOAT *AL,
                                               const HPRLP_FLOAT *AU,
                                               const HPRLP_FLOAT *AX,
                                               const HPRLP_FLOAT *last_Y,
                                               const HPRLP_FLOAT *sigma,
                                               const HPRLP_FLOAT *halpern_fact1,
                                               const HPRLP_FLOAT *halpern_fact2,
                                               const unsigned char *active,
                                               HPRLP_FLOAT lambda_max,
                                               int m,
                                               int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int k = t / m;
    if (!active[k]) return;
    HPRLP_FLOAT fact1 = lambda_max * sigma[k];
    HPRLP_FLOAT yi = Y[t];
    HPRLP_FLOAT v = AX[t] - fact1 * yi;
    HPRLP_FLOAT d = fmax(AL[t] - v, fmin(AU[t] - v, 0.0));
    HPRLP_FLOAT yb = d / fact1;
    HPRLP_FLOAT yh = 2.0 * yb - yi;
    Y[t] = halpern_fact2[k] * yh + halpern_fact1[k] * last_Y[t];
}

__global__ void compute_batched_Rd_kernel(const HPRLP_FLOAT *col_norm,
                                          const HPRLP_FLOAT *ATY,
                                          const HPRLP_FLOAT *Z_bar,
                                          const HPRLP_FLOAT *C,
                                          HPRLP_FLOAT *RD,
                                          int n,
                                          int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int i = t % n;
    RD[t] = (C[t] - ATY[t] - Z_bar[t]) * col_norm[i];
}

__global__ void compute_batched_Rp_kernel(const HPRLP_FLOAT *row_norm,
                                          HPRLP_FLOAT *RP,
                                          const HPRLP_FLOAT *AL,
                                          const HPRLP_FLOAT *AU,
                                          const HPRLP_FLOAT *AX,
                                          int m,
                                          int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int i = t % m;
    HPRLP_FLOAT v = AX[t];
    RP[t] = row_norm[i] * fmax(fmin(AU[t] - v, 0.0), AL[t] - v);
}

__global__ void compute_batched_lu_violation_kernel(const HPRLP_FLOAT *col_norm,
                                                    HPRLP_FLOAT *DX,
                                                    const HPRLP_FLOAT *X_bar,
                                                    const HPRLP_FLOAT *L,
                                                    const HPRLP_FLOAT *U,
                                                    int n,
                                                    int total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int i = t % n;
    HPRLP_FLOAT x = X_bar[t];
    HPRLP_FLOAT violation = x < L[t] ? L[t] - x : (x > U[t] ? x - U[t] : 0.0);
    DX[t] = violation / col_norm[i];
}

__global__ void batched_restart_movement_kernel(HPRLP_FLOAT *DX,
                                                HPRLP_FLOAT *DY,
                                                const HPRLP_FLOAT *X_bar,
                                                const HPRLP_FLOAT *Y_bar,
                                                const HPRLP_FLOAT *last_X,
                                                const HPRLP_FLOAT *last_Y,
                                                int n,
                                                int m,
                                                int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total_x = n * B;
    int total_y = m * B;
    if (t < total_x) DX[t] = X_bar[t] - last_X[t];
    if (t < total_y) DY[t] = Y_bar[t] - last_Y[t];
}

__global__ void do_batched_restart_kernel(HPRLP_FLOAT *X,
                                          HPRLP_FLOAT *Y,
                                          HPRLP_FLOAT *last_X,
                                          HPRLP_FLOAT *last_Y,
                                          const HPRLP_FLOAT *X_bar,
                                          const HPRLP_FLOAT *Y_bar,
                                          const unsigned char *restart_flags,
                                          int n,
                                          int m,
                                          int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total_x = n * B;
    int total_y = m * B;
    if (t < total_x) {
        int k = t / n;
        if (restart_flags[k]) {
            X[t] = X_bar[t];
            last_X[t] = X_bar[t];
        }
    }
    if (t < total_y) {
        int k = t / m;
        if (restart_flags[k]) {
            Y[t] = Y_bar[t];
            last_Y[t] = Y_bar[t];
        }
    }
}

int blocks(int n) { return (n + numThreads - 1) / numThreads; }

void cuda_malloc_zero(HPRLP_FLOAT **ptr, int len) {
    CUDA_CHECK(cudaMalloc(ptr, static_cast<size_t>(len) * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemset(*ptr, 0, static_cast<size_t>(len) * sizeof(HPRLP_FLOAT)));
}

HPRLP_FLOAT bound_norm_host(const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU, int m, int stride, int k) {
    long double sum = 0.0;
    const int offset = k * stride;
    for (int i = 0; i < m; ++i) {
        HPRLP_FLOAT lo = AL[offset + i];
        HPRLP_FLOAT hi = AU[offset + i];
        HPRLP_FLOAT a = std::isinf(lo) && lo < 0 ? 0.0 : std::abs(lo);
        HPRLP_FLOAT b = std::isinf(hi) && hi > 0 ? 0.0 : std::abs(hi);
        HPRLP_FLOAT v = std::max(a, b);
        sum += static_cast<long double>(v) * v;
    }
    return std::sqrt(static_cast<HPRLP_FLOAT>(sum));
}

HPRLP_FLOAT column_norm_host(const HPRLP_FLOAT *X, int n, int k) {
    long double sum = 0.0;
    const int offset = k * n;
    for (int i = 0; i < n; ++i) {
        HPRLP_FLOAT v = X[offset + i];
        sum += static_cast<long double>(v) * v;
    }
    return std::sqrt(static_cast<HPRLP_FLOAT>(sum));
}

HPRLP_batched_results make_batched_error(const char *status, int m, int n, int B) {
    HPRLP_batched_results r;
    r.m = m;
    r.n = n;
    r.batch_size = B;
    if (B > 0) {
        r.status = static_cast<char*>(std::calloc(static_cast<size_t>(B) * 64, sizeof(char)));
        for (int k = 0; k < B; ++k) {
            std::strncpy(r.status + 64 * k, status, 63);
        }
    }
    return r;
}

void copy_result_status(char *dst, int k, const char *status) {
    std::strncpy(dst + 64 * k, status, 63);
    dst[64 * k + 63] = '\0';
}

void free_batched_lp_device(BatchedLPDevice *lp) {
    if (!lp) return;
    if (lp->C) cudaFree(lp->C);
    if (lp->AL) cudaFree(lp->AL);
    if (lp->AU) cudaFree(lp->AU);
    if (lp->L) cudaFree(lp->L);
    if (lp->U) cudaFree(lp->U);
}

void destroy_spmm(BatchedSpMM *spmm) {
    if (!spmm) return;
    if (spmm->buffer) cudaFree(spmm->buffer);
    if (spmm->X_descr) cusparseDestroyDnMat(spmm->X_descr);
    if (spmm->X_hat_descr) cusparseDestroyDnMat(spmm->X_hat_descr);
    if (spmm->X_bar_descr) cusparseDestroyDnMat(spmm->X_bar_descr);
    if (spmm->DX_descr) cusparseDestroyDnMat(spmm->DX_descr);
    if (spmm->AX_descr) cusparseDestroyDnMat(spmm->AX_descr);
    if (spmm->Y_descr) cusparseDestroyDnMat(spmm->Y_descr);
    if (spmm->Y_bar_descr) cusparseDestroyDnMat(spmm->Y_bar_descr);
    if (spmm->ATY_descr) cusparseDestroyDnMat(spmm->ATY_descr);
    if (spmm->A_descr) cusparseDestroySpMat(spmm->A_descr);
    if (spmm->AT_descr) cusparseDestroySpMat(spmm->AT_descr);
    if (spmm->handle) cusparseDestroy(spmm->handle);
}

void free_batched_workspace(BatchedWorkspace *ws) {
    if (!ws) return;
    destroy_spmm(&ws->spmm);
    if (ws->cublas) cublasDestroy(ws->cublas);
    if (ws->X) cudaFree(ws->X);
    if (ws->X_hat) cudaFree(ws->X_hat);
    if (ws->X_bar) cudaFree(ws->X_bar);
    if (ws->DX) cudaFree(ws->DX);
    if (ws->Y) cudaFree(ws->Y);
    if (ws->Y_hat) cudaFree(ws->Y_hat);
    if (ws->Y_bar) cudaFree(ws->Y_bar);
    if (ws->DY) cudaFree(ws->DY);
    if (ws->Y_obj) cudaFree(ws->Y_obj);
    if (ws->Z_bar) cudaFree(ws->Z_bar);
    if (ws->RP) cudaFree(ws->RP);
    if (ws->RD) cudaFree(ws->RD);
    if (ws->ATY) cudaFree(ws->ATY);
    if (ws->AX) cudaFree(ws->AX);
    if (ws->last_X) cudaFree(ws->last_X);
    if (ws->last_Y) cudaFree(ws->last_Y);
    if (ws->sigma) cudaFree(ws->sigma);
    if (ws->halpern_fact1) cudaFree(ws->halpern_fact1);
    if (ws->halpern_fact2) cudaFree(ws->halpern_fact2);
    if (ws->active) cudaFree(ws->active);
    if (ws->restart_flags) cudaFree(ws->restart_flags);
    if (ws->stream) cudaStreamDestroy(ws->stream);
}

void prepare_spmm(BatchedWorkspace *ws, const LP_info_gpu *shared_lp) {
    BatchedSpMM *sp = &ws->spmm;
    CUSPARSE_CHECK(cusparseCreate(&sp->handle));
    CUSPARSE_CHECK(cusparseSetStream(sp->handle, ws->stream));
    CUSPARSE_CHECK(cusparseCreateCsr(&sp->A_descr, ws->m, ws->n, shared_lp->A->numElements,
                                     shared_lp->A->rowPtr, shared_lp->A->colIndex, shared_lp->A->value,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&sp->AT_descr, ws->n, ws->m, shared_lp->AT->numElements,
                                     shared_lp->AT->rowPtr, shared_lp->AT->colIndex, shared_lp->AT->value,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->X_descr, ws->n, ws->B, ws->n, ws->X, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->X_hat_descr, ws->n, ws->B, ws->n, ws->X_hat, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->X_bar_descr, ws->n, ws->B, ws->n, ws->X_bar, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->DX_descr, ws->n, ws->B, ws->n, ws->DX, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->AX_descr, ws->m, ws->B, ws->m, ws->AX, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->Y_descr, ws->m, ws->B, ws->m, ws->Y, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->Y_bar_descr, ws->m, ws->B, ws->m, ws->Y_bar, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&sp->ATY_descr, ws->n, ws->B, ws->n, ws->ATY, CUDA_R_64F, CUSPARSE_ORDER_COL));

    size_t size_a = 0;
    size_t size_at = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(sp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &sp->alpha, sp->A_descr, sp->X_bar_descr,
                                           &sp->beta, sp->AX_descr, CUDA_R_64F,
                                           CUSPARSE_SPMM_CSR_ALG1, &size_a));
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(sp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &sp->alpha, sp->AT_descr, sp->Y_bar_descr,
                                           &sp->beta, sp->ATY_descr, CUDA_R_64F,
                                           CUSPARSE_SPMM_CSR_ALG1, &size_at));
    sp->buffer_size = std::max(size_a, size_at);
    if (sp->buffer_size > 0) CUDA_CHECK(cudaMalloc(&sp->buffer, sp->buffer_size));
}

void spmm_A(BatchedWorkspace *ws, cusparseDnMatDescr_t source) {
    BatchedSpMM *sp = &ws->spmm;
    CUSPARSE_CHECK(cusparseSpMM(sp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &sp->alpha, sp->A_descr, source,
                                &sp->beta, sp->AX_descr, CUDA_R_64F,
                                CUSPARSE_SPMM_CSR_ALG1, sp->buffer));
}

void spmm_AT(BatchedWorkspace *ws, cusparseDnMatDescr_t source) {
    BatchedSpMM *sp = &ws->spmm;
    CUSPARSE_CHECK(cusparseSpMM(sp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &sp->alpha, sp->AT_descr, source,
                                &sp->beta, sp->ATY_descr, CUDA_R_64F,
                                CUSPARSE_SPMM_CSR_ALG1, sp->buffer));
}

void allocate_batched_workspace(BatchedWorkspace *ws,
                                const LP_info_gpu *shared_lp,
                                const BatchedScaling &scaling,
                                HPRLP_FLOAT lambda_max,
                                int B) {
    ws->m = shared_lp->m;
    ws->n = shared_lp->n;
    ws->B = B;
    ws->lambda_max = lambda_max;
    CUDA_CHECK(cudaStreamCreate(&ws->stream));
    CUBLAS_CHECK(cublasCreate(&ws->cublas));
    CUBLAS_CHECK(cublasSetStream(ws->cublas, ws->stream));

    int nB = ws->n * B;
    int mB = ws->m * B;
    cuda_malloc_zero(&ws->X, nB);
    cuda_malloc_zero(&ws->X_hat, nB);
    cuda_malloc_zero(&ws->X_bar, nB);
    cuda_malloc_zero(&ws->DX, nB);
    cuda_malloc_zero(&ws->Y, mB);
    cuda_malloc_zero(&ws->Y_hat, mB);
    cuda_malloc_zero(&ws->Y_bar, mB);
    cuda_malloc_zero(&ws->DY, mB);
    cuda_malloc_zero(&ws->Y_obj, mB);
    cuda_malloc_zero(&ws->Z_bar, nB);
    cuda_malloc_zero(&ws->RP, mB);
    cuda_malloc_zero(&ws->RD, nB);
    cuda_malloc_zero(&ws->ATY, nB);
    cuda_malloc_zero(&ws->AX, mB);
    cuda_malloc_zero(&ws->last_X, nB);
    cuda_malloc_zero(&ws->last_Y, mB);
    CUDA_CHECK(cudaMalloc(&ws->sigma, B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&ws->halpern_fact1, B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&ws->halpern_fact2, B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&ws->active, B * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&ws->restart_flags, B * sizeof(unsigned char)));

    std::vector<HPRLP_FLOAT> sigma(B, 1.0);
    std::vector<unsigned char> active(B, 1);
    std::vector<unsigned char> flags(B, 0);
    std::vector<HPRLP_FLOAT> h1(B, 0.5), h2(B, 0.5);
    for (int k = 0; k < B; ++k) {
        if (scaling.norm_b[k] > 1.0e-8 && scaling.norm_c[k] > 1.0e-8) {
            sigma[k] = scaling.norm_b[k] / scaling.norm_c[k];
        }
    }
    CUDA_CHECK(cudaMemcpy(ws->sigma, sigma.data(), B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->active, active.data(), B * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->restart_flags, flags.data(), B * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->halpern_fact1, h1.data(), B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->halpern_fact2, h2.data(), B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));

    prepare_spmm(ws, shared_lp);
}

BatchedRestartHost initialize_restart(const BatchedWorkspace &ws) {
    BatchedRestartHost r;
    int B = ws.B;
    r.restart_flag.assign(B, 0);
    r.first_restart.assign(B, 1);
    r.last_gap.assign(B, std::numeric_limits<HPRLP_FLOAT>::infinity());
    r.current_gap.assign(B, std::numeric_limits<HPRLP_FLOAT>::infinity());
    r.save_gap.assign(B, std::numeric_limits<HPRLP_FLOAT>::infinity());
    r.best_gap.assign(B, std::numeric_limits<HPRLP_FLOAT>::infinity());
    r.best_sigma.assign(B, 1.0);
    r.inner.assign(B, 0);
    r.sufficient.assign(B, 0);
    r.necessary.assign(B, 0);
    r.long_restart.assign(B, 0);
    r.times.assign(B, 0);
    r.halpern_fact1.assign(B, 0.5);
    r.halpern_fact2.assign(B, 0.5);
    r.restart_flags.assign(B, 0);
    r.sigma.assign(B, 1.0);
    CUDA_CHECK(cudaMemcpy(r.sigma.data(), ws.sigma, B * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    r.best_sigma = r.sigma;
    return r;
}

void upload_halpern_factors(BatchedWorkspace *ws, BatchedRestartHost *restart) {
    for (int k = 0; k < ws->B; ++k) {
        restart->halpern_fact1[k] = 1.0 / (restart->inner[k] + 2.0);
        restart->halpern_fact2[k] = 1.0 - restart->halpern_fact1[k];
    }
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_fact1, restart->halpern_fact1.data(), ws->B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_fact2, restart->halpern_fact2.data(), ws->B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice, ws->stream));
}

BatchedResidualHost allocate_residuals(int B) {
    BatchedResidualHost r;
    r.primal_obj.assign(B, 0.0);
    r.dual_obj.assign(B, 0.0);
    r.err_Rp.assign(B, 0.0);
    r.err_Rd.assign(B, 0.0);
    r.rel_gap.assign(B, 0.0);
    r.kkt_error.assign(B, std::numeric_limits<HPRLP_FLOAT>::infinity());
    return r;
}

void compute_residuals(BatchedWorkspace *ws,
                       const BatchedLPDevice &batch,
                       const BatchedScaling &scaling,
                       BatchedResidualHost *residuals,
                       int iter) {
    spmm_AT(ws, ws->spmm.Y_bar_descr);
    compute_batched_Rd_kernel<<<blocks(ws->n * ws->B), numThreads, 0, ws->stream>>>(
        scaling.col_norm, ws->ATY, ws->Z_bar, batch.C, ws->RD, ws->n, ws->n * ws->B);

    spmm_A(ws, ws->spmm.X_bar_descr);
    compute_batched_Rp_kernel<<<blocks(ws->m * ws->B), numThreads, 0, ws->stream>>>(
        scaling.row_norm, ws->RP, batch.AL, batch.AU, ws->AX, ws->m, ws->m * ws->B);

    if (iter == 0) {
        compute_batched_lu_violation_kernel<<<blocks(ws->n * ws->B), numThreads, 0, ws->stream>>>(
            scaling.col_norm, ws->DX, ws->X_bar, batch.L, batch.U, ws->n, ws->n * ws->B);
    }

    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    for (int k = 0; k < ws->B; ++k) {
        HPRLP_FLOAT dot_cx = 0.0;
        HPRLP_FLOAT dot_yy = 0.0;
        HPRLP_FLOAT dot_xz = 0.0;
        HPRLP_FLOAT rd_norm = 0.0;
        HPRLP_FLOAT rp_norm = 0.0;
        CUBLAS_CHECK(cublasDdot(ws->cublas, ws->n, batch.C + k * ws->n, 1, ws->X_bar + k * ws->n, 1, &dot_cx));
        CUBLAS_CHECK(cublasDdot(ws->cublas, ws->m, ws->Y_obj + k * ws->m, 1, ws->Y_bar + k * ws->m, 1, &dot_yy));
        CUBLAS_CHECK(cublasDdot(ws->cublas, ws->n, ws->X_bar + k * ws->n, 1, ws->Z_bar + k * ws->n, 1, &dot_xz));
        CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->n, ws->RD + k * ws->n, 1, &rd_norm));
        CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->m, ws->RP + k * ws->m, 1, &rp_norm));
        HPRLP_FLOAT obj_scale = scaling.b_scale[k] * scaling.c_scale[k];
        residuals->primal_obj[k] = obj_scale * dot_cx + batch.obj_constants[k];
        residuals->dual_obj[k] = obj_scale * (dot_yy + dot_xz) + batch.obj_constants[k];
        residuals->err_Rd[k] = scaling.c_scale[k] * rd_norm / scaling.norm_c_org[k];
        residuals->err_Rp[k] = scaling.b_scale[k] * rp_norm / scaling.norm_b_org[k];
        if (iter == 0) {
            HPRLP_FLOAT lu_norm = 0.0;
            CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->n, ws->DX + k * ws->n, 1, &lu_norm));
            residuals->err_Rp[k] = std::max(residuals->err_Rp[k], scaling.b_scale[k] * lu_norm);
        }
        residuals->rel_gap[k] = std::abs(residuals->primal_obj[k] - residuals->dual_obj[k]) /
            (1.0 + std::abs(residuals->primal_obj[k]) + std::abs(residuals->dual_obj[k]));
        residuals->kkt_error[k] = std::max(residuals->err_Rp[k], std::max(residuals->err_Rd[k], residuals->rel_gap[k]));
    }
}

std::vector<HPRLP_FLOAT> compute_weighted_norm(BatchedWorkspace *ws) {
    spmm_A(ws, ws->spmm.DX_descr);
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    std::vector<HPRLP_FLOAT> sigma(ws->B);
    std::vector<HPRLP_FLOAT> weighted(ws->B, 0.0);
    CUDA_CHECK(cudaMemcpy(sigma.data(), ws->sigma, ws->B * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    for (int k = 0; k < ws->B; ++k) {
        HPRLP_FLOAT dot_ax_dy = 0.0;
        HPRLP_FLOAT dy_norm = 0.0;
        HPRLP_FLOAT dx_norm = 0.0;
        CUBLAS_CHECK(cublasDdot(ws->cublas, ws->m, ws->AX + k * ws->m, 1, ws->DY + k * ws->m, 1, &dot_ax_dy));
        CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->m, ws->DY + k * ws->m, 1, &dy_norm));
        CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->n, ws->DX + k * ws->n, 1, &dx_norm));
        HPRLP_FLOAT dot_prod = 2.0 * dot_ax_dy;
        HPRLP_FLOAT dy_sq = dy_norm * dy_norm;
        HPRLP_FLOAT dx_sq = dx_norm * dx_norm;
        HPRLP_FLOAT value = sigma[k] * (ws->lambda_max * dy_sq) + dx_sq / sigma[k] + dot_prod;
        if (value < 0.0 && dy_sq > 0.0) {
            HPRLP_FLOAT candidate_lambda = -(dot_prod + dx_sq / sigma[k]) / (sigma[k] * dy_sq) * 1.05;
            ws->lambda_max = std::max(ws->lambda_max, candidate_lambda);
            value = sigma[k] * (ws->lambda_max * dy_sq) + dx_sq / sigma[k] + dot_prod;
        }
        weighted[k] = std::sqrt(std::max(value, 0.0));
    }
    return weighted;
}

void compute_restart_movement_norms(BatchedWorkspace *ws,
                                    std::vector<HPRLP_FLOAT> *primal_move,
                                    std::vector<HPRLP_FLOAT> *dual_move) {
    int total = std::max(ws->n * ws->B, ws->m * ws->B);
    batched_restart_movement_kernel<<<blocks(total), numThreads, 0, ws->stream>>>(
        ws->DX, ws->DY, ws->X_bar, ws->Y_bar, ws->last_X, ws->last_Y, ws->n, ws->m, ws->B);
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    primal_move->assign(ws->B, 0.0);
    dual_move->assign(ws->B, 0.0);
    for (int k = 0; k < ws->B; ++k) {
        CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->n, ws->DX + k * ws->n, 1, &(*primal_move)[k]));
        CUBLAS_CHECK(cublasDnrm2(ws->cublas, ws->m, ws->DY + k * ws->m, 1, &(*dual_move)[k]));
    }
}

void check_restart(BatchedRestartHost *restart, int iter, int check_iter,
                   const std::vector<HPRLP_FLOAT> &sigma,
                   const std::vector<unsigned char> &active) {
    for (int k = 0; k < static_cast<int>(active.size()); ++k) {
        if (!active[k]) continue;
        if (restart->first_restart[k]) {
            if (iter == check_iter) {
                restart->first_restart[k] = 0;
                restart->restart_flag[k] = 1;
                restart->best_gap[k] = restart->current_gap[k];
                restart->best_sigma[k] = sigma[k];
            }
        } else if (iter % check_iter == 0) {
            if (restart->current_gap[k] < 0.0) restart->current_gap[k] = 1.0e-6;
            if (restart->current_gap[k] <= 0.2 * restart->last_gap[k]) {
                restart->sufficient[k] += 1;
                restart->restart_flag[k] = 1;
            }
            if (restart->current_gap[k] <= 0.6 * restart->last_gap[k] && restart->current_gap[k] > restart->save_gap[k]) {
                restart->necessary[k] += 1;
                restart->restart_flag[k] = 2;
            }
            if (restart->inner[k] >= 0.2 * iter) {
                restart->long_restart[k] += 1;
                restart->restart_flag[k] = 3;
            }
            if (restart->best_gap[k] > restart->current_gap[k]) {
                restart->best_gap[k] = restart->current_gap[k];
                restart->best_sigma[k] = sigma[k];
            }
            restart->save_gap[k] = restart->current_gap[k];
        }
    }
}

void update_sigma(BatchedRestartHost *restart,
                  BatchedWorkspace *ws,
                  const BatchedResidualHost &residuals,
                  const std::vector<unsigned char> &active) {
    bool any = false;
    for (int flag : restart->restart_flag) any = any || (flag >= 1 && flag <= 3);
    if (!any) return;

    std::vector<HPRLP_FLOAT> primal_move;
    std::vector<HPRLP_FLOAT> dual_move;
    compute_restart_movement_norms(ws, &primal_move, &dual_move);
    HPRLP_FLOAT sqrt_lambda = std::sqrt(ws->lambda_max);
    for (int k = 0; k < ws->B; ++k) {
        if (!active[k]) continue;
        if (restart->restart_flag[k] >= 1 && restart->restart_flag[k] <= 3) {
            if (primal_move[k] > 1.0e-16 && dual_move[k] > 1.0e-16 && primal_move[k] < 1.0e12 && dual_move[k] < 1.0e12) {
                HPRLP_FLOAT ratio = (primal_move[k] / dual_move[k]) / sqrt_lambda;
                HPRLP_FLOAT fact = std::exp(-0.05 * (restart->current_gap[k] / restart->best_gap[k]));
                HPRLP_FLOAT temp1 = std::max(std::min(residuals.err_Rd[k], residuals.err_Rp[k]),
                                             std::min(residuals.rel_gap[k], restart->current_gap[k]));
                HPRLP_FLOAT sigma_cand = std::exp(fact * std::log(ratio) + (1.0 - fact) * std::log(restart->best_sigma[k]));
                HPRLP_FLOAT ratio_infeas = residuals.err_Rd[k] / residuals.err_Rp[k];
                HPRLP_FLOAT kappa = 1.0;
                if (temp1 > 9.0e-10) {
                    kappa = 1.0;
                } else if (temp1 > 5.0e-10) {
                    kappa = std::max(std::min(std::sqrt(ratio_infeas), 100.0), 1.0e-2);
                } else {
                    kappa = std::max(std::min(ratio_infeas, 100.0), 1.0e-2);
                }
                restart->sigma[k] = kappa * sigma_cand;
            } else {
                restart->sigma[k] = 1.0;
            }
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(ws->sigma, restart->sigma.data(), ws->B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice, ws->stream));
}

bool do_restart(BatchedRestartHost *restart, BatchedWorkspace *ws, const std::vector<unsigned char> &active) {
    bool any = false;
    for (int k = 0; k < ws->B; ++k) {
        restart->restart_flags[k] = restart->restart_flag[k] > 0 ? 1 : 0;
        any = any || restart->restart_flags[k];
    }
    CUDA_CHECK(cudaMemcpyAsync(ws->restart_flags, restart->restart_flags.data(), ws->B * sizeof(unsigned char), cudaMemcpyHostToDevice, ws->stream));
    if (any) {
        int total = std::max(ws->n * ws->B, ws->m * ws->B);
        do_batched_restart_kernel<<<blocks(total), numThreads, 0, ws->stream>>>(
            ws->X, ws->Y, ws->last_X, ws->last_Y, ws->X_bar, ws->Y_bar,
            ws->restart_flags, ws->n, ws->m, ws->B);
        for (int k = 0; k < ws->B; ++k) {
            if (active[k] && restart->restart_flag[k] > 0) {
                restart->times[k] += 1;
                restart->inner[k] = 0;
                restart->save_gap[k] = std::numeric_limits<HPRLP_FLOAT>::infinity();
            }
        }
    }
    return any;
}

void update_x_z(BatchedWorkspace *ws, const BatchedLPDevice &batch, bool check) {
    spmm_AT(ws, ws->spmm.Y_descr);
    if (check) {
        update_x_z_check_batched_kernel<<<blocks(ws->n * ws->B), numThreads, 0, ws->stream>>>(
            ws->DX, ws->X, ws->Z_bar, ws->X_bar, ws->X_hat, batch.L, batch.U, ws->ATY,
            batch.C, ws->last_X, ws->sigma, ws->halpern_fact1, ws->halpern_fact2,
            ws->active, ws->n, ws->n * ws->B);
    } else {
        update_x_z_normal_batched_kernel<<<blocks(ws->n * ws->B), numThreads, 0, ws->stream>>>(
            ws->X, ws->X_hat, batch.L, batch.U, ws->ATY, batch.C, ws->last_X,
            ws->sigma, ws->halpern_fact1, ws->halpern_fact2, ws->active, ws->n, ws->n * ws->B);
    }
}

void update_y(BatchedWorkspace *ws, const BatchedLPDevice &batch, bool check) {
    spmm_A(ws, ws->spmm.X_hat_descr);
    if (check) {
        update_y_check_batched_kernel<<<blocks(ws->m * ws->B), numThreads, 0, ws->stream>>>(
            ws->DY, ws->Y_bar, ws->Y_hat, ws->Y, ws->Y_obj, batch.AL, batch.AU, ws->AX,
            ws->last_Y, ws->sigma, ws->halpern_fact1, ws->halpern_fact2, ws->active,
            ws->lambda_max, ws->m, ws->m * ws->B);
    } else {
        update_y_normal_batched_kernel<<<blocks(ws->m * ws->B), numThreads, 0, ws->stream>>>(
            ws->Y, batch.AL, batch.AU, ws->AX, ws->last_Y, ws->sigma,
            ws->halpern_fact1, ws->halpern_fact2, ws->active, ws->lambda_max, ws->m, ws->m * ws->B);
    }
}

BatchedLPDevice build_batched_lp_device(const LP_info_cpu *model,
                                        int B,
                                        const HPRLP_FLOAT *C,
                                        const HPRLP_FLOAT *AL,
                                        const HPRLP_FLOAT *AU,
                                        const HPRLP_FLOAT *L,
                                        const HPRLP_FLOAT *U,
                                        const HPRLP_FLOAT *obj_constants,
                                        const HPRLP_parameters *param,
                                        BatchedScaling *scaling) {
    int m = model->m;
    int n = model->n;
    std::vector<HPRLP_FLOAT> hC(C, C + n * B);
    std::vector<HPRLP_FLOAT> hAL(AL, AL + m * B);
    std::vector<HPRLP_FLOAT> hAU(AU, AU + m * B);
    std::vector<HPRLP_FLOAT> hL(L, L + n * B);
    std::vector<HPRLP_FLOAT> hU(U, U + n * B);

    scaling->b_scale.assign(B, 1.0);
    scaling->c_scale.assign(B, 1.0);
    scaling->norm_b.assign(B, 0.0);
    scaling->norm_c.assign(B, 0.0);
    scaling->norm_b_org.assign(B, 1.0);
    scaling->norm_c_org.assign(B, 1.0);

    for (int k = 0; k < B; ++k) {
        scaling->norm_b_org[k] = 1.0 + bound_norm_host(hAL.data(), hAU.data(), m, m, k);
        scaling->norm_c_org[k] = 1.0 + column_norm_host(hC.data(), n, k);
        for (int i = 0; i < m; ++i) {
            HPRLP_FLOAT rn = scaling->row_norm_host[i];
            hAL[k * m + i] /= rn;
            hAU[k * m + i] /= rn;
        }
        for (int i = 0; i < n; ++i) {
            HPRLP_FLOAT cn = scaling->col_norm_host[i];
            hC[k * n + i] /= cn;
            hL[k * n + i] *= cn;
            hU[k * n + i] *= cn;
        }
    }

    if (param->use_bc_scaling) {
        for (int k = 0; k < B; ++k) {
            scaling->b_scale[k] = 1.0 + bound_norm_host(hAL.data(), hAU.data(), m, m, k);
            scaling->c_scale[k] = 1.0 + column_norm_host(hC.data(), n, k);
            for (int i = 0; i < m; ++i) {
                hAL[k * m + i] /= scaling->b_scale[k];
                hAU[k * m + i] /= scaling->b_scale[k];
            }
            for (int i = 0; i < n; ++i) {
                hC[k * n + i] /= scaling->c_scale[k];
                hL[k * n + i] /= scaling->b_scale[k];
                hU[k * n + i] /= scaling->b_scale[k];
            }
        }
    }

    for (int k = 0; k < B; ++k) {
        scaling->norm_b[k] = bound_norm_host(hAL.data(), hAU.data(), m, m, k);
        scaling->norm_c[k] = column_norm_host(hC.data(), n, k);
        for (int i = 0; i < m; ++i) {
            HPRLP_FLOAT &lo = hAL[k * m + i];
            HPRLP_FLOAT &hi = hAU[k * m + i];
            if (std::isinf(lo) && lo < 0) lo = -kInfReplacement;
            if (std::isinf(hi) && hi > 0) hi = kInfReplacement;
        }
        for (int i = 0; i < n; ++i) {
            HPRLP_FLOAT &lo = hL[k * n + i];
            HPRLP_FLOAT &hi = hU[k * n + i];
            if (std::isinf(lo) && lo < 0) lo = -kInfReplacement;
            if (std::isinf(hi) && hi > 0) hi = kInfReplacement;
        }
    }

    BatchedLPDevice batch;
    batch.m = m;
    batch.n = n;
    batch.B = B;
    batch.obj_constants.assign(B, model->obj_constant);
    if (obj_constants) {
        batch.obj_constants.assign(obj_constants, obj_constants + B);
    }
    CUDA_CHECK(cudaMalloc(&batch.C, n * B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&batch.AL, m * B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&batch.AU, m * B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&batch.L, n * B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&batch.U, n * B * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(batch.C, hC.data(), n * B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(batch.AL, hAL.data(), m * B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(batch.AU, hAU.data(), m * B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(batch.L, hL.data(), n * B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(batch.U, hU.data(), n * B * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    return batch;
}

void collect_results(BatchedWorkspace *ws,
                     const BatchedScaling &scaling,
                     const BatchedResidualHost &residuals,
                     const std::vector<std::string> &status,
                     const std::vector<int> &iter,
                     HPRLP_FLOAT setup_time,
                     HPRLP_FLOAT solve_time,
                     HPRLP_FLOAT power_time,
                     HPRLP_batched_results *out) {
    int nB = ws->n * ws->B;
    int mB = ws->m * ws->B;
    std::vector<HPRLP_FLOAT> hX(nB), hY(mB), hZ(nB);
    CUDA_CHECK(cudaMemcpy(hX.data(), ws->X_bar, nB * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hY.data(), ws->Y_bar, mB * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hZ.data(), ws->Z_bar, nB * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));

    out->m = ws->m;
    out->n = ws->n;
    out->batch_size = ws->B;
    out->x = static_cast<HPRLP_FLOAT*>(std::malloc(nB * sizeof(HPRLP_FLOAT)));
    out->y = static_cast<HPRLP_FLOAT*>(std::malloc(mB * sizeof(HPRLP_FLOAT)));
    out->z = static_cast<HPRLP_FLOAT*>(std::malloc(nB * sizeof(HPRLP_FLOAT)));
    out->primal_obj = static_cast<HPRLP_FLOAT*>(std::malloc(ws->B * sizeof(HPRLP_FLOAT)));
    out->residuals = static_cast<HPRLP_FLOAT*>(std::malloc(ws->B * sizeof(HPRLP_FLOAT)));
    out->gap = static_cast<HPRLP_FLOAT*>(std::malloc(ws->B * sizeof(HPRLP_FLOAT)));
    out->iter = static_cast<int*>(std::malloc(ws->B * sizeof(int)));
    out->status = static_cast<char*>(std::calloc(static_cast<size_t>(ws->B) * 64, sizeof(char)));

    for (int k = 0; k < ws->B; ++k) {
        for (int i = 0; i < ws->n; ++i) {
            int idx = k * ws->n + i;
            out->x[idx] = (hX[idx] / scaling.col_norm_host[i]) * scaling.b_scale[k];
            out->z[idx] = (hZ[idx] * scaling.col_norm_host[i]) * scaling.c_scale[k];
        }
        for (int i = 0; i < ws->m; ++i) {
            int idx = k * ws->m + i;
            out->y[idx] = (hY[idx] / scaling.row_norm_host[i]) * scaling.c_scale[k];
        }
        out->primal_obj[k] = residuals.primal_obj[k];
        out->residuals[k] = residuals.kkt_error[k];
        out->gap[k] = residuals.rel_gap[k];
        out->iter[k] = iter[k];
        copy_result_status(out->status, k, status[k].c_str());
    }
    out->setup_time = setup_time;
    out->solve_time = solve_time;
    out->power_time = power_time;
    out->time = setup_time + solve_time;
}

} // namespace

extern "C" HPRLP_batched_results solve_batched(const LP_info_cpu *model,
                                                int batch_size,
                                                const HPRLP_FLOAT *C,
                                                const HPRLP_FLOAT *AL,
                                                const HPRLP_FLOAT *AU,
                                                const HPRLP_FLOAT *l,
                                                const HPRLP_FLOAT *u,
                                                const HPRLP_FLOAT *obj_constants,
                                                const HPRLP_parameters *param) {
    if (!model || !model->A || batch_size <= 0 || !C || !AL || !AU || !l || !u) {
        return make_batched_error("ERROR", model ? model->m : 0, model ? model->n : 0, std::max(batch_size, 0));
    }

    HPRLP_parameters default_param;
    HPRLP_parameters actual = param ? *param : default_param;
    actual.use_presolve = false;
    CUDA_CHECK(cudaSetDevice(actual.device_number));

    auto setup_start = time_now();

    std::vector<HPRLP_FLOAT> zero_m(model->m, 0.0);
    std::vector<HPRLP_FLOAT> zero_n(model->n, 0.0);
    LP_info_cpu matrix_cpu{};
    matrix_cpu.m = model->m;
    matrix_cpu.n = model->n;
    matrix_cpu.A = model->A;
    matrix_cpu.AL = zero_m.data();
    matrix_cpu.AU = zero_m.data();
    matrix_cpu.c = zero_n.data();
    matrix_cpu.l = zero_n.data();
    matrix_cpu.u = zero_n.data();
    matrix_cpu.obj_constant = 0.0;

    LP_info_gpu shared_lp{};
    copy_lpinfo_to_device(&matrix_cpu, &shared_lp);

    cublasHandle_t setup_cublas = nullptr;
    CUBLAS_CHECK(cublasCreate(&setup_cublas));
    Scaling_info shared_scaling{};
    HPRLP_parameters matrix_param = actual;
    matrix_param.use_bc_scaling = false;
    matrix_param.CUSPARSE_spmv = true;
    scaling(&shared_lp, &shared_scaling, &matrix_param, setup_cublas);

    BatchedScaling batched_scaling;
    batched_scaling.row_norm = shared_scaling.row_norm;
    batched_scaling.col_norm = shared_scaling.col_norm;
    batched_scaling.row_norm_host.resize(model->m);
    batched_scaling.col_norm_host.resize(model->n);
    CUDA_CHECK(cudaMemcpy(batched_scaling.row_norm_host.data(), shared_scaling.row_norm, model->m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(batched_scaling.col_norm_host.data(), shared_scaling.col_norm, model->n * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));

    BatchedLPDevice batch = build_batched_lp_device(model, batch_size, C, AL, AU, l, u,
                                                    obj_constants, &actual, &batched_scaling);

    HPRLP_workspace_gpu power_ws{};
    power_ws.m = model->m;
    power_ws.n = model->n;
    allocate_memory(&power_ws, &shared_lp, &matrix_param);
    auto power_start = time_now();
    HPRLP_FLOAT lambda_max = power_method_cusparse(&power_ws, 5000, 1.0e-4) * 1.01;
    HPRLP_FLOAT power_time = time_since(power_start);
    free_workspace(&power_ws);
    CUBLAS_CHECK(cublasDestroy(setup_cublas));

    BatchedWorkspace ws{};
    allocate_batched_workspace(&ws, &shared_lp, batched_scaling, lambda_max, batch_size);
    HPRLP_FLOAT setup_time = time_since(setup_start);

    auto solve_start = time_now();
    BatchedResidualHost residuals = allocate_residuals(batch_size);
    BatchedRestartHost restart = initialize_restart(ws);
    std::vector<std::string> status(batch_size, "CONTINUE");
    std::vector<int> final_iter(batch_size, actual.max_iter);
    std::vector<unsigned char> active(batch_size, 1);
    int check_iter = std::max(actual.check_iter, 1);
    HPRLP_batched_results out{};

    for (int iter = 0; iter <= actual.max_iter; ++iter) {
        bool periodic_check = (iter % check_iter) == 0;
        HPRLP_FLOAT elapsed = time_since(solve_start);
        bool residual_check = periodic_check;
        if (residual_check) {
            if (periodic_check && iter > 0) {
                restart.current_gap = compute_weighted_norm(&ws);
            }
            compute_residuals(&ws, batch, batched_scaling, &residuals, iter);
            for (int k = 0; k < batch_size; ++k) {
                if (active[k] && residuals.kkt_error[k] <= actual.stop_tol) {
                    status[k] = "OPTIMAL";
                    final_iter[k] = iter;
                    active[k] = 0;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(ws.active, active.data(), batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice, ws.stream));
        }

        bool all_done = true;
        for (const std::string &s : status) all_done = all_done && (s != "CONTINUE");
        if (all_done) {
            collect_results(&ws, batched_scaling, residuals, status, final_iter,
                            setup_time, time_since(solve_start), power_time, &out);
            break;
        }

        if (iter >= actual.max_iter || elapsed >= actual.time_limit) {
            const char *final_status = elapsed >= actual.time_limit ? "TIME_LIMIT" : "ITER_LIMIT";
            for (int k = 0; k < batch_size; ++k) {
                if (status[k] == "CONTINUE") {
                    status[k] = final_status;
                    final_iter[k] = iter;
                    active[k] = 0;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(ws.active, active.data(), batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice, ws.stream));
            collect_results(&ws, batched_scaling, residuals, status, final_iter,
                            setup_time, time_since(solve_start), power_time, &out);
            break;
        }

        std::fill(restart.restart_flag.begin(), restart.restart_flag.end(), 0);
        if (periodic_check) {
            CUDA_CHECK(cudaMemcpy(restart.sigma.data(), ws.sigma, batch_size * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
            check_restart(&restart, iter, check_iter, restart.sigma, active);
        }

        update_sigma(&restart, &ws, residuals, active);
        bool restarted = do_restart(&restart, &ws, active);
        bool to_check = ((iter + 1) % check_iter) == 0 || restarted;
        to_check = to_check || ((iter + 1) % step(iter + 1) == 0);

        upload_halpern_factors(&ws, &restart);
        update_x_z(&ws, batch, to_check);
        update_y(&ws, batch, to_check);
        CUDA_CHECK(cudaStreamSynchronize(ws.stream));

        for (int k = 0; k < batch_size; ++k) {
            if (active[k]) restart.inner[k] += 1;
        }
        if (restarted) {
            std::vector<HPRLP_FLOAT> last_gap = compute_weighted_norm(&ws);
            for (int k = 0; k < batch_size; ++k) {
                if (restart.restart_flag[k] > 0) restart.last_gap[k] = last_gap[k];
            }
        }
    }

    free_batched_workspace(&ws);
    free_batched_lp_device(&batch);
    free_scaling_info(&shared_scaling);
    free_lp_info(&shared_lp);
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

extern "C" void free_batched_results(HPRLP_batched_results *results) {
    if (!results) return;
    if (results->x) std::free(results->x);
    if (results->y) std::free(results->y);
    if (results->z) std::free(results->z);
    if (results->primal_obj) std::free(results->primal_obj);
    if (results->residuals) std::free(results->residuals);
    if (results->gap) std::free(results->gap);
    if (results->iter) std::free(results->iter);
    if (results->status) std::free(results->status);
    *results = HPRLP_batched_results{};
}
