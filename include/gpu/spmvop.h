#ifndef HPRLP_GPU_SPMVOP_H
#define HPRLP_GPU_SPMVOP_H

#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <cstddef>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13030
#define HPRLP_HAS_CUSPARSE_SPMVOP 1
#else
#define HPRLP_HAS_CUSPARSE_SPMVOP 0
#endif

struct HPRLP_spmvop {
#if HPRLP_HAS_CUSPARSE_SPMVOP
    cusparseSpMVOpDescr_t descriptor = nullptr;
    cusparseSpMVOpPlan_t plan = nullptr;
    cusparseSpMVOpAlg_t algorithm = CUSPARSE_SPMVOP_ALG1;
#else
    // Legacy cuSPARSE keeps the matrix descriptor outside the operation.
    // It remains owned by the workspace that created it.
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_CSR_ALG2;
    cudaDataType_t compute_type = CUDA_R_64F;
#endif
    std::size_t buffer_size = 0;
    void *buffer = nullptr;
};

inline void hprlp_destroy_spmvop(HPRLP_spmvop *op) {
    if (op == nullptr) return;
#if HPRLP_HAS_CUSPARSE_SPMVOP
    if (op->plan != nullptr) {
        cusparseSpMVOp_destroyPlan(op->plan);
    }
    if (op->descriptor != nullptr) {
        cusparseSpMVOp_destroyDescr(op->descriptor);
    }
#endif
    if (op->buffer != nullptr) {
        cudaFree(op->buffer);
    }
    *op = HPRLP_spmvop{};
}

inline cusparseStatus_t hprlp_prepare_spmvop(
        cusparseHandle_t handle,
        cusparseSpMatDescr_t matrix,
        cusparseDnVecDescr_t input,
        cusparseDnVecDescr_t addend,
        cusparseDnVecDescr_t output,
        cudaDataType_t compute_type,
        HPRLP_spmvop *op) {
    if (op == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;

    hprlp_destroy_spmvop(op);
#if HPRLP_HAS_CUSPARSE_SPMVOP
    cusparseStatus_t status = cusparseSpMVOp_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix, input, addend,
        output, compute_type, op->algorithm, &op->buffer_size);
#else
    // cusparseSpMV applies beta to its output in place. Every HPR-LP call
    // uses the output itself as the addend, which is exactly that contract.
    if (addend != output) return CUSPARSE_STATUS_NOT_SUPPORTED;
    op->matrix = matrix;
    op->compute_type = compute_type;

    const float one_f = 1.0f;
    const float zero_f = 0.0f;
    const double one_d = 1.0;
    const double zero_d = 0.0;
    const void *one = compute_type == CUDA_R_32F
        ? static_cast<const void *>(&one_f)
        : static_cast<const void *>(&one_d);
    const void *zero = compute_type == CUDA_R_32F
        ? static_cast<const void *>(&zero_f)
        : static_cast<const void *>(&zero_d);
    cusparseStatus_t status = cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one, matrix, input, zero,
        output, compute_type, op->algorithm, &op->buffer_size);
#endif
    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    if (op->buffer_size > 0) {
        const cudaError_t cuda_status = cudaMalloc(&op->buffer, op->buffer_size);
        if (cuda_status != cudaSuccess) {
            *op = HPRLP_spmvop{};
            return CUSPARSE_STATUS_ALLOC_FAILED;
        }
    }

#if HPRLP_HAS_CUSPARSE_SPMVOP
    status = cusparseSpMVOp_createDescr(
        handle, &op->descriptor, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix,
        input, addend, output, compute_type, op->algorithm, op->buffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        hprlp_destroy_spmvop(op);
        return status;
    }

    status = cusparseSpMVOp_createPlan(
        handle, op->descriptor, &op->plan, nullptr, 0);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        hprlp_destroy_spmvop(op);
    }
#else
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    status = cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one, matrix, input, zero,
        output, compute_type, op->algorithm, op->buffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        hprlp_destroy_spmvop(op);
    }
#endif
#endif
    return status;
}

inline cusparseStatus_t hprlp_run_spmvop(
        cusparseHandle_t handle,
        const HPRLP_spmvop &op,
        const void *alpha,
        const void *beta,
        cusparseDnVecDescr_t input,
        cusparseDnVecDescr_t addend,
        cusparseDnVecDescr_t output) {
#if HPRLP_HAS_CUSPARSE_SPMVOP
    if (op.plan == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    return cusparseSpMVOp(
        handle, op.plan, alpha, beta, input, addend, output);
#else
    if (op.matrix == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (addend != output) return CUSPARSE_STATUS_NOT_SUPPORTED;
    return cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, op.matrix, input,
        beta, output, op.compute_type, op.algorithm, op.buffer);
#endif
}

#endif
