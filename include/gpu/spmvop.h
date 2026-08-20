#ifndef HPRLP_GPU_SPMVOP_H
#define HPRLP_GPU_SPMVOP_H

#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <cstddef>

#if !defined(CUDART_VERSION) || CUDART_VERSION < 13030
#error "HPR-LP-C requires CUDA Toolkit 13.3 or newer for cusparseSpMVOp ALG1"
#endif

struct HPRLP_spmvop {
    cusparseSpMVOpDescr_t descriptor = nullptr;
    cusparseSpMVOpPlan_t plan = nullptr;
    cusparseSpMVOpAlg_t algorithm = CUSPARSE_SPMVOP_ALG1;
    std::size_t buffer_size = 0;
    void *buffer = nullptr;
};

inline void hprlp_destroy_spmvop(HPRLP_spmvop *op) {
    if (op == nullptr) return;
    if (op->plan != nullptr) {
        cusparseSpMVOp_destroyPlan(op->plan);
    }
    if (op->descriptor != nullptr) {
        cusparseSpMVOp_destroyDescr(op->descriptor);
    }
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
    cusparseStatus_t status = cusparseSpMVOp_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix, input, addend,
        output, compute_type, op->algorithm, &op->buffer_size);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    if (op->buffer_size > 0) {
        const cudaError_t cuda_status = cudaMalloc(&op->buffer, op->buffer_size);
        if (cuda_status != cudaSuccess) {
            *op = HPRLP_spmvop{};
            return CUSPARSE_STATUS_ALLOC_FAILED;
        }
    }

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
    if (op.plan == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    return cusparseSpMVOp(
        handle, op.plan, alpha, beta, input, addend, output);
}

#endif
