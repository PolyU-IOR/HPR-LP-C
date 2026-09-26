#ifndef HPRLP_GPU_SPMVOP_H
#define HPRLP_GPU_SPMVOP_H

#include <cuda_runtime_api.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13030
#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#endif

#include <cusparse.h>
#include <cstdint>

#include <cstddef>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13030
#define HPRLP_HAS_CUSPARSE_SPMVOP 1
#define HPRLP_CUSPARSE_BACKEND_NAME "cusparseSpMVOp ALG1"
#else
#define HPRLP_HAS_CUSPARSE_SPMVOP 0
#define HPRLP_CUSPARSE_BACKEND_NAME "cusparseSpMV CSR ALG2"
#endif

struct HPRLP_spmvop {
#if HPRLP_HAS_CUSPARSE_SPMVOP
    cusparseSpMVOpDescr_t descriptor = nullptr;
    cusparseSpMVOpPlan_t plan = nullptr;
    cusparseSpMVOpAlg_t algorithm = CUSPARSE_SPMVOP_ALG1;
#endif
    // The legacy path also handles CSR index pairs unsupported by SpMVOp,
    // notably 64-bit offsets with 64-bit column IDs.
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseSpMVAlg_t legacy_algorithm = CUSPARSE_SPMV_CSR_ALG2;
    cudaDataType_t compute_type = CUDA_R_64F;
    bool use_legacy_spmv = !HPRLP_HAS_CUSPARSE_SPMVOP;
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
    op->matrix = matrix;
    op->compute_type = compute_type;
#if HPRLP_HAS_CUSPARSE_SPMVOP
    std::int64_t rows = 0;
    std::int64_t columns = 0;
    std::int64_t nonzeros = 0;
    void *row_offsets = nullptr;
    void *column_indices = nullptr;
    void *values = nullptr;
    cusparseIndexType_t offsets_type = CUSPARSE_INDEX_32I;
    cusparseIndexType_t indices_type = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType_t value_type = CUDA_R_64F;
    cusparseStatus_t status = cusparseCsrGet(
        matrix, &rows, &columns, &nonzeros, &row_offsets, &column_indices,
        &values, &offsets_type, &indices_type, &index_base, &value_type);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    // SpMVOp accepts 32/32 and 64/32, but not 64/64. cuSPARSE SpMV
    // accepts 64/64 and is used for that pair even on newer toolkits.
    op->use_legacy_spmv =
        !((offsets_type == CUSPARSE_INDEX_32I &&
           indices_type == CUSPARSE_INDEX_32I) ||
          (offsets_type == CUSPARSE_INDEX_64I &&
           indices_type == CUSPARSE_INDEX_32I));
#else
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
#endif

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

    if (op->use_legacy_spmv) {
        // cusparseSpMV applies beta to its output in place.
        if (addend != output) return CUSPARSE_STATUS_NOT_SUPPORTED;
        status = cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one, matrix, input,
            zero, output, compute_type, op->legacy_algorithm,
            &op->buffer_size);
    } else {
#if HPRLP_HAS_CUSPARSE_SPMVOP
        status = cusparseSpMVOp_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix, input, addend,
            output, compute_type, op->algorithm, &op->buffer_size);
#else
        return CUSPARSE_STATUS_NOT_SUPPORTED;
#endif
    }
    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    if (op->buffer_size > 0) {
        const cudaError_t cuda_status = cudaMalloc(&op->buffer, op->buffer_size);
        if (cuda_status != cudaSuccess) {
            hprlp_destroy_spmvop(op);
            return CUSPARSE_STATUS_ALLOC_FAILED;
        }
    }

    if (op->use_legacy_spmv) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
        status = cusparseSpMV_preprocess(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one, matrix, input,
            zero, output, compute_type, op->legacy_algorithm, op->buffer);
        if (status != CUSPARSE_STATUS_SUCCESS) hprlp_destroy_spmvop(op);
#endif
        return status;
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
    if (op.use_legacy_spmv) {
        if (op.matrix == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
        if (addend != output) return CUSPARSE_STATUS_NOT_SUPPORTED;
        return cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, op.matrix,
            input, beta, output, op.compute_type, op.legacy_algorithm,
            op.buffer);
    }
#if HPRLP_HAS_CUSPARSE_SPMVOP
    if (op.plan == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    return cusparseSpMVOp(
        handle, op.plan, alpha, beta, input, addend, output);
#else
    return CUSPARSE_STATUS_NOT_SUPPORTED;
#endif
}

#endif
