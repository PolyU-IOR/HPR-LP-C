#ifndef HPRLP_CUDA_CHECK_H
#define HPRLP_CUDA_CHECK_H

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>

/**
 * CUDA Error Checking Macros
 * 
 * These macros provide convenient error checking for CUDA, cuBLAS, cuSPARSE,
 * and cuSOLVER API calls. Use these macros to wrap API calls and automatically
 * check for errors.
 */

#define CHECK_kernel(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA API failed at line " << __LINE__ << " with error: " \
                  << cudaGetErrorString(status) << " (" << status << ")\n";    \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "CUSPARSE API failed at line " << __LINE__ << " with error: " \
                  << cusparseGetErrorString(status) << " (" << status << ")\n"; \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        std::cerr << "CUSOLVER API failed at line " << __LINE__ << " with error: " \
                  << cusolverSpGetErrorString(status) << " (" << status << ")\n"; \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::cerr << "CUDA error " << err_ << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::cerr << "cublas error " << err_ << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

#endif /* HPRLP_CUDA_CHECK_H */