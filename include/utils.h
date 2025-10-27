#ifndef UTILS_H
#define UTILS_H

#include "structs.h"
#include <chrono>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "cuda_kernels/HPR_cuda_kernels.cuh"
#include "cuda_kernels/cuda_check.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>

// Forward declarations
struct HPRLP_workspace_gpu;
struct HPRLP_results;
struct Scaling_info;

/*-------- Can consider using template --------*/
void set_vector_value_device(HPRLP_FLOAT *x, int n, HPRLP_FLOAT value);

void create_zero_vector_device(HPRLP_FLOAT* &x, int n);

void gen_conceptual_b(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n);


/*----------   1. divide: false result = x ⊙ y     2. divide: true result = x ⊘ y  ---------- */
void vector_dot_product(HPRLP_FLOAT *x, HPRLP_FLOAT *y, HPRLP_FLOAT *result, int n, bool divide);

void diag_AAT(sparseMatrix A, HPRLP_FLOAT *result);

/*----------   1. norm = 99, compute row-wise inf-norm of CSR sparse matrix A.    2. norm = 1, compute l1 norm instead*/
void CSR_A_row_norm(const sparseMatrix *A, HPRLP_FLOAT *result, int norm);

void mul_CSR_A_row(sparseMatrix *A, HPRLP_FLOAT *x, bool divide);         // row-wise scaling

void mul_CSR_AT_row(sparseMatrix *A, HPRLP_FLOAT *x, bool divide);        // col-wise scaling

void transfer_CSR_matrix(const sparseMatrix *A, sparseMatrix* d_A);


/*---------- z = ax (a simple extension to in-place <Dscal> operation) ----------*/
void ax(HPRLP_FLOAT a,  HPRLP_FLOAT *x, HPRLP_FLOAT *z, int n, cublasHandle_t cublasHandle);

/*---------- z = ax + y (a simple extension to in-place <Daxpy> operation) ----------*/
void axpy(HPRLP_FLOAT a, const HPRLP_FLOAT* x, const HPRLP_FLOAT* y, HPRLP_FLOAT* z, int len);

/*---------- z = ax + by ----------*/
void axpby(HPRLP_FLOAT a, const HPRLP_FLOAT *x, HPRLP_FLOAT b, const HPRLP_FLOAT *y, HPRLP_FLOAT *z, int len);

/*---------- return \|x\|_{2} (call cublasDnrm2) ----------*/
HPRLP_FLOAT l2_norm(const HPRLP_FLOAT *x, int n, cublasHandle_t cublasHandle);

/*---------- return <x, y> (call cublasDdot) ----------*/
HPRLP_FLOAT inner_product(const HPRLP_FLOAT *x, const HPRLP_FLOAT *y, int n, cublasHandle_t cublasHandle);

/*. copy device memory. */
void vMemcpy_device(HPRLP_FLOAT *dst, HPRLP_FLOAT *src, int n);

int step(int iter);

/*record device data to files*/
void record_to_file(HPRLP_FLOAT* deviceptr, int len, const std::string filename);

void showDevVec(HPRLP_FLOAT *devPtr, int len);

std::chrono::steady_clock::time_point time_now();

HPRLP_FLOAT time_since(std::chrono::steady_clock::time_point clock_begin);

// Collect solution vectors from GPU workspace to output structure
void collect_solution(HPRLP_workspace_gpu *workspace, Scaling_info *scaling_info, HPRLP_results *output);

// CSR Matrix transpose (host utility)
void CSR_transpose_host(sparseMatrix A, sparseMatrix *AT);

#endif