#ifndef HPRLP_STRUCTS_H
#define HPRLP_STRUCTS_H

#include <cstdint>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <limits>
#include <string>
#include <vector>

// Type definitions
#define HPRLP_FLOAT double

// struct of CSR matrix
// We need these values for constructing CUDA CSR sparse matrix through "cusparseCreateCsr".
struct sparseMatrix {
    int row, col;
    int numElements;
    int *colIndex;
    int *rowPtr;
    HPRLP_FLOAT *value;
};

// struct for parameters
struct HPRLP_parameters {
    int max_iter = INT32_MAX;
    HPRLP_FLOAT stop_tol = 1e-4;
    HPRLP_FLOAT time_limit = 3600.0;
    int device_number = 0;
    int check_iter = 150;
    bool CUSPARSE_spmv = false;
    bool autotune_verbose = false;

    /* ----------Scaling Controllers---------- */
    bool use_Ruiz_scaling = true;
    bool use_Pock_Chambolle_scaling = true;
    bool use_bc_scaling = true;
    bool use_presolve = true;
};


// struct for output
struct HPRLP_results {
    HPRLP_FLOAT residuals;
    HPRLP_FLOAT primal_obj;
    HPRLP_FLOAT gap;

    // Default to 'not achive'
    HPRLP_FLOAT time4 = 0.0;          
    HPRLP_FLOAT time6 = 0.0;
    HPRLP_FLOAT time8 = 0.0;
    HPRLP_FLOAT time = 0.0;            
    int iter4 = 0;                                    
    int iter6 = 0;
    int iter8 = 0;
    int iter = 0;  

    char status[64];  // Status string: "OPTIMAL", "TIME_LIMIT", "ITER_LIMIT", "ERROR", etc.

    // Solution vectors (allocated on host)
    HPRLP_FLOAT *x = nullptr;     // Primal solution
    HPRLP_FLOAT *y = nullptr;     // Dual solution
    HPRLP_FLOAT *z = nullptr;     // Bound dual solution
};


struct CUSPARSE_spmv_A {
    cusparseHandle_t cusparseHandle;
    cusparseOperation_t _operator;
    HPRLP_FLOAT alpha;
    HPRLP_FLOAT beta;
    cusparseSpMatDescr_t A_cusparseDescr;
    cusparseDnVecDescr_t x_hat_cusparseDescr;
    cusparseDnVecDescr_t x_bar_cusparseDescr;
    cusparseDnVecDescr_t x_temp_cusparseDescr;
    cusparseDnVecDescr_t Ax_cusparseDescr;
    cudaDataType_t computeType;
    cusparseSpMVAlg_t alg;
    size_t buffersize;
    void *buffer;
};


struct CUSPARSE_spmv_AT {
    cusparseHandle_t cusparseHandle;
    cusparseOperation_t _operator;
    HPRLP_FLOAT alpha;
    HPRLP_FLOAT beta;
    cusparseSpMatDescr_t AT_cusparseDescr;
    cusparseDnVecDescr_t y_bar_cusparseDescr;
    cusparseDnVecDescr_t y_cusparseDescr;
    cusparseDnVecDescr_t ATy_cusparseDescr;
    cudaDataType_t computeType;
    cusparseSpMVAlg_t alg;
    size_t buffersize;
    void *buffer;
};


// struct for GPU workspace
struct HPRLP_workspace_gpu {
    int m, n;
    HPRLP_FLOAT *x, *y, *z;
    HPRLP_FLOAT *last_x, *last_y;
    HPRLP_FLOAT *x_temp, *y_temp;
    HPRLP_FLOAT *x_bar, *y_bar, *z_bar;
    HPRLP_FLOAT *x_hat, *y_hat;
    HPRLP_FLOAT *y_obj;        // The vector y_obj, used for computing the dual objective function variable
    sparseMatrix *A, *AT;
    CUSPARSE_spmv_A *spmv_A;
    CUSPARSE_spmv_AT *spmv_AT;
    HPRLP_FLOAT *AL;
    HPRLP_FLOAT *AU;
    HPRLP_FLOAT *c;
    HPRLP_FLOAT *l;
    HPRLP_FLOAT *u;

    HPRLP_FLOAT sigma;
    HPRLP_FLOAT lambda_max;    // The value of λ_max(AA^T), the maximum eigenvalue of the matrix AA^T

    HPRLP_FLOAT *Rd;           // The vector Rp, normally used to store the vector b-Ax
    HPRLP_FLOAT *Rp;           // The vector Rd, normally used to store the vector c-A^Ty-z

    HPRLP_FLOAT *Ax;
    HPRLP_FLOAT *ATy;

    // Dynamic scalar parameters for GPU kernels:
    // [sigma, lambda_max*sigma, 1/(lambda_max*sigma), 1/sigma]
    HPRLP_FLOAT *Halpern_params = nullptr;

    // Device-side Halpern iteration state.
    int *halpern_inner = nullptr;
    HPRLP_FLOAT *halpern_factors = nullptr;

    // Pinned host mirrors used for lazy scalar uploads.
    HPRLP_FLOAT *iter_params_host = nullptr;
    int *halpern_inner_host = nullptr;
    HPRLP_FLOAT *halpern_factors_host = nullptr;

    // Track the last uploaded runtime scalars to avoid redundant H2D copies.
    HPRLP_FLOAT uploaded_sigma = std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT uploaded_lambda_max = std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();

    cudaGraph_t graph;            // CUDA Graph for capturing the main iteration
    cudaGraphExec_t graph_exec;
    bool graph_initialized=false;

    cudaStream_t stream;

    bool check;                 // Normally used to indicate whether the termination conditions should be checked
    
    bool use_custom_fused_x = false;
    bool use_custom_fused_y = false;

    uint8_t *x_bound_type = nullptr;
    uint8_t *y_bound_type = nullptr;

    int *A_rows_short = nullptr;
    int *A_rows_medium = nullptr;
    int *AT_rows_short = nullptr;
    int *AT_rows_medium = nullptr;

    int num_A_rows_short = 0;
    int num_A_rows_medium = 0;
    int num_AT_rows_short = 0;
    int num_AT_rows_medium = 0;

    cublasHandle_t cublasHandle;

    // 10 slots (0-indexed):
    //  0: dot(Ax, y_temp) — used by compute_weighted_norm and compute_residuals gap
    //  1: dot(y_temp, y_temp) — same
    //  2: dot(x_temp, x_temp) — same
    //  3: Rd nrm2
    //  4: Rp nrm2
    //  5: restart-gap dot(A*x_temp, y_temp)  [compute_residuals compute_gap path]
    //  6: restart-gap dot(y_temp, y_temp)
    //  7: restart-gap dot(x_temp, x_temp)
    //  8: movement nrm2(x_temp)  [for sigma update]
    //  9: movement nrm2(y_temp)  [for sigma update]
    HPRLP_FLOAT *reduction_scalars = nullptr;       // device buffer, 10 elements
    HPRLP_FLOAT *reduction_scalars_host = nullptr;  // pinned host buffer, 10 elements

    // CUBLAS handle configured with CUBLAS_POINTER_MODE_DEVICE for async queued ops.
    cublasHandle_t cublasHandle_device = nullptr;
};


struct HPRLP_restart {
    int restart_flag;           // indicate which restart condition is satisfied, 1: sufficient, 2: necessary, 3: long
    bool first_restart = true;
    HPRLP_FLOAT last_gap = std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT current_gap = std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT save_gap = std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT best_gap = std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT best_sigma;
    int inner = 0;
    int sufficient = 0;
    int necessary = 0;
    int _long = 0;
    int times = 0;
};


struct LP_info_cpu {
    int m, n;
    sparseMatrix *A;  // AT will be generated on GPU from A
    HPRLP_FLOAT *AL;
    HPRLP_FLOAT *AU;
    HPRLP_FLOAT *c;
    HPRLP_FLOAT *l;
    HPRLP_FLOAT *u;
    HPRLP_FLOAT obj_constant;
};


struct LP_info_gpu {
    int m, n;
    sparseMatrix *A, *AT;
    HPRLP_FLOAT *AL;
    HPRLP_FLOAT *AU;
    HPRLP_FLOAT *c;
    HPRLP_FLOAT *l;
    HPRLP_FLOAT *u;
    HPRLP_FLOAT obj_constant;
};


struct HPRLP_residuals {
    HPRLP_FLOAT err_Rp_org_bar;
    HPRLP_FLOAT err_Rd_org_bar;
    HPRLP_FLOAT primal_obj_bar;
    HPRLP_FLOAT dual_obj_bar;
    HPRLP_FLOAT rel_gap_bar;
    bool is_updated;
    HPRLP_FLOAT KKTx_and_gap_org_bar;
};


struct Scaling_info {
    HPRLP_FLOAT *l_org;
    HPRLP_FLOAT *u_org;
    HPRLP_FLOAT *row_norm;
    HPRLP_FLOAT *col_norm;
    HPRLP_FLOAT b_scale;
    HPRLP_FLOAT c_scale;
    HPRLP_FLOAT norm_b;
    HPRLP_FLOAT norm_c;
    HPRLP_FLOAT norm_b_org;
    HPRLP_FLOAT norm_c_org;
};

/**
 * High-level LP data structure with explicit array sizes.
 * This is designed for easy interfacing with Python, Julia, MATLAB.
 * 
 * Arrays are owned by the caller and should NOT be freed by the solver.
 */
struct HPRLP_LP_Data {
    // Problem dimensions
    int m;              // Number of constraints
    int n;              // Number of variables
    int nnz;            // Number of non-zeros in constraint matrix
    
    // Constraint matrix in CSR or CSC format
    int *rowPtr;        // Size: m+1 (CSR) or n+1 (CSC)
    int *colIndex;      // Size: nnz
    HPRLP_FLOAT *values;  // Size: nnz
    bool is_csc;        // true if CSC format, false if CSR format
    
    // Constraint bounds: AL <= A*x <= AU
    HPRLP_FLOAT *AL;    // Size: m (lower bounds)
    HPRLP_FLOAT *AU;    // Size: m (upper bounds)
    
    // Variable bounds: l <= x <= u
    HPRLP_FLOAT *l;     // Size: n (lower bounds)
    HPRLP_FLOAT *u;     // Size: n (upper bounds)
    
    // Objective: minimize c'*x
    HPRLP_FLOAT *c;     // Size: n (objective coefficients)
};

#endif