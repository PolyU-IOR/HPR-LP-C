#include "HPRLP.h"
#include "io/mps_reader.h"
#include "support/dimension_limits.h"

#include <exception>
#include <iostream>
#include <limits>
#include <vector>

/** Create an LP model from caller-owned arrays, preserving obj_constant. */
LP_info_cpu* create_model_from_arrays_with_obj_constant(
    int m, int n, int nnz,
    const int *rowPtr, const int *colIndex,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const HPRLP_FLOAT *c, HPRLP_FLOAT obj_constant,
    bool is_csc) {
    // Validate inputs
    if (m <= 0 || n <= 0 || nnz <= 0) {
        std::cerr << "[error] Invalid dimensions: m=" << m << ", n=" << n << ", nnz=" << nnz << std::endl;
        return nullptr;
    }

    if (!rowPtr || !colIndex || !values || !AL || !AU || !l || !u || !c) {
        std::cerr << "[error] Null pointer in input arrays" << std::endl;
        return nullptr;
    }

    // Allocate model structure
    // Value-initialize every pointer so a parser failure can be cleaned up safely.
    LP_info_cpu* model = new LP_info_cpu{};
    if (!model) {
        std::cerr << "[error] Failed to allocate model structure" << std::endl;
        return nullptr;
    }

    // Initialize fields
    model->A = nullptr;
    model->m = 0;
    model->n = 0;
    model->obj_constant = 0.0;

    // Create CSRMatrix structure
    CSRMatrix csr_input{};
    if (is_csc) {
        // Convert CSC to CSR: need to transpose
        // CSC format means rowPtr is actually column pointers, colIndex is row indices
        // We need to convert this to CSR format

        // Create a temporary sparseMatrix for CSC (which is CSR of A^T)
        sparseMatrix csc_sparse{};
        csc_sparse.row = n;  // CSC: rows in storage = columns in A^T
        csc_sparse.col = m;  // CSC: cols in storage = rows in A^T
        csc_sparse.numElements = nnz;
        csc_sparse.rowPtr = (int*)malloc((static_cast<std::size_t>(n) + 1) * sizeof(int));
        csc_sparse.colIndex = (int*)malloc(nnz * sizeof(int));
        csc_sparse.value = (HPRLP_FLOAT*)malloc(nnz * sizeof(HPRLP_FLOAT));

        if (!csc_sparse.rowPtr || !csc_sparse.colIndex || !csc_sparse.value) {
            std::cerr << "[error] Memory allocation failed for CSC matrix" << std::endl;
            if (csc_sparse.rowPtr) free(csc_sparse.rowPtr);
            if (csc_sparse.colIndex) free(csc_sparse.colIndex);
            if (csc_sparse.value) free(csc_sparse.value);
            delete model;
            return nullptr;
        }

        memcpy(csc_sparse.rowPtr, rowPtr, (static_cast<std::size_t>(n) + 1) * sizeof(int));
        memcpy(csc_sparse.colIndex, colIndex, nnz * sizeof(int));
        memcpy(csc_sparse.value, values, nnz * sizeof(HPRLP_FLOAT));

        // Transpose to get CSR (transpose of A^T = A)
        sparseMatrix csr_sparse{};
        CSR_transpose_host(csc_sparse, &csr_sparse);

        // Free temporary CSC
        free(csc_sparse.rowPtr);
        free(csc_sparse.colIndex);
        free(csc_sparse.value);

        // Now convert sparseMatrix to CSRMatrix
        csr_input.nrows = m;
        csr_input.ncols = n;
        csr_input.nnz = nnz;
        csr_input.row_ptr = csr_sparse.rowPtr;
        csr_input.row_ptr64 = csr_sparse.rowPtr64;
        csr_input.col_idx = csr_sparse.colIndex;
        csr_input.values = csr_sparse.value;
    } else {
        // Already CSR, just copy
        csr_input.nrows = m;
        csr_input.ncols = n;
        csr_input.nnz = nnz;
        csr_input.row_ptr = (int*)malloc((static_cast<std::size_t>(m) + 1) * sizeof(int));
        csr_input.row_ptr64 = nullptr;
        csr_input.col_idx = (int*)malloc(nnz * sizeof(int));
        csr_input.values = (HPRLP_FLOAT*)malloc(nnz * sizeof(HPRLP_FLOAT));

        if (!csr_input.row_ptr || !csr_input.col_idx || !csr_input.values) {
            std::cerr << "[error] Memory allocation failed for CSR matrix" << std::endl;
            if (csr_input.row_ptr) free(csr_input.row_ptr);
            if (csr_input.col_idx) free(csr_input.col_idx);
            if (csr_input.values) free(csr_input.values);
            delete model;
            return nullptr;
        }

        memcpy(csr_input.row_ptr, rowPtr, (static_cast<std::size_t>(m) + 1) * sizeof(int));
        memcpy(csr_input.col_idx, colIndex, nnz * sizeof(int));
        memcpy(csr_input.values, values, nnz * sizeof(HPRLP_FLOAT));
    }

    // Call build_model_from_arrays to build the model directly (no preprocessing)
    try {
        build_model_from_arrays(
            &csr_input, AL, AU, l, u, c, obj_constant, model);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to build model: " << e.what() << std::endl;
        free(csr_input.row_ptr);
        free(csr_input.col_idx);
        free(csr_input.values);
        delete model;
        return nullptr;
    }

    // Free temporary CSR matrix
    free(csr_input.row_ptr);
    free(csr_input.col_idx);
    free(csr_input.values);

    // Verify model was created successfully
    if (!model->A || model->m <= 0 || model->n <= 0) {
        std::cerr << "[error] Model creation failed" << std::endl;
        free_lp_info_cpu(model);
        delete model;
        return nullptr;
    }

    return model;
}

/** Create an LP model from caller-owned arrays. */
LP_info_cpu* create_model_from_arrays(int m, int n, int nnz,
                                      const int *rowPtr, const int *colIndex,
                                      const HPRLP_FLOAT *values,
                                      const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                      const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                      const HPRLP_FLOAT *c,
                                      bool is_csc) {
    return create_model_from_arrays_with_obj_constant(
        m, n, nnz, rowPtr, colIndex, values,
        AL, AU, l, u, c, 0.0, is_csc);
}


LP_info_cpu* create_model_from_arrays64_with_obj_constant(
    std::int64_t m, std::int64_t n, std::int64_t nnz,
    const std::int64_t *rowPtr, const int *colIndex,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const HPRLP_FLOAT *c, HPRLP_FLOAT obj_constant,
    bool is_csc) {
    if (!hprlp_dimensions_fit_int32(m, n)) {
        return nullptr;
    }
    if (m <= 0 || n <= 0 || nnz <= 0 || !rowPtr || !colIndex ||
        !values || !AL || !AU || !l || !u || !c) {
        std::cerr << "[error] Invalid dimensions or null pointer in 64-bit input"
                  << std::endl;
        return nullptr;
    }
    if (static_cast<std::uint64_t>(nnz) >
        std::numeric_limits<std::size_t>::max() / sizeof(HPRLP_FLOAT)) {
        std::cerr << "[error] nnz exceeds addressable array size"
                  << std::endl;
        return nullptr;
    }
    const int stored_rows = is_csc ? n : m;
    if (rowPtr[0] != 0 || rowPtr[stored_rows] != nnz) {
        std::cerr << "[error] Invalid 64-bit CSR/CSC offsets" << std::endl;
        return nullptr;
    }

    for (int i = 0; i < stored_rows; ++i) {
        if (rowPtr[i] < 0 || rowPtr[i] > rowPtr[i + 1] ||
            rowPtr[i + 1] > nnz) {
            std::cerr << "[error] Invalid 64-bit CSR/CSC offsets"
                      << std::endl;
            return nullptr;
        }
    }

    // Preserve the original implementation and storage for ordinary models.
    if (nnz <= std::numeric_limits<int>::max()) {
        std::vector<int> offsets(static_cast<std::size_t>(stored_rows) + 1);
        for (std::size_t i = 0;
             i <= static_cast<std::size_t>(stored_rows); ++i) {
            if (rowPtr[i] < 0 ||
                rowPtr[i] > std::numeric_limits<int>::max() ||
                (i > 0 && rowPtr[i] < rowPtr[i - 1])) {
                std::cerr << "[error] Invalid 64-bit CSR/CSC offsets"
                          << std::endl;
                return nullptr;
            }
            offsets[i] = static_cast<int>(rowPtr[i]);
        }
        return create_model_from_arrays_with_obj_constant(
            m, n, static_cast<int>(nnz), offsets.data(), colIndex, values,
            AL, AU, l, u, c, obj_constant, is_csc);
    }

    const int index_bound = is_csc ? m : n;
    for (std::int64_t entry = 0; entry < nnz; ++entry) {
        if (colIndex[entry] < 0 || colIndex[entry] >= index_bound) {
            std::cerr << "[error] Invalid CSR/CSC index at entry "
                      << entry << std::endl;
            return nullptr;
        }
    }

    LP_info_cpu *model = new LP_info_cpu{};
    CSRMatrix csr_input{};
    sparseMatrix transposed{};
    try {
        if (is_csc) {
            sparseMatrix csc{};
            csc.row = n;
            csc.col = m;
            csc.numElements = nnz;
            csc.colIndex = const_cast<int *>(colIndex);
            csc.rowPtr64 = const_cast<std::int64_t *>(rowPtr);
            csc.value = const_cast<HPRLP_FLOAT *>(values);
            CSR_transpose_host(csc, &transposed);
            csr_input.nrows = m;
            csr_input.ncols = n;
            csr_input.nnz = nnz;
            csr_input.row_ptr64 = transposed.rowPtr64;
            csr_input.col_idx = transposed.colIndex;
            csr_input.values = transposed.value;
        } else {
            csr_input.nrows = m;
            csr_input.ncols = n;
            csr_input.nnz = nnz;
            csr_input.row_ptr64 = const_cast<std::int64_t *>(rowPtr);
            csr_input.col_idx = const_cast<int *>(colIndex);
            csr_input.values = const_cast<HPRLP_FLOAT *>(values);
        }
        build_model_from_arrays(
            &csr_input, AL, AU, l, u, c, obj_constant, model);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to build 64-bit model: " << e.what()
                  << std::endl;
    }
    free(transposed.rowPtr);
    free(transposed.rowPtr64);
    free(transposed.colIndex);
    free(transposed.value);

    if (!model->A || model->m <= 0 || model->n <= 0) {
        free_lp_info_cpu(model);
        delete model;
        return nullptr;
    }
    return model;
}

LP_info_cpu* create_model_from_arrays64(
    std::int64_t m, std::int64_t n, std::int64_t nnz,
    const std::int64_t *rowPtr, const int *colIndex,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
    const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const HPRLP_FLOAT *c, bool is_csc) {
    return create_model_from_arrays64_with_obj_constant(
        m, n, nnz, rowPtr, colIndex, values, AL, AU, l, u, c, 0.0,
        is_csc);
}


/**
 * Create an LP model from an MPS file
 */
LP_info_cpu* create_model_from_mps(const char* mps_file_path, HPRLP_FLOAT* read_time_out) {
    if (read_time_out) {
        *read_time_out = 0.0;
    }
    // Validate input
    if (!mps_file_path) {
        std::cerr << "[error] Null MPS file path pointer" << std::endl;
        return nullptr;
    }

    // Allocate model structure
    LP_info_cpu* model = new LP_info_cpu;
    if (!model) {
        std::cerr << "[error] Failed to allocate model structure" << std::endl;
        return nullptr;
    }

    // Initialize fields
    model->A = nullptr;
    model->m = 0;
    model->n = 0;

    // Time the complete MPS read/build operation at the public API boundary.
    const auto read_start = time_now();
    try {
        build_model_from_mps(mps_file_path, model);
    } catch (const std::exception& e) {
        const HPRLP_FLOAT read_time = time_since(read_start);
        if (read_time_out) {
            *read_time_out = read_time;
        }
        std::cerr << "[error] Failed to read MPS file: " << e.what() << std::endl;
        delete model;
        return nullptr;
    }
    const HPRLP_FLOAT read_time = time_since(read_start);
    if (read_time_out) {
        *read_time_out = read_time;
    }
    // Verify model was created successfully
    if (!model->A || model->m <= 0 || model->n <= 0) {
        std::cerr << "[error] Invalid model from MPS file" << std::endl;
        free_lp_info_cpu(model);
        delete model;
        return nullptr;
    }

    return model;
}

/**
 * Solve an LP model with given parameters
 */
void free_model(LP_info_cpu *model) {
    if (!model) return;

    // Free LP data structures
    free_lp_info_cpu(model);

    // Free the model structure itself
    delete model;
}
