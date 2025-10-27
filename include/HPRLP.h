#ifndef HPRLP_H 
#define HPRLP_H

/**
 * @file HPRLP.h
 * @brief Main API for HPRLP (Halpern-Peaceman-Rachford Linear Programming) solver
 * 
 * This header provides the primary interface for solving linear programming problems
 * using GPU acceleration with the Halpern-Peaceman-Rachford splitting method.
 * 
 * @author HPRLP Contributors
 * @version 0.1.0
 */

#include "structs.h"        // Data structures for LP problems
#include "scaling.h"        // Matrix/vector scaling algorithms
#include "power_iteration.h"  // Eigenvalue computation
#include "main_iterate.h"   // Core HPR algorithm
#include "utils.h"          // Utility functions
#include "preprocess.h"     // Memory allocation and preprocessing

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Core LP solver using internal LP_info_cpu structure
 * 
 * This is the low-level solver interface that takes pre-processed LP data
 * in the internal LP_info_cpu format. For most users, use the model-based API:
 * create_model_from_arrays() or create_model_from_mps() followed by solve().
 * 
 * @param lp_info_cpu Pointer to LP problem data on CPU
 * @param param Solver parameters (tolerances, iterations, etc.)
 * @return HPRLP_results containing solution, status, and timing information
 * 
 * @note This function initializes the CUDA device automatically based on param->device_number
 * @see create_model_from_arrays(), create_model_from_mps(), solve()
 */
HPRLP_results HPRLP_main_solve(const LP_info_cpu *lp_info_cpu, const HPRLP_parameters *param);

/* ============================================================================
 * New Model-Based API (v0.2+)
 * ============================================================================
 * This is the recommended API for language bindings and advanced users.
 * It separates model construction from solving, providing better control
 * and allowing model reuse with different parameters.
 */

/**
 * @brief Create an LP model from raw arrays
 * 
 * Constructs an LP_info_cpu structure from constraint matrix and bounds arrays.
 * This function allocates memory and performs preprocessing (row deletion, etc.).
 * 
 * The model represents an LP of the form:
 * @code
 *   minimize    c'*x
 *   subject to  AL <= A*x <= AU
 *               l <= x <= u
 * @endcode
 * 
 * @param m Number of constraints (must be > 0)
 * @param n Number of variables (must be > 0)
 * @param nnz Number of non-zero elements in constraint matrix
 * @param rowPtr Row pointer array (CSR format: size m+1) or column pointer (CSC format: size n+1)
 * @param colIndex Column index array (CSR format: size nnz) or row index (CSC format: size nnz)
 * @param values Non-zero values array (size nnz)
 * @param AL Lower bounds for constraints (size m, use -INFINITY for unbounded)
 * @param AU Upper bounds for constraints (size m, use INFINITY for unbounded)
 * @param l Lower bounds for variables (size n, use -INFINITY for unbounded)
 * @param u Upper bounds for variables (size n, use INFINITY for unbounded)
 * @param c Objective function coefficients (size n)
 * @param is_csc If true, input matrix is in CSC format; if false, CSR format (default: false)
 * @return Pointer to LP_info_cpu model, or NULL on error
 * 
 * @note The returned model must be freed using free_model()
 * @note Input arrays are copied internally, caller retains ownership
 * @note Empty rows and unconstrained rows are automatically removed
 * 
 * @code{.cpp}
 * // Example: Create model from arrays
 * int m = 2, n = 2, nnz = 4;
 * int rowPtr[] = {0, 2, 4};
 * int colIndex[] = {0, 1, 0, 1};
 * double values[] = {1.0, 2.0, 3.0, 1.0};
 * double AL[] = {-INFINITY, -INFINITY};
 * double AU[] = {10.0, 12.0};
 * double l[] = {0.0, 0.0};
 * double u[] = {INFINITY, INFINITY};
 * double c[] = {-3.0, -5.0};
 * 
 * LP_info_cpu* model = create_model_from_arrays(m, n, nnz, 
 *                                                rowPtr, colIndex, values,
 *                                                AL, AU, l, u, c, false);
 * if (model) {
 *     // Model created successfully
 *     printf("Model has %d constraints, %d variables\n", model->m, model->n);
 * }
 * @endcode
 * 
 * @see solve(), free_model()
 */
LP_info_cpu* create_model_from_arrays(int m, int n, int nnz,
                                      const int *rowPtr, const int *colIndex, 
                                      const HPRLP_FLOAT *values,
                                      const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                                      const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                                      const HPRLP_FLOAT *c,
                                      bool is_csc = false);

/**
 * @brief Create an LP model from an MPS file
 * 
 * Reads and parses an MPS format file, constructing an LP_info_cpu model.
 * This function handles all file I/O and preprocessing.
 * 
 * @param mps_file_path Path to the MPS format file (null-terminated string)
 * @return Pointer to LP_info_cpu model, or NULL on error
 * 
 * @note The returned model must be freed using free_model()
 * @note Supports both fixed and free MPS format
 * @note Automatically removes empty and unconstrained rows
 * 
 * @code{.cpp}
 * // Example: Create model from MPS file
 * LP_info_cpu* model = create_model_from_mps("problem.mps");
 * if (model) {
 *     printf("Loaded model: %d constraints, %d variables\n", 
 *            model->m, model->n);
 *     printf("Objective constant: %.6f\n", model->obj_constant);
 * } else {
 *     fprintf(stderr, "Failed to load MPS file\n");
 * }
 * @endcode
 * 
 * @see solve(), free_model()
 */
LP_info_cpu* create_model_from_mps(const char* mps_file_path);

/**
 * @brief Solve an LP model with given parameters
 * 
 * Takes a pre-constructed LP model and solves it with the specified parameters.
 * This function can be called multiple times with different parameters on the
 * same model.
 * 
 * @param model Pointer to LP_info_cpu model (created by create_model_*)
 * @param param Solver parameters, or NULL for defaults
 * @return HPRLP_results structure containing solution and statistics
 * 
 * @note The model is not modified by this function
 * @note The caller must free result.x and result.y after use
 * @note If param is NULL, default parameters are used
 * 
 * @code{.cpp}
 * // Example: Solve model with different parameters
 * LP_info_cpu* model = create_model_from_mps("problem.mps");
 * 
 * // Solve with tight tolerance
 * HPRLP_parameters param1;
 * param1.stop_tol = 1e-8;
 * param1.max_iter = 100000;
 * HPRLP_results result1 = solve(model, &param1);
 * 
 * // Solve with different device
 * HPRLP_parameters param2;
 * param2.device_number = 1;
 * HPRLP_results result2 = solve(model, &param2);
 * 
 * // Cleanup
 * free(result1.x); free(result1.y);
 * free(result2.x); free(result2.y);
 * free_model(model);
 * @endcode
 * 
 * @see create_model_from_arrays(), create_model_from_mps(), free_model()
 */
HPRLP_results solve(const LP_info_cpu *model, const HPRLP_parameters *param);

/**
 * @brief Free an LP model
 * 
 * Releases all memory associated with an LP model created by create_model_*.
 * After calling this function, the model pointer is invalid and should not be used.
 * 
 * @param model Pointer to LP_info_cpu model to free, can be NULL (no-op)
 * 
 * @note This function is safe to call with NULL pointer
 * @note Does NOT free result.x and result.y from solve()
 * 
 * @code{.cpp}
 * LP_info_cpu* model = create_model_from_mps("problem.mps");
 * // ... use model ...
 * free_model(model);
 * model = NULL;  // Good practice
 * @endcode
 * 
 * @see create_model_from_arrays(), create_model_from_mps()
 */
void free_model(LP_info_cpu *model);

#ifdef __cplusplus
}
#endif

#endif