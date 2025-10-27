/**
 * @file constants.h
 * @brief Named constants and configuration parameters for HPRLP
 * 
 * This file defines all magic numbers and configuration constants used
 * throughout the HPRLP library, making the code more maintainable and
 * self-documenting.
 */

#ifndef HPRLP_CONSTANTS_H
#define HPRLP_CONSTANTS_H

namespace hprlp {
namespace constants {

// ============================================================================
// CUDA Configuration
// ============================================================================

/**
 * @brief Number of threads per CUDA block
 * 
 * This value (256) is chosen for optimal occupancy on most modern GPUs.
 * It provides a good balance between:
 * - Register usage per thread
 * - Shared memory availability
 * - Warp scheduling efficiency
 */
constexpr int CUDA_THREADS_PER_BLOCK = 256;

/**
 * @brief Calculate number of blocks needed for n elements
 * @param n Number of elements to process
 * @return Number of CUDA blocks required
 */
inline int cuda_num_blocks(int n) {
    return (n + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
}

// ============================================================================
// Algorithm Parameters
// ============================================================================

/**
 * @brief Number of Ruiz scaling iterations
 * 
 * 10 iterations typically achieve good matrix conditioning without
 * excessive computational cost. Based on:
 * Ruiz, D. (2001). "A scaling algorithm to equilibrate both rows and columns norms in matrices"
 */
constexpr int RUIZ_SCALING_ITERATIONS = 10;

/**
 * @brief Default maximum iterations for power method eigenvalue computation
 * 
 * Power iteration for computing λ_max(AA^T). 50,000 iterations ensures
 * convergence for most practical problems.
 */
constexpr int POWER_METHOD_MAX_ITER = 50000;

/**
 * @brief Default tolerance for power method convergence
 * 
 * Relative change in eigenvalue estimate below this threshold indicates convergence
 */
constexpr double POWER_METHOD_TOLERANCE = 1e-4;

/**
 * @brief Safety factor for eigenvalue estimate
 * 
 * Multiply λ_max by 1.01 to ensure algorithm stability.
 * This slight overestimate prevents numerical issues in step-size selection.
 */
constexpr double EIGENVALUE_SAFETY_FACTOR = 1.01;

// ============================================================================
// Restart Strategy Parameters
// ============================================================================

/**
 * @brief Sufficient decrease ratio for restart
 * 
 * Restart when current_gap <= 0.2 * last_gap (80% improvement)
 */
constexpr double RESTART_SUFFICIENT_RATIO = 0.2;

/**
 * @brief Necessary restart ratio threshold
 * 
 * Restart when gap shows modest improvement: current_gap <= 0.6 * last_gap
 * but also current_gap > 1.0 * save_gap (no improvement over saved gap)
 */
constexpr double RESTART_NECESSARY_RATIO = 0.6;

/**
 * @brief Long iteration restart threshold
 * 
 * Force restart when inner iterations exceed 20% of total iterations,
 * preventing prolonged periods without restart
 */
constexpr double RESTART_LONG_ITERATION_RATIO = 0.2;

// ============================================================================
// Convergence Tolerances
// ============================================================================

/**
 * @brief Convergence tolerance levels for timing checkpoints
 */
constexpr double TOLERANCE_LEVEL_4 = 1e-4;  ///< Standard convergence
constexpr double TOLERANCE_LEVEL_6 = 1e-6;  ///< High precision
constexpr double TOLERANCE_LEVEL_8 = 1e-8;  ///< Very high precision

/**
 * @brief Default convergence tolerance for termination
 */
constexpr double DEFAULT_STOPPING_TOLERANCE = 1e-4;

/**
 * @brief Default time limit in seconds
 */
constexpr double DEFAULT_TIME_LIMIT = 3600.0;

/**
 * @brief Default check interval for convergence testing
 * 
 * Check convergence every 150 iterations to balance between:
 * - Computational overhead of convergence checks
 * - Responsiveness to convergence
 */
constexpr int DEFAULT_CHECK_INTERVAL = 150;

// ============================================================================
// Memory and Data Structure Parameters
// ============================================================================

/**
 * @brief Initial capacity for MPS reader data structures
 * 
 * Pre-allocate space for 8192 variables/constraints to minimize
 * dynamic memory reallocation during MPS file parsing
 */
constexpr int MPS_INITIAL_CAPACITY = 8192;

/**
 * @brief Initial capacity for non-zero elements in MPS reader
 * 
 * Pre-allocate space for 100,000 non-zeros in sparse matrices
 */
constexpr int MPS_INITIAL_NNZ_CAPACITY = 100000;

/**
 * @brief Number of hash buckets for name-index mapping
 * 
 * Power of 2 for efficient modulo operation (4096 buckets)
 */
constexpr int NAME_MAP_HASH_BUCKETS = 4096;

/**
 * @brief Initial capacity for name-index map
 */
constexpr int NAME_MAP_INITIAL_CAPACITY = 1024;

// ============================================================================
// Numerical Constants
// ============================================================================

/**
 * @brief Small epsilon for numerical comparisons
 */
constexpr double NUMERICAL_EPSILON = 1e-12;

/**
 * @brief Infinity representation for unbounded constraints/variables
 */
constexpr double NUMERICAL_INFINITY = 1e20;

// ============================================================================
// Sparse Matrix Format Parameters
// ============================================================================

/**
 * @brief CSR (Compressed Sparse Row) format identifier
 */
constexpr int SPARSE_FORMAT_CSR = 0;

/**
 * @brief CSC (Compressed Sparse Column) format identifier
 */
constexpr int SPARSE_FORMAT_CSC = 1;

} // namespace constants
} // namespace hprlp

// For backwards compatibility, also define macros
#define numThreads hprlp::constants::CUDA_THREADS_PER_BLOCK
#define numBlocks(n) hprlp::constants::cuda_num_blocks(n)

#endif /* HPRLP_CONSTANTS_H */
