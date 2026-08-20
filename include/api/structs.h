#ifndef HPRLP_STRUCTS_H
#define HPRLP_STRUCTS_H

#include <cstdint>
#include <cublas_v2.h>
#include <limits>
#include <string>
#include <vector>

#include "gpu/spmvop.h"
#include "gpu/preprocessing/operators/structured/grid_slack_laplacian_operator.h"
#include "gpu/preprocessing/operators/structured/windowed_stencil_operator.h"

// Type definitions
#define HPRLP_FLOAT double

enum HPRLP_presolve_backend {
    HPRLP_PRESOLVER_PSLP = 0,
    HPRLP_PRESOLVER_GPU = 1,
    HPRLP_PRESOLVER_NONE = 2
};

enum HPRLP_x_bar_mask : std::uint8_t {
    HPRLP_XBAR_AT_LOWER = 1,
    HPRLP_XBAR_AT_UPPER = 2,
    HPRLP_XBAR_INTERIOR = 3
};

struct HPRLP_reduced_matrix_state;

// struct of CSR matrix
// We need these values for constructing CUDA CSR sparse matrix through "cusparseCreateCsr".
struct sparseMatrix {
    int row, col;
    int numElements;
    int *colIndex;
    int *rowPtr;
    HPRLP_FLOAT *value;
};

struct HPRLP_structured_operator_gpu {
    int coefficient_bias = 0;
    bool coefficient_has_escape = false;
    int coefficient_escape_value = 0;

    int dense_row_count = 0;
    int dense_col_count = 0;
    uint16_t *dense_rows = nullptr;
    uint16_t *dense_cols = nullptr;
    uint16_t *dense_local_cols = nullptr;
    uint16_t *dense_A_values_u16 = nullptr;
    uint16_t *dense_AT_values_u16 = nullptr;

    int sparse_row_count = 0;
    uint16_t *sparse_rows = nullptr;
    uint16_t *sparse_col0 = nullptr;
    uint16_t *sparse_col1 = nullptr;
    int8_t *sparse_second_sign = nullptr;

    int short_AT_output_count = 0;
    int short_AT_nonzeros = 0;
    uint16_t *short_AT_output_cols = nullptr;
    int *short_AT_row_ptr = nullptr;
    uint16_t *short_AT_rows = nullptr;
    uint16_t *short_AT_values_u16 = nullptr;
};

struct HPRLP_unit_coltile_gpu {
    int rows = 0;
    int columns = 0;
    int tile_cols = 0;
    int tile_count = 0;
    int *row_tile_offsets = nullptr;
    uint16_t *local_cols = nullptr;
};

struct HPRLP_signed_unit_operator_gpu {
    bool A_uses_u16 = false;
    bool AT_uses_u16 = false;
    bool A_split_u16_ready = false;
    bool AT_split_u16_ready = false;
    uint16_t *A_entries_u16 = nullptr;
    uint32_t *A_entries_u32 = nullptr;
    uint16_t *AT_entries_u16 = nullptr;
    uint32_t *AT_entries_u32 = nullptr;
    uint16_t *A_split_indices_u16 = nullptr;
    uint8_t *A_split_negative_u8 = nullptr;
    uint16_t *AT_split_indices_u16 = nullptr;
    uint8_t *AT_split_negative_u8 = nullptr;
    int A_degree2_run_row_begin = 0;
    int A_degree2_run_row_count = 0;
    int A_degree2_run_entry_begin = 0;
    int AT_degree3_run_row_begin = 0;
    int AT_degree3_run_row_count = 0;
    int AT_degree3_run_entry_begin = 0;
};

struct HPRLP_row_template_matrix_gpu {
    int row_count = 0;
    int template_count = 0;
    int encoded_row_count = 0;
    int maximum_row_degree = 0;
    std::uint8_t *row_template_ids = nullptr;
    int *row_bases = nullptr;
    int *template_ptr = nullptr;
    int *template_offsets = nullptr;
    HPRLP_FLOAT *template_values = nullptr;
    int fallback_short_count = 0;
    int fallback_warp_count = 0;
    int fallback_block_count = 0;
    int *fallback_short_rows = nullptr;
    int *fallback_warp_rows = nullptr;
    int *fallback_block_rows = nullptr;
    int *fallback_row_ptr = nullptr;
    int *fallback_col_indices = nullptr;
    HPRLP_FLOAT *fallback_values = nullptr;
};

struct HPRLP_row_template_operator_gpu {
    HPRLP_row_template_matrix_gpu A;
    HPRLP_row_template_matrix_gpu AT;
};

struct HPRLP_affine_block_matrix_gpu {
    int row_count = 0;
    int block_count = 0;
    int encoded_row_count = 0;
    int maximum_row_degree = 0;
    int *block_row_begin = nullptr;
    int *block_row_count = nullptr;
    int *block_entry_ptr = nullptr;
    int *entry_base_columns = nullptr;
    int *entry_column_strides = nullptr;
    HPRLP_FLOAT *entry_values = nullptr;
    int fallback_short_count = 0;
    int fallback_warp_count = 0;
    int fallback_block_count = 0;
    int *fallback_short_rows = nullptr;
    int *fallback_warp_rows = nullptr;
    int *fallback_block_rows = nullptr;
    int *fallback_row_ptr = nullptr;
    int *fallback_col_indices = nullptr;
    HPRLP_FLOAT *fallback_values = nullptr;
};

struct HPRLP_affine_block_operator_gpu {
    HPRLP_affine_block_matrix_gpu A;
    HPRLP_affine_block_matrix_gpu AT;
};

// Exact, lossless records for the static vectors consumed by the specialized
// observation/stencil normal kernels.  Each model entry stores one byte; the
// small record tables retain the original binary64 values verbatim.
struct HPRLP_factorized_x_static_record {
    HPRLP_FLOAT lower;
    HPRLP_FLOAT upper;
    HPRLP_FLOAT objective;
    std::uint8_t bound_type;
    std::uint8_t padding[7] = {};
};

struct HPRLP_factorized_y_static_record {
    HPRLP_FLOAT lower;
    HPRLP_FLOAT upper;
    std::uint8_t bound_type;
    std::uint8_t padding[7] = {};
};

enum class HPRLPPackedDictionaryStorage : uint8_t {
    None,
    PackedU32,
    SeparateU16U8,
    SeparateU16U16,
    SeparateU32U8,
    SeparateU32U16
};

struct HPRLPPackedStatePlan {
    int x_zero_lower_boxed_begin = 0;
    int x_zero_lower_boxed_count = 0;
    int x_objective_zero_begin = 0;
    int x_objective_zero_count = 0;
    int y_lower_only_begin = 0;
    int y_lower_only_count = 0;
    int y_upper_only_begin = 0;
    int y_upper_only_count = 0;
    int y_equality_begin = 0;
    int y_equality_count = 0;
};

constexpr int HPRLP_FIXED_DEGREE_MAX_RUNS = 8;

struct HPRLPFixedDegreeRun {
    int row_begin = 0;
    int row_count = 0;
    int entry_begin = 0;
    int degree = 0;
    std::uint32_t *packed_entries_soa = nullptr;
};

// A matrix-derived plan for long contiguous runs whose rows all have the
// same small degree.  The run kernels derive each CSR entry range from four
// scalars, so normal iterations do not read row pointers or row-id arrays for
// the covered rows.  The three fallback lists form an exact disjoint cover of
// every row not represented by a run.
struct HPRLPFixedDegreeRunPlan {
    bool ready = false;
    int threads = 256;
    int run_count = 0;
    HPRLPFixedDegreeRun runs[HPRLP_FIXED_DEGREE_MAX_RUNS]{};
    long long covered_rows = 0;
    long long covered_nnz = 0;
    int fallback_short_count = 0;
    int fallback_warp_count = 0;
    int fallback_block_count = 0;
    int *fallback_short_rows = nullptr;
    int *fallback_warp_rows = nullptr;
    int *fallback_block_rows = nullptr;
};

enum class HPRLPXBackend : uint8_t {
    ScaledCusparse,
    GenericFused,
    UnitFactorized,
    SignedUnitPacked,
    SignedUnitSplitU16,
    PackedDictionary,
    FixedDegreePackedDictionary,
    StructuredOriginal,
    FactorizedStencil,
    GridSlackLaplacian,
    RowTemplate,
    AffineBlock
};

enum class HPRLPYBackend : uint8_t {
    ScaledCusparse,
    GenericFused,
    SegmentedFused,
    UnitFactorized,
    SignedUnitPacked,
    SignedUnitPackedCombined,
    StructuredOriginal,
    FactorizedStencil,
    UnitColTile,
    UnitColTileZeroBitset,
    UnitActiveScatter,
    GridSlackLaplacian,
    PackedDictionary,
    FixedDegreePackedDictionary,
    RowTemplate,
    AffineBlock
};

// struct for parameters
struct HPRLP_parameters {
    int max_iter = INT32_MAX;
    HPRLP_FLOAT stop_tol = 1e-6;
    HPRLP_FLOAT time_limit = 1000.0;
    int device_number = 0;
    int check_iter = 150;
    bool CUSPARSE_spmv = false;
    bool autotune_verbose = false;

    bool enable_progress_monitor = true;
    bool enable_progress_control = true;
    bool enable_sigma_rebalance_restart = true;
    bool use_progress_restart_guard = false;
    int restart_cooldown_checks = 0;
    bool debug_restart = false;
    bool debug_sigma = false;
    HPRLP_FLOAT fixed_sigma =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();

    /* ----------Scaling Controllers---------- */
    bool use_CR_scaling = true;
    bool use_Ruiz_scaling = true;
    bool use_Pock_Chambolle_scaling = true;
    bool use_bc_scaling = true;
    bool use_presolve = true;
    bool enable_gpu_folding = true;
    HPRLP_presolve_backend presolver = HPRLP_PRESOLVER_GPU;
    bool use_reduced_matrix = true;
    bool auto_reduced_compression_policy = false;

    // Keep normal logs concise unless detailed implementation diagnostics are
    // explicitly requested.  specified_parameter_mask is populated by CLI
    // frontends so quiet logs can still echo options supplied by the user.
    bool print_debug_info = false;
    std::uint64_t specified_parameter_mask = 0;
};

enum HPRLP_parameter_print_bit : std::uint64_t {
    HPRLP_PARAM_MAX_ITER = 1ULL << 0,
    HPRLP_PARAM_STOP_TOL = 1ULL << 1,
    HPRLP_PARAM_TIME_LIMIT = 1ULL << 2,
    HPRLP_PARAM_DEVICE = 1ULL << 3,
    HPRLP_PARAM_CHECK_ITER = 1ULL << 4,
    HPRLP_PARAM_CUSPARSE_SPMV = 1ULL << 5,
    HPRLP_PARAM_AUTOTUNE_VERBOSE = 1ULL << 6,
    HPRLP_PARAM_PROGRESS_MONITOR = 1ULL << 7,
    HPRLP_PARAM_PROGRESS_CONTROL = 1ULL << 8,
    HPRLP_PARAM_SIGMA_REBALANCE = 1ULL << 9,
    HPRLP_PARAM_RESTART_GUARD = 1ULL << 10,
    HPRLP_PARAM_RESTART_COOLDOWN = 1ULL << 11,
    HPRLP_PARAM_DEBUG_RESTART = 1ULL << 12,
    HPRLP_PARAM_DEBUG_SIGMA = 1ULL << 13,
    HPRLP_PARAM_FIXED_SIGMA = 1ULL << 14,
    HPRLP_PARAM_CR_SCALING = 1ULL << 15,
    HPRLP_PARAM_RUIZ_SCALING = 1ULL << 16,
    HPRLP_PARAM_POCK_SCALING = 1ULL << 17,
    HPRLP_PARAM_BC_SCALING = 1ULL << 18,
    HPRLP_PARAM_PRESOLVER = 1ULL << 19,
    HPRLP_PARAM_GPU_FOLDING = 1ULL << 20,
    HPRLP_PARAM_REDUCED_MATRIX = 1ULL << 21,
    HPRLP_PARAM_AUTO_MEMORY_POLICY = 1ULL << 22,
    HPRLP_PARAM_PRINT_DEBUG_INFO = 1ULL << 23,
};


// Timing breakdown for a single HPRLP solve.
struct HPRLP_time {
    HPRLP_FLOAT total_time = 0.0;
    HPRLP_FLOAT presolve_time = 0.0;
    HPRLP_FLOAT setup_time = 0.0;
    HPRLP_FLOAT scaling_time = 0.0;
    HPRLP_FLOAT analyze_time = 0.0;
    HPRLP_FLOAT power_iteration_time = 0.0;
    HPRLP_FLOAT solve_time = 0.0;
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
    HPRLP_FLOAT folding_time = 0.0;
    HPRLP_FLOAT presolve_time = 0.0;
    int iter4 = 0;                                    
    int iter6 = 0;
    int iter8 = 0;
    int iter = 0;  

    char status[64];  // Status string: "OPTIMAL", "TIME_LIMIT", "ITER_LIMIT", "ERROR", etc.

    // Solution vectors (allocated on host)
    HPRLP_FLOAT *x = nullptr;     // Primal solution
    HPRLP_FLOAT *y = nullptr;     // Dual solution
    HPRLP_FLOAT *z = nullptr;     // Bound dual solution

    HPRLP_time timing;

    HPRLP_FLOAT interior_percentage = 100.0;
    int reduced_activation_checks = 0;
    int reduced_active_iterations = 0;
    int reduced_first_iteration = -1;
    int reduced_rebuilds = 0;
    HPRLP_FLOAT reduced_build_time = 0.0;
    int reduced_last_trigger_iteration = -1;
    HPRLP_FLOAT reduced_last_free_ratio = 1.0;
    HPRLP_FLOAT reduced_last_trigger_residual =
        std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT reduced_last_trigger_sigma =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    int reduced_last_free_columns = 0;
    HPRLP_FLOAT reduced_active_iteration_ratio = 0.0;
    HPRLP_FLOAT reduced_average_column_ratio = 1.0;
    HPRLP_FLOAT reduced_average_nnz_ratio = 1.0;
    HPRLP_FLOAT reduced_minimum_column_ratio = 1.0;
    HPRLP_FLOAT reduced_minimum_nnz_ratio = 1.0;
};


struct HPRLP_batched_results {
    int m = 0;
    int n = 0;
    int batch_size = 0;

    // Column-major host arrays: x/z are n x batch_size, y is m x batch_size.
    HPRLP_FLOAT *x = nullptr;
    HPRLP_FLOAT *y = nullptr;
    HPRLP_FLOAT *z = nullptr;

    HPRLP_FLOAT *primal_obj = nullptr;
    HPRLP_FLOAT *residuals = nullptr;
    HPRLP_FLOAT *gap = nullptr;
    int *iter = nullptr;

    // batch_size contiguous status strings, each 64 bytes.
    char *status = nullptr;

    HPRLP_FLOAT time = 0.0;
    HPRLP_FLOAT setup_time = 0.0;
    HPRLP_FLOAT solve_time = 0.0;
    HPRLP_FLOAT power_time = 0.0;
};


struct CUSPARSE_spmvop_A {
    cusparseHandle_t cusparseHandle;
    HPRLP_FLOAT alpha;
    HPRLP_FLOAT beta;
    cusparseSpMatDescr_t A_cusparseDescr;
    cusparseDnVecDescr_t x_hat_cusparseDescr;
    cusparseDnVecDescr_t x_bar_cusparseDescr;
    cusparseDnVecDescr_t x_temp_cusparseDescr;
    cusparseDnVecDescr_t Ax_cusparseDescr;
    cudaDataType_t computeType;
    HPRLP_spmvop operation;
    cusparseSpMatDescr_t unit_A_cusparseDescr = nullptr;
    cusparseDnVecDescr_t unit_x_hat_cusparseDescr = nullptr;
    HPRLP_spmvop unit_operation;
};


struct CUSPARSE_spmvop_AT {
    cusparseHandle_t cusparseHandle;
    HPRLP_FLOAT alpha;
    HPRLP_FLOAT beta;
    cusparseSpMatDescr_t AT_cusparseDescr;
    cusparseDnVecDescr_t y_bar_cusparseDescr;
    cusparseDnVecDescr_t y_cusparseDescr;
    cusparseDnVecDescr_t ATy_cusparseDescr;
    cudaDataType_t computeType;
    HPRLP_spmvop operation;
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
    CUSPARSE_spmvop_A *spmv_A;
    CUSPARSE_spmvop_AT *spmv_AT;
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
    HPRLP_FLOAT *halpern_factor_batch = nullptr;
    // Graph capture snapshots kernel arguments. The batched graph points each
    // update pair at a precomputed factor pair, so it suppresses the ordinary
    // per-iteration factor-advance kernel during capture.

    // Pinned host mirrors used for lazy scalar uploads.
    HPRLP_FLOAT *iter_params_host = nullptr;
    int *halpern_inner_host = nullptr;
    HPRLP_FLOAT *halpern_factors_host = nullptr;

    // Track the last uploaded runtime scalars to avoid redundant H2D copies.
    HPRLP_FLOAT uploaded_sigma = std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT uploaded_lambda_max = std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();

    cudaGraph_t graph;            // CUDA Graph for capturing the main iteration
    cudaGraphExec_t graph_exec;
    cudaGraph_t graph_batch;
    cudaGraphExec_t graph_exec_batch;
    bool graph_initialized=false;
    // The canonical full-state check update always uses the same cuSPARSE
    // operators and vector storage.  Keep a separate graph so reduced-mode
    // checkpoints do not repeatedly submit the same five operations.
    cudaGraph_t check_graph;
    cudaGraphExec_t check_graph_exec;
    bool check_graph_initialized=false;

    cudaStream_t stream;

    bool check;                 // Normally used to indicate whether the termination conditions should be checked
    
    HPRLPXBackend x_backend = HPRLPXBackend::ScaledCusparse;
    HPRLPYBackend y_backend = HPRLPYBackend::ScaledCusparse;
    // Compact reduced iterations choose between their generic fused and
    // cuSPARSE implementations independently of the full-state backend.
    bool reduced_use_fused_x = false;
    bool reduced_use_fused_y = false;
    bool reduced_backend_autotune_done = false;

    uint8_t *x_bound_type = nullptr;
    uint8_t *y_bound_type = nullptr;

    int *A_rows_short = nullptr;
    int *A_rows_medium = nullptr;
    int *A_rows_long = nullptr;
    int *AT_rows_short = nullptr;
    int *AT_rows_medium = nullptr;
    int *AT_rows_long = nullptr;

    int num_A_rows_short = 0;
    int num_A_rows_medium = 0;
    int num_A_rows_long = 0;
    int num_AT_rows_short = 0;
    int num_AT_rows_medium = 0;
    int num_AT_rows_long = 0;
    int max_A_row_nnz = 0;
    int max_AT_row_nnz = 0;

    // General load-balancing plan for highly skewed A rows.  Rows are
    // admitted only from their CSR lengths; model names and fingerprints are
    // never consulted.  Each admitted row is split into fixed, disjoint CSR
    // segments, reduced to deterministic partial sums, then finalized in a
    // second kernel.  Other long rows retain the ordinary one-block path.
    bool segmented_A_ready = false;
    int segmented_A_threshold = 0;
    int segmented_A_tile_entries = 0;
    int segmented_A_threads = 0;
    int segmented_A_row_count = 0;
    int segmented_A_tile_count = 0;
    int segmented_A_fallback_long_count = 0;
    long long segmented_A_nnz = 0;
    int *segmented_A_rows = nullptr;
    int *segmented_A_row_tile_ptr = nullptr;
    int *segmented_A_tile_begin = nullptr;
    int *segmented_A_tile_end = nullptr;
    int *segmented_A_fallback_long_rows = nullptr;
    HPRLP_FLOAT *segmented_A_partials = nullptr;

    bool segmented_AT_ready = false;
    int segmented_AT_threshold = 0;
    int segmented_AT_tile_entries = 0;
    int segmented_AT_threads = 0;
    int segmented_AT_row_count = 0;
    int segmented_AT_tile_count = 0;
    int segmented_AT_fallback_long_count = 0;
    long long segmented_AT_nnz = 0;
    int *segmented_AT_rows = nullptr;
    int *segmented_AT_row_tile_ptr = nullptr;
    int *segmented_AT_tile_begin = nullptr;
    int *segmented_AT_tile_end = nullptr;
    int *segmented_AT_fallback_long_rows = nullptr;
    HPRLP_FLOAT *segmented_AT_partials = nullptr;

    HPRLPFixedDegreeRunPlan fixed_degree_A_plan{};
    HPRLPFixedDegreeRunPlan fixed_degree_AT_plan{};

    bool all_positive_unit_coefficients = false;
    int8_t uniform_unit_sign = 0;
    bool all_zero_lower_unbounded_variables = false;
    bool unit_operator_x_ready = false;
    bool unit_operator_y_ready = false;
    bool signed_unit_operator_ready = false;
    // Runtime-certified state specialization for large signed-unit models.
    // The plan is valid only when every x variable is boxed with raw +0
    // lower bound, c has a long raw +0 run, and a long y-row run is
    // upper-bounded by raw +0.  Normal signed kernels may then omit the
    // corresponding redundant vector reads; canonical check kernels remain
    // unchanged.
    bool signed_state_plan_ready = false;
    bool signed_state_specialization_enabled = false;
    int signed_x_zero_objective_run_begin = 0;
    int signed_x_zero_objective_run_count = 0;
    int signed_y_upper_zero_run_begin = 0;
    int signed_y_upper_zero_run_count = 0;
    // Selected signed/signed normal iterations produce the transformed
    // x-hat cache consumed by signed Y directly.  In that mode the ordinary
    // x_hat array is reconstructed only by the canonical check path.
    bool signed_single_state_enabled = false;
    bool signed_zero_skip_monitor_enabled = false;
    bool signed_zero_skip_enabled = false;
    bool unit_coltile_ready = false;
    bool dictionary_operator_x_ready = false;
    bool dictionary_operator_y_ready = false;
    bool structured_operator_ready = false;
    bool row_template_operator_ready = false;
    bool affine_block_operator_ready = false;
    bool windowed_stencil_operator_ready = false;
    HPRLPWindowedStencilShape windowed_stencil_shape{};
    // Certified only from the scaled bound/objective arrays and the inferred
    // windowed-stencil partition.  It permits a branch-free normal kernel;
    // model names, paths, and fingerprints are not inputs.
    bool windowed_stencil_state_ready = false;
    std::uint8_t windowed_observation_first_bound_type = 0;
    std::uint8_t windowed_observation_second_bound_type = 0;
    bool grid_slack_laplacian_operator_ready = false;
    HPRLPGridSlackLaplacianShape grid_slack_laplacian_shape{};
    HPRLP_structured_operator_gpu *structured_operator = nullptr;
    HPRLP_row_template_operator_gpu *row_template_operator = nullptr;
    HPRLP_affine_block_operator_gpu *affine_block_operator = nullptr;
    HPRLP_unit_coltile_gpu *unit_coltile = nullptr;
    HPRLP_signed_unit_operator_gpu *signed_unit_operator = nullptr;
    int coefficient_dictionary_size = 0;
    HPRLP_FLOAT *coefficient_dictionary = nullptr;
    uint8_t *AT_value_codes = nullptr;
    HPRLPPackedDictionaryStorage packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    unsigned packed_dictionary_index_bits = 0;
    unsigned packed_dictionary_code_bits = 0;
    uint32_t *AT_dictionary_packed_u32 = nullptr;
    uint16_t *AT_dictionary_indices_u16 = nullptr;
    uint32_t *AT_dictionary_indices_u32 = nullptr;
    uint8_t *AT_dictionary_codes_u8 = nullptr;
    uint16_t *AT_dictionary_codes_u16 = nullptr;
    int A_coefficient_dictionary_size = 0;
    HPRLP_FLOAT *A_coefficient_dictionary = nullptr;
    HPRLPPackedDictionaryStorage A_packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    unsigned A_packed_dictionary_code_bits = 0;
    uint32_t *A_dictionary_packed_u32 = nullptr;
    uint32_t *A_dictionary_indices_u32 = nullptr;
    uint16_t *A_dictionary_codes_u16 = nullptr;
    HPRLPPackedStatePlan packed_state_plan{};
    uint16_t *unit_AT_col_index_u16 = nullptr;
    uint16_t *A_col_index_u16 = nullptr;
    HPRLP_FLOAT *inverse_row_norm = nullptr;
    HPRLP_FLOAT *inverse_col_norm = nullptr;
    HPRLP_FLOAT *factor_row_norm = nullptr;
    HPRLP_FLOAT *factor_col_norm = nullptr;
    HPRLP_FLOAT *unit_scaled_y = nullptr;
    HPRLP_FLOAT *unit_A_values = nullptr;
    HPRLP_FLOAT *unit_scaled_x_hat = nullptr;
    HPRLP_FLOAT *factorized_stencil_dense_partials = nullptr;
    unsigned int *factorized_stencil_dense_counter = nullptr;
    bool factorized_last_block_enabled = false;
    bool factorized_static_records_ready = false;
    int factorized_x_static_record_count = 0;
    int factorized_y_static_record_count = 0;
    std::uint8_t *factorized_x_static_codes = nullptr;
    std::uint8_t *factorized_y_static_codes = nullptr;
    HPRLP_factorized_x_static_record *factorized_x_static_records = nullptr;
    HPRLP_factorized_y_static_record *factorized_y_static_records = nullptr;
    std::uint32_t *unit_scaled_x_zero_bits = nullptr;
    unsigned long long *unit_scaled_x_positive_zero_count = nullptr;
    // 1 means the transformed value is not raw +0.  Raw -0 is deliberately
    // active so zero skipping cannot erase its sign semantics.
    uint8_t *unit_scaled_x_hat_nonzero = nullptr;
    unsigned long long *signed_xhat_positive_zero_count = nullptr;
    unsigned long long *signed_xhat_positive_zero_count_host = nullptr;

    cublasHandle_t cublasHandle;

    // 14 slots (0-indexed):
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
    // 10: progress dot(dx_k, dx_(k-1))
    // 11: progress dot(dy_k, dy_(k-1))
    // 12: progress dot(dx_k, dx_k)
    // 13: progress dot(dy_k, dy_k)
    HPRLP_FLOAT *reduction_scalars = nullptr;       // device buffer, 14 elements
    HPRLP_FLOAT *reduction_scalars_host = nullptr;  // pinned host buffer, 14 elements

    // CUBLAS handle configured with CUBLAS_POINTER_MODE_DEVICE for async queued ops.
    cublasHandle_t cublasHandle_device = nullptr;

    // Allocated only when the runtime reduced-column path is enabled.
    HPRLP_reduced_matrix_state *reduced_matrix = nullptr;
};


struct HPRLP_restart {
    int restart_flag;           // indicate which restart condition is satisfied, 1: sufficient, 2: necessary, 3: long
    bool first_restart = true;
    // Edge-trigger the feasibility-converged diagnostic once per region.
    bool sigma_feasibility_skip_reported = false;
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
    bool all_positive_unit_coefficients = false;
    int8_t uniform_unit_sign = 0;
    bool mixed_signed_unit_coefficients = false;
    bool all_zero_lower_unbounded_variables = false;
    bool signed_state_plan_ready = false;
    int signed_x_zero_objective_run_begin = 0;
    int signed_x_zero_objective_run_count = 0;
    int signed_y_upper_zero_run_begin = 0;
    int signed_y_upper_zero_run_count = 0;
    bool has_original_coefficient_dictionary = false;
    int coefficient_dictionary_size = 0;
    HPRLP_FLOAT *coefficient_dictionary = nullptr;
    uint8_t *AT_value_codes = nullptr;
    HPRLPPackedDictionaryStorage packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    unsigned packed_dictionary_index_bits = 0;
    unsigned packed_dictionary_code_bits = 0;
    uint32_t *AT_dictionary_packed_u32 = nullptr;
    uint16_t *AT_dictionary_indices_u16 = nullptr;
    uint32_t *AT_dictionary_indices_u32 = nullptr;
    uint8_t *AT_dictionary_codes_u8 = nullptr;
    uint16_t *AT_dictionary_codes_u16 = nullptr;
    int A_coefficient_dictionary_size = 0;
    HPRLP_FLOAT *A_coefficient_dictionary = nullptr;
    HPRLPPackedDictionaryStorage A_packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    unsigned A_packed_dictionary_code_bits = 0;
    uint32_t *A_dictionary_packed_u32 = nullptr;
    uint32_t *A_dictionary_indices_u32 = nullptr;
    uint16_t *A_dictionary_codes_u16 = nullptr;
    HPRLPPackedStatePlan packed_state_plan{};
    HPRLP_structured_operator_gpu *structured_operator = nullptr;
    HPRLP_row_template_operator_gpu *row_template_operator = nullptr;
    HPRLP_affine_block_operator_gpu *affine_block_operator = nullptr;
    bool windowed_stencil_operator = false;
    HPRLPWindowedStencilShape windowed_stencil_shape{};
    bool grid_slack_laplacian_operator = false;
    HPRLPGridSlackLaplacianShape grid_slack_laplacian_shape{};
    HPRLP_unit_coltile_gpu *unit_coltile = nullptr;
    HPRLP_signed_unit_operator_gpu *signed_unit_operator = nullptr;
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
