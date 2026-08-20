#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {

// Mirrors Julia `PresolveParams` in src/presolve/presolve_structs.jl.
struct PresolveParams {
  int max_iters = 10;
  double max_time = __builtin_huge_val();
  bool verbose = false;
  bool debug_checks = false;
  bool trace_enabled = false;
  bool record_postsolve_tape = true;
  bool record_postsolve_tape_cpu = false;

  double feasibility_tol = 1.0e-6;
  double bound_tol = 1.0e-6;
  double zero_tol = 1.0e-10;
  double primal_propagation_min_tighten_abs = 1.0e-2;
  int primal_propagation_max_rounds = 64;
  int primal_propagation_max_bound_only_rounds = 64;
  std::int64_t primal_propagation_bound_only_nnz_round_budget = 8000000;
  bool post_propagation_phase_barrier = true;
  bool doubleton_eq_single_batch_per_iter = false;
  int doubleton_eq_max_fill_in_proxy = 10;
  bool doubleton_eq_scan = true;
  int doubleton_eq_min_selected_per_batch = 256;
  double doubleton_eq_min_selected_ratio = 0.005;
  int doubleton_eq_max_batch_rounds = 0;
  double doubleton_eq_max_time = 0.0;

  bool enable_close_bounds = true;
  bool enable_empty_rows = true;
  bool enable_singleton_rows = true;
  bool enable_activity_checks = true;
  bool enable_primal_propagation = true;
  bool enable_parallel_rows = true;
  bool enable_empty_cols = true;
  bool enable_singleton_cols_eq = true;
  bool enable_singleton_cols_dual_infer = true;
  bool enable_doubleton_eq = true;
  bool enable_linear_eq_agg = false;
  bool enable_dual_fix = true;
  bool enable_parallel_cols = true;
  bool enable_fme_projection = false;
  bool enable_structural_l1_substitution = true;
  double structural_l1_residual_bound_as_free_min = 1.0e8;
  bool enable_redundant_bounds = false;
  bool enable_folding = false;
  double folding_tolerance = 1.0e-8;

  bool use_tiered_scheduler = true;
  bool enable_tiered_bootstrap = true;
  int tiered_cleanup_max_rounds = 2;
  double tiered_light_continue_ratio = 0.995;
  double tiered_cycle_stop_ratio = 0.999;
  int tiered_max_light_streak = 3;
  int tiered_global_period = 2;
};

// Mirrors Julia `LP_info_gpu` in src/core/structs.jl.
struct DeviceCsrMatrix {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::int32_t nnz = 0;
  std::int32_t* rowPtr = nullptr;
  std::int32_t* colVal = nullptr;
  double* nzVal = nullptr;
};

struct LPInfoGpu {
  DeviceCsrMatrix A;
  DeviceCsrMatrix AT;
  double* c = nullptr;
  double* AL = nullptr;
  double* AU = nullptr;
  double* l = nullptr;
  double* u = nullptr;
  double obj_constant = 0.0;
  std::int32_t AT_leading_slack = 0;
  std::int32_t* AT_slack_after = nullptr;
};

// Mirrors Julia `PresolveStats_gpu`.
struct PresolveStatsGpu {
  std::int32_t* row_nnz = nullptr;
  std::uint8_t* empty_row_mask = nullptr;
  std::uint8_t* singleton_row_mask = nullptr;
  std::int32_t* singleton_row_col = nullptr;
  double* singleton_row_val = nullptr;

  std::int32_t* col_nnz = nullptr;
  std::uint8_t* empty_col_mask = nullptr;
  std::uint8_t* singleton_col_mask = nullptr;
  std::int32_t* singleton_col_row = nullptr;
  double* singleton_col_val = nullptr;

  bool row_nnz_valid = false;
  bool col_nnz_valid = false;
  std::int32_t* structural_screen_counts = nullptr;
  std::uint8_t* structural_screen_flag = nullptr;
  std::int32_t structural_screen_pattern_code = -1;
  bool structural_screen_valid = false;
};

struct StructuralL1SplitRecovery {
  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  double rho = 1.0;
};

struct StructuralOuterPairRecovery {
  std::int32_t bound_col = -1;
  std::int32_t free_col = -1;
};

struct StructuralLinkedSlackRecovery {
  std::int32_t slack_col = -1;
  std::int32_t t_col = -1;
  double factor = 0.0;
};

struct StructuralMaxSlackRecovery {
  std::int32_t slack_col = -1;
  std::vector<std::int32_t> t_cols;
  std::vector<double> factors;
};

struct StructuralL1PrimalRecoveryStep {
  std::string pattern;
  std::vector<StructuralL1SplitRecovery> splits;
  std::vector<StructuralOuterPairRecovery> outer_pairs;
  std::vector<StructuralLinkedSlackRecovery> linked_slacks;
  std::vector<StructuralMaxSlackRecovery> max_slacks;
};

enum class PostsolveReductionType : std::int32_t {
  FixedCol = 0,
  FixedColInf = 1,
  SubCol = 2,
  ParallelCol = 3,
  ParallelRow = 4,
  DeletedRow = 5,
  AddedRow = 6,
  AddedRows = 7,
  LhsChange = 8,
  RhsChange = 9,
  EqToIneq = 10,
  BoundChangeNoRow = 11,
  BoundChangeTheRow = 12,
  DoubletonEq = 13,
  FmeCol = 14,
};

enum class PostsolveDualMode : std::uint8_t {
  None = 0,
  Exact = 1,
  Minimal = 2,
};

struct PostsolveTape {
  std::vector<std::int32_t> types;
  std::vector<std::int32_t> index_starts{0};
  std::vector<std::int32_t> value_starts{0};
  std::vector<std::uint8_t> dual_modes;
  std::vector<std::int32_t> indices;
  std::vector<double> vals;
};

struct PostsolveTapeGpu {
  PostsolveTapeGpu() = default;
  ~PostsolveTapeGpu();

  PostsolveTapeGpu(const PostsolveTapeGpu&) = delete;
  PostsolveTapeGpu& operator=(const PostsolveTapeGpu&) = delete;
  PostsolveTapeGpu(PostsolveTapeGpu&& other) noexcept;
  PostsolveTapeGpu& operator=(PostsolveTapeGpu&& other) noexcept;

  void reset() noexcept;

  std::int32_t* types = nullptr;
  std::int32_t* index_starts = nullptr;
  std::int32_t* value_starts = nullptr;
  std::uint8_t* dual_modes = nullptr;
  std::int32_t* indices = nullptr;
  double* vals = nullptr;
  std::int32_t record_count = 0;
  std::int32_t index_count = 0;
  std::int32_t value_count = 0;
  std::int32_t record_capacity = 0;
  std::int32_t index_capacity = 0;
  std::int32_t value_capacity = 0;
};

struct PresolveRecordGpu {
  std::int32_t m0 = 0;
  std::int32_t n0 = 0;
  std::int32_t m1 = 0;
  std::int32_t n1 = 0;

  std::vector<std::int32_t> row_org2red;
  std::vector<std::int32_t> row_red2org;
  std::vector<std::int32_t> col_org2red;
  std::vector<std::int32_t> col_red2org;

  std::vector<std::int32_t> fixed_idx;
  std::vector<double> fixed_val;
  std::vector<std::int32_t> removed_row_idx;
  std::vector<std::int32_t> removed_col_idx;

  double obj_constant_old = 0.0;
  double obj_constant_new = 0.0;

  std::vector<StructuralL1PrimalRecoveryStep> structural_primal_recoveries;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

struct GpuPostsolveResult {
  double* x_org = nullptr;
  double* y_org = nullptr;
  double* z_org = nullptr;
  std::int32_t n0 = 0;
  std::int32_t m0 = 0;
};

// Mirrors Julia `PresolvePlan_gpu` fields used by the LP rules.
struct PresolvePlanGpu {
  std::uint8_t* keep_row_mask = nullptr;
  std::uint8_t* keep_col_mask = nullptr;

  DeviceCsrMatrix new_A;
  bool has_new_A = false;
  std::int32_t new_AT_leading_slack = 0;
  std::int32_t* new_AT_slack_after = nullptr;

  double* new_c = nullptr;
  double* new_AL = nullptr;
  double* new_AU = nullptr;
  double* new_l = nullptr;
  double* new_u = nullptr;
  double obj_constant_delta = 0.0;

  bool has_change = false;
  bool has_row_action = false;
  bool has_col_action = false;
  bool has_infeasible = false;
  bool has_unbounded = false;

  bool has_structural_primal_recovery = false;
  bool tape_gpu_mirrors_cpu = false;
  StructuralL1PrimalRecoveryStep structural_primal_recovery;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

}  // namespace gpu_presolver::presolve
