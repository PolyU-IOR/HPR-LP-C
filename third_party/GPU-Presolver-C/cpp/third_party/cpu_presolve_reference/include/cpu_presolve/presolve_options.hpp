#pragma once

namespace cpu_presolve {

enum class PresolveScheduler {
  kFixed,
  kTiered,
};

struct PresolveOptions {
  PresolveScheduler scheduler = PresolveScheduler::kTiered;
  int max_iterations = 10;
  double bound_tolerance = 1.0e-6;
  double feasibility_tolerance = 1.0e-6;
  double zero_tolerance = 1.0e-10;
  bool enable_close_bounds = true;
  bool enable_empty_rows = true;
  bool enable_singleton_rows = true;
  bool enable_activity_checks = true;
  bool enable_primal_propagation = true;
  bool enable_parallel_rows = true;
  bool enable_dual_fix = true;
  bool enable_empty_cols = true;
  bool enable_singleton_cols_dual_infer = true;
  bool enable_singleton_cols_eq = true;
  bool enable_doubleton_eq = true;
  bool enable_redundant_bounds = false;
  bool enable_parallel_cols = true;
  bool enable_structural_l1_substitution = true;
  int doubleton_eq_max_reductions = -1;
  int doubleton_eq_max_fill_in_proxy = 10;
  int doubleton_eq_min_selected_per_batch = 256;
  double doubleton_eq_min_selected_ratio = 0.005;
  bool doubleton_eq_batch_mode = false;
};

}  // namespace cpu_presolve
