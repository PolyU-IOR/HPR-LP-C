#include "test_common.hpp"

#include "cpu_presolve/presolver.hpp"

#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using cpu_presolve::CscMatrix;
using cpu_presolve::LpModel;
using cpu_presolve::PresolveOptions;
using cpu_presolve::PresolveScheduler;
using cpu_presolve::PresolveStatus;
using cpu_presolve::Presolver;

namespace {

void disable_reference_expansion_rules(PresolveOptions& options) {
  options.enable_parallel_rows = false;
  options.enable_singleton_cols_dual_infer = false;
  options.enable_singleton_cols_eq = false;
  options.enable_doubleton_eq = false;
  options.enable_parallel_cols = false;
}

std::vector<std::string> profiled_rule_names(const LpModel& model, const PresolveOptions& options) {
  std::ostringstream profile_output;
  std::streambuf* old_cerr = std::cerr.rdbuf(profile_output.rdbuf());
  setenv("CPU_PRESOLVE_PROFILE", "1", 1);
  static_cast<void>(Presolver::run(model, options));
  unsetenv("CPU_PRESOLVE_PROFILE");
  std::cerr.rdbuf(old_cerr);

  std::vector<std::string> names;
  std::istringstream lines(profile_output.str());
  std::string line;
  while (std::getline(lines, line)) {
    const std::string prefix = "PROFILE_BEGIN\t";
    if (line.rfind(prefix, 0) != 0) {
      continue;
    }
    const std::size_t start = prefix.size();
    const std::size_t stop = line.find('\t', start);
    names.push_back(line.substr(start, stop == std::string::npos ? stop : stop - start));
  }
  return names;
}

std::string profiled_output(const LpModel& model, const PresolveOptions& options) {
  std::ostringstream profile_output;
  std::streambuf* old_cerr = std::cerr.rdbuf(profile_output.rdbuf());
  setenv("CPU_PRESOLVE_PROFILE", "1", 1);
  static_cast<void>(Presolver::run(model, options));
  unsetenv("CPU_PRESOLVE_PROFILE");
  std::cerr.rdbuf(old_cerr);
  return profile_output.str();
}

std::size_t find_rule_after(const std::vector<std::string>& names,
                            const std::string& rule,
                            std::size_t start) {
  for (std::size_t i = start; i < names.size(); ++i) {
    if (names[i] == rule) {
      return i;
    }
  }
  return names.size();
}

}  // namespace

CPU_PRESOLVE_TEST(tiered_scheduler_profiles_cleanup_as_single_trivial_closure) {
  CscMatrix csc(
      1,
      2,
      {0, 1, 1},
      {0},
      {1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {1.0},
      {1.0},
      {1.0, 0.0},
      {1.0, 10.0},
      0.0);

  PresolveOptions options;
  options.enable_structural_l1_substitution = false;
  options.enable_singleton_rows = true;
  options.enable_close_bounds = true;
  options.enable_empty_rows = true;
  options.enable_empty_cols = true;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_singleton_cols_dual_infer = false;
  options.enable_singleton_cols_eq = false;
  options.enable_doubleton_eq = false;
  options.enable_dual_fix = false;
  options.enable_parallel_cols = false;

  std::ostringstream profile_output;
  std::streambuf* old_cerr = std::cerr.rdbuf(profile_output.rdbuf());
  setenv("CPU_PRESOLVE_PROFILE", "1", 1);
  const auto result = Presolver::run(model, options);
  unsetenv("CPU_PRESOLVE_PROFILE");
  std::cerr.rdbuf(old_cerr);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());

  const std::string profile = profile_output.str();
  CPU_PRESOLVE_REQUIRE(profile.find("PROFILE_BEGIN\ttrivial_cleanup") != std::string::npos);
}

CPU_PRESOLVE_TEST(close_bounds_fixes_column_and_shifts_rows) {
  CscMatrix csc(
      2,
      3,
      {0, 2, 3, 4},
      {0, 1, 0, 1},
      {2.0, -1.0, 3.0, 4.0});

  LpModel model(
      csc,
      {5.0, 7.0, 11.0},
      {4.0, 6.0},
      {20.0, 30.0},
      {2.0, 0.0, 1.0},
      {2.0 + 1.0e-10, 10.0, 3.0},
      1.5);

  PresolveOptions options;
  options.bound_tolerance = 1.0e-9;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  const double fixed_value = 2.0 + 0.5e-10;
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.fixed_columns().size() == 1);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[0].original_col == 0);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[0].value, fixed_value, 1.0e-12);

  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({3.0, 4.0}));
  CPU_PRESOLVE_REQUIRE(reduced.objective() == std::vector<double>({7.0, 11.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_lower() == std::vector<double>({0.0, 1.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_upper() == std::vector<double>({10.0, 3.0}));

  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[0], 4.0 - 2.0 * fixed_value, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 20.0 - 2.0 * fixed_value, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[1], 6.0 + fixed_value, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[1], 30.0 + fixed_value, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 1.5 + 5.0 * fixed_value, 1.0e-12);
}

CPU_PRESOLVE_TEST(empty_rows_removes_feasible_empty_rows) {
  CscMatrix csc(
      3,
      2,
      {0, 1, 2},
      {1, 2},
      {2.0, 3.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {-1.0, 0.0, 0.0},
      {1.0, 10.0, 12.0},
      {0.0, 0.0},
      {10.0, 10.0},
      0.0);

  PresolveOptions options;
  options.enable_close_bounds = false;
  options.enable_empty_rows = true;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({2.0, 3.0}));
  CPU_PRESOLVE_REQUIRE(reduced.row_lower() == std::vector<double>({0.0, 0.0}));
  CPU_PRESOLVE_REQUIRE(reduced.row_upper() == std::vector<double>({10.0, 12.0}));
}

CPU_PRESOLVE_TEST(empty_rows_marks_infeasible_empty_row) {
  CscMatrix csc(
      2,
      1,
      {0, 1},
      {1},
      {2.0});

  LpModel model(
      csc,
      {1.0},
      {1.0, 0.0},
      {2.0, 10.0},
      {0.0},
      {10.0},
      0.0);

  PresolveOptions options;
  options.enable_close_bounds = false;
  options.enable_empty_rows = true;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.feasibility_tolerance = 1.0e-9;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kInfeasible);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(empty_cols_fixes_by_objective_direction_and_removes_columns) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      2,
      4,
      {0, 2, 2, 2, 2},
      {0, 1},
      {1.0, 2.0});

  LpModel model(
      csc,
      {1.0, 3.0, -4.0, 0.0},
      {0.0, 1.0},
      {10.0, 20.0},
      {0.0, 2.0, -10.0, -inf},
      {10.0, 10.0, 5.0, 7.0},
      2.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = true;
  options.zero_tolerance = 1.0e-12;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.fixed_columns().size() == 3);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[0].original_col == 1);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[1].original_col == 2);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[2].original_col == 3);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[0].value, 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[1].value, 5.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[2].value, 7.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(result.objective_shift(), -14.0, 1.0e-12);

  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 2.0}));
  CPU_PRESOLVE_REQUIRE(reduced.objective() == std::vector<double>({1.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_lower() == std::vector<double>({0.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_upper() == std::vector<double>({10.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), -12.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(empty_cols_marks_unbounded_for_improving_direction_without_bound) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      1,
      {0, 0},
      {},
      {});

  LpModel model(
      csc,
      {3.0},
      {0.0},
      {0.0},
      {-inf},
      {10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kUnbounded);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(singleton_rows_tightens_variable_bounds_and_removes_rows) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      2,
      2,
      {0, 2, 3},
      {0, 1, 1},
      {2.0, 1.0, 3.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {4.0, -inf},
      {10.0, 20.0},
      {0.0, 0.0},
      {10.0, 10.0},
      0.0);

  PresolveOptions options;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_empty_cols = false;
  options.enable_singleton_rows = true;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 3.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 5.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[1], 10.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE(reduced.row_lower().size() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.row_upper().size() == 1);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 20.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(singleton_rows_marks_infeasible_when_bounds_cross) {
  CscMatrix csc(
      1,
      1,
      {0, 1},
      {0},
      {1.0});

  LpModel model(
      csc,
      {1.0},
      {5.0},
      {6.0},
      {0.0},
      {4.0},
      0.0);

  PresolveOptions options;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_empty_cols = false;
  options.enable_singleton_rows = true;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kInfeasible);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(activity_checks_removes_fully_redundant_rows) {
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {-1.0},
      {5.0},
      {0.0, 0.0},
      {1.0, 2.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = true;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 0);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx().empty());
  CPU_PRESOLVE_REQUIRE(reduced.csc().values().empty());
}

CPU_PRESOLVE_TEST(activity_checks_drops_redundant_row_sides) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      2,
      2,
      {0, 2, 4},
      {0, 1, 0, 1},
      {1.0, 1.0, 1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {-1.0, 0.5},
      {1.5, 3.0},
      {0.0, 0.0},
      {1.0, 1.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = true;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.row_lower()[0] == -inf);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 1.5, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[1], 0.5, 1.0e-12);
  CPU_PRESOLVE_REQUIRE(reduced.row_upper()[1] == inf);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 2, 4}));
}

CPU_PRESOLVE_TEST(tiered_scheduler_cleans_medium_propagation_as_one_block) {
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {-1.0},
      {5.0},
      {0.0, 0.0},
      {1.0, 2.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_structural_l1_substitution = false;
  options.enable_close_bounds = false;
  options.enable_empty_rows = true;
  options.enable_singleton_rows = true;
  options.enable_activity_checks = true;
  options.enable_primal_propagation = true;
  options.enable_dual_fix = false;
  options.enable_empty_cols = true;

  const std::vector<std::string> names = profiled_rule_names(model, options);
  bool found_core_medium_block = false;
  for (std::size_t i = 1; i + 2 < names.size(); ++i) {
    if (names[i - 1] == "trivial_cleanup" &&
        names[i] == "activity_checks" &&
        names[i + 1] == "primal_propagation" &&
        names[i + 2] == "trivial_cleanup") {
      found_core_medium_block = true;
      break;
    }
  }
  CPU_PRESOLVE_REQUIRE(found_core_medium_block);
}

CPU_PRESOLVE_TEST(activity_checks_marks_infeasible_unreachable_row) {
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {3.5},
      {5.0},
      {0.0, 0.0},
      {1.0, 1.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = true;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kInfeasible);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(primal_propagation_tightens_bounds_from_row_activity) {
  CscMatrix csc(
      2,
      2,
      {0, 2, 4},
      {0, 1, 0, 1},
      {2.0, 2.0, 1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {10.0, -100.0},
      {100.0, 10.0},
      {0.0, 2.0},
      {10.0, 4.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = true;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 3.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 4.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[1], 4.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(tiered_primal_propagation_keeps_small_valid_tightenings) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {0.0, 0.0},
      {-inf},
      {9.995},
      {0.0, 0.0},
      {10.0, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.scheduler = PresolveScheduler::kTiered;
  options.enable_structural_l1_substitution = false;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = true;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 9.995, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[1], 9.995, 1.0e-12);
}

CPU_PRESOLVE_TEST(primal_propagation_marks_infeasible_when_implied_bounds_cross) {
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {10.0},
      {100.0},
      {0.0, 0.0},
      {4.0, 4.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = true;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kInfeasible);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(dual_fix_fixes_unlocked_improving_column_and_shifts_rows) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {3.0, -1.0},
      {-inf},
      {10.0},
      {2.0, 0.0},
      {8.0, 10.0},
      1.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = true;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.fixed_columns().size() == 1);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[0].original_col == 0);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[0].value, 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(result.objective_shift(), 6.0, 1.0e-12);

  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0}));
  CPU_PRESOLVE_REQUIRE(reduced.row_lower()[0] == -inf);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 8.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 7.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(dual_fix_marks_unbounded_when_improving_bound_missing) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      1,
      {0, 1},
      {0},
      {1.0});

  LpModel model(
      csc,
      {3.0},
      {-inf},
      {10.0},
      {-inf},
      {8.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = true;
  options.enable_empty_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kUnbounded);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(redundant_bounds_drops_implied_upper_and_lower_bounds) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      2,
      4,
      {0, 1, 2, 3, 4},
      {0, 0, 1, 1},
      {1.0, 1.0, 1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 2.0, 3.0, 4.0},
      {-inf, 5.0},
      {5.0, inf},
      {-inf, 0.0, 5.0, 0.0},
      {5.0, 10.0, inf, 0.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.col_lower()[0] == -inf);
  CPU_PRESOLVE_REQUIRE(reduced.col_upper()[0] == inf);
  CPU_PRESOLVE_REQUIRE(reduced.col_lower()[2] == -inf);
  CPU_PRESOLVE_REQUIRE(reduced.col_upper()[2] == inf);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[1], 10.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[3], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[3], 0.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(parallel_rows_merges_scalar_multiple_rows) {
  CscMatrix csc(
      2,
      2,
      {0, 2, 4},
      {0, 1, 0, 1},
      {1.0, 2.0, 1.0, 2.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {0.0, 2.0},
      {10.0, 12.0},
      {0.0, 0.0},
      {10.0, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_rows = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 1.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[0], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 6.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(tiered_scheduler_does_not_cleanup_after_parallel_rows) {
  CscMatrix csc(
      2,
      2,
      {0, 2, 4},
      {0, 1, 0, 1},
      {1.0, 2.0, 1.0, 2.0});

  LpModel model(
      csc,
      {1.0, 2.0},
      {0.0, 2.0},
      {10.0, 12.0},
      {0.0, 0.0},
      {10.0, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_structural_l1_substitution = false;
  options.enable_close_bounds = false;
  options.enable_empty_rows = true;
  options.enable_singleton_rows = true;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = true;
  options.enable_redundant_bounds = false;
  options.enable_parallel_rows = true;
  options.enable_parallel_cols = true;

  const std::vector<std::string> names = profiled_rule_names(model, options);
  bool found_parallel_block = false;
  for (std::size_t i = 0; i + 1 < names.size(); ++i) {
    if (names[i] == "parallel_rows" && names[i + 1] == "parallel_cols") {
      found_parallel_block = true;
      break;
    }
  }
  CPU_PRESOLVE_REQUIRE(found_parallel_block);
}

CPU_PRESOLVE_TEST(parallel_rows_marks_infeasible_disjoint_intervals) {
  CscMatrix csc(
      2,
      1,
      {0, 2},
      {0, 1},
      {1.0, 2.0});

  LpModel model(
      csc,
      {1.0},
      {0.0, 10.0},
      {1.0, 12.0},
      {0.0},
      {10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_rows = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kInfeasible);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(parallel_rows_merges_only_matching_signature_candidates) {
  CscMatrix csc(
      5,
      3,
      {0, 3, 6, 9},
      {0, 1, 4, 0, 2, 4, 1, 2, 3},
      {1.0, 1.0, 3.0, 2.0, 5.0, 6.0, 4.0, 7.0, 6.0});

  LpModel model(
      csc,
      {1.0, 2.0, 3.0},
      {0.0, 0.0, 0.0, 0.0, 9.0},
      {100.0, 100.0, 100.0, 100.0, 18.0},
      {0.0, 0.0, 0.0},
      {10.0, 10.0, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_rows = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 4);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 3);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 2, 4, 7}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1, 0, 2, 1, 2, 3}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 1.0, 2.0, 5.0, 4.0, 7.0, 6.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[0], 3.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 6.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(parallel_cols_merges_objective_compatible_columns) {
  CscMatrix csc(
      2,
      3,
      {0, 2, 4, 5},
      {0, 1, 0, 1, 1},
      {1.0, 2.0, 2.0, 4.0, 3.0});

  LpModel model(
      csc,
      {3.0, 6.0, 5.0},
      {0.0, 0.0},
      {20.0, 30.0},
      {1.0, 2.0, 0.0},
      {5.0, 4.0, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 2, 3}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 2.0, 3.0}));
  CPU_PRESOLVE_REQUIRE(reduced.objective() == std::vector<double>({3.0, 5.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 5.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 13.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[1], 10.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(parallel_cols_fixes_objective_incompatible_source_column) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 3.0},
      {0.0},
      {10.0},
      {0.0, 2.0},
      {inf, 5.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.fixed_columns().size() == 1);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[0].original_col == 1);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[0].value, 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(result.objective_shift(), 6.0, 1.0e-12);

  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[0], -2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 8.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 6.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(parallel_cols_marks_unbounded_when_required_fixing_bound_is_missing) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 3.0},
      {0.0},
      {10.0},
      {0.0, -inf},
      {inf, 5.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kUnbounded);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(parallel_cols_merges_only_matching_signature_candidates) {
  CscMatrix csc(
      3,
      5,
      {0, 2, 4, 6, 7, 9},
      {0, 1, 0, 2, 1, 2, 2, 0, 1},
      {1.0, 2.0, 5.0, 7.0, 4.0, 8.0, 6.0, 3.0, 6.0});

  LpModel model(
      csc,
      {2.0, 1.0, 3.0, 4.0, 6.0},
      {0.0, 0.0, 0.0},
      {100.0, 100.0, 100.0},
      {1.0, 0.0, 0.0, 0.0, 2.0},
      {5.0, 10.0, 10.0, 10.0, 4.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 3);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 4);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 2, 4, 6, 7}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1, 0, 2, 1, 2, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 2.0, 5.0, 7.0, 4.0, 8.0, 6.0}));
  CPU_PRESOLVE_REQUIRE(reduced.objective() == std::vector<double>({2.0, 1.0, 3.0, 4.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 7.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 17.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(singleton_cols_eq_eliminates_free_equality_singleton_column) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 2.0});

  LpModel model(
      csc,
      {3.0, 5.0},
      {10.0},
      {10.0},
      {-inf, 0.0},
      {inf, 4.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_singleton_cols_eq = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 0);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx().empty());
  CPU_PRESOLVE_REQUIRE(reduced.csc().values().empty());
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[0], -1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 30.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(singleton_cols_eq_keeps_one_sided_bound_as_row_side) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {2.0, 7.0},
      {5.0},
      {5.0},
      {0.0, 0.0},
      {10.0, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_singleton_cols_eq = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0}));
  CPU_PRESOLVE_REQUIRE(reduced.row_lower()[0] == -inf);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 5.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[0], 5.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 10.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(singleton_cols_dual_infer_tightens_inequality_to_active_equality) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {2.0, 0.0},
      {1.0},
      {10.0},
      {-inf, 0.0},
      {inf, 3.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_singleton_cols_dual_infer = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[0], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
}

CPU_PRESOLVE_TEST(singleton_cols_dual_infer_marks_direct_unbounded_direction) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {2.0, 0.0},
      {-inf},
      {10.0},
      {-inf, 0.0},
      {inf, 3.0},
      0.0);

  PresolveOptions options;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_singleton_cols_dual_infer = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kUnbounded);
  CPU_PRESOLVE_REQUIRE(!result.changed());
}

CPU_PRESOLVE_TEST(singleton_cols_profiles_single_queue_combined_pass) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      1,
      2,
      {0, 1, 2},
      {0, 0},
      {1.0, 1.0});

  LpModel model(
      csc,
      {2.0, 0.0},
      {1.0},
      {10.0},
      {-inf, 0.0},
      {inf, 3.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_structural_l1_substitution = false;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_singleton_cols_dual_infer = true;
  options.enable_singleton_cols_eq = true;

  const std::string profile = profiled_output(model, options);
  CPU_PRESOLVE_REQUIRE(profile.find("PROFILE_END\tsingleton_cols") != std::string::npos);
  CPU_PRESOLVE_REQUIRE(profile.find("PROFILE_END\tsingleton_cols\t") != std::string::npos);
  CPU_PRESOLVE_REQUIRE(profile.find("mode=single_queue") != std::string::npos);
}

CPU_PRESOLVE_TEST(doubleton_eq_substitutes_column_updates_rows_objective_and_bounds) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      3,
      3,
      {0, 2, 4, 6},
      {0, 2, 0, 1, 1, 2},
      {2.0, 1.0, 3.0, 5.0, 1.0, 1.0});

  LpModel model(
      csc,
      {4.0, 1.0, 2.0},
      {12.0, -inf, -inf},
      {12.0, 20.0, 30.0},
      {0.0, 0.0, 0.0},
      {10.0, 10.0, 100.0},
      0.0);

  PresolveOptions options;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_doubleton_eq = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 2, 4}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 1, 0, 1}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.csc().values()[0], -10.0 / 3.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.csc().values()[1], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.csc().values()[2], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.csc().values()[3], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[1], 30.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[0], 10.0 / 3.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[1], 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 6.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 4.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(doubleton_eq_queues_chained_substitutions_in_one_pass) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      4,
      5,
      {0, 2, 3, 5, 7, 8},
      {0, 2, 0, 1, 3, 1, 2, 3},
      {2.0, 1.0, 3.0, 5.0, 1.0, 7.0, 4.0, 1.0});

  LpModel model(
      csc,
      {4.0, 1.0, 6.0, 2.0, 5.0},
      {12.0, 20.0, -inf, 28.0},
      {12.0, 20.0, 30.0, 28.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {10.0, 10.0, 10.0, 10.0, 100.0},
      0.0);

  PresolveOptions options;
  options.scheduler = PresolveScheduler::kFixed;
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_singleton_cols_dual_infer = false;
  options.enable_singleton_cols_eq = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_structural_l1_substitution = false;
  options.enable_doubleton_eq = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.csc().values()[0], 1.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.csc().values()[1], -20.0 / 7.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_lower()[0], -inf, 0.0);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 130.0 / 7.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[0], 10.0 / 3.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[1], -3.0 / 7.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[0], 6.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[1], 4.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 1048.0 / 7.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(structural_l1_split_rewrites_strict_three_row_blocks) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      3,
      2,
      {0, 2, 5},
      {1, 2, 0, 1, 2},
      {1.0, 1.0, 1.0, -1.0, 1.0});

  LpModel model(
      csc,
      {2.0, 0.5},
      {1.0, 0.0, 0.0},
      {1.0, inf, inf},
      {0.0, -inf},
      {inf, inf},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_structural_l1_substitution = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, -1.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[0], 2.5, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[1], 1.5, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE(reduced.col_upper()[0] == inf);
  CPU_PRESOLVE_REQUIRE(reduced.col_upper()[1] == inf);
}

CPU_PRESOLVE_TEST(structural_l1_graph_rewrites_four_row_epigraph_with_removable_slack) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      4,
      4,
      {0, 3, 6, 7, 8},
      {1, 2, 3, 0, 1, 2, 3, 0},
      {1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 2.0});

  LpModel model(
      csc,
      {2.0, 0.5, 0.0, 7.0},
      {5.0, 0.0, 0.0, 0.0},
      {5.0, inf, inf, inf},
      {0.0, -inf, 0.0, 0.0},
      {inf, inf, inf, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_structural_l1_substitution = true;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 3);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2, 3}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, -1.0, 2.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[0], 2.5, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[1], 1.5, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.objective()[2], 7.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[0], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[1], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_lower()[2], 0.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE(reduced.col_upper()[0] == inf);
  CPU_PRESOLVE_REQUIRE(reduced.col_upper()[1] == inf);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.col_upper()[2], 10.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(default_options_enable_structural_l1_for_graph_reduction) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      4,
      3,
      {0, 3, 6, 7},
      {1, 2, 3, 0, 1, 2, 3},
      {1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0});

  LpModel model(
      csc,
      {1.0, 0.0, 0.0},
      {4.0, 0.0, 0.0, 0.0},
      {4.0, inf, inf, inf},
      {0.0, -inf, 0.0},
      {inf, inf, inf},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  options.enable_close_bounds = false;
  options.enable_empty_rows = false;
  options.enable_singleton_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.reduced_model().num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(result.reduced_model().num_cols() == 2);
}

CPU_PRESOLVE_TEST(default_scheduler_runs_structural_l1_before_bound_propagation) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      4,
      4,
      {0, 3, 6, 7, 8},
      {1, 2, 3, 0, 1, 2, 3, 0},
      {1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 2.0});

  LpModel model(
      csc,
      {2.0, 0.5, 0.0, 7.0},
      {5.0, 0.0, 0.0, 0.0},
      {5.0, inf, inf, inf},
      {0.0, -inf, 0.0, 0.0},
      {inf, inf, inf, 10.0},
      0.0);

  PresolveOptions options;
  disable_reference_expansion_rules(options);
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.reduced_model().num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(result.reduced_model().num_cols() == 3);
}

CPU_PRESOLVE_TEST(tiered_scheduler_cleans_fixed_column_after_singleton_row_tightening) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      2,
      3,
      {0, 2, 3, 4},
      {0, 1, 1, 1},
      {1.0, 1.0, 1.0, 1.0});

  LpModel model(
      csc,
      {3.0, 0.0, 0.0},
      {2.0, -inf},
      {2.0, 10.0},
      {0.0, 0.0, 0.0},
      {10.0, 10.0, 10.0},
      1.0);

  PresolveOptions options;
  options.scheduler = PresolveScheduler::kTiered;
  options.enable_empty_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_singleton_cols_dual_infer = false;
  options.enable_singleton_cols_eq = false;
  options.enable_doubleton_eq = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_structural_l1_substitution = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.fixed_columns().size() == 1);
  CPU_PRESOLVE_REQUIRE(result.fixed_columns()[0].original_col == 0);
  CPU_PRESOLVE_REQUIRE_NEAR(result.fixed_columns()[0].value, 2.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(result.objective_shift(), 6.0, 1.0e-12);

  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 1.0}));
  CPU_PRESOLVE_REQUIRE(reduced.objective() == std::vector<double>({0.0, 0.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_lower() == std::vector<double>({0.0, 0.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_upper() == std::vector<double>({10.0, 10.0}));
  CPU_PRESOLVE_REQUIRE(std::isinf(reduced.row_lower()[0]) && reduced.row_lower()[0] < 0.0);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 8.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 7.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(fixed_scheduler_keeps_single_pass_without_cleanup_recirculation) {
  const double inf = std::numeric_limits<double>::infinity();
  CscMatrix csc(
      2,
      3,
      {0, 2, 3, 4},
      {0, 1, 1, 1},
      {1.0, 1.0, 1.0, 1.0});

  LpModel model(
      csc,
      {3.0, 0.0, 0.0},
      {2.0, -inf},
      {2.0, 10.0},
      {0.0, 0.0, 0.0},
      {10.0, 10.0, 10.0},
      1.0);

  PresolveOptions options;
  options.scheduler = PresolveScheduler::kFixed;
  options.enable_empty_rows = false;
  options.enable_activity_checks = false;
  options.enable_primal_propagation = false;
  options.enable_parallel_rows = false;
  options.enable_dual_fix = false;
  options.enable_empty_cols = false;
  options.enable_singleton_cols_dual_infer = false;
  options.enable_singleton_cols_eq = false;
  options.enable_doubleton_eq = false;
  options.enable_redundant_bounds = false;
  options.enable_parallel_cols = false;
  options.enable_structural_l1_substitution = false;
  const auto result = Presolver::run(model, options);

  CPU_PRESOLVE_REQUIRE(result.status() == PresolveStatus::kOk);
  CPU_PRESOLVE_REQUIRE(result.changed());
  CPU_PRESOLVE_REQUIRE(result.fixed_columns().empty());
  CPU_PRESOLVE_REQUIRE_NEAR(result.objective_shift(), 0.0, 1.0e-12);

  const LpModel& reduced = result.reduced_model();
  CPU_PRESOLVE_REQUIRE(reduced.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(reduced.num_cols() == 3);
  CPU_PRESOLVE_REQUIRE(reduced.csc().col_ptr() == std::vector<int>({0, 1, 2, 3}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().row_idx() == std::vector<int>({0, 0, 0}));
  CPU_PRESOLVE_REQUIRE(reduced.csc().values() == std::vector<double>({1.0, 1.0, 1.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_lower() == std::vector<double>({2.0, 0.0, 0.0}));
  CPU_PRESOLVE_REQUIRE(reduced.col_upper() == std::vector<double>({2.0, 10.0, 10.0}));
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.row_upper()[0], 10.0, 1.0e-12);
  CPU_PRESOLVE_REQUIRE_NEAR(reduced.obj_constant(), 1.0, 1.0e-12);
}

CPU_PRESOLVE_TEST(default_options_match_gpu_lp_reference_switches) {
  const PresolveOptions options;

  CPU_PRESOLVE_REQUIRE(options.scheduler == PresolveScheduler::kTiered);
  CPU_PRESOLVE_REQUIRE(options.max_iterations == 10);
  CPU_PRESOLVE_REQUIRE_NEAR(options.feasibility_tolerance, 1.0e-6, 0.0);
  CPU_PRESOLVE_REQUIRE_NEAR(options.bound_tolerance, 1.0e-6, 0.0);
  CPU_PRESOLVE_REQUIRE_NEAR(options.zero_tolerance, 1.0e-10, 0.0);
  CPU_PRESOLVE_REQUIRE(options.enable_close_bounds);
  CPU_PRESOLVE_REQUIRE(options.enable_empty_rows);
  CPU_PRESOLVE_REQUIRE(options.enable_singleton_rows);
  CPU_PRESOLVE_REQUIRE(options.enable_activity_checks);
  CPU_PRESOLVE_REQUIRE(options.enable_primal_propagation);
  CPU_PRESOLVE_REQUIRE(options.enable_parallel_rows);
  CPU_PRESOLVE_REQUIRE(options.enable_empty_cols);
  CPU_PRESOLVE_REQUIRE(options.enable_singleton_cols_dual_infer);
  CPU_PRESOLVE_REQUIRE(options.enable_singleton_cols_eq);
  CPU_PRESOLVE_REQUIRE(options.enable_doubleton_eq);
  CPU_PRESOLVE_REQUIRE(options.enable_dual_fix);
  CPU_PRESOLVE_REQUIRE(options.enable_parallel_cols);
  CPU_PRESOLVE_REQUIRE(!options.enable_redundant_bounds);
  CPU_PRESOLVE_REQUIRE(options.enable_structural_l1_substitution);
}
