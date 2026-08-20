#include "test_common.hpp"

#include "cpu_presolve_c/cpu_presolve.h"

CPU_PRESOLVE_TEST(c_api_runs_presolve_from_csc) {
  const int col_ptr[] = {0, 1};
  const int row_idx[] = {0};
  const double values[] = {2.0};
  const double objective[] = {3.0};
  const double row_lower[] = {0.0};
  const double row_upper[] = {10.0};
  const double col_lower[] = {1.0};
  const double col_upper[] = {1.0};

  cpu_presolve_model_t* model = cpu_presolve_create_lp_from_csc(
      1,
      1,
      col_ptr,
      row_idx,
      values,
      1,
      objective,
      row_lower,
      row_upper,
      col_lower,
      col_upper,
      0.0);
  CPU_PRESOLVE_REQUIRE(model != nullptr);

  cpu_presolve_options_t* options = cpu_presolve_create_options();
  CPU_PRESOLVE_REQUIRE(options != nullptr);
  cpu_presolve_result_t* result = cpu_presolve_run(model, options);
  CPU_PRESOLVE_REQUIRE(result != nullptr);

  CPU_PRESOLVE_REQUIRE(cpu_presolve_result_status(result) == CPU_PRESOLVE_STATUS_OK);
  CPU_PRESOLVE_REQUIRE(cpu_presolve_result_changed(result) == 1);
  CPU_PRESOLVE_REQUIRE(cpu_presolve_result_num_rows(result) == 0);
  CPU_PRESOLVE_REQUIRE(cpu_presolve_result_num_cols(result) == 0);
  CPU_PRESOLVE_REQUIRE_NEAR(cpu_presolve_result_objective_constant(result), 3.0, 1.0e-12);

  cpu_presolve_free_result(result);
  cpu_presolve_free_options(options);
  cpu_presolve_free_model(model);
}
