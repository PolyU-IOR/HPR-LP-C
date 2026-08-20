#include "cpu_presolve_c/cpu_presolve.h"

#include "cpu_presolve/lp_model.hpp"
#include "cpu_presolve/presolver.hpp"

#include <memory>
#include <vector>

struct cpu_presolve_model {
  cpu_presolve::LpModel model;
};

struct cpu_presolve_options {
  cpu_presolve::PresolveOptions options;
};

struct cpu_presolve_result {
  cpu_presolve::PresolveResult result;
};

namespace {

std::vector<int> copy_int_array(const int* data, int count) {
  if (count == 0) {
    return {};
  }
  return std::vector<int>(data, data + count);
}

std::vector<double> copy_double_array(const double* data, int count) {
  if (count == 0) {
    return {};
  }
  return std::vector<double>(data, data + count);
}

cpu_presolve_status_t to_c_status(cpu_presolve::PresolveStatus status) {
  switch (status) {
    case cpu_presolve::PresolveStatus::kOk:
      return CPU_PRESOLVE_STATUS_OK;
    case cpu_presolve::PresolveStatus::kInfeasible:
      return CPU_PRESOLVE_STATUS_INFEASIBLE;
    case cpu_presolve::PresolveStatus::kUnbounded:
      return CPU_PRESOLVE_STATUS_UNBOUNDED;
  }
  return CPU_PRESOLVE_STATUS_ERROR;
}

}  // namespace

cpu_presolve_model_t* cpu_presolve_create_lp_from_csc(
    int num_rows,
    int num_cols,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz,
    const double* objective,
    const double* row_lower,
    const double* row_upper,
    const double* col_lower,
    const double* col_upper,
    double obj_constant) {
  try {
    if (num_rows < 0 || num_cols < 0 || nnz < 0) {
      return nullptr;
    }
    if ((num_cols + 1 > 0 && col_ptr == nullptr) ||
        (nnz > 0 && (row_idx == nullptr || values == nullptr)) ||
        (num_cols > 0 && (objective == nullptr || col_lower == nullptr || col_upper == nullptr)) ||
        (num_rows > 0 && (row_lower == nullptr || row_upper == nullptr))) {
      return nullptr;
    }

    auto model = std::make_unique<cpu_presolve_model>(cpu_presolve_model{
        cpu_presolve::LpModel(
            cpu_presolve::CscMatrix(
                num_rows,
                num_cols,
                copy_int_array(col_ptr, num_cols + 1),
                copy_int_array(row_idx, nnz),
                copy_double_array(values, nnz)),
            copy_double_array(objective, num_cols),
            copy_double_array(row_lower, num_rows),
            copy_double_array(row_upper, num_rows),
            copy_double_array(col_lower, num_cols),
            copy_double_array(col_upper, num_cols),
            obj_constant)});
    return model.release();
  } catch (...) {
    return nullptr;
  }
}

cpu_presolve_options_t* cpu_presolve_create_options(void) {
  try {
    return new cpu_presolve_options{cpu_presolve::PresolveOptions{}};
  } catch (...) {
    return nullptr;
  }
}

void cpu_presolve_options_set_close_bounds(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_close_bounds = enabled != 0;
  }
}

void cpu_presolve_options_set_empty_rows(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_empty_rows = enabled != 0;
  }
}

void cpu_presolve_options_set_empty_cols(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_empty_cols = enabled != 0;
  }
}

void cpu_presolve_options_set_singleton_rows(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_singleton_rows = enabled != 0;
  }
}

void cpu_presolve_options_set_activity_checks(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_activity_checks = enabled != 0;
  }
}

void cpu_presolve_options_set_primal_propagation(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_primal_propagation = enabled != 0;
  }
}

void cpu_presolve_options_set_dual_fix(cpu_presolve_options_t* options, int enabled) {
  if (options != nullptr) {
    options->options.enable_dual_fix = enabled != 0;
  }
}

cpu_presolve_result_t* cpu_presolve_run(
    const cpu_presolve_model_t* model,
    const cpu_presolve_options_t* options) {
  try {
    if (model == nullptr) {
      return nullptr;
    }
    const cpu_presolve::PresolveOptions run_options =
        options == nullptr ? cpu_presolve::PresolveOptions{} : options->options;
    return new cpu_presolve_result{cpu_presolve::Presolver::run(model->model, run_options)};
  } catch (...) {
    return nullptr;
  }
}

cpu_presolve_status_t cpu_presolve_result_status(const cpu_presolve_result_t* result) {
  if (result == nullptr) {
    return CPU_PRESOLVE_STATUS_ERROR;
  }
  return to_c_status(result->result.status());
}

int cpu_presolve_result_changed(const cpu_presolve_result_t* result) {
  return result != nullptr && result->result.changed() ? 1 : 0;
}

int cpu_presolve_result_num_rows(const cpu_presolve_result_t* result) {
  return result == nullptr ? -1 : result->result.reduced_model().num_rows();
}

int cpu_presolve_result_num_cols(const cpu_presolve_result_t* result) {
  return result == nullptr ? -1 : result->result.reduced_model().num_cols();
}

double cpu_presolve_result_objective_constant(const cpu_presolve_result_t* result) {
  return result == nullptr ? 0.0 : result->result.reduced_model().obj_constant();
}

const int* cpu_presolve_result_csc_col_ptr(const cpu_presolve_result_t* result) {
  return result == nullptr ? nullptr : result->result.reduced_model().csc().col_ptr().data();
}

const int* cpu_presolve_result_csc_row_idx(const cpu_presolve_result_t* result) {
  return result == nullptr ? nullptr : result->result.reduced_model().csc().row_idx().data();
}

const double* cpu_presolve_result_csc_values(const cpu_presolve_result_t* result) {
  return result == nullptr ? nullptr : result->result.reduced_model().csc().values().data();
}

int cpu_presolve_result_csc_nnz(const cpu_presolve_result_t* result) {
  return result == nullptr ? -1 : result->result.reduced_model().csc().nnz();
}

void cpu_presolve_free_model(cpu_presolve_model_t* model) {
  delete model;
}

void cpu_presolve_free_options(cpu_presolve_options_t* options) {
  delete options;
}

void cpu_presolve_free_result(cpu_presolve_result_t* result) {
  delete result;
}
