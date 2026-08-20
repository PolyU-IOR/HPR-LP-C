#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpu_presolve_model cpu_presolve_model_t;
typedef struct cpu_presolve_options cpu_presolve_options_t;
typedef struct cpu_presolve_result cpu_presolve_result_t;

typedef enum cpu_presolve_status {
  CPU_PRESOLVE_STATUS_OK = 0,
  CPU_PRESOLVE_STATUS_INFEASIBLE = 1,
  CPU_PRESOLVE_STATUS_UNBOUNDED = 2,
  CPU_PRESOLVE_STATUS_ERROR = 3
} cpu_presolve_status_t;

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
    double obj_constant);

cpu_presolve_options_t* cpu_presolve_create_options(void);
void cpu_presolve_options_set_close_bounds(cpu_presolve_options_t* options, int enabled);
void cpu_presolve_options_set_empty_rows(cpu_presolve_options_t* options, int enabled);
void cpu_presolve_options_set_empty_cols(cpu_presolve_options_t* options, int enabled);
void cpu_presolve_options_set_singleton_rows(cpu_presolve_options_t* options, int enabled);
void cpu_presolve_options_set_activity_checks(cpu_presolve_options_t* options, int enabled);
void cpu_presolve_options_set_primal_propagation(cpu_presolve_options_t* options, int enabled);
void cpu_presolve_options_set_dual_fix(cpu_presolve_options_t* options, int enabled);

cpu_presolve_result_t* cpu_presolve_run(
    const cpu_presolve_model_t* model,
    const cpu_presolve_options_t* options);

cpu_presolve_status_t cpu_presolve_result_status(const cpu_presolve_result_t* result);
int cpu_presolve_result_changed(const cpu_presolve_result_t* result);
int cpu_presolve_result_num_rows(const cpu_presolve_result_t* result);
int cpu_presolve_result_num_cols(const cpu_presolve_result_t* result);
double cpu_presolve_result_objective_constant(const cpu_presolve_result_t* result);

const int* cpu_presolve_result_csc_col_ptr(const cpu_presolve_result_t* result);
const int* cpu_presolve_result_csc_row_idx(const cpu_presolve_result_t* result);
const double* cpu_presolve_result_csc_values(const cpu_presolve_result_t* result);
int cpu_presolve_result_csc_nnz(const cpu_presolve_result_t* result);

void cpu_presolve_free_model(cpu_presolve_model_t* model);
void cpu_presolve_free_options(cpu_presolve_options_t* options);
void cpu_presolve_free_result(cpu_presolve_result_t* result);

#ifdef __cplusplus
}
#endif
