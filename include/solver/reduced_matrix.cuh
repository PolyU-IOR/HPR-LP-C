#ifndef HPRLP_REDUCED_MATRIX_CUH
#define HPRLP_REDUCED_MATRIX_CUH

#include "api/structs.h"

constexpr HPRLP_FLOAT HPRLP_REDUCED_ENTER_RESIDUAL = 1e-2;
constexpr HPRLP_FLOAT HPRLP_REDUCED_ENTER_RATIO = 0.5;
constexpr int HPRLP_REDUCED_ENTER_STABLE = 3;
constexpr HPRLP_FLOAT HPRLP_REDUCED_ENTER_FAST_RATIO = 0.25;
constexpr int HPRLP_REDUCED_REBUILD_INTERVAL = 1000;
constexpr HPRLP_FLOAT HPRLP_REDUCED_REBUILD_FAST_RATIO = 0.05;
constexpr int HPRLP_REDUCED_REBUILD_FAST_INTERVAL = 300;

void hprlp_initialize_reduced_matrix_state(HPRLP_workspace_gpu *workspace);
void hprlp_free_reduced_matrix_state(HPRLP_workspace_gpu *workspace);

bool hprlp_refresh_reduced_matrix_mask(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_parameters *parameters,
    const HPRLP_residuals *residuals,
    bool force_refresh = false);

void hprlp_update_reduced_matrix_mode(
    HPRLP_workspace_gpu *workspace,
    LP_info_gpu *lp,
    Scaling_info *scaling,
    const HPRLP_parameters *parameters,
    const HPRLP_residuals *residuals,
    int iteration,
    bool residuals_refreshed,
    bool next_iteration_checks);

void hprlp_maybe_reset_reduced_matrix_mask_on_restart(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_parameters *parameters,
    const HPRLP_residuals *residuals,
    int iteration,
    int restart_flag);

void hprlp_flush_reduced_matrix_state(HPRLP_workspace_gpu *workspace);
void hprlp_sync_reduced_matrix_state_from_full(
    HPRLP_workspace_gpu *workspace);
int hprlp_launch_reduced_matrix_iterations(
    HPRLP_workspace_gpu *workspace,
    int iteration,
    int restart_flag,
    int requested_iterations);
bool hprlp_reduced_matrix_active(const HPRLP_workspace_gpu *workspace);
void hprlp_print_reduced_matrix_profile(
    const HPRLP_workspace_gpu *workspace);
void hprlp_collect_reduced_matrix_results(
    const HPRLP_workspace_gpu *workspace,
    HPRLP_results *results);

#endif
