#ifndef HPRLP_MAIN_ITERATE_H
#define HPRLP_MAIN_ITERATE_H

#include "api/structs.h"
#include "solver/restart_control.h"
#include "support/utils.h"
#include "cuda_kernels/HPR_cuda_kernels.cuh"


// Compute relative KKT error
void residual_compute_Rp_cusparse(HPRLP_workspace_gpu *ws, Scaling_info *scaling);

void residual_compute_Rd_cusparse(HPRLP_workspace_gpu *ws, Scaling_info *scaling);

// Collect KKT residuals.  When compute_gap=true, also computes restart_info->current_gap
// (batched with the same device→host fetch to avoid an extra sync).
// restart_info may be nullptr when compute_gap=false (e.g. autotune probes).
void compute_residuals(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling,
                       HPRLP_residuals *residual, int iter,
                       HPRLP_restart *restart_info = nullptr, bool compute_gap = false);

// Whether restart
void check_restart(HPRLP_restart *restart_info, int iter,
                   const HPRLP_parameters *param, HPRLP_FLOAT sigma,
                   const HPRLP_progress_metrics &progress);

bool check_sigma_rebalance_restart(
    HPRLP_restart *restart_info, const HPRLP_residuals *residuals,
    int iter, const HPRLP_parameters *param,
    const HPRLP_control_state &control,
    const HPRLP_progress_metrics &progress,
    HPRLP_FLOAT ratio_threshold);

// Whether stop 
std::string check_stopping(HPRLP_residuals *residuals, int iter, std::chrono::steady_clock::time_point t_start, const HPRLP_parameters *param);

void update_sigma(HPRLP_restart *restart_info, HPRLP_workspace_gpu *ws,
                  HPRLP_residuals *residuals,
                  const HPRLP_parameters *param);

// Perform restart
void do_restart(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info);

void update_zx_check_gpu(HPRLP_workspace_gpu *ws);
void update_zx_normal_gpu(HPRLP_workspace_gpu *ws);

void update_y_check_gpu(HPRLP_workspace_gpu *ws);
void update_y_normal_gpu(HPRLP_workspace_gpu *ws);
void advance_halpern_factors(HPRLP_workspace_gpu *ws);

void reset_halpern_runtime_params(HPRLP_workspace_gpu *ws);
void upload_halpern_iter_params_if_needed(HPRLP_workspace_gpu *ws);
void upload_halpern_restart_params(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info);

// Compute M-weighted norm ||dx, dy||_M for x_temp/y_temp using queued dots + one fetch.
HPRLP_FLOAT compute_weighted_norm(HPRLP_workspace_gpu *ws);

void autotune_custom_update_backends(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling, const HPRLP_parameters *param);
void autotune_reduced_update_backends(
    HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling,
    const HPRLP_parameters *param);
void monitor_signed_zero_skip_backend(HPRLP_workspace_gpu *ws);

#endif
