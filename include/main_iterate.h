#ifndef HPRLP_MAIN_ITERATE_H
#define HPRLP_MAIN_ITERATE_H

#include "structs.h"
#include "utils.h"
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
void check_restart(HPRLP_restart *restart_info, int iter, int check_iter, HPRLP_FLOAT sigma);

// Whether stop 
std::string check_stopping(HPRLP_residuals *residuals, int iter, std::chrono::steady_clock::time_point t_start, const HPRLP_parameters *param);

void update_sigma(HPRLP_restart *restart_info, HPRLP_workspace_gpu *ws, HPRLP_residuals *residuals);

// Perform restart
void do_restart(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info);

void update_zx_check_gpu(HPRLP_workspace_gpu *ws);
void update_zx_normal_gpu(HPRLP_workspace_gpu *ws);

void update_y_check_gpu(HPRLP_workspace_gpu *ws);
void update_y_normal_gpu(HPRLP_workspace_gpu *ws);

void reset_halpern_runtime_params(HPRLP_workspace_gpu *ws);
void upload_halpern_iter_params_if_needed(HPRLP_workspace_gpu *ws);
void upload_halpern_restart_params(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info);

// Compute M-weighted norm ||dx, dy||_M for x_temp/y_temp using queued dots + one fetch.
HPRLP_FLOAT compute_weighted_norm(HPRLP_workspace_gpu *ws);

void autotune_custom_update_backends(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling, const HPRLP_parameters *param);

#endif