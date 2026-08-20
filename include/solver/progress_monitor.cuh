#ifndef HPRLP_PROGRESS_MONITOR_CUH
#define HPRLP_PROGRESS_MONITOR_CUH

#include "api/structs.h"
#include "solver/restart_control.h"

#include <cstdint>

struct HPRLP_progress_monitor_gpu {
    std::uint8_t *previous_x_state = nullptr;
    std::uint8_t *previous_y_state = nullptr;
    HPRLP_FLOAT *previous_x_bar = nullptr;
    HPRLP_FLOAT *previous_y_bar = nullptr;
    HPRLP_FLOAT *previous_step_x = nullptr;
    HPRLP_FLOAT *previous_step_y = nullptr;
    HPRLP_FLOAT *current_step_x = nullptr;
    HPRLP_FLOAT *current_step_y = nullptr;
    unsigned long long *counters = nullptr;
    unsigned long long *counters_host = nullptr;
    bool direction_initialized = false;
    HPRLP_FLOAT previous_dx_norm_sq = 0.0;
    HPRLP_FLOAT previous_dy_norm_sq = 0.0;
    HPRLP_progress_metrics metrics;
};

void hprlp_initialize_progress_monitor(
    HPRLP_progress_monitor_gpu *monitor,
    HPRLP_workspace_gpu *workspace);
void hprlp_queue_progress_sample(
    HPRLP_progress_monitor_gpu *monitor,
    HPRLP_workspace_gpu *workspace);
void hprlp_finalize_progress_sample(
    HPRLP_progress_monitor_gpu *monitor,
    const HPRLP_workspace_gpu *workspace);
void hprlp_free_progress_monitor(HPRLP_progress_monitor_gpu *monitor);

#endif
