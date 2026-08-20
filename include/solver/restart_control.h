#ifndef HPRLP_RESTART_CONTROL_H
#define HPRLP_RESTART_CONTROL_H

#include "api/structs.h"

#include <limits>

constexpr int HPRLP_PROGRESS_MONITOR_INTERVAL = 150;
constexpr int HPRLP_SIGMA_REBALANCE_RESTART_FLAG = 4;
constexpr int HPRLP_SIGMA_TRIAL_ROLLBACK_FLAG = 5;
constexpr int HPRLP_PHASE2_REBOUND_RESTART_FLAG = 6;
constexpr int HPRLP_ONE_SIDED_STALL_RESTART_FLAG = 7;
constexpr int HPRLP_ONE_SIDED_STALL_CHECKS = 150;
constexpr HPRLP_FLOAT HPRLP_ONE_SIDED_STALL_TOL_MULTIPLIER = 100.0;
// A one-sided correction can initially trade a small increase in its target
// residual for a better-balanced trajectory.  The SpMVOp MCF path exhibited a
// 2.4% transient rebound that subsequently recovers, while the former 0.5%
// threshold rolled it back and exposed ordinary sigma updates to a large
// escalation.  Retain modest transients and roll back material rebounds.
constexpr HPRLP_FLOAT HPRLP_ONE_SIDED_STALL_REBOUND_FACTOR = 1.03;
// Preserve the established queued norm below 2^27 elements; ultra-wide
// movement uses a completion-enforcing host-result norm.
constexpr int HPRLP_ULTRAWIDE_MOVEMENT_LENGTH = 1 << 27;

constexpr bool hprlp_use_host_movement_norm(int length) {
    return length >= HPRLP_ULTRAWIDE_MOVEMENT_LENGTH;
}

enum class HPRLPPhaseEvent {
    None,
    Entered,
    ExitedPrimalRebound,
    ExitedActiveSet,
    ExitedDirection
};

struct HPRLP_progress_metrics {
    HPRLP_FLOAT active_set_change_ratio =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT step_direction_cosine =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT primal_zero_move_ratio =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT stationary_bound_ratio =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
};

struct HPRLP_control_state {
    bool phase2_identified = false;
    int phase2_confirmations = 0;
    int phase2_age_checks = 0;
    HPRLP_FLOAT phase2_primal_reference =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT phase2_entry_primal =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLPPhaseEvent phase2_event = HPRLPPhaseEvent::None;

    bool sigma_trial_active = false;
    int sigma_trial_iter = 0;
    HPRLP_FLOAT sigma_trial_sigma_before =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT sigma_trial_rp_before =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT sigma_trial_rd_before =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT sigma_trial_kkt_before =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    bool sigma_trial_blocked = false;
    bool sigma_phase_settled_blocked = false;
    bool sigma_trial_accepted_once = false;
    bool sigma_trial_rollback_blocked = false;
    bool sigma_trial_floor_recovery_used = false;
    int sigma_trial_last_resolution_iter = 0;

    bool one_sided_stall_active = false;
    int one_sided_stall_side = 0;
    int one_sided_stall_best_iter = 0;
    HPRLP_FLOAT one_sided_stall_best_target =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    int one_sided_stall_last_correction_side = 0;
    int one_sided_stall_direction_transitions = 0;
    bool one_sided_stall_blocked = false;
    bool one_sided_stall_trial_active = false;
    int one_sided_stall_trial_iter = 0;
    int one_sided_stall_trial_side = 0;
    HPRLP_FLOAT one_sided_stall_trial_sigma_before =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT one_sided_stall_trial_target_before =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
};

struct HPRLP_phase_update {
    bool identified;
    int confirmations;
    HPRLPPhaseEvent event;
    HPRLP_FLOAT primal_threshold;
};

enum class HPRLPTrialReason {
    Accepted,
    AcceptedMonotoneProgress,
    AcceptedFeasibilitySlack,
    TargetResidualStable,
    InvalidMetrics,
    KktRebound,
    TargetResidualRebound,
    TargetResidualStalled,
    ImbalanceNotContracting,
    Phase2Revoked
};

struct HPRLP_trial_assessment {
    bool accepted;
    HPRLPTrialReason reason;
};

struct HPRLP_sigma_update_result {
    HPRLP_FLOAT sigma;
    HPRLP_FLOAT residual_ratio;
    HPRLP_FLOAT log_step;
    bool valid;
    const char *reason;
};

struct HPRLP_one_sided_stall_update {
    bool active;
    int side;
    int best_iter;
    HPRLP_FLOAT best_target;
    bool trigger;
};

HPRLP_control_state hprlp_initialize_control_state(int check_iter);

HPRLP_FLOAT hprlp_bounded_direction_cosine(
    HPRLP_FLOAT numerator,
    HPRLP_FLOAT current_norm_sq,
    HPRLP_FLOAT previous_norm_sq);

bool hprlp_phase2_frozen_face_ready(const HPRLP_progress_metrics &progress);
bool hprlp_phase2_direction_ready(const HPRLP_progress_metrics &progress);

HPRLP_phase_update hprlp_update_phase2_state(
    bool identified,
    int confirmations,
    HPRLP_FLOAT primal_residual,
    HPRLP_FLOAT primal_reference,
    const HPRLP_progress_metrics &progress);

bool hprlp_sigma_settled_block_should_clear(HPRLPPhaseEvent event);
bool hprlp_sigma_path_should_block(
    const HPRLP_progress_metrics &progress,
    HPRLP_FLOAT rp,
    HPRLP_FLOAT rd);

bool hprlp_sigma_trial_floor_recovery_ready(
    const HPRLP_control_state &control,
    int iter,
    int check_iter,
    const HPRLP_progress_metrics &progress,
    HPRLP_FLOAT rp,
    HPRLP_FLOAT rd,
    HPRLP_FLOAT stopping_tolerance);

HPRLP_sigma_update_result hprlp_safeguarded_sigma_update(
    HPRLP_FLOAT old_sigma,
    HPRLP_FLOAT sigma_candidate,
    HPRLP_FLOAT blend_factor,
    bool enable_progress_control,
    HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual,
    HPRLP_FLOAT stopping_tolerance);

HPRLP_sigma_update_result hprlp_residual_balanced_sigma_update(
    HPRLP_FLOAT old_sigma,
    HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual);

HPRLP_one_sided_stall_update hprlp_update_one_sided_stall(
    bool active,
    int side,
    int best_iter,
    HPRLP_FLOAT best_target,
    int iter,
    HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual,
    HPRLP_FLOAT stopping_tolerance,
    int stall_window_iterations);

bool hprlp_one_sided_stall_trial_should_rollback(
    int side,
    HPRLP_FLOAT target_before,
    HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual,
    HPRLP_FLOAT stopping_tolerance);

HPRLP_sigma_update_result hprlp_one_sided_stall_sigma_update(
    HPRLP_FLOAT old_sigma,
    HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual);

HPRLP_trial_assessment hprlp_assess_sigma_trial(
    HPRLP_FLOAT rp_before,
    HPRLP_FLOAT rd_before,
    HPRLP_FLOAT kkt_before,
    HPRLP_FLOAT rp_after,
    HPRLP_FLOAT rd_after,
    HPRLP_FLOAT kkt_after,
    HPRLP_FLOAT stopping_tolerance);

bool hprlp_sigma_trial_should_keep_terminal(
    HPRLP_FLOAT rp_before,
    HPRLP_FLOAT rd_before,
    HPRLP_FLOAT kkt_before,
    HPRLP_FLOAT rp_after,
    HPRLP_FLOAT rd_after,
    HPRLP_FLOAT kkt_after);


const char *hprlp_phase_event_name(HPRLPPhaseEvent event);
const char *hprlp_trial_reason_name(HPRLPTrialReason reason);

#endif
