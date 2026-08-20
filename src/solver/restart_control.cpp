#include "solver/restart_control.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr HPRLP_FLOAT kFloor = 1e-12;
constexpr HPRLP_FLOAT kNumericalFloor = 1e-8;
HPRLP_FLOAT clamp_value(HPRLP_FLOAT x, HPRLP_FLOAT lo, HPRLP_FLOAT hi) {
    return std::max(lo, std::min(hi, x));
}
bool valid_metric(HPRLP_FLOAT x) { return std::isfinite(x) && x >= 0.0; }
HPRLP_FLOAT imbalance(HPRLP_FLOAT rp, HPRLP_FLOAT rd) {
    return std::abs(std::log(clamp_value(
        std::max(rd, kFloor) / std::max(rp, kFloor), 1e-8, 1e8)));
}
}

HPRLP_control_state hprlp_initialize_control_state(int check_iter) {
    HPRLP_control_state state;
    state.sigma_trial_last_resolution_iter = -5 * check_iter;
    return state;
}

HPRLP_FLOAT hprlp_bounded_direction_cosine(
    HPRLP_FLOAT numerator, HPRLP_FLOAT current_norm_sq,
    HPRLP_FLOAT previous_norm_sq) {
    if (!(current_norm_sq > 0.0) || !(previous_norm_sq > 0.0))
        return std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    const HPRLP_FLOAT denominator =
        std::sqrt(current_norm_sq) * std::sqrt(previous_norm_sq);
    if (!std::isfinite(denominator) ||
        denominator <= std::numeric_limits<HPRLP_FLOAT>::epsilon())
        return std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    return clamp_value(numerator / denominator, -1.0, 1.0);
}

bool hprlp_phase2_frozen_face_ready(const HPRLP_progress_metrics &p) {
    return std::isfinite(p.active_set_change_ratio) &&
        p.active_set_change_ratio <= 1e-5 &&
        std::isfinite(p.primal_zero_move_ratio) &&
        p.primal_zero_move_ratio >= 0.995 &&
        std::isfinite(p.stationary_bound_ratio) &&
        p.stationary_bound_ratio >= 0.99 &&
        p.stationary_bound_ratio <= p.primal_zero_move_ratio;
}

bool hprlp_phase2_direction_ready(const HPRLP_progress_metrics &p) {
    if (std::isfinite(p.step_direction_cosine))
        return p.step_direction_cosine >= 0.9;
    return std::isnan(p.step_direction_cosine) &&
        hprlp_phase2_frozen_face_ready(p);
}

HPRLP_phase_update hprlp_update_phase2_state(
    bool identified, int confirmations, HPRLP_FLOAT primal_residual,
    HPRLP_FLOAT primal_reference, const HPRLP_progress_metrics &p) {
    const HPRLP_FLOAT threshold =
        std::isfinite(primal_reference) && primal_reference >= 0.0
        ? std::max(1e-6, std::min(1e-2, 0.01 * primal_reference)) : 1e-6;
    const bool primal_valid = std::isfinite(primal_residual);
    const HPRLP_FLOAT rp = primal_valid ? std::abs(primal_residual)
        : std::numeric_limits<HPRLP_FLOAT>::infinity();
    if (identified) {
        HPRLPPhaseEvent event = HPRLPPhaseEvent::None;
        if (!primal_valid || rp > 5.0 * threshold)
            event = HPRLPPhaseEvent::ExitedPrimalRebound;
        else if (std::isfinite(p.active_set_change_ratio) &&
                 p.active_set_change_ratio > 1e-2)
            event = HPRLPPhaseEvent::ExitedActiveSet;
        else if (std::isfinite(p.step_direction_cosine) &&
                 p.step_direction_cosine < 0.5)
            event = HPRLPPhaseEvent::ExitedDirection;
        if (event != HPRLPPhaseEvent::None) return {false, 0, event, threshold};
        return {true, 0, HPRLPPhaseEvent::None, threshold};
    }
    const bool ready = primal_valid && rp <= threshold &&
        std::isfinite(p.active_set_change_ratio) &&
        p.active_set_change_ratio <= 1e-3 && hprlp_phase2_direction_ready(p);
    confirmations = ready ? confirmations + 1 : 0;
    if (confirmations >= 5)
        return {true, 0, HPRLPPhaseEvent::Entered, threshold};
    return {false, confirmations, HPRLPPhaseEvent::None, threshold};
}

bool hprlp_sigma_settled_block_should_clear(HPRLPPhaseEvent event) {
    return event == HPRLPPhaseEvent::ExitedActiveSet ||
        event == HPRLPPhaseEvent::ExitedPrimalRebound;
}

bool hprlp_sigma_path_should_block(
    const HPRLP_progress_metrics &p, HPRLP_FLOAT rp, HPRLP_FLOAT rd) {
    return std::isfinite(p.active_set_change_ratio) &&
        std::isfinite(p.step_direction_cosine) &&
        (p.step_direction_cosine > 0.993 || p.active_set_change_ratio < 1e-5) &&
        std::min(std::abs(rp), std::abs(rd)) > 1e-7;
}

bool hprlp_sigma_trial_floor_recovery_ready(
    const HPRLP_control_state &c, int iter, int check_iter,
    const HPRLP_progress_metrics &p, HPRLP_FLOAT rp, HPRLP_FLOAT rd,
    HPRLP_FLOAT tolerance) {
    return c.sigma_trial_blocked && c.sigma_trial_rollback_blocked &&
        c.sigma_trial_accepted_once && !c.sigma_trial_floor_recovery_used &&
        c.phase2_identified && c.phase2_age_checks >= 5 &&
        iter - c.sigma_trial_last_resolution_iter >= 100 * check_iter &&
        std::isfinite(p.active_set_change_ratio) &&
        p.active_set_change_ratio <= 1e-5 && std::isfinite(rp) &&
        std::isfinite(rd) && std::min(std::abs(rp), std::abs(rd)) <= 1e-10 &&
        std::isfinite(tolerance) && tolerance > 0.0 &&
        std::max(std::abs(rp), std::abs(rd)) > tolerance;
}

HPRLP_sigma_update_result hprlp_safeguarded_sigma_update(
    HPRLP_FLOAT old_sigma, HPRLP_FLOAT candidate, HPRLP_FLOAT blend,
    bool enabled, HPRLP_FLOAT residual_primal, HPRLP_FLOAT residual_dual,
    HPRLP_FLOAT tolerance) {
    if (!enabled) return {candidate, 1.0, blend, true, "legacy"};
    if (!(std::isfinite(old_sigma) && old_sigma > 0.0)) old_sigma = 1.0;
    if (!(std::isfinite(candidate) && candidate > 0.0))
        return {old_sigma, 1.0, 0.0, false, "invalid_candidate_preserve"};
    const HPRLP_FLOAT rp = std::abs(residual_primal);
    const HPRLP_FLOAT rd = std::abs(residual_dual);
    const bool valid = std::isfinite(rp) && std::isfinite(rd);
    const HPRLP_FLOAT max_min_ratio = valid ? std::max(rp, rd) /
        std::max(std::min(rp, rd), kFloor) : 1.0;
    const bool tolerance_imbalance = valid && std::isfinite(tolerance) &&
        tolerance > 0.0 && std::min(rp, rd) <= tolerance &&
        std::max(rp, rd) > tolerance;
    const bool opposes = (rd > rp && candidate < old_sigma) ||
                         (rp > rd && candidate > old_sigma);
    if (tolerance_imbalance && opposes) {
        // Ordinary restarts already have a finite movement-based candidate.
        // Do not reverse it merely because one feasibility residual crossed
        // the stopping tolerance. Residual balancing belongs to the explicit
        // flag-4 phase trial, where the change is assessed and rolled back.
        return {candidate, max_min_ratio, blend, true,
                "ordinary_restart_candidate_preserved"};
    }
    if (valid && std::min(rp, rd) <= 1e-8 && max_min_ratio >= 4.0 && opposes)
        return {old_sigma, max_min_ratio, 0.0, true,
                "residual_floor_direction_guard"};
    return {candidate, max_min_ratio, blend, true, "legacy_progress"};
}

HPRLP_sigma_update_result hprlp_residual_balanced_sigma_update(
    HPRLP_FLOAT old_sigma, HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual) {
    const HPRLP_FLOAT rp = std::abs(residual_primal);
    const HPRLP_FLOAT rd = std::abs(residual_dual);
    if (!(std::isfinite(old_sigma) && old_sigma > 0.0) ||
        !valid_metric(rp) || !valid_metric(rd)) {
        const HPRLP_FLOAT fallback = std::isfinite(old_sigma) && old_sigma > 0.0
            ? old_sigma : 1.0;
        return {fallback, NAN, NAN, false, "invalid_residual_balance"};
    }
    const HPRLP_FLOAT ratio = clamp_value(
        std::max(rd, kFloor) / std::max(rp, kFloor), 1e-8, 1e8);
    const bool severe = ratio >= 100.0 || ratio <= 0.01;
    const HPRLP_FLOAT cap = severe && std::min(rp, rd) <= 1e-7
        ? std::log(8.0) : std::log(4.0);
    const HPRLP_FLOAT log_step = clamp_value(0.5 * std::log(ratio), -cap, cap);
    const HPRLP_FLOAT sigma = old_sigma * std::exp(log_step);
    return std::isfinite(sigma) && sigma > 0.0
        ? HPRLP_sigma_update_result{sigma, ratio, log_step, true,
                                    "phase2_residual_balance"}
        : HPRLP_sigma_update_result{old_sigma, ratio, log_step, false,
                                    "invalid_residual_balance"};
}

HPRLP_one_sided_stall_update hprlp_update_one_sided_stall(
    bool active, int side, int best_iter, HPRLP_FLOAT best_target, int iter,
    HPRLP_FLOAT residual_primal, HPRLP_FLOAT residual_dual,
    HPRLP_FLOAT stopping_tolerance, int stall_window_iterations) {
    const HPRLP_FLOAT rp = std::abs(residual_primal);
    const HPRLP_FLOAT rd = std::abs(residual_dual);
    const bool one_sided = std::isfinite(rp) && std::isfinite(rd) &&
        std::isfinite(stopping_tolerance) && stopping_tolerance > 0.0 &&
        std::min(rp, rd) <= stopping_tolerance &&
        std::max(rp, rd) > stopping_tolerance &&
        std::max(rp, rd) <=
            std::nextafter(
                HPRLP_ONE_SIDED_STALL_TOL_MULTIPLIER * stopping_tolerance,
                std::numeric_limits<HPRLP_FLOAT>::infinity());
    if (!one_sided)
        return {false, 0, iter,
                std::numeric_limits<HPRLP_FLOAT>::quiet_NaN(), false};
    const int current_side = rp > stopping_tolerance ? 1 : -1;
    const HPRLP_FLOAT target = std::max(rp, rd);
    if (!active || side != current_side || !std::isfinite(best_target) ||
        best_target <= 0.0 || iter < best_iter ||
        target <= 0.9 * best_target)
        return {true, current_side, iter, target, false};
    if (stall_window_iterations > 0 &&
        iter - best_iter >= stall_window_iterations)
        return {true, current_side, iter, target, true};
    return {true, current_side, best_iter, best_target, false};
}

bool hprlp_one_sided_stall_trial_should_rollback(
    int side, HPRLP_FLOAT target_before, HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual, HPRLP_FLOAT stopping_tolerance) {
    if ((side != 1 && side != -1) || !valid_metric(target_before) ||
        !(std::isfinite(stopping_tolerance) && stopping_tolerance > 0.0))
        return true;
    const HPRLP_FLOAT target = side == 1
        ? std::abs(residual_primal) : std::abs(residual_dual);
    if (!valid_metric(target)) return true;
    if (target <= stopping_tolerance) return false;
    return target >
        HPRLP_ONE_SIDED_STALL_REBOUND_FACTOR * target_before;
}

HPRLP_sigma_update_result hprlp_one_sided_stall_sigma_update(
    HPRLP_FLOAT old_sigma, HPRLP_FLOAT residual_primal,
    HPRLP_FLOAT residual_dual) {
    if (!(std::isfinite(old_sigma) && old_sigma > 0.0)) old_sigma = 1.0;
    const HPRLP_FLOAT rp = std::abs(residual_primal);
    const HPRLP_FLOAT rd = std::abs(residual_dual);
    if (!valid_metric(rp) || !valid_metric(rd))
        return {old_sigma, NAN, 0.0, false,
                "invalid_one_sided_stall_residuals"};
    const HPRLP_FLOAT ratio = clamp_value(
        std::max(rd, kFloor) / std::max(rp, kFloor), 1e-8, 1e8);
    const HPRLP_FLOAT log_step = clamp_value(
        0.5 * std::log(ratio), -std::log(4.0), std::log(4.0));
    const HPRLP_FLOAT sigma = old_sigma * std::exp(log_step);
    return std::isfinite(sigma) && sigma > 0.0
        ? HPRLP_sigma_update_result{sigma, ratio, log_step, true,
                                    "one_sided_stall_balance"}
        : HPRLP_sigma_update_result{old_sigma, ratio, log_step, false,
                                    "invalid_one_sided_stall_balance"};
}

HPRLP_trial_assessment hprlp_assess_sigma_trial(
    HPRLP_FLOAT rp0, HPRLP_FLOAT rd0, HPRLP_FLOAT kkt0,
    HPRLP_FLOAT rp1, HPRLP_FLOAT rd1, HPRLP_FLOAT kkt1,
    HPRLP_FLOAT tolerance) {
    const HPRLP_FLOAT values[] = {rp0, rd0, kkt0, rp1, rd1, kkt1};
    for (HPRLP_FLOAT v : values)
        if (!valid_metric(v)) return {false, HPRLPTrialReason::InvalidMetrics};
    const bool increase = rd0 >= rp0;
    const HPRLP_FLOAT target0 = increase ? rd0 : rp0;
    const HPRLP_FLOAT target1 = increase ? rd1 : rp1;
    if (kkt1 > 1.05 * std::max(kkt0, kFloor))
        return {false, HPRLPTrialReason::KktRebound};
    if (std::min(rp0, rd0) <= kNumericalFloor) {
        const bool stable = target1 <= 1.01 * std::max(target0, kFloor);
        return {stable, stable ? HPRLPTrialReason::TargetResidualStable
            : HPRLPTrialReason::TargetResidualRebound};
    }
    const bool target = target1 <= 0.85 * std::max(target0, kFloor);
    const HPRLP_FLOAT imbalance0 = imbalance(rp0, rd0);
    const HPRLP_FLOAT imbalance1 = imbalance(rp1, rd1);
    const bool contraction = imbalance1 <= 0.9 * imbalance0;
    const bool monotone = rp1 < rp0 && rd1 < rd0 && imbalance1 < imbalance0;
    const HPRLP_FLOAT non_target = increase ? rp1 : rd1;
    const bool slack = target1 <= 0.98 * std::max(target0, kFloor) &&
        std::isfinite(tolerance) && tolerance > 0.0 &&
        non_target <= 0.5 * tolerance && imbalance1 < imbalance0;
    if (slack && !(target && (contraction || monotone)))
        return {true, HPRLPTrialReason::AcceptedFeasibilitySlack};
    if (!target) return {false, HPRLPTrialReason::TargetResidualStalled};
    if (!(contraction || monotone))
        return {false, HPRLPTrialReason::ImbalanceNotContracting};
    return {true, contraction ? HPRLPTrialReason::Accepted
                              : HPRLPTrialReason::AcceptedMonotoneProgress};
}

bool hprlp_sigma_trial_should_keep_terminal(
    HPRLP_FLOAT rp0, HPRLP_FLOAT rd0, HPRLP_FLOAT kkt0,
    HPRLP_FLOAT rp1, HPRLP_FLOAT rd1, HPRLP_FLOAT kkt1) {
    const HPRLP_FLOAT values[] = {rp0, rd0, kkt0, rp1, rd1, kkt1};
    for (HPRLP_FLOAT v : values) if (!valid_metric(v)) return false;
    if (std::min(rp0, rd0) <= kNumericalFloor) return false;
    const bool increase = rd0 >= rp0;
    const HPRLP_FLOAT target0 = increase ? rd0 : rp0;
    const HPRLP_FLOAT target1 = increase ? rd1 : rp1;
    return target1 <= 0.98 * std::max(target0, kFloor) &&
        rp1 < rp0 && rd1 < rd0 &&
        kkt1 <= 1.05 * std::max(kkt0, kFloor) &&
        imbalance(rp1, rd1) > imbalance(rp0, rd0);
}

const char *hprlp_phase_event_name(HPRLPPhaseEvent e) {
    switch (e) {
        case HPRLPPhaseEvent::Entered: return "entered";
        case HPRLPPhaseEvent::ExitedPrimalRebound: return "exited_primal_rebound";
        case HPRLPPhaseEvent::ExitedActiveSet: return "exited_active_set";
        case HPRLPPhaseEvent::ExitedDirection: return "exited_direction";
        default: return "none";
    }
}

const char *hprlp_trial_reason_name(HPRLPTrialReason r) {
    switch (r) {
        case HPRLPTrialReason::Accepted: return "accepted";
        case HPRLPTrialReason::AcceptedMonotoneProgress: return "accepted_monotone_progress";
        case HPRLPTrialReason::AcceptedFeasibilitySlack: return "accepted_feasibility_slack";
        case HPRLPTrialReason::TargetResidualStable: return "target_residual_stable";
        case HPRLPTrialReason::InvalidMetrics: return "invalid_trial_metrics";
        case HPRLPTrialReason::KktRebound: return "kkt_rebound";
        case HPRLPTrialReason::TargetResidualRebound: return "target_residual_rebound";
        case HPRLPTrialReason::TargetResidualStalled: return "target_residual_stalled";
        case HPRLPTrialReason::ImbalanceNotContracting: return "imbalance_not_contracting";
        case HPRLPTrialReason::Phase2Revoked: return "phase2_revoked";
    }
    return "unknown";
}
