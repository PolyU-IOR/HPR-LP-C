#include "solver/restart_control.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

static bool near(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b));
}

int main() {
    assert(!hprlp_use_host_movement_norm(5000000));
    assert(!hprlp_use_host_movement_norm(
        HPRLP_ULTRAWIDE_MOVEMENT_LENGTH - 1));
    assert(hprlp_use_host_movement_norm(
        HPRLP_ULTRAWIDE_MOVEMENT_LENGTH));
    assert(hprlp_use_host_movement_norm(208479461));
    assert(near(hprlp_bounded_direction_cosine(2.0, 4.0, 1.0), 1.0));
    assert(std::isnan(hprlp_bounded_direction_cosine(0.0, 0.0, 1.0)));

    HPRLP_progress_metrics frozen;
    frozen.active_set_change_ratio = 1e-6;
    frozen.step_direction_cosine = NAN;
    frozen.primal_zero_move_ratio = 0.999;
    frozen.stationary_bound_ratio = 0.995;
    assert(hprlp_phase2_frozen_face_ready(frozen));
    assert(hprlp_phase2_direction_ready(frozen));

    HPRLP_progress_metrics coherent;
    coherent.active_set_change_ratio = 1e-4;
    coherent.step_direction_cosine = 0.95;
    bool identified = false;
    int confirmations = 0;
    for (int i = 0; i < 5; ++i) {
        const HPRLP_phase_update update = hprlp_update_phase2_state(
            identified, confirmations, 1e-5, 1e-2, coherent);
        identified = update.identified;
        confirmations = update.confirmations;
    }
    assert(identified);
    coherent.active_set_change_ratio = 0.02;
    const HPRLP_phase_update exit = hprlp_update_phase2_state(
        identified, confirmations, 1e-5, 1e-2, coherent);
    assert(!exit.identified &&
           exit.event == HPRLPPhaseEvent::ExitedActiveSet);

    const HPRLP_sigma_update_result ordinary =
        hprlp_safeguarded_sigma_update(
            1.0, 2.0, 0.5, true, 16.0, 1.0, 2.0);
    assert(near(ordinary.sigma, 2.0));
    assert(std::string(ordinary.reason) ==
           "ordinary_restart_candidate_preserved");
    const HPRLP_sigma_update_result correct_direction =
        hprlp_safeguarded_sigma_update(
            1.0, 0.5, 0.5, true, 16.0, 1.0, 2.0);
    assert(near(correct_direction.sigma, 0.5));
    const HPRLP_sigma_update_result floor_guard =
        hprlp_safeguarded_sigma_update(
            1.0, 0.5, 0.5, true, 1e-10, 1e-4, 1e-12);
    assert(near(floor_guard.sigma, 1.0));

    HPRLP_one_sided_stall_update stall = hprlp_update_one_sided_stall(
        false, 0, 0, NAN, 100, 9e-5, 1e-7, 1e-6, 1000);
    assert(stall.active && stall.side == 1 &&
           !stall.trigger && stall.best_iter == 100);
    stall = hprlp_update_one_sided_stall(
        stall.active, stall.side, stall.best_iter, stall.best_target,
        500, 8e-5, 1e-7, 1e-6, 1000);
    assert(stall.active && !stall.trigger && stall.best_iter == 500);
    stall = hprlp_update_one_sided_stall(
        stall.active, stall.side, stall.best_iter, stall.best_target,
        1500, 7.5e-5, 1e-7, 1e-6, 1000);
    assert(stall.active && stall.trigger && stall.best_iter == 1500);
    stall = hprlp_update_one_sided_stall(
        stall.active, stall.side, stall.best_iter, stall.best_target,
        1600, 5e-7, 4e-7, 1e-6, 1000);
    assert(!stall.active && !stall.trigger);
    stall = hprlp_update_one_sided_stall(
        false, 0, 0, NAN, 1700, 1e-2, 1e-8, 1e-6, 1000);
    assert(!stall.active && !stall.trigger);
    stall = hprlp_update_one_sided_stall(
        false, 0, 0, NAN, 1800, 1e-7, 8e-5, 1e-6, 1000);
    assert(stall.active && stall.side == -1 && !stall.trigger);
    const HPRLP_sigma_update_result stalled =
        hprlp_one_sided_stall_sigma_update(1.0, 1e-4, 1e-7);
    assert(stalled.valid && near(stalled.sigma, 0.25));
    assert(hprlp_one_sided_stall_trial_should_rollback(
        1, 3.2811845146051878e-5, 3.40e-5, 4.3e-7,
        1e-6));
    // The SpMVOp mcf_5000_50_500 path has a 2.4% transient target rebound.
    // Retaining this trial avoids the harmful rollback/sigma escalation.
    assert(!hprlp_one_sided_stall_trial_should_rollback(
        1, 6.0632251957256252e-5, 6.2085737360319029e-5,
        1.2138415667442344e-6, 1e-6));
    assert(!hprlp_one_sided_stall_trial_should_rollback(
        1, 2.9615297324568047e-6, 2.9627491643706875e-6, 8.7e-7,
        1e-6));
    assert(!hprlp_one_sided_stall_trial_should_rollback(
        1, 2.9e-6, 9e-7, 2e-6, 1e-6));

    const HPRLP_sigma_update_result capped =
        hprlp_residual_balanced_sigma_update(1.0, 1.0, 16.0);
    assert(capped.valid && near(capped.sigma, 4.0));
    const HPRLP_sigma_update_result severe =
        hprlp_residual_balanced_sigma_update(1.0, 1e-9, 1e-3);
    assert(severe.valid && near(severe.sigma, 8.0));

    const HPRLP_trial_assessment accepted = hprlp_assess_sigma_trial(
        1.0, 10.0, 10.0, 0.9, 8.0, 10.2, 1e-4);
    assert(accepted.accepted);
    const HPRLP_trial_assessment rejected = hprlp_assess_sigma_trial(
        1.0, 10.0, 10.0, 0.9, 9.9, 11.0, 1e-4);
    assert(!rejected.accepted && rejected.reason == HPRLPTrialReason::KktRebound);
    const HPRLP_trial_assessment floor_stable = hprlp_assess_sigma_trial(
        1e-10, 1.0, 1.0, 2e-10, 1.005, 1.0, 1e-8);
    assert(floor_stable.accepted &&
           floor_stable.reason == HPRLPTrialReason::TargetResidualStable);
    assert(hprlp_sigma_trial_should_keep_terminal(
        1.0, 10.0, 10.0, 0.1, 9.5, 10.0));

    HPRLP_control_state control = hprlp_initialize_control_state(150);
    control.sigma_trial_blocked = true;
    control.sigma_trial_rollback_blocked = true;
    control.sigma_trial_accepted_once = true;
    control.phase2_identified = true;
    control.phase2_age_checks = 5;
    control.sigma_trial_last_resolution_iter = 0;
    coherent.active_set_change_ratio = 1e-6;
    assert(hprlp_sigma_trial_floor_recovery_ready(
        control, 15000, 150, coherent, 1e-11, 1e-3, 1e-4));

    std::cout << "restart_control_test: PASS\n";
    return 0;
}
