void do_restart(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info) {
    if (restart_info->restart_flag <= 0) return;
    vMemcpy_device(ws->last_x, ws->x_bar, ws->n);
    vMemcpy_device(ws->last_y, ws->y_bar, ws->m);
    vMemcpy_device(ws->x, ws->x_bar, ws->n);
    vMemcpy_device(ws->y, ws->y_bar, ws->m);
    restart_info->last_gap = restart_info->current_gap;
    restart_info->inner = 0;
    restart_info->times += 1;
    restart_info->save_gap = std::numeric_limits<HPRLP_FLOAT>::infinity();
}

void check_restart(HPRLP_restart *r, int iter,
                   const HPRLP_parameters *p, HPRLP_FLOAT sigma,
                   const HPRLP_progress_metrics &progress) {
    r->restart_flag = 0;
    if (r->first_restart) {
        if (iter == p->check_iter) {
            r->first_restart = false;
            r->restart_flag = 1;
            r->best_gap = r->current_gap;
            r->best_sigma = sigma;
            r->last_gap = r->current_gap;
        }
        return;
    }
    if (iter % p->check_iter != 0) return;

    const bool cooldown_ready = std::isfinite(r->current_gap) &&
        r->current_gap >= 0.0 &&
        (!std::isfinite(progress.active_set_change_ratio) ||
         progress.active_set_change_ratio <= 0.1);
    if (p->enable_progress_control && p->restart_cooldown_checks > 0 &&
        cooldown_ready &&
        r->inner <= p->restart_cooldown_checks * p->check_iter) {
        if (r->best_gap > r->current_gap) {
            r->best_gap = r->current_gap;
            r->best_sigma = sigma;
        }
        r->last_gap = r->current_gap;
        r->save_gap = r->current_gap;
        return;
    }
    if (r->current_gap < 0.0) {
        r->current_gap = 1e-6;
        std::cout << "current_gap < 0" << std::endl;
    }
    if (r->current_gap <= 0.2 * r->last_gap) {
        ++r->sufficient;
        r->restart_flag = 1;
    }
    if (r->current_gap <= 0.6 * r->last_gap &&
        r->current_gap > r->save_gap) {
        ++r->necessary;
        r->restart_flag = 2;
    }
    const HPRLP_FLOAT threshold = 0.2 * iter;
    const bool due = r->inner >= threshold;
    const HPRLP_FLOAT relative_decrease =
        std::isfinite(r->save_gap) && r->save_gap > 0.0
        ? (r->save_gap - r->current_gap) / r->save_gap
        : std::numeric_limits<HPRLP_FLOAT>::infinity();
    const bool progress_defers = p->use_progress_restart_guard &&
        std::isfinite(progress.active_set_change_ratio) &&
        progress.active_set_change_ratio >= 1e-4 &&
        progress.active_set_change_ratio <= 1e-2 &&
        std::isfinite(progress.step_direction_cosine) &&
        progress.step_direction_cosine >= 0.95 &&
        progress.step_direction_cosine < 0.99;
    const bool defer = due && r->current_gap < r->save_gap &&
        relative_decrease >= 0.01 && progress_defers &&
        r->inner - threshold < 0.8 * p->check_iter;
    if (due && !defer) {
        ++r->_long;
        r->restart_flag = 3;
    }
    if (r->best_gap > r->current_gap) {
        r->best_gap = r->current_gap;
        r->best_sigma = sigma;
    }
    r->save_gap = r->current_gap;
}

bool check_sigma_rebalance_restart(
    HPRLP_restart *r, const HPRLP_residuals *residuals, int iter,
    const HPRLP_parameters *p, const HPRLP_control_state &control,
    const HPRLP_progress_metrics &progress, HPRLP_FLOAT ratio_threshold) {
    if (r->restart_flag != 0 || !control.phase2_identified ||
        control.phase2_age_checks < 5 || r->inner < 5 * p->check_iter ||
        !std::isfinite(progress.active_set_change_ratio) ||
        progress.active_set_change_ratio > 1e-3 ||
        !hprlp_phase2_direction_ready(progress) ||
        (std::isfinite(progress.step_direction_cosine) &&
         progress.step_direction_cosine > 1.0)) return false;

    const HPRLP_FLOAT rp = std::abs(residuals->err_Rp_org_bar);
    const HPRLP_FLOAT rd = std::abs(residuals->err_Rd_org_bar);
    const HPRLP_FLOAT kkt = std::abs(residuals->KKTx_and_gap_org_bar);
    if (!std::isfinite(rp) || !std::isfinite(rd) || !std::isfinite(kkt))
        return false;
    if (std::isfinite(p->stop_tol) && p->stop_tol > 0.0 &&
        std::max(rp, rd) <= p->stop_tol) {
        if ((p->debug_restart || p->debug_sigma) &&
            !r->sigma_feasibility_skip_reported) {
            std::cout << "Sigma rebalance skipped: feasibility residuals already converged"
                      << ", residual_primal=" << rp
                      << ", residual_dual=" << rd
                      << ", stopping_tolerance=" << p->stop_tol
                      << ", kkt=" << kkt << std::endl;
            r->sigma_feasibility_skip_reported = true;
        }
        return false;
    }
    r->sigma_feasibility_skip_reported = false;
    const bool settled = std::isfinite(progress.step_direction_cosine) &&
        progress.step_direction_cosine > 0.993;
    if (settled && std::min(rp, rd) > 1e-7) return false;
    if (kkt > (settled ? 1e-3 : 1.5e-2)) return false;
    if (std::isfinite(control.phase2_entry_primal) &&
        control.phase2_entry_primal >= 0.0 &&
        rp > 1.25 * std::max(control.phase2_entry_primal, 1e-12))
        return false;
    const HPRLP_FLOAT ratio = std::max(rd, 1e-12) / std::max(rp, 1e-12);
    const HPRLP_FLOAT threshold = std::max(ratio_threshold, 1.0);
    if (ratio < threshold && ratio > 1.0 / threshold) return false;
    r->restart_flag = HPRLP_SIGMA_REBALANCE_RESTART_FLAG;
    if (p->debug_restart || p->debug_sigma)
        std::cout << "Sigma rebalance: restart at iteration " << iter
                  << ", residual_ratio_dual_over_primal=" << ratio << std::endl;
    return true;
}

void update_sigma(HPRLP_restart *r, HPRLP_workspace_gpu *ws,
                  HPRLP_residuals *residuals,
                  const HPRLP_parameters *p) {
    if (r->restart_flag < 1 || r->restart_flag > 4) return;
    if (std::isfinite(p->fixed_sigma) && p->fixed_sigma > 0.0) {
        ws->sigma = p->fixed_sigma;
        return;
    }
    const HPRLP_FLOAT old_sigma = ws->sigma;
    if (r->restart_flag == HPRLP_SIGMA_REBALANCE_RESTART_FLAG &&
        p->enable_progress_control) {
        const HPRLP_sigma_update_result result =
            hprlp_residual_balanced_sigma_update(
                old_sigma, residuals->err_Rp_org_bar,
                residuals->err_Rd_org_bar);
        if (result.valid) {
            ws->sigma = result.sigma;
            if (p->debug_sigma) {
                std::cout << std::scientific << std::setprecision(16)
                          << "Sigma update: restart_flag=" << r->restart_flag
                          << ", old_sigma=" << old_sigma
                          << ", sigma_candidate=n/a"
                          << ", residual_primal=" << residuals->err_Rp_org_bar
                          << ", residual_dual=" << residuals->err_Rd_org_bar
                          << ", relative_gap=" << residuals->rel_gap_bar
                          << ", reason=" << result.reason
                          << ", new_sigma=" << ws->sigma
                          << std::defaultfloat << std::setprecision(2)
                          << std::endl;
            }
            return;
        }
    }
    const bool use_host_movement_norm =
        hprlp_use_host_movement_norm(ws->n);
    axpby(1.0, ws->x_bar, -1.0, ws->last_x, ws->x_temp, ws->n,
          ws->stream);
    axpby(1.0, ws->y_bar, -1.0, ws->last_y, ws->y_temp, ws->m,
          ws->stream);
    HPRLP_FLOAT primal_move_host = 0.0;
    if (use_host_movement_norm) {
        // Complete the ordinary 32-bit cublasDnrm2 at the call site for
        // ultra-wide primal movement. This preserves norm arithmetic while
        // avoiding the unstable queued device-result execution window.
        primal_move_host = l2_norm(ws->x_temp, ws->n, ws->cublasHandle);
    } else {
        queue_nrm2(ws->reduction_scalars, 8, ws->x_temp, ws->n,
                   ws->cublasHandle_device);
    }
    queue_nrm2(ws->reduction_scalars, 9, ws->y_temp, ws->m,
               ws->cublasHandle_device);
    fetch_reduction_scalars(ws);
    const HPRLP_FLOAT primal_move = use_host_movement_norm
        ? primal_move_host : ws->reduction_scalars_host[8];
    const HPRLP_FLOAT dual_move = ws->reduction_scalars_host[9];
    HPRLP_FLOAT candidate = std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    HPRLP_FLOAT blend = 0.0;
    const HPRLP_FLOAT best_sigma =
        std::isfinite(r->best_sigma) && r->best_sigma > 0.0
        ? r->best_sigma : old_sigma;
    if (primal_move > 1e-16 && dual_move > 1e-16 &&
        primal_move < 1e12 && dual_move < 1e12 &&
        std::isfinite(ws->lambda_max) && ws->lambda_max > 0.0 &&
        std::isfinite(best_sigma) && best_sigma > 0.0) {
        const HPRLP_FLOAT ratio =
            (primal_move / dual_move) / std::sqrt(ws->lambda_max);
        const HPRLP_FLOAT gap_ratio =
            std::isfinite(r->current_gap) && std::isfinite(r->best_gap) &&
            r->best_gap > 0.0
            ? std::max(r->current_gap / r->best_gap, 0.0) : 1.0;
        blend = std::exp(-0.05 * gap_ratio);
        candidate = std::exp(blend * std::log(ratio) +
                             (1.0 - blend) * std::log(best_sigma));
    }
    const HPRLP_sigma_update_result result = hprlp_safeguarded_sigma_update(
        old_sigma, candidate, blend, p->enable_progress_control,
        residuals->err_Rp_org_bar, residuals->err_Rd_org_bar, p->stop_tol);
    ws->sigma = result.sigma;
    if (!(std::isfinite(ws->sigma) && ws->sigma > 0.0)) ws->sigma = 1.0;
    if (p->debug_sigma) {
        std::cout << std::scientific << std::setprecision(16)
                          << "Sigma update: restart_flag=" << r->restart_flag
                  << ", old_sigma=" << old_sigma
                  << ", primal_move=" << primal_move
                  << ", dual_move=" << dual_move
                  << ", best_sigma=" << best_sigma
                  << ", sigma_candidate=" << candidate
                  << ", residual_primal=" << residuals->err_Rp_org_bar
                  << ", residual_dual=" << residuals->err_Rd_org_bar
                  << ", relative_gap=" << residuals->rel_gap_bar
                  << ", reason=" << result.reason
                  << ", new_sigma=" << ws->sigma
                          << std::defaultfloat << std::setprecision(2)
                          << std::endl;
    }
}

std::string check_stopping(
    HPRLP_residuals *residuals, int iter,
    std::chrono::steady_clock::time_point t_start,
    const HPRLP_parameters *param) {
    if (residuals->KKTx_and_gap_org_bar < param->stop_tol) return "OPTIMAL";
    if (iter >= param->max_iter) return "ITER_LIMIT";
    if (time_since(t_start) > param->time_limit) return "TIME_LIMIT";
    return "CONTINUE";
}
