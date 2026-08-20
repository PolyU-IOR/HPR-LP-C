void do_restart(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info) {
    if (restart_info->restart_flag > 0) {
        vMemcpy_device(ws->last_x, ws->x_bar, ws->n);
        vMemcpy_device(ws->last_y, ws->y_bar, ws->m);
        vMemcpy_device(ws->x, ws->x_bar, ws->n);
        vMemcpy_device(ws->y, ws->y_bar, ws->m);
        restart_info->inner = 0;
        restart_info->times += 1;
        restart_info->save_gap = std::numeric_limits<HPRLP_FLOAT>::infinity();
    }
}

void check_restart(HPRLP_restart *restart_info, int iter, int check_iter, HPRLP_FLOAT sigma) {
    restart_info->restart_flag = 0;

    if(restart_info->first_restart) {
        if(iter == check_iter) {
            restart_info->first_restart = false;
            restart_info->restart_flag = 1;
            restart_info->best_gap = restart_info->current_gap;
            restart_info->best_sigma = sigma;
        }
    } else {
        if(iter % check_iter == 0) {
            if(restart_info->current_gap < 0) {
                restart_info->current_gap = 1e-6;
                std::cout << "current_gap < 0" << std::endl;
            }

            if(restart_info->current_gap <= 0.2 * restart_info->last_gap) {
                restart_info->sufficient += 1;
                restart_info->restart_flag = 1;
            }

            if((restart_info->current_gap <= 0.6 * restart_info->last_gap) && (restart_info->current_gap > 1.00 * restart_info->save_gap)) {
                restart_info->necessary += 1;
                restart_info->restart_flag = 2;
            }

            if(restart_info->inner >= 0.2 * iter) {
                restart_info->_long += 1;
                restart_info->restart_flag = 3;
            }

            if(restart_info->best_gap > restart_info->current_gap) {
                restart_info->best_gap = restart_info->current_gap;
                restart_info->best_sigma = sigma;
            }

            restart_info->save_gap = restart_info->current_gap;
        }
    }
}


void update_sigma(HPRLP_restart *restart_info, HPRLP_workspace_gpu *ws, HPRLP_residuals *residuals) {
    if(restart_info->restart_flag > 0) {
        // Compute movement vectors into x_temp / y_temp.
        const bool use_host_movement_norm =
            hprlp_use_host_movement_norm(ws->n);
        axpby(1.0, ws->x_bar, -1.0, ws->last_x, ws->x_temp, ws->n,
              ws->stream);
        axpby(1.0, ws->y_bar, -1.0, ws->last_y, ws->y_temp, ws->m,
              ws->stream);

        // Queue nrm2 into slots 8-9 (device-mode, async), then fetch once.
        HPRLP_FLOAT primal_move_host = 0.0;
        if (use_host_movement_norm) {
            primal_move_host = l2_norm(
                ws->x_temp, ws->n, ws->cublasHandle);
        } else {
            queue_nrm2(ws->reduction_scalars, 8, ws->x_temp, ws->n,
                       ws->cublasHandle_device);
        }
        queue_nrm2(ws->reduction_scalars, 9, ws->y_temp, ws->m, ws->cublasHandle_device);
        fetch_reduction_scalars(ws);

        HPRLP_FLOAT primal_move = use_host_movement_norm
            ? primal_move_host : ws->reduction_scalars_host[8];
        HPRLP_FLOAT dual_move   = ws->reduction_scalars_host[9];

        if (primal_move > 1e-16 && dual_move > 1e-16 && primal_move < 1e12 && dual_move < 1e12) {
            HPRLP_FLOAT pm_over_dm = primal_move / dual_move;
            HPRLP_FLOAT sqrt_lambda = sqrt(ws->lambda_max);
            HPRLP_FLOAT ratio       = pm_over_dm / sqrt_lambda;
            HPRLP_FLOAT fact        = std::exp(-0.05 * (restart_info->current_gap / restart_info->best_gap));
            HPRLP_FLOAT temp1       = std::max(std::min(residuals->err_Rd_org_bar, residuals->err_Rp_org_bar),
                                               std::min(residuals->rel_gap_bar, restart_info->current_gap));
            HPRLP_FLOAT sigma_cand  = std::exp(fact * std::log(ratio) + (1 - fact) * std::log(restart_info->best_sigma));
            HPRLP_FLOAT kappa;
            if (temp1 > 9e-10) {
                kappa = 1.0;
            } else if (temp1 > 5e-10) {
                HPRLP_FLOAT ratio_infeas = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
                kappa = std::max(std::min(std::sqrt(ratio_infeas), 100.0), 1e-2);
            } else {
                HPRLP_FLOAT ratio_infeas = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
                kappa = std::max(std::min(ratio_infeas, 100.0), 1e-2);
            }
            ws->sigma = kappa * sigma_cand;
        } else {
            ws->sigma = 1.0;
        }
    }
}

std::string check_stopping(HPRLP_residuals *residuals, int iter, std::chrono::steady_clock::time_point t_start, const HPRLP_parameters *param) {
    if (residuals->KKTx_and_gap_org_bar < param->stop_tol) {
        return "OPTIMAL";
    }

    if (iter >= param->max_iter) {
        return "ITER_LIMIT";
    }

    if (time_since(t_start) > param->time_limit) {
        return "TIME_LIMIT";
    }

    return "CONTINUE";
}
