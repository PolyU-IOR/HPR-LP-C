#include "main_iterate.h"
#include "cuda_kernels/cuda_check.h"
#include <iostream>

namespace {

constexpr int kFusedThreads = 512;
constexpr int kWarpsPerBlock = kFusedThreads / 32;

struct BackendCandidate {
    bool use_x;
    bool use_y;
};

}

void reset_halpern_runtime_params(HPRLP_workspace_gpu *ws) {
    HPRLP_FLOAT fact1 = ws->lambda_max * ws->sigma;
    ws->iter_params_host[0] = ws->sigma;
    ws->iter_params_host[1] = fact1;
    ws->iter_params_host[2] = 1.0 / fact1;
    ws->iter_params_host[3] = 1.0 / ws->sigma;
    ws->halpern_inner_host[0] = 0;
    ws->halpern_factors_host[0] = 0.5;
    ws->halpern_factors_host[1] = 0.5;

    CUDA_CHECK(cudaMemcpyAsync(ws->Halpern_params, ws->iter_params_host, 4 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_inner, ws->halpern_inner_host, sizeof(int),
                               cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_factors, ws->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));

    ws->uploaded_sigma = ws->sigma;
    ws->uploaded_lambda_max = ws->lambda_max;
}

void upload_halpern_iter_params_if_needed(HPRLP_workspace_gpu *ws) {
    if (ws->sigma == ws->uploaded_sigma && ws->lambda_max == ws->uploaded_lambda_max) {
        return;
    }

    HPRLP_FLOAT y_fact1 = ws->lambda_max * ws->sigma;
    ws->iter_params_host[0] = ws->sigma;
    ws->iter_params_host[1] = y_fact1;
    ws->iter_params_host[2] = 1.0 / y_fact1;
    ws->iter_params_host[3] = 1.0 / ws->sigma;
    CUDA_CHECK(cudaMemcpyAsync(ws->Halpern_params, ws->iter_params_host, 4 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));
    ws->uploaded_sigma = ws->sigma;
    ws->uploaded_lambda_max = ws->lambda_max;
}

void upload_halpern_restart_params(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info) {
    if (restart_info->restart_flag <= 0) {
        return;
    }

    ws->halpern_inner_host[0] = 0.0;
    ws->halpern_factors_host[0] = 0.5;
    ws->halpern_factors_host[1] = 1.0 - ws->halpern_factors_host[0];
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_inner, ws->halpern_inner_host, sizeof(int),
                               cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_factors, ws->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));
}

void advance_halpern_factors(HPRLP_workspace_gpu *ws) {
    advance_halpern_factors_kernel<<<1, 1, 0, ws->stream>>>(ws->halpern_inner, ws->halpern_factors);
}

namespace {

void restore_device_state(HPRLP_workspace_gpu *ws,
                          HPRLP_FLOAT *x_save,
                          HPRLP_FLOAT *x_hat_save,
                          HPRLP_FLOAT *x_bar_save,
                          HPRLP_FLOAT *y_save,
                          HPRLP_FLOAT *y_hat_save,
                          HPRLP_FLOAT *y_bar_save,
                          HPRLP_FLOAT *y_obj_save,
                          HPRLP_FLOAT *z_bar_save,
                          HPRLP_FLOAT *Ax_save,
                          HPRLP_FLOAT *ATy_save) {
    vMemcpy_device(ws->x, x_save, ws->n);
    vMemcpy_device(ws->x_hat, x_hat_save, ws->n);
    vMemcpy_device(ws->x_bar, x_bar_save, ws->n);
    vMemcpy_device(ws->y, y_save, ws->m);
    vMemcpy_device(ws->y_hat, y_hat_save, ws->m);
    vMemcpy_device(ws->y_bar, y_bar_save, ws->m);
    vMemcpy_device(ws->y_obj, y_obj_save, ws->m);
    vMemcpy_device(ws->z_bar, z_bar_save, ws->n);
    vMemcpy_device(ws->Ax, Ax_save, ws->m);
    vMemcpy_device(ws->ATy, ATy_save, ws->n);
}

void save_device_state(HPRLP_workspace_gpu *ws,
                       HPRLP_FLOAT **x_save,
                       HPRLP_FLOAT **x_hat_save,
                       HPRLP_FLOAT **x_bar_save,
                       HPRLP_FLOAT **y_save,
                       HPRLP_FLOAT **y_hat_save,
                       HPRLP_FLOAT **y_bar_save,
                       HPRLP_FLOAT **y_obj_save,
                       HPRLP_FLOAT **z_bar_save,
                       HPRLP_FLOAT **Ax_save,
                       HPRLP_FLOAT **ATy_save) {
    create_zero_vector_device(*x_save, ws->n);
    create_zero_vector_device(*x_hat_save, ws->n);
    create_zero_vector_device(*x_bar_save, ws->n);
    create_zero_vector_device(*y_save, ws->m);
    create_zero_vector_device(*y_hat_save, ws->m);
    create_zero_vector_device(*y_bar_save, ws->m);
    create_zero_vector_device(*y_obj_save, ws->m);
    create_zero_vector_device(*z_bar_save, ws->n);
    create_zero_vector_device(*Ax_save, ws->m);
    create_zero_vector_device(*ATy_save, ws->n);

    vMemcpy_device(*x_save, ws->x, ws->n);
    vMemcpy_device(*x_hat_save, ws->x_hat, ws->n);
    vMemcpy_device(*x_bar_save, ws->x_bar, ws->n);
    vMemcpy_device(*y_save, ws->y, ws->m);
    vMemcpy_device(*y_hat_save, ws->y_hat, ws->m);
    vMemcpy_device(*y_bar_save, ws->y_bar, ws->m);
    vMemcpy_device(*y_obj_save, ws->y_obj, ws->m);
    vMemcpy_device(*z_bar_save, ws->z_bar, ws->n);
    vMemcpy_device(*Ax_save, ws->Ax, ws->m);
    vMemcpy_device(*ATy_save, ws->ATy, ws->n);
}

void free_saved_device_state(HPRLP_FLOAT *x_save,
                             HPRLP_FLOAT *x_hat_save,
                             HPRLP_FLOAT *x_bar_save,
                             HPRLP_FLOAT *y_save,
                             HPRLP_FLOAT *y_hat_save,
                             HPRLP_FLOAT *y_bar_save,
                             HPRLP_FLOAT *y_obj_save,
                             HPRLP_FLOAT *z_bar_save,
                             HPRLP_FLOAT *Ax_save,
                             HPRLP_FLOAT *ATy_save) {
    cudaFree(x_save);
    cudaFree(x_hat_save);
    cudaFree(x_bar_save);
    cudaFree(y_save);
    cudaFree(y_hat_save);
    cudaFree(y_bar_save);
    cudaFree(y_obj_save);
    cudaFree(z_bar_save);
    cudaFree(Ax_save);
    cudaFree(ATy_save);
}

void update_x_z_normal_fused_bucket_gpu(HPRLP_workspace_gpu *ws) {
    if (ws->num_AT_rows_short > 0) {
        fused_update_x_z_rows_short_kernel<<<(ws->num_AT_rows_short + kFusedThreads - 1) / kFusedThreads, kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x, ws->y,
            ws->AT->rowPtr, ws->AT->colIndex, ws->AT->value, ws->Halpern_params, ws->halpern_factors,
            ws->AT_rows_short, ws->num_AT_rows_short);
    }
    if (ws->num_AT_rows_medium > 0) {
        fused_update_x_z_rows_warp_kernel<<<(ws->num_AT_rows_medium + kWarpsPerBlock - 1) / kWarpsPerBlock, kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x, ws->y,
            ws->AT->rowPtr, ws->AT->colIndex, ws->AT->value, ws->Halpern_params, ws->halpern_factors,
            ws->AT_rows_medium, ws->num_AT_rows_medium);
    }
}

void update_y_normal_fused_bucket_gpu(HPRLP_workspace_gpu *ws) {
    if (ws->num_A_rows_short > 0) {
        fused_update_y_rows_short_kernel<<<(ws->num_A_rows_short + kFusedThreads - 1) / kFusedThreads, kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y, ws->x_hat,
            ws->A->rowPtr, ws->A->colIndex, ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_short, ws->num_A_rows_short);
    }
    if (ws->num_A_rows_medium > 0) {
        fused_update_y_rows_warp_kernel<<<(ws->num_A_rows_medium + kWarpsPerBlock - 1) / kWarpsPerBlock, kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y, ws->x_hat,
            ws->A->rowPtr, ws->A->colIndex, ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_medium, ws->num_A_rows_medium);
    }

    advance_halpern_factors(ws);
}

BackendCandidate choose_backend(HPRLP_FLOAT ref_metric,
                                long long ref_time_ns,
                                const std::vector<std::pair<BackendCandidate, std::pair<HPRLP_FLOAT, long long>>> &candidates) {
    BackendCandidate choice{false, false};
    HPRLP_FLOAT allowed_metric = ref_metric + std::max(static_cast<HPRLP_FLOAT>(1e-12), std::abs(ref_metric) * static_cast<HPRLP_FLOAT>(0.01));
    double best_time = std::numeric_limits<double>::infinity();
    for (const auto &candidate : candidates) {
        HPRLP_FLOAT metric = candidate.second.first;
        long long time_ns = candidate.second.second;
        if (!std::isfinite(metric) || metric > allowed_metric || time_ns <= 0) {
            continue;
        }
        if (time_ns <= static_cast<long long>(ref_time_ns * 0.95) && static_cast<double>(time_ns) < best_time) {
            best_time = static_cast<double>(time_ns);
            choice = candidate.first;
        }
    }
    return choice;
}

}

void residual_compute_Rp_cusparse(HPRLP_workspace_gpu *ws, Scaling_info *scaling) {

    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                        &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_bar_cusparseDescr, 
                        &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                        ws->spmv_A->alg, ws->spmv_A->buffer);

    residual_compute_Rp_kernel<<<numBlocks(ws->m), numThreads>>>(scaling->row_norm, ws->Rp, ws->AL, ws->AU, ws->Ax, ws->m);
}


void residual_compute_Rd_cusparse(HPRLP_workspace_gpu *ws, Scaling_info *scaling) {

    cusparseSpMV(ws->spmv_AT->cusparseHandle, ws->spmv_AT->_operator,
                    &ws->spmv_AT->alpha, ws->spmv_AT->AT_cusparseDescr, ws->spmv_AT->y_bar_cusparseDescr, 
                    &ws->spmv_AT->beta, ws->spmv_AT->ATy_cusparseDescr, ws->spmv_AT->computeType,
                    ws->spmv_AT->alg, ws->spmv_AT->buffer);

    residual_compute_Rd_kernel<<<numBlocks(ws->n), numThreads>>>(scaling->col_norm, ws->ATy, ws->z_bar, ws->c, ws->Rd, ws->n);
}


void collect_residuals(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling,
                       HPRLP_residuals *residual, int iter,
                       HPRLP_restart *restart_info, bool compute_gap) {
    int n = ws->n;
    int m = ws->m;
    HPRLP_FLOAT obj_scale = scaling->b_scale * scaling->c_scale;

    // Queue slots 0-2: objective components (non-blocking, device-mode CUBLAS).
    queue_dot(ws->reduction_scalars, 0, ws->c, ws->x_bar, n, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 1, ws->y_obj, ws->y_bar, m, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 2, ws->x_bar, ws->z_bar, n, ws->cublasHandle_device);

    // Queue slots 5-7: restart-gap terms (if requested).
    // Uses x_temp = x_bar - last_x and y_temp = y_bar - last_y set by update_sigma.
    // SpMV A*x_temp → Ax (will be overwritten by Rp SpMV later, but dots 5-7 are
    // already enqueued before that happens — stream-ordering ensures correctness).
    if (compute_gap) {
        cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                     &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr,
                     ws->spmv_A->x_temp_cusparseDescr,
                     &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr,
                     ws->spmv_A->computeType, ws->spmv_A->alg, ws->spmv_A->buffer);
        queue_dot(ws->reduction_scalars, 5, ws->Ax, ws->y_temp, m, ws->cublasHandle_device);
        queue_dot(ws->reduction_scalars, 6, ws->y_temp, ws->y_temp, m, ws->cublasHandle_device);
        queue_dot(ws->reduction_scalars, 7, ws->x_temp, ws->x_temp, n, ws->cublasHandle_device);
    }

    // Compute Rd residual kernel, queue slot 3.
    residual_compute_Rd_cusparse(ws, scaling);
    queue_nrm2(ws->reduction_scalars, 3, ws->Rd, n, ws->cublasHandle_device);

    // Compute Rp residual kernel (overwrites Ax), queue slot 4.
    residual_compute_Rp_cusparse(ws, scaling);
    queue_nrm2(ws->reduction_scalars, 4, ws->Rp, m, ws->cublasHandle_device);

    if (iter == 0) {
        residual_compute_lu_kernel<<<numBlocks(ws->n), numThreads>>>(
            scaling->col_norm, ws->x_temp, ws->x_bar, ws->l, ws->u, ws->n);
    }

    // Single device→host fetch: one cudaMemcpyAsync + stream sync.
    fetch_reduction_scalars(ws);

    // Derive residuals from the host buffer.
    residual->primal_obj_bar = obj_scale * ws->reduction_scalars_host[0] + lp->obj_constant;
    residual->dual_obj_bar = obj_scale * (ws->reduction_scalars_host[1] +
                                          ws->reduction_scalars_host[2]) + lp->obj_constant;
    residual->rel_gap_bar = std::abs(residual->primal_obj_bar - residual->dual_obj_bar) /
                            (1.0 + std::abs(residual->primal_obj_bar) +
                             std::abs(residual->dual_obj_bar));

    residual->err_Rd_org_bar = scaling->c_scale * ws->reduction_scalars_host[3] /
                               scaling->norm_c_org;
    residual->err_Rp_org_bar = scaling->b_scale * ws->reduction_scalars_host[4] /
                               scaling->norm_b_org;

    if (iter == 0) {
        // x_temp was overwritten by residual_compute_lu_kernel above.
        residual->err_Rp_org_bar = std::max(residual->err_Rp_org_bar,
            scaling->b_scale * l2_norm(ws->x_temp, n, ws->cublasHandle));
    }

    residual->KKTx_and_gap_org_bar = std::max(std::max(residual->err_Rd_org_bar, residual->err_Rp_org_bar), residual->rel_gap_bar);

    if (compute_gap && restart_info != nullptr) {
        HPRLP_FLOAT dot_prod      = 2.0 * ws->reduction_scalars_host[5];
        HPRLP_FLOAT dy_squarenorm = ws->reduction_scalars_host[6];
        HPRLP_FLOAT dx_squarenorm = ws->reduction_scalars_host[7];
        HPRLP_FLOAT weighted_norm = ws->sigma * (ws->lambda_max * dy_squarenorm) +
                                    dx_squarenorm / ws->sigma + dot_prod;
        if (weighted_norm < 0) {
            std::cout << "The estimated maximum eigenvalue is too small! Current value is " << ws->lambda_max << "\n";
            ws->lambda_max = -(dot_prod + dx_squarenorm / ws->sigma) / (ws->sigma * dy_squarenorm) * 1.05;
            std::cout << "The new estimated maximum eigenvalue is " << ws->lambda_max << "\n";
            weighted_norm = sqrt(-(dot_prod + dx_squarenorm / ws->sigma) * 0.05);
        } else {
            weighted_norm = sqrt(weighted_norm);
        }
        restart_info->current_gap = weighted_norm;
    }
}


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
        axpby(1.0, ws->x_bar, -1.0, ws->last_x, ws->x_temp, ws->n);
        axpby(1.0, ws->y_bar, -1.0, ws->last_y, ws->y_temp, ws->m);

        // Queue nrm2 into slots 8-9 (device-mode, async), then fetch once.
        queue_nrm2(ws->reduction_scalars, 8, ws->x_temp, ws->n, ws->cublasHandle_device);
        queue_nrm2(ws->reduction_scalars, 9, ws->y_temp, ws->m, ws->cublasHandle_device);
        fetch_reduction_scalars(ws);

        HPRLP_FLOAT primal_move = ws->reduction_scalars_host[8];
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

void update_zx_check_gpu(HPRLP_workspace_gpu *ws) {
    cusparseSpMV(ws->spmv_AT->cusparseHandle, ws->spmv_AT->_operator,
                 &ws->spmv_AT->alpha, ws->spmv_AT->AT_cusparseDescr, ws->spmv_AT->y_cusparseDescr,
                 &ws->spmv_AT->beta, ws->spmv_AT->ATy_cusparseDescr, ws->spmv_AT->computeType,
                 ws->spmv_AT->alg, ws->spmv_AT->buffer);

    update_zx_check_kernel<<<numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x_temp, ws->x, ws->z_bar, ws->x_bar, ws->x_hat,
    ws->l, ws->u, ws->ATy, ws->c, ws->last_x,
    ws->Halpern_params, ws->halpern_factors, ws->n);
}

void update_zx_normal_gpu(HPRLP_workspace_gpu *ws) {
    if (ws->use_custom_fused_x) {
        update_x_z_normal_fused_bucket_gpu(ws);
        return;
    }

    cusparseSpMV(ws->spmv_AT->cusparseHandle, ws->spmv_AT->_operator,
                 &ws->spmv_AT->alpha, ws->spmv_AT->AT_cusparseDescr, ws->spmv_AT->y_cusparseDescr,
                 &ws->spmv_AT->beta, ws->spmv_AT->ATy_cusparseDescr, ws->spmv_AT->computeType,
                 ws->spmv_AT->alg, ws->spmv_AT->buffer);

    update_zx_normal_kernel<<<numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x, ws->x_hat,
        ws->l, ws->u, ws->ATy, ws->c, ws->last_x,
        ws->Halpern_params, ws->halpern_factors, ws->n);
}

void update_y_check_gpu(HPRLP_workspace_gpu *ws) {
    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_hat_cusparseDescr, 
                &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                ws->spmv_A->alg, ws->spmv_A->buffer);

    update_y_check_kernel<<<numBlocks(ws->m), numThreads, 0, ws->stream>>>(
        ws->y_temp, ws->y_bar, ws->y, ws->y_obj, ws->AL, ws->AU,
        ws->Ax, ws->last_y, ws->Halpern_params, ws->halpern_factors, ws->m);

    advance_halpern_factors(ws);
}


void update_y_normal_gpu(HPRLP_workspace_gpu *ws) {
    if (ws->use_custom_fused_y) {
        update_y_normal_fused_bucket_gpu(ws);
        return;
    }

    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_hat_cusparseDescr, 
                &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                ws->spmv_A->alg, ws->spmv_A->buffer);

    update_y_normal_kernel<<<numBlocks(ws->m), numThreads, 0, ws->stream>>>(
        ws->y, ws->AL, ws->AU, ws->Ax,
        ws->last_y, ws->Halpern_params, ws->halpern_factors, ws->m);

    advance_halpern_factors(ws);
}


// Compute the M-weighted norm using x_temp (dx) and y_temp (dy).
// Batches the three inner products into one device→host fetch (no extra sync beyond that).
HPRLP_FLOAT compute_weighted_norm(HPRLP_workspace_gpu *ws) {

    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                            &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_temp_cusparseDescr, 
                            &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                            ws->spmv_A->alg, ws->spmv_A->buffer);

    // Queue three dots into device slots 0-2 (non-blocking, device-pointer mode).
    queue_dot(ws->reduction_scalars, 0, ws->Ax,     ws->y_temp, ws->m, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 1, ws->y_temp, ws->y_temp, ws->m, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 2, ws->x_temp, ws->x_temp, ws->n, ws->cublasHandle_device);

    // Single fetch: one cudaMemcpyAsync + stream sync.
    fetch_reduction_scalars(ws);

    HPRLP_FLOAT dot_prod      = 2.0 * ws->reduction_scalars_host[0];
    HPRLP_FLOAT dy_squarenorm = ws->reduction_scalars_host[1];
    HPRLP_FLOAT dx_squarenorm = ws->reduction_scalars_host[2];

    HPRLP_FLOAT weighted_norm = ws->sigma * (ws->lambda_max * dy_squarenorm) +
                                dx_squarenorm / ws->sigma + dot_prod;
    if (weighted_norm < 0) {
        std::cout << "The estimated value of lambda_max is too small!\n";
        ws->lambda_max = -(dot_prod + (dx_squarenorm) / ws->sigma) / (ws->sigma * (dy_squarenorm)) * 1.05;
        weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws->sigma) * 0.05);
    } else {
        weighted_norm = sqrt(weighted_norm);
    }
    return weighted_norm;
}

void autotune_custom_update_backends(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling, const HPRLP_parameters *param) {
    if (param->CUSPARSE_spmv) {
        ws->use_custom_fused_x = false;
        ws->use_custom_fused_y = false;
        return;
    }

    std::vector<BackendCandidate> candidates{{false, false}, {true, false}, {false, true}, {true, true}};

    if (param->autotune_verbose) {
        std::cout << "AUTO-SELECT custom backends (" << std::min(param->max_iter, param->check_iter)
                  << " iterations per candidate) ..." << std::endl;
        int total_A = ws->num_A_rows_short + ws->num_A_rows_medium;
        int total_AT = ws->num_AT_rows_short + ws->num_AT_rows_medium;
        std::cout << "  A row buckets: short=" << ws->num_A_rows_short << ", medium=" << ws->num_A_rows_medium
                  << ", total=" << total_A << std::endl;
        std::cout << "  AT row buckets: short=" << ws->num_AT_rows_short << ", medium=" << ws->num_AT_rows_medium
                  << ", total=" << total_AT << std::endl;
    }

    HPRLP_FLOAT *x_save = nullptr, *x_hat_save = nullptr, *x_bar_save = nullptr;
    HPRLP_FLOAT *y_save = nullptr, *y_hat_save = nullptr, *y_bar_save = nullptr, *y_obj_save = nullptr;
    HPRLP_FLOAT *z_bar_save = nullptr, *Ax_save = nullptr, *ATy_save = nullptr;
    save_device_state(ws, &x_save, &x_hat_save, &x_bar_save, &y_save, &y_hat_save, &y_bar_save, &y_obj_save, &z_bar_save, &Ax_save, &ATy_save);

    std::vector<std::pair<BackendCandidate, std::pair<HPRLP_FLOAT, long long>>> results;
    results.reserve(candidates.size());
    int bench_iters = std::min(param->max_iter, param->check_iter);

    auto run_probe = [&]() {
        for (int i = 0; i < bench_iters; ++i) {
            update_zx_normal_gpu(ws);
            update_y_normal_gpu(ws);
        }
        update_zx_check_gpu(ws);
        update_y_check_gpu(ws);
        HPRLP_residuals residual{};
        collect_residuals(ws, lp, scaling, &residual, bench_iters);
        CUDA_CHECK(cudaStreamSynchronize(ws->stream));
        return residual.KKTx_and_gap_org_bar;
    };

    for (const auto &candidate : candidates) {
        restore_device_state(ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
        reset_halpern_runtime_params(ws);
        ws->use_custom_fused_x = candidate.use_x;
        ws->use_custom_fused_y = candidate.use_y;
        run_probe();

        restore_device_state(ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
        reset_halpern_runtime_params(ws);
        CUDA_CHECK(cudaStreamSynchronize(ws->stream));
        auto start = std::chrono::steady_clock::now();
        HPRLP_FLOAT merit = run_probe();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        results.push_back({candidate, {merit, elapsed}});

        if (param->autotune_verbose) {
            std::cout << "  candidate x=" << (candidate.use_x ? "fused" : "cusparse")
                      << ", y=" << (candidate.use_y ? "fused" : "cusparse")
                      << " -> " << (static_cast<double>(elapsed) / 1e6) << " ms, merit=" << merit << std::endl;
        }
    }

    BackendCandidate selected = choose_backend(results.front().second.first, results.front().second.second,
                                               std::vector<std::pair<BackendCandidate, std::pair<HPRLP_FLOAT, long long>>>(results.begin() + 1, results.end()));

    restore_device_state(ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
    reset_halpern_runtime_params(ws);
    ws->use_custom_fused_x = selected.use_x;
    ws->use_custom_fused_y = selected.use_y;

    if (param->autotune_verbose) {
        std::cout << "AUTO-SELECT selected x=" << (selected.use_x ? "fused" : "cusparse")
                  << ", y=" << (selected.use_y ? "fused" : "cusparse") << std::endl;
    }

    free_saved_device_state(x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
}
