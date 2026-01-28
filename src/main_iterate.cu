#include "main_iterate.h"
#include "cuda_kernels/cuda_check.h"
#include <iostream>

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


void collect_residuals(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling, HPRLP_residuals *residual, int iter) {
    int n = ws->n;
    int m = ws->m;

    // 1. Obj
    HPRLP_FLOAT obj_scale = scaling->b_scale * scaling->c_scale;
    residual->primal_obj_bar = obj_scale * inner_product(ws->c, ws->x_bar, n, ws->cublasHandle) + lp->obj_constant;
    residual->dual_obj_bar = obj_scale * (inner_product(ws->y_obj, ws->y_bar, m, ws->cublasHandle) +
                                          inner_product(ws->x_bar, ws->z_bar, n, ws->cublasHandle)) + lp->obj_constant;
    residual->rel_gap_bar = std::abs(residual->primal_obj_bar - residual->dual_obj_bar) / (1.0 + std::abs(residual->primal_obj_bar) + std::abs(residual->dual_obj_bar));

        
    // 2. check the dual feasibility
    residual_compute_Rd_cusparse(ws, scaling);
    residual->err_Rd_org_bar = scaling->c_scale * l2_norm(ws->Rd, n, ws->cublasHandle) / scaling->norm_c_org;


    // 3.check the primal feasibility
    residual_compute_Rp_cusparse(ws, scaling);
    residual->err_Rp_org_bar = scaling->b_scale * l2_norm(ws->Rp, m, ws->cublasHandle) / scaling->norm_b_org;

    if(iter == 0) {
        residual_compute_lu_kernel<<<numBlocks(ws->n), numThreads>>>(scaling->col_norm, ws->x_temp, ws->x_bar, ws->l, ws->u, ws->n);
        residual->err_Rp_org_bar = std::max(residual->err_Rp_org_bar, scaling->b_scale * l2_norm(ws->x_temp, n, ws->cublasHandle));
    }

    residual->KKTx_and_gap_org_bar = std::max(std::max(residual->err_Rd_org_bar, residual->err_Rp_org_bar), residual->rel_gap_bar);
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
        axpby(1.0, ws->x_bar, -1.0, ws->last_x, ws->x_temp, ws->n);
        axpby(1.0, ws->y_bar, -1.0, ws->last_y, ws->y_temp, ws->m);

        HPRLP_FLOAT primal_move = l2_norm(ws->x_temp, ws->n, ws->cublasHandle);
        HPRLP_FLOAT dual_move = l2_norm(ws->y_temp, ws->m, ws->cublasHandle);
        if (primal_move > 1e-16 && dual_move > 1e-16 && primal_move < 1e12 && dual_move < 1e12) {
            HPRLP_FLOAT pm_over_dm = primal_move / dual_move;
            HPRLP_FLOAT sqrt_lambda = sqrt(ws->lambda_max);
            HPRLP_FLOAT ratio = pm_over_dm / sqrt_lambda;
            HPRLP_FLOAT fact = std::exp(-0.05 * (restart_info->current_gap / restart_info->best_gap));
            HPRLP_FLOAT temp1 = std::max(std::min(residuals->err_Rd_org_bar, residuals->err_Rp_org_bar),
                                            std::min(residuals->rel_gap_bar, restart_info->current_gap));
            
            HPRLP_FLOAT sigma_cand = std::exp(fact * std::log(ratio) + (1 - fact) * std::log(restart_info->best_sigma));
            HPRLP_FLOAT kappa;
            if(temp1 > 9e-10) {
                kappa = 1.0;
            }
            else if(temp1 > 5e-10) {
                HPRLP_FLOAT ratio_infeas_org = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
                kappa = std::max(std::min(std::sqrt(ratio_infeas_org), 100.0), 1e-2);
            }
            else{
                HPRLP_FLOAT ratio_infeas_org = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
                kappa = std::max(std::min(ratio_infeas_org, 100.0), 1e-2);
            }
            ws->sigma = kappa * sigma_cand;
        }
        else ws->sigma = 1.0;
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

<<<<<<< HEAD
void update_zx_check_gpu(HPRLP_workspace_gpu *ws) {
    cusparseSpMV(ws->spmv_AT->cusparseHandle, ws->spmv_AT->_operator,
                 &ws->spmv_AT->alpha, ws->spmv_AT->AT_cusparseDescr, ws->spmv_AT->y_cusparseDescr,
                 &ws->spmv_AT->beta, ws->spmv_AT->ATy_cusparseDescr, ws->spmv_AT->computeType,
                 ws->spmv_AT->alg, ws->spmv_AT->buffer);

    update_zx_check_kernel<<<numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x_temp, ws->x, ws->z_bar, ws->x_bar, ws->x_hat,
        ws->l, ws->u, ws->sigma, ws->ATy, ws->c, ws->last_x,
        ws->Halpern_params, ws->n);
}

void update_zx_normal_gpu(HPRLP_workspace_gpu *ws) {
    cusparseSpMV(ws->spmv_AT->cusparseHandle, ws->spmv_AT->_operator,
                 &ws->spmv_AT->alpha, ws->spmv_AT->AT_cusparseDescr, ws->spmv_AT->y_cusparseDescr,
                 &ws->spmv_AT->beta, ws->spmv_AT->ATy_cusparseDescr, ws->spmv_AT->computeType,
                 ws->spmv_AT->alg, ws->spmv_AT->buffer);

    update_zx_normal_kernel<<<numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x, ws->x_hat,
        ws->l, ws->u, ws->sigma,
        ws->ATy, ws->c, ws->last_x,
        ws->Halpern_params, ws->n);
}

void update_y_check_gpu(HPRLP_workspace_gpu *ws) {
    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_hat_cusparseDescr, 
                &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                ws->spmv_A->alg, ws->spmv_A->buffer);
=======

void update_z_x(HPRLP_workspace_gpu *ws, HPRLP_FLOAT fact1, HPRLP_FLOAT fact2) {
    cusparseSpMV(ws->spmv_AT->cusparseHandle, ws->spmv_AT->_operator,
                    &ws->spmv_AT->alpha, ws->spmv_AT->AT_cusparseDescr, ws->spmv_AT->y_cusparseDescr, 
                    &ws->spmv_AT->beta, ws->spmv_AT->ATy_cusparseDescr, ws->spmv_AT->computeType,
                    ws->spmv_AT->alg, ws->spmv_AT->buffer);

    // directly combine Halpern step here 
    if(ws->check){
        compute_zx_kernel<<<numBlocks(ws->n), numThreads>>>(ws->x_temp, ws->x, ws->z_bar, ws->x_bar, ws->x_hat,
                                                            ws->l, ws->u, ws->sigma, ws->ATy, ws->c, ws->last_x,
                                                            fact1, fact2, ws->n);
    }
    else{
        compute_x_kernel<<<numBlocks(ws->n), numThreads>>>(ws->x, ws->x_hat, ws->l, ws->u, ws->sigma,
                                                            ws->ATy, ws->c, ws->last_x, fact1, fact2, ws->n);
    }
}


void update_y(HPRLP_workspace_gpu *ws, HPRLP_FLOAT halpern_fact1, HPRLP_FLOAT halpern_fact2) {

    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                            &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_hat_cusparseDescr, 
                            &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                            ws->spmv_A->alg, ws->spmv_A->buffer);
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
    
    HPRLP_FLOAT fact1 = ws->lambda_max * ws->sigma;
    HPRLP_FLOAT fact2 = 1.0 / fact1;

<<<<<<< HEAD
    update_y_check_kernel<<<numBlocks(ws->m), numThreads, 0, ws->stream>>>(
        ws->y_temp, ws->y_bar, ws->y, ws->y_obj, ws->AL, ws->AU,
        ws->Ax, fact1, fact2, ws->last_y, ws->Halpern_params, ws->m);
}


void update_y_normal_gpu(HPRLP_workspace_gpu *ws) {
    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_hat_cusparseDescr, 
                &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                ws->spmv_A->alg, ws->spmv_A->buffer);
    
    HPRLP_FLOAT fact1 = ws->lambda_max * ws->sigma;
    HPRLP_FLOAT fact2 = 1.0 / fact1;

    update_y_normal_kernel<<<numBlocks(ws->m), numThreads, 0, ws->stream>>>(
        ws->y, ws->AL, ws->AU, ws->Ax, fact1, fact2, 
        ws->last_y, ws->Halpern_params, ws->m);
=======
    if(ws->check){
        compute_y_1_kernel<<<numBlocks(ws->m), numThreads>>>(ws->y_temp, ws->y_bar, ws->y, ws->y_obj, ws->AL, ws->AU,
                                                            ws->Ax, fact1, fact2, ws->last_y, halpern_fact1, halpern_fact2, ws->m);
    }
    else{
        compute_y_2_kernel<<<numBlocks(ws->m), numThreads>>>(ws->y, ws->AL, ws->AU, ws->Ax, fact1, fact2, ws->last_y, halpern_fact1, halpern_fact2, ws->m);
    }
>>>>>>> fbb102f935dec8faba4968ef6258196134cb9a4e
}


HPRLP_FLOAT compute_weighted_norm(HPRLP_workspace_gpu *ws) {

    cusparseSpMV(ws->spmv_A->cusparseHandle, ws->spmv_A->_operator,
                            &ws->spmv_A->alpha, ws->spmv_A->A_cusparseDescr, ws->spmv_A->x_temp_cusparseDescr, 
                            &ws->spmv_A->beta, ws->spmv_A->Ax_cusparseDescr, ws->spmv_A->computeType,
                            ws->spmv_A->alg, ws->spmv_A->buffer);

    HPRLP_FLOAT dot_prod = 2 * inner_product(ws->Ax, ws->y_temp, ws->m, ws->cublasHandle);
    HPRLP_FLOAT dy_squarenorm = inner_product(ws->y_temp, ws->y_temp, ws->m, ws->cublasHandle);
    HPRLP_FLOAT dx_squarenorm = inner_product(ws->x_temp, ws->x_temp, ws->n, ws->cublasHandle);

    HPRLP_FLOAT weighted_norm = ws->sigma * (ws->lambda_max * dy_squarenorm) + (dx_squarenorm) / ws->sigma + dot_prod;

    if (weighted_norm < 0) {
        std::cout << "The estimated value of lambda_max is too small!\n";
        ws->lambda_max = -(dot_prod + (dx_squarenorm) / ws->sigma) / (ws->sigma * (dy_squarenorm)) * 1.05;
        weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws->sigma) * 0.05);
    } else {
        weighted_norm = sqrt(weighted_norm);
    }
    return weighted_norm;
}
