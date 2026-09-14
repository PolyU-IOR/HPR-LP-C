void residual_compute_Rp_cusparse(HPRLP_workspace_gpu *ws, Scaling_info *scaling) {

    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_A->cusparseHandle, ws->spmv_A->operation,
        &ws->spmv_A->alpha, &ws->spmv_A->beta,
        ws->spmv_A->x_bar_cusparseDescr, ws->spmv_A->Ax_cusparseDescr,
        ws->spmv_A->Ax_cusparseDescr));

    residual_compute_Rp_kernel<<<HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0, ws->stream>>>(
        scaling->row_norm, ws->Rp, ws->AL, ws->AU, ws->Ax, ws->m);
}


void residual_compute_Rd_cusparse(HPRLP_workspace_gpu *ws, Scaling_info *scaling) {

    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_AT->cusparseHandle, ws->spmv_AT->operation,
        &ws->spmv_AT->alpha, &ws->spmv_AT->beta,
        ws->spmv_AT->y_bar_cusparseDescr,
        ws->spmv_AT->ATy_cusparseDescr,
        ws->spmv_AT->ATy_cusparseDescr));

    residual_compute_Rd_kernel<<<HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0, ws->stream>>>(
        scaling->col_norm, ws->ATy, ws->z_bar, ws->c, ws->Rd, ws->n);
}


void compute_residuals(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling,
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
        CUSPARSE_CHECK(hprlp_run_spmvop(
            ws->spmv_A->cusparseHandle, ws->spmv_A->operation,
            &ws->spmv_A->alpha, &ws->spmv_A->beta,
            ws->spmv_A->x_temp_cusparseDescr,
            ws->spmv_A->Ax_cusparseDescr,
            ws->spmv_A->Ax_cusparseDescr));
        queue_dot(ws->reduction_scalars, 5, ws->Ax, ws->y_temp, m, ws->cublasHandle_device);
        queue_dot(ws->reduction_scalars, 6, ws->y_temp, ws->y_temp, m, ws->cublasHandle_device);
        queue_dot(ws->reduction_scalars, 7, ws->x_temp, ws->x_temp, n, ws->cublasHandle_device);
    }

    // Keep residual and restart reductions on the canonical stream.  Their
    // results control adaptive restarts, so cross-stream reduction scheduling
    // must not perturb the numerical iteration path.
    residual_compute_Rd_cusparse(ws, scaling);
    queue_nrm2(ws->reduction_scalars, 3, ws->Rd, n,
               ws->cublasHandle_device);

    // Compute Rp residual kernel (overwrites Ax), queue slot 4.
    residual_compute_Rp_cusparse(ws, scaling);
    queue_nrm2(ws->reduction_scalars, 4, ws->Rp, m, ws->cublasHandle_device);

    if (iter == 0) {
        residual_compute_lu_kernel<<<
            HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0, ws->stream>>>(
                scaling->col_norm, ws->x_temp, ws->x_bar, ws->l, ws->u,
                ws->n);
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
