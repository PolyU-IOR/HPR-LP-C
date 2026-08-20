HPRLP_FLOAT compute_weighted_norm(HPRLP_workspace_gpu *ws) {

    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_A->cusparseHandle, ws->spmv_A->operation,
        &ws->spmv_A->alpha, &ws->spmv_A->beta,
        ws->spmv_A->x_temp_cusparseDescr, ws->spmv_A->Ax_cusparseDescr,
        ws->spmv_A->Ax_cusparseDescr));

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
