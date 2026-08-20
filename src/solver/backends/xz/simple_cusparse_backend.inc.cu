void update_zx_check_cusparse_gpu(HPRLP_workspace_gpu *ws) {
    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_AT->cusparseHandle, ws->spmv_AT->operation,
        &ws->spmv_AT->alpha, &ws->spmv_AT->beta,
        ws->spmv_AT->y_cusparseDescr, ws->spmv_AT->ATy_cusparseDescr,
        ws->spmv_AT->ATy_cusparseDescr));

    update_zx_check_kernel<<<numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x_temp, ws->x, ws->z_bar, ws->x_bar, ws->x_hat,
    ws->l, ws->u, ws->ATy, ws->c, ws->last_x,
    ws->Halpern_params, ws->halpern_factors, ws->n);
}

void update_zx_normal_cusparse_gpu(HPRLP_workspace_gpu *ws) {
    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_AT->cusparseHandle, ws->spmv_AT->operation,
        &ws->spmv_AT->alpha, &ws->spmv_AT->beta,
        ws->spmv_AT->y_cusparseDescr, ws->spmv_AT->ATy_cusparseDescr,
        ws->spmv_AT->ATy_cusparseDescr));

    update_zx_normal_kernel<<<numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x, ws->x_hat,
        ws->l, ws->u, ws->ATy, ws->c, ws->last_x,
        ws->Halpern_params, ws->halpern_factors, ws->n);
}
