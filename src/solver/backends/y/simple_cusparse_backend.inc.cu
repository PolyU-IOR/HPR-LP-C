void update_y_check_cusparse_gpu(HPRLP_workspace_gpu *ws) {
    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_A->cusparseHandle, ws->spmv_A->operation,
        &ws->spmv_A->alpha, &ws->spmv_A->beta,
        ws->spmv_A->x_hat_cusparseDescr, ws->spmv_A->Ax_cusparseDescr,
        ws->spmv_A->Ax_cusparseDescr));

    update_y_check_kernel<<<HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0, ws->stream>>>(
        ws->y_temp, ws->y_bar, ws->y, ws->y_obj, ws->AL, ws->AU,
        ws->Ax, ws->last_y, ws->Halpern_params, ws->halpern_factors, ws->m);

}

void update_y_normal_cusparse_gpu(HPRLP_workspace_gpu *ws) {
    CUSPARSE_CHECK(hprlp_run_spmvop(
        ws->spmv_A->cusparseHandle, ws->spmv_A->operation,
        &ws->spmv_A->alpha, &ws->spmv_A->beta,
        ws->spmv_A->x_hat_cusparseDescr, ws->spmv_A->Ax_cusparseDescr,
        ws->spmv_A->Ax_cusparseDescr));

    update_y_normal_kernel<<<HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0, ws->stream>>>(
        ws->y, ws->AL, ws->AU, ws->Ax,
        ws->last_y, ws->Halpern_params, ws->halpern_factors, ws->m);

}
