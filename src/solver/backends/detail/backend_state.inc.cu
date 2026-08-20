namespace {

void refresh_paired_scaled_cache(HPRLP_workspace_gpu *ws) {
    const bool signed_pair =
        is_signed_x_backend(ws->x_backend) &&
        is_signed_y_backend(ws->y_backend) &&
        ws->signed_unit_operator_ready;
    const bool factorized_stencil_pair =
        ws->x_backend == HPRLPXBackend::FactorizedStencil &&
        ws->y_backend == HPRLPYBackend::FactorizedStencil &&
        ws->windowed_stencil_operator_ready;
    const bool packed_dictionary_pair =
        is_dictionary_x_backend(ws->x_backend) &&
        is_dictionary_y_backend(ws->y_backend) &&
        ws->dictionary_operator_x_ready &&
        ws->dictionary_operator_y_ready;
    const bool row_template_pair =
        ws->x_backend == HPRLPXBackend::RowTemplate &&
        ws->y_backend == HPRLPYBackend::RowTemplate &&
        ws->row_template_operator_ready;
    const bool affine_block_pair =
        ws->x_backend == HPRLPXBackend::AffineBlock &&
        ws->y_backend == HPRLPYBackend::AffineBlock &&
        ws->affine_block_operator_ready;
    const bool unit_coltile_pair =
        ws->x_backend == HPRLPXBackend::UnitFactorized &&
        ws->y_backend == HPRLPYBackend::UnitColTile &&
        ws->unit_operator_x_ready && ws->unit_coltile_ready;
    if ((!signed_pair && !factorized_stencil_pair &&
         !packed_dictionary_pair && !row_template_pair &&
         !affine_block_pair && !unit_coltile_pair) ||
        ws->inverse_row_norm == nullptr ||
        ws->inverse_col_norm == nullptr ||
        ws->unit_scaled_y == nullptr ||
        ws->unit_scaled_x_hat == nullptr) {
        return;
    }
    vector_dot_product_kernel<<<numBlocks(ws->m), numThreads, 0,
                                ws->stream>>>(
        ws->y, ws->inverse_row_norm, ws->unit_scaled_y, ws->m, false);
    vector_dot_product_kernel<<<numBlocks(ws->n), numThreads, 0,
                                ws->stream>>>(
        ws->x_hat, ws->inverse_col_norm, ws->unit_scaled_x_hat, ws->n,
        false);
    if (ws->unit_scaled_x_hat_nonzero != nullptr) {
        mark_raw_positive_zero_flags_kernel<<<
            numBlocks(ws->n), numThreads, 0, ws->stream>>>(
            ws->unit_scaled_x_hat, ws->unit_scaled_x_hat_nonzero, ws->n);
    }
}

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


} // namespace
