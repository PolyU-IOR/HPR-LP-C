void update_y_check_gpu(HPRLP_workspace_gpu *ws) {
    update_y_check_cusparse_gpu(ws);

    // Check mode bypasses signed kernels, so rebuild paired caches before a
    // subsequent signed/signed normal iteration.
    refresh_paired_scaled_cache(ws);
}

void update_y_normal_gpu(HPRLP_workspace_gpu *ws) {
    if (ws->y_backend == HPRLPYBackend::GridSlackLaplacian &&
        ws->grid_slack_laplacian_operator_ready) {
        update_y_normal_grid_slack_laplacian_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::FactorizedStencil &&
               ws->windowed_stencil_operator_ready &&
               ws->factorized_static_records_ready &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->unit_scaled_y != nullptr &&
               ws->inverse_row_norm != nullptr) {
        update_y_normal_factorized_stencil_gpu(ws);
    } else if (
        ws->y_backend == HPRLPYBackend::FixedDegreePackedDictionary &&
        ws->dictionary_operator_y_ready && ws->fixed_degree_A_plan.ready &&
        ws->A_packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32 &&
        ws->unit_scaled_x_hat != nullptr &&
        ws->inverse_col_norm != nullptr &&
        ws->inverse_row_norm != nullptr) {
        update_y_normal_fixed_degree_packed_dictionary_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::PackedDictionary &&
               ws->dictionary_operator_y_ready &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->inverse_col_norm != nullptr &&
               ws->inverse_row_norm != nullptr) {
        update_y_normal_packed_dictionary_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::SegmentedFused &&
               ws->segmented_A_ready) {
        update_y_normal_segmented_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::GenericFused) {
        update_y_normal_generic_fused_gpu(ws);
    } else if (is_signed_y_backend(ws->y_backend) &&
               ws->signed_unit_operator_ready &&
               ws->signed_unit_operator != nullptr) {
        update_y_normal_signed_unit_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::StructuredOriginal &&
               ws->structured_operator_ready) {
        update_y_normal_structured_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::AffineBlock &&
               ws->affine_block_operator_ready &&
               ws->affine_block_operator != nullptr &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->unit_scaled_y != nullptr &&
               ws->inverse_row_norm != nullptr) {
        update_y_normal_affine_block_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::RowTemplate &&
               ws->row_template_operator_ready &&
               ws->row_template_operator != nullptr &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->unit_scaled_y != nullptr &&
               ws->inverse_row_norm != nullptr) {
        update_y_normal_row_template_gpu(ws);
    } else if (
        ws->y_backend == HPRLPYBackend::UnitActiveScatter &&
        ws->x_backend == HPRLPXBackend::UnitFactorized &&
        ws->unit_operator_x_ready && ws->unit_operator_y_ready &&
        ws->unit_AT_col_index_u16 != nullptr &&
        hprlp_all_rows_fit_scalar(
            ws->n, ws->max_AT_row_nnz, HPRLP_UNIT_SCALAR_ROW_MAX_NNZ)) {
        update_y_normal_unit_active_scatter_gpu(ws);
    } else if (
        (ws->y_backend == HPRLPYBackend::UnitColTile ||
         ws->y_backend == HPRLPYBackend::UnitColTileZeroBitset) &&
        ws->unit_coltile_ready && ws->unit_coltile != nullptr &&
        ws->inverse_row_norm != nullptr &&
        ws->inverse_col_norm != nullptr &&
        ws->unit_scaled_x_hat != nullptr &&
        (ws->y_backend != HPRLPYBackend::UnitColTileZeroBitset ||
         ws->unit_scaled_x_zero_bits != nullptr)) {
        update_y_normal_unit_coltile_gpu(ws);
    } else if (ws->y_backend == HPRLPYBackend::UnitFactorized &&
               ws->unit_operator_y_ready) {
        update_y_normal_unit_factorized_gpu(ws);
    } else {
        update_y_normal_cusparse_gpu(ws);
    }
}
