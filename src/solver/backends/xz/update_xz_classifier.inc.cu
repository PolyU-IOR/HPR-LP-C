void update_zx_check_gpu(HPRLP_workspace_gpu *ws) {
    update_zx_check_cusparse_gpu(ws);
}

void update_zx_normal_gpu(HPRLP_workspace_gpu *ws) {
    if (ws->x_backend == HPRLPXBackend::ScaledCusparse) {
        update_zx_normal_cusparse_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::GridSlackLaplacian &&
               ws->grid_slack_laplacian_operator_ready) {
        update_xz_normal_grid_slack_laplacian_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::FactorizedStencil &&
               ws->windowed_stencil_operator_ready &&
               ws->factorized_static_records_ready &&
               ws->unit_scaled_y != nullptr &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->factorized_stencil_dense_partials != nullptr &&
               ws->factorized_stencil_dense_counter != nullptr &&
               ws->inverse_col_norm != nullptr) {
        update_xz_normal_factorized_stencil_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::AffineBlock &&
               ws->affine_block_operator_ready &&
               ws->affine_block_operator != nullptr &&
               ws->unit_scaled_y != nullptr &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->inverse_col_norm != nullptr) {
        update_xz_normal_affine_block_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::RowTemplate &&
               ws->row_template_operator_ready &&
               ws->row_template_operator != nullptr &&
               ws->unit_scaled_y != nullptr &&
               ws->unit_scaled_x_hat != nullptr &&
               ws->inverse_col_norm != nullptr) {
        update_xz_normal_row_template_gpu(ws);
    } else if (is_signed_x_backend(ws->x_backend) &&
               ws->signed_unit_operator_ready &&
               ws->signed_unit_operator != nullptr) {
        update_xz_normal_signed_unit_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::UnitFactorized &&
               ws->unit_operator_x_ready) {
        update_xz_normal_unit_factorized_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::StructuredOriginal &&
               ws->structured_operator_ready) {
        update_xz_normal_structured_gpu(ws);
    } else if (
        ws->x_backend == HPRLPXBackend::FixedDegreePackedDictionary &&
        ws->dictionary_operator_x_ready && ws->fixed_degree_AT_plan.ready &&
        ws->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32) {
        update_xz_normal_fixed_degree_packed_dictionary_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::PackedDictionary &&
               ws->dictionary_operator_x_ready &&
               ws->packed_dictionary_storage ==
                   HPRLPPackedDictionaryStorage::PackedU32) {
        update_xz_normal_packed_dictionary_u32_gpu(ws);
    } else if (ws->x_backend == HPRLPXBackend::PackedDictionary &&
               ws->dictionary_operator_x_ready &&
               ws->packed_dictionary_storage ==
                   HPRLPPackedDictionaryStorage::SeparateU32U16) {
        update_xz_normal_packed_dictionary_u32_u16_gpu(ws);
    } else {
        update_xz_normal_generic_fused_gpu(ws);
    }
}
