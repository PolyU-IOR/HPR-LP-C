namespace {

void update_y_normal_structured_gpu(HPRLP_workspace_gpu *ws) {
    HPRLP_structured_operator_gpu *op = ws->structured_operator;
    vector_dot_product_kernel<<<HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0,
                                ws->stream>>>(
        ws->x_hat, ws->inverse_col_norm, ws->unit_scaled_x_hat, ws->n, false);
    structured_update_y_sparse_kernel<<<
        (op->sparse_row_count + kFusedThreads - 1) / kFusedThreads,
        kFusedThreads, 0, ws->stream>>>(
        ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
        ws->unit_scaled_x_hat, ws->inverse_row_norm, op->sparse_rows,
        op->sparse_col0, op->sparse_col1, op->sparse_second_sign,
        ws->Halpern_params, ws->halpern_factors, op->sparse_row_count);
    structured_update_y_dense_kernel<<<
        (op->dense_row_count + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kFusedThreads, 0, ws->stream>>>(
        ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
        ws->unit_scaled_x_hat, ws->inverse_row_norm, op->dense_rows,
        op->dense_cols, op->dense_local_cols, op->dense_A_values_u16,
        ws->Halpern_params, ws->halpern_factors, op->dense_row_count,
        op->dense_col_count, op->coefficient_bias,
        op->coefficient_has_escape, op->coefficient_escape_value);
}

} // namespace
namespace {

void update_y_normal_grid_slack_laplacian_gpu(HPRLP_workspace_gpu *ws) {
        grid_slack_laplacian_update_y_kernel<<<
            HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->value,
            ws->grid_slack_laplacian_shape, ws->Halpern_params,
            ws->halpern_factors, ws->m);
}

} // namespace
namespace {

void update_y_normal_factorized_stencil_gpu(HPRLP_workspace_gpu *ws) {
        const HPRLPWindowedStencilShape shape =
            ws->windowed_stencil_shape;
        if (ws->windowed_stencil_state_ready) {
            windowed_stencil_state_update_y_kernel<<<
                shape.observation_height + shape.grid_height,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->last_y, ws->factorized_y_static_codes,
                ws->factorized_y_static_records, ws->AL, ws->AU,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, shape,
                ws->windowed_observation_first_bound_type,
                ws->windowed_observation_second_bound_type,
                ws->Halpern_params, ws->halpern_factors);
        } else {
            windowed_stencil_update_y_kernel<<<
                shape.observation_height + shape.grid_height,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->last_y, ws->factorized_y_static_codes,
                ws->factorized_y_static_records, ws->AL, ws->AU,
                ws->y_bound_type, ws->unit_scaled_x_hat,
                ws->inverse_row_norm, ws->unit_scaled_y, shape,
                ws->Halpern_params, ws->halpern_factors);
        }
}

} // namespace
namespace {

void update_y_normal_affine_block_gpu(HPRLP_workspace_gpu *ws) {
        const HPRLP_affine_block_matrix_gpu &op =
            ws->affine_block_operator->A;
        affine_block_update_y_kernel<<<
            op.block_count, kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            ws->unit_scaled_y, op.block_row_begin, op.block_row_count,
            op.block_entry_ptr, op.entry_base_columns,
            op.entry_column_strides, op.entry_values,
            ws->Halpern_params, ws->halpern_factors);
        if (op.fallback_short_count > 0) {
            row_template_update_y_fallback_short_kernel<<<
                (op.fallback_short_count + kFusedThreads - 1) /
                    kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_short_rows, op.fallback_short_count);
        }
        if (op.fallback_warp_count > 0) {
            row_template_update_y_fallback_warp_kernel<<<
                (op.fallback_warp_count + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_warp_rows, op.fallback_warp_count);
        }
        if (op.fallback_block_count > 0) {
            row_template_update_y_fallback_block_kernel<<<
                op.fallback_block_count, kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_block_rows, op.fallback_block_count);
        }
}

} // namespace
namespace {

void update_y_normal_row_template_gpu(HPRLP_workspace_gpu *ws) {
        const HPRLP_row_template_matrix_gpu &op =
            ws->row_template_operator->A;
        row_template_update_y_all_kernel<<<
            (ws->m + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            ws->unit_scaled_y, op.row_template_ids, op.row_bases,
            op.template_ptr, op.template_offsets, op.template_values,
            ws->Halpern_params, ws->halpern_factors, ws->m);
        if (op.fallback_short_count > 0) {
            row_template_update_y_fallback_short_kernel<<<
                (op.fallback_short_count + kFusedThreads - 1) /
                    kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_short_rows, op.fallback_short_count);
        }
        if (op.fallback_warp_count > 0) {
            row_template_update_y_fallback_warp_kernel<<<
                (op.fallback_warp_count + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_warp_rows, op.fallback_warp_count);
        }
        if (op.fallback_block_count > 0) {
            row_template_update_y_fallback_block_kernel<<<
                op.fallback_block_count, kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                ws->unit_scaled_y, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_block_rows, op.fallback_block_count);
        }
}

} // namespace
