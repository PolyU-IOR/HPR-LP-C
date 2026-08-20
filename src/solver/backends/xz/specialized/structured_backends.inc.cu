namespace {

void update_xz_normal_grid_slack_laplacian_gpu(HPRLP_workspace_gpu *ws) {
        grid_slack_laplacian_update_x_kernel<<<
            numBlocks(ws->n), numThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c,
            ws->last_x, ws->y, ws->AT->rowPtr, ws->AT->value,
            ws->grid_slack_laplacian_shape, ws->Halpern_params,
            ws->halpern_factors, ws->n);
}

} // namespace
namespace {

void update_xz_normal_factorized_stencil_gpu(HPRLP_workspace_gpu *ws) {
        const HPRLPWindowedStencilShape shape =
            ws->windowed_stencil_shape;
        if (ws->windowed_stencil_state_ready) {
            windowed_stencil_state_update_x_grid_kernel<<<
                shape.grid_height, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->last_x, ws->factorized_x_static_codes,
                ws->factorized_x_static_records, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->unit_scaled_y,
                ws->inverse_col_norm,
                ws->unit_scaled_x_hat,
                ws->factorized_stencil_dense_partials,
                ws->factorized_stencil_dense_counter, shape,
                ws->Halpern_params, ws->halpern_factors);
        } else {
            windowed_stencil_update_x_grid_kernel<<<
                shape.grid_height, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->last_x, ws->factorized_x_static_codes,
                ws->factorized_x_static_records, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->unit_scaled_y,
                ws->inverse_col_norm, ws->unit_scaled_x_hat,
                ws->factorized_stencil_dense_partials,
                ws->factorized_stencil_dense_counter, shape,
                ws->Halpern_params, ws->halpern_factors);
        }
}

} // namespace
namespace {

void update_xz_normal_affine_block_gpu(HPRLP_workspace_gpu *ws) {
        const HPRLP_affine_block_matrix_gpu &op =
            ws->affine_block_operator->AT;
        affine_block_update_x_kernel<<<
            op.block_count, kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
            ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
            ws->unit_scaled_x_hat, op.block_row_begin,
            op.block_row_count, op.block_entry_ptr,
            op.entry_base_columns, op.entry_column_strides,
            op.entry_values, ws->Halpern_params, ws->halpern_factors);
        if (op.fallback_short_count > 0) {
            row_template_update_x_fallback_short_kernel<<<
                (op.fallback_short_count + kFusedThreads - 1) /
                    kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_short_rows, op.fallback_short_count);
        }
        if (op.fallback_warp_count > 0) {
            row_template_update_x_fallback_warp_kernel<<<
                (op.fallback_warp_count + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_warp_rows, op.fallback_warp_count);
        }
        if (op.fallback_block_count > 0) {
            row_template_update_x_fallback_block_kernel<<<
                op.fallback_block_count, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_block_rows, op.fallback_block_count);
        }
}

} // namespace
namespace {

void update_xz_normal_row_template_gpu(HPRLP_workspace_gpu *ws) {
        const HPRLP_row_template_matrix_gpu &op =
            ws->row_template_operator->AT;
        row_template_update_x_all_kernel<<<
            (ws->n + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
            ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
            ws->unit_scaled_x_hat, op.row_template_ids, op.row_bases,
            op.template_ptr, op.template_offsets, op.template_values,
            ws->Halpern_params, ws->halpern_factors, ws->n);
        if (op.fallback_short_count > 0) {
            row_template_update_x_fallback_short_kernel<<<
                (op.fallback_short_count + kFusedThreads - 1) /
                    kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_short_rows, op.fallback_short_count);
        }
        if (op.fallback_warp_count > 0) {
            row_template_update_x_fallback_warp_kernel<<<
                (op.fallback_warp_count + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_warp_rows, op.fallback_warp_count);
        }
        if (op.fallback_block_count > 0) {
            row_template_update_x_fallback_block_kernel<<<
                op.fallback_block_count, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->l, ws->u, ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, op.fallback_row_ptr,
                op.fallback_col_indices, op.fallback_values,
                ws->Halpern_params, ws->halpern_factors,
                op.fallback_block_rows, op.fallback_block_count);
        }
}

} // namespace
namespace {

void update_xz_normal_structured_gpu(HPRLP_workspace_gpu *ws) {
        HPRLP_structured_operator_gpu *op = ws->structured_operator;
        vector_dot_product_kernel<<<numBlocks(ws->m), numThreads, 0,
                                    ws->stream>>>(
            ws->y, ws->inverse_row_norm, ws->unit_scaled_y, ws->m, false);
        structured_update_x_short_kernel<<<
            (op->short_AT_output_count + kFusedThreads - 1) /
                kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c,
            ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
            op->short_AT_output_cols, op->short_AT_row_ptr,
            op->short_AT_rows, op->short_AT_values_u16,
            ws->Halpern_params, ws->halpern_factors,
            op->coefficient_bias, op->coefficient_has_escape,
            op->coefficient_escape_value,
            op->short_AT_output_count);
        structured_update_x_dense_kernel<<<
            op->dense_col_count, kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c,
            ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
            op->dense_cols, op->dense_rows, op->dense_AT_values_u16,
            ws->Halpern_params, ws->halpern_factors,
            op->coefficient_bias, op->coefficient_has_escape,
            op->coefficient_escape_value,
            op->dense_row_count, op->dense_col_count);
}

} // namespace
