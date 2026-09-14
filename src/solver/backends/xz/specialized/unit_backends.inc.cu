namespace {

// Signed-unit is one specialized normal X/Z backend family.

void update_xz_normal_signed_unit_gpu(HPRLP_workspace_gpu *ws) {
        HPRLP_signed_unit_operator_gpu *op = ws->signed_unit_operator;
        HPRLP_FLOAT *normal_x_hat_output =
            (ws->signed_single_state_enabled &&
             is_signed_y_backend(ws->y_backend))
                ? nullptr
                : ws->x_hat;
        const bool emit_zero_flags =
            ws->signed_zero_skip_monitor_enabled &&
            ws->unit_scaled_x_hat_nonzero != nullptr;
        // In a signed/signed pair, the preceding signed Y update produced
        // this cache. Mixed pairs retain the standalone scaling fallback.
        if (!is_signed_y_backend(ws->y_backend)) {
            vector_dot_product_kernel<<<HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0,
                                        ws->stream>>>(
                ws->y, ws->inverse_row_norm, ws->unit_scaled_y, ws->m,
                false);
        }
        if (ws->x_backend == HPRLPXBackend::SignedUnitSplitU16) {
            const HPRLP_signed_unit_x_split_u16_view_gpu view{
                ws->n, ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->unit_scaled_x_hat,
                emit_zero_flags ? ws->unit_scaled_x_hat_nonzero : nullptr,
                ws->inverse_col_norm, ws->AT->rowPtr,
                op->AT_split_indices_u16, op->AT_split_negative_u8};
            hprlp_enqueue_signed_unit_x_split_u16_scalar(
                view, ws->Halpern_params, ws->halpern_factors,
                kSignedScalarThreads, ws->stream);
            return;
        }
        if (hprlp_all_rows_fit_scalar(
                ws->n, ws->max_AT_row_nnz,
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ)) {
            if (op->AT_uses_u16 || op->AT_degree3_run_row_count == 0) {
                const HPRLP_signed_unit_x_packed_view_gpu view{
                    ws->n, ws->x, normal_x_hat_output, ws->l, ws->u,
                    ws->x_bound_type, ws->c, ws->last_x,
                    ws->unit_scaled_y, ws->unit_scaled_x_hat,
                    emit_zero_flags
                        ? ws->unit_scaled_x_hat_nonzero : nullptr,
                    ws->inverse_col_norm, ws->AT->rowPtr,
                    op->AT_uses_u16 ? op->AT_entries_u16 : nullptr,
                    op->AT_uses_u16 ? nullptr : op->AT_entries_u32};
                hprlp_enqueue_signed_unit_x_packed_scalar(
                    view, ws->Halpern_params, ws->halpern_factors,
                    kSignedScalarThreads, ws->stream);
            } else if (emit_zero_flags) {
                if (ws->signed_state_specialization_enabled) {
                    signed_unit_update_x_all_scalar_degree3_run_state_specialized_flagged_u32_kernel<<<
                        (ws->n + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->x, normal_x_hat_output, ws->u, ws->c,
                        ws->last_x, ws->unit_scaled_y,
                        ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero,
                        ws->inverse_col_norm, ws->AT->rowPtr,
                        op->AT_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->n,
                        ws->signed_x_zero_objective_run_begin,
                        ws->signed_x_zero_objective_run_count,
                        op->AT_degree3_run_row_begin,
                        op->AT_degree3_run_row_count,
                        op->AT_degree3_run_entry_begin);
                } else {
                    signed_unit_update_x_all_scalar_degree3_run_flagged_u32_kernel<<<
                        (ws->n + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->x, normal_x_hat_output, ws->l, ws->u,
                        ws->x_bound_type, ws->c, ws->last_x,
                        ws->unit_scaled_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero,
                        ws->inverse_col_norm, ws->AT->rowPtr,
                        op->AT_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->n,
                        op->AT_degree3_run_row_begin,
                        op->AT_degree3_run_row_count,
                        op->AT_degree3_run_entry_begin);
                }
            } else if (ws->signed_state_specialization_enabled) {
                signed_unit_update_x_all_scalar_degree3_run_state_specialized_u32_kernel<<<
                    (ws->n + kSignedScalarThreads - 1) /
                        kSignedScalarThreads,
                    kSignedScalarThreads, 0, ws->stream>>>(
                    ws->x, normal_x_hat_output, ws->u, ws->c,
                    ws->last_x, ws->unit_scaled_y,
                    ws->unit_scaled_x_hat, ws->inverse_col_norm,
                    ws->AT->rowPtr, op->AT_entries_u32,
                    ws->Halpern_params, ws->halpern_factors, ws->n,
                    ws->signed_x_zero_objective_run_begin,
                    ws->signed_x_zero_objective_run_count,
                    op->AT_degree3_run_row_begin,
                    op->AT_degree3_run_row_count,
                    op->AT_degree3_run_entry_begin);
            } else {
                signed_unit_update_x_all_scalar_degree3_run_u32_kernel<<<
                    (ws->n + kSignedScalarThreads - 1) /
                        kSignedScalarThreads,
                    kSignedScalarThreads, 0, ws->stream>>>(
                    ws->x, normal_x_hat_output, ws->l, ws->u,
                    ws->x_bound_type, ws->c, ws->last_x,
                    ws->unit_scaled_y, ws->unit_scaled_x_hat,
                    ws->inverse_col_norm, ws->AT->rowPtr,
                    op->AT_entries_u32, ws->Halpern_params,
                    ws->halpern_factors, ws->n,
                    op->AT_degree3_run_row_begin,
                    op->AT_degree3_run_row_count,
                    op->AT_degree3_run_entry_begin);
            }
            return;
        }
        const HPRLP_signed_unit_x_packed_bucket_view_gpu view{
            {ws->n, ws->x, normal_x_hat_output, ws->l, ws->u,
             ws->x_bound_type, ws->c, ws->last_x, ws->unit_scaled_y,
             ws->unit_scaled_x_hat,
             emit_zero_flags
                 ? ws->unit_scaled_x_hat_nonzero : nullptr,
             ws->inverse_col_norm, ws->AT->rowPtr,
             op->AT_uses_u16 ? op->AT_entries_u16 : nullptr,
             op->AT_uses_u16 ? nullptr : op->AT_entries_u32},
            ws->AT_rows_short, ws->num_AT_rows_short,
            ws->AT_rows_medium, ws->num_AT_rows_medium,
            ws->AT_rows_long, ws->num_AT_rows_long};
        hprlp_enqueue_signed_unit_x_packed_bucketed(
            view, ws->Halpern_params, ws->halpern_factors,
            kSignedScalarThreads, kFusedThreads, ws->stream);
}

} // namespace
namespace {

void update_xz_normal_unit_factorized_gpu(HPRLP_workspace_gpu *ws) {
        const bool paired_unit_coltile =
            ws->y_backend == HPRLPYBackend::UnitColTile &&
            ws->unit_coltile_ready && ws->unit_scaled_x_hat != nullptr;
        if (!paired_unit_coltile) {
            vector_dot_product_kernel<<<HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0,
                                        ws->stream>>>(
                ws->y, ws->inverse_row_norm, ws->unit_scaled_y, ws->m,
                false);
        }
        HPRLP_FLOAT *scaled_x_hat_output =
            paired_unit_coltile ? ws->unit_scaled_x_hat : nullptr;
        if (ws->y_backend == HPRLPYBackend::UnitActiveScatter &&
            ws->unit_AT_col_index_u16 != nullptr &&
            hprlp_all_rows_fit_scalar(
                ws->n, ws->max_AT_row_nnz,
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ)) {
            const HPRLP_unit_factorized_x_view_gpu view{
                ws->m, ws->n,
                ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type,
                ws->c, ws->last_x, ws->unit_scaled_y,
                ws->inverse_col_norm, nullptr, ws->Ax, ws->AT->rowPtr,
                ws->unit_AT_col_index_u16};
            hprlp_enqueue_unit_factorized_x_scalar(
                view, ws->Halpern_params, ws->halpern_factors,
                ws->uniform_unit_sign, HPRLP_NUM_THREADS, ws->stream);
            return;
        }
        if (ws->all_zero_lower_unbounded_variables && ws->num_AT_rows_short == ws->n) {
            fused_update_x_z_all_short_unit_nonnegative_kernel<<<HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0, ws->stream>>>(
                ws->x, ws->x_hat, ws->c, ws->last_x, ws->unit_scaled_y,
                ws->inverse_col_norm, scaled_x_hat_output,
                ws->AT->rowPtr, ws->unit_AT_col_index_u16,
                ws->Halpern_params, ws->halpern_factors,
                ws->uniform_unit_sign, ws->n);
            return;
        }
        if (hprlp_all_rows_fit_scalar(
                ws->n, ws->max_AT_row_nnz,
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ)) {
            const HPRLP_unit_factorized_x_view_gpu view{
                ws->m, ws->n,
                ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type,
                ws->c, ws->last_x, ws->unit_scaled_y,
                ws->inverse_col_norm, scaled_x_hat_output, nullptr,
                ws->AT->rowPtr, ws->unit_AT_col_index_u16};
            hprlp_enqueue_unit_factorized_x_scalar(
                view, ws->Halpern_params, ws->halpern_factors,
                ws->uniform_unit_sign, HPRLP_NUM_THREADS, ws->stream);
            return;
        }
        if (ws->num_AT_rows_short > 0) {
            fused_update_x_z_rows_short_unit_kernel<<<(ws->num_AT_rows_short + kFusedThreads - 1) / kFusedThreads, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, ws->AT->rowPtr,
                ws->unit_AT_col_index_u16,
                ws->Halpern_params, ws->halpern_factors,
                ws->uniform_unit_sign, ws->AT_rows_short,
                ws->num_AT_rows_short);
        }
        if (ws->num_AT_rows_medium > 0) {
            fused_update_x_z_rows_warp_unit_kernel<<<(ws->num_AT_rows_medium + kWarpsPerBlock - 1) / kWarpsPerBlock, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, ws->AT->rowPtr,
                ws->unit_AT_col_index_u16,
                ws->Halpern_params, ws->halpern_factors,
                ws->uniform_unit_sign, ws->AT_rows_medium,
                ws->num_AT_rows_medium);
        }
        if (ws->num_AT_rows_long > 0) {
            fused_update_x_z_rows_block_unit_kernel<<<ws->num_AT_rows_long, kFusedThreads, 0, ws->stream>>>(
                ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, ws->AT->rowPtr,
                ws->unit_AT_col_index_u16,
                ws->Halpern_params, ws->halpern_factors,
                ws->uniform_unit_sign, ws->AT_rows_long,
                ws->num_AT_rows_long);
        }
}

} // namespace
