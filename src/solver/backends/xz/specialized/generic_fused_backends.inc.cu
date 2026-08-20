namespace {

void update_xz_normal_generic_fused_gpu(HPRLP_workspace_gpu *ws) {
    if (hprlp_all_rows_short(
            ws->n, ws->num_AT_rows_short, ws->num_AT_rows_medium,
            ws->num_AT_rows_long)) {
        fused_update_x_z_all_short_kernel<<<
            (ws->n + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type,
            ws->c, ws->last_x, ws->y, ws->AT->rowPtr,
            ws->AT->colIndex, ws->AT->value, ws->Halpern_params,
            ws->halpern_factors, ws->n);
        return;
    }

    if (hprlp_use_direct_short_rows(ws->n, ws->num_AT_rows_short)) {
        fused_update_x_z_direct_short_kernel<<<
            (ws->n + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type,
            ws->c, ws->last_x, ws->y, ws->AT->rowPtr,
            ws->AT->colIndex, ws->AT->value, ws->Halpern_params,
            ws->halpern_factors, ws->n);
    } else if (ws->num_AT_rows_short > 0) {
        fused_update_x_z_rows_short_kernel<<<
            (ws->num_AT_rows_short + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type,
            ws->c, ws->last_x, ws->y, ws->AT->rowPtr,
            ws->AT->colIndex, ws->AT->value, ws->Halpern_params,
            ws->halpern_factors, ws->AT_rows_short,
            ws->num_AT_rows_short);
    }
    if (ws->num_AT_rows_medium > 0) {
        fused_update_x_z_rows_warp_kernel<<<(ws->num_AT_rows_medium + kWarpsPerBlock - 1) / kWarpsPerBlock, kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x, ws->y,
            ws->AT->rowPtr, ws->AT->colIndex, ws->AT->value, ws->Halpern_params, ws->halpern_factors,
            ws->AT_rows_medium, ws->num_AT_rows_medium);
    }
    if (ws->segmented_AT_ready) {
        if (ws->segmented_AT_fallback_long_count > 0) {
            fused_update_x_z_rows_block_kernel<<<
                ws->segmented_AT_fallback_long_count, kFusedThreads, 0,
                ws->stream>>>(
                ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type,
                ws->c, ws->last_x, ws->y, ws->AT->rowPtr,
                ws->AT->colIndex, ws->AT->value, ws->Halpern_params,
                ws->halpern_factors,
                ws->segmented_AT_fallback_long_rows,
                ws->segmented_AT_fallback_long_count);
        }
        segmented_update_y_partial_kernel<<<
            ws->segmented_AT_tile_count, ws->segmented_AT_threads, 0,
            ws->stream>>>(
            ws->segmented_AT_partials, ws->y, ws->AT->colIndex,
            ws->AT->value, ws->segmented_AT_tile_begin,
            ws->segmented_AT_tile_end, ws->segmented_AT_tile_count);
        segmented_update_x_finalize_kernel<<<
            ws->segmented_AT_row_count, ws->segmented_AT_threads, 0,
            ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c,
            ws->last_x, ws->segmented_AT_partials,
            ws->segmented_AT_rows, ws->segmented_AT_row_tile_ptr,
            ws->Halpern_params, ws->halpern_factors,
            ws->segmented_AT_row_count);
    } else if (ws->num_AT_rows_long > 0) {
        fused_update_x_z_rows_block_kernel<<<ws->num_AT_rows_long, kFusedThreads, 0, ws->stream>>>(
            ws->x, ws->x_hat, ws->l, ws->u, ws->x_bound_type, ws->c, ws->last_x, ws->y,
            ws->AT->rowPtr, ws->AT->colIndex, ws->AT->value, ws->Halpern_params, ws->halpern_factors,
            ws->AT_rows_long, ws->num_AT_rows_long);
    }
}

} // namespace
