namespace {

void update_y_normal_generic_fused_gpu(HPRLP_workspace_gpu *ws) {
    if (hprlp_all_rows_short(
            ws->m, ws->num_A_rows_short, ws->num_A_rows_medium,
            ws->num_A_rows_long)) {
        if (ws->A_col_index_u16 != nullptr) {
            fused_update_y_all_short_u16_kernel<<<
                (ws->m + kFusedThreads - 1) / kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type,
                ws->last_y, ws->x_hat, ws->A->rowPtr,
                ws->A_col_index_u16, ws->A->value,
                ws->Halpern_params, ws->halpern_factors, ws->m);
        } else {
            fused_update_y_all_short_kernel<<<
                (ws->m + kFusedThreads - 1) / kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type,
                ws->last_y, ws->x_hat, ws->A->rowPtr,
                ws->A->colIndex, ws->A->value,
                ws->Halpern_params, ws->halpern_factors, ws->m);
        }
        return;
    }

    if (hprlp_use_direct_short_rows(ws->m, ws->num_A_rows_short)) {
        fused_update_y_direct_short_kernel<<<
            (ws->m + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->rowPtr, ws->A->colIndex,
            ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->m);
    } else if (ws->num_A_rows_short > 0) {
        fused_update_y_rows_short_kernel<<<
            (ws->num_A_rows_short + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->rowPtr, ws->A->colIndex,
            ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_short, ws->num_A_rows_short);
    }
    if (ws->num_A_rows_medium > 0) {
        fused_update_y_rows_warp_kernel<<<(ws->num_A_rows_medium + kWarpsPerBlock - 1) / kWarpsPerBlock, kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y, ws->x_hat,
            ws->A->rowPtr, ws->A->colIndex, ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_medium, ws->num_A_rows_medium);
    }
    if (ws->num_A_rows_long > 0) {
        fused_update_y_rows_block_kernel<<<ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y, ws->x_hat,
            ws->A->rowPtr, ws->A->colIndex, ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_long, ws->num_A_rows_long);
    }

}

void update_y_normal_segmented_gpu(HPRLP_workspace_gpu *ws) {
    if (hprlp_use_direct_short_rows(ws->m, ws->num_A_rows_short)) {
        fused_update_y_direct_short_kernel<<<
            (ws->m + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->rowPtr, ws->A->colIndex,
            ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->m);
    } else if (ws->num_A_rows_short > 0) {
        fused_update_y_rows_short_kernel<<<
            (ws->num_A_rows_short + kFusedThreads - 1) / kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->rowPtr, ws->A->colIndex,
            ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_short, ws->num_A_rows_short);
    }
    if (ws->num_A_rows_medium > 0) {
        fused_update_y_rows_warp_kernel<<<
            (ws->num_A_rows_medium + kWarpsPerBlock - 1) /
                kWarpsPerBlock,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->rowPtr, ws->A->colIndex,
            ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->A_rows_medium, ws->num_A_rows_medium);
    }
    if (ws->segmented_A_fallback_long_count > 0) {
        fused_update_y_rows_block_kernel<<<
            ws->segmented_A_fallback_long_count, kFusedThreads, 0,
            ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->x_hat, ws->A->rowPtr, ws->A->colIndex,
            ws->A->value, ws->Halpern_params, ws->halpern_factors,
            ws->segmented_A_fallback_long_rows,
            ws->segmented_A_fallback_long_count);
    }
    segmented_update_y_partial_kernel<<<
        ws->segmented_A_tile_count, ws->segmented_A_threads, 0,
        ws->stream>>>(
        ws->segmented_A_partials, ws->x_hat, ws->A->colIndex,
        ws->A->value, ws->segmented_A_tile_begin,
        ws->segmented_A_tile_end, ws->segmented_A_tile_count);
    segmented_update_y_finalize_kernel<<<
        ws->segmented_A_row_count, ws->segmented_A_threads, 0,
        ws->stream>>>(
        ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
        ws->segmented_A_partials, ws->segmented_A_rows,
        ws->segmented_A_row_tile_ptr, ws->Halpern_params,
        ws->halpern_factors, ws->segmented_A_row_count);

}

} // namespace
