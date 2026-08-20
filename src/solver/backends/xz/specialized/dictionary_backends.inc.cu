namespace {

void update_xz_normal_fixed_degree_packed_dictionary_gpu(HPRLP_workspace_gpu *ws) {
        const bool paired_dictionary_y =
            is_dictionary_y_backend(ws->y_backend) &&
            ws->dictionary_operator_y_ready;
        const HPRLPPackedStatePlan active_packed_state_plan =
            paired_dictionary_y ? ws->packed_state_plan
                                : HPRLPPackedStatePlan{};
        HPRLP_FLOAT *scaled_x_hat_output =
            paired_dictionary_y ? ws->unit_scaled_x_hat : nullptr;
        HPRLP_FLOAT *normal_x_hat_output =
            paired_dictionary_y ? nullptr : ws->x_hat;
        if (!paired_dictionary_y) {
            vector_dot_product_kernel<<<numBlocks(ws->m), numThreads, 0,
                                        ws->stream>>>(
                ws->y, ws->inverse_row_norm, ws->unit_scaled_y,
                ws->m, false);
        }
        const HPRLPFixedDegreeRunPlan &plan =
            ws->fixed_degree_AT_plan;
        for (int run_index = 0; run_index < plan.run_count; ++run_index) {
            const HPRLPFixedDegreeRun &run = plan.runs[run_index];
            packed_dictionary_update_x_fixed_degree_run_kernel<<<
                (run.row_count + plan.threads - 1) / plan.threads,
                plan.threads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, active_packed_state_plan,
                ws->coefficient_dictionary,
                run.packed_entries_soa,
                ws->packed_dictionary_code_bits, ws->Halpern_params,
                ws->halpern_factors, run.row_begin, run.row_count,
                run.degree);
        }
        if (plan.fallback_short_count > 0) {
            packed_dictionary_update_x_rows_short_kernel<<<
                (plan.fallback_short_count + kFusedThreads - 1) /
                    kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, active_packed_state_plan,
                ws->coefficient_dictionary,
                ws->AT_dictionary_packed_u32, ws->AT->rowPtr,
                ws->packed_dictionary_code_bits, ws->Halpern_params,
                ws->halpern_factors, plan.fallback_short_rows,
                plan.fallback_short_count);
        }
        if (plan.fallback_warp_count > 0) {
            packed_dictionary_update_x_rows_warp_kernel<<<
                (plan.fallback_warp_count + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, active_packed_state_plan,
                ws->coefficient_dictionary,
                ws->AT_dictionary_packed_u32, ws->AT->rowPtr,
                ws->packed_dictionary_code_bits, ws->Halpern_params,
                ws->halpern_factors, plan.fallback_warp_rows,
                plan.fallback_warp_count);
        }
        if (plan.fallback_block_count > 0) {
            packed_dictionary_update_x_rows_block_kernel<<<
                plan.fallback_block_count, kFusedThreads, 0,
                ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, active_packed_state_plan,
                ws->coefficient_dictionary,
                ws->AT_dictionary_packed_u32, ws->AT->rowPtr,
                ws->packed_dictionary_code_bits, ws->Halpern_params,
                ws->halpern_factors, plan.fallback_block_rows,
                plan.fallback_block_count);
        }
}

} // namespace
namespace {

void update_xz_normal_packed_dictionary_u32_gpu(HPRLP_workspace_gpu *ws) {
        const bool paired_dictionary_y =
            is_dictionary_y_backend(ws->y_backend) &&
            ws->dictionary_operator_y_ready;
        const HPRLPPackedStatePlan active_packed_state_plan =
            paired_dictionary_y ? ws->packed_state_plan
                                : HPRLPPackedStatePlan{};
        HPRLP_FLOAT *scaled_x_hat_output =
            paired_dictionary_y ? ws->unit_scaled_x_hat : nullptr;
        HPRLP_FLOAT *normal_x_hat_output =
            paired_dictionary_y ? nullptr : ws->x_hat;
        if (!paired_dictionary_y) {
            vector_dot_product_kernel<<<numBlocks(ws->m), numThreads, 0,
                                        ws->stream>>>(
                ws->y, ws->inverse_row_norm, ws->unit_scaled_y,
                ws->m, false);
        }
        const HPRLP_packed_dictionary_x_view_gpu view{
            ws->x, normal_x_hat_output, ws->l, ws->u,
            ws->x_bound_type, ws->c, ws->last_x, ws->unit_scaled_y,
            ws->inverse_col_norm, scaled_x_hat_output,
            active_packed_state_plan, ws->coefficient_dictionary,
            ws->AT_dictionary_packed_u32, ws->AT->rowPtr,
            ws->packed_dictionary_code_bits,
            ws->AT_rows_short, ws->num_AT_rows_short,
            ws->AT_rows_medium, ws->num_AT_rows_medium,
            nullptr, 0, false};
        hprlp_enqueue_packed_dictionary_x(
            view, ws->Halpern_params, ws->halpern_factors,
            kFusedThreads, ws->stream);
        if (ws->segmented_AT_ready) {
            if (ws->segmented_AT_fallback_long_count > 0) {
                packed_dictionary_update_x_rows_block_kernel<<<
                    ws->segmented_AT_fallback_long_count,
                    kFusedThreads, 0, ws->stream>>>(
                    ws->x, normal_x_hat_output, ws->l, ws->u,
                    ws->x_bound_type, ws->c, ws->last_x,
                    ws->unit_scaled_y, ws->inverse_col_norm,
                    scaled_x_hat_output, active_packed_state_plan,
                    ws->coefficient_dictionary,
                    ws->AT_dictionary_packed_u32, ws->AT->rowPtr,
                    ws->packed_dictionary_code_bits,
                    ws->Halpern_params, ws->halpern_factors,
                    ws->segmented_AT_fallback_long_rows,
                    ws->segmented_AT_fallback_long_count);
            }
            packed_dictionary_segmented_y_partial_kernel<<<
                ws->segmented_AT_tile_count, ws->segmented_AT_threads, 0,
                ws->stream>>>(
                ws->segmented_AT_partials, ws->unit_scaled_y,
                ws->coefficient_dictionary,
                ws->AT_dictionary_packed_u32,
                ws->segmented_AT_tile_begin,
                ws->segmented_AT_tile_end,
                ws->packed_dictionary_code_bits,
                ws->segmented_AT_tile_count);
            packed_dictionary_segmented_x_finalize_kernel<<<
                ws->segmented_AT_row_count, ws->segmented_AT_threads, 0,
                ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->inverse_col_norm, scaled_x_hat_output,
                active_packed_state_plan, ws->segmented_AT_partials,
                ws->segmented_AT_rows,
                ws->segmented_AT_row_tile_ptr, ws->Halpern_params,
                ws->halpern_factors, ws->segmented_AT_row_count);
        } else if (ws->num_AT_rows_long > 0) {
            packed_dictionary_update_x_rows_block_kernel<<<
                ws->num_AT_rows_long, kFusedThreads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c, ws->last_x,
                ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output, active_packed_state_plan,
                ws->coefficient_dictionary,
                ws->AT_dictionary_packed_u32, ws->AT->rowPtr,
                ws->packed_dictionary_code_bits,
                ws->Halpern_params, ws->halpern_factors,
                ws->AT_rows_long, ws->num_AT_rows_long);
        }
}

} // namespace
namespace {

void update_xz_normal_packed_dictionary_u32_u16_gpu(HPRLP_workspace_gpu *ws) {
        const bool paired_dictionary_y =
            is_dictionary_y_backend(ws->y_backend) &&
            ws->dictionary_operator_y_ready;
        const HPRLPPackedStatePlan active_packed_state_plan =
            paired_dictionary_y ? ws->packed_state_plan
                                : HPRLPPackedStatePlan{};
        HPRLP_FLOAT *scaled_x_hat_output =
            paired_dictionary_y ? ws->unit_scaled_x_hat : nullptr;
        HPRLP_FLOAT *normal_x_hat_output =
            paired_dictionary_y ? nullptr : ws->x_hat;
        if (!paired_dictionary_y) {
            vector_dot_product_kernel<<<numBlocks(ws->m), numThreads, 0,
                                        ws->stream>>>(
                ws->y, ws->inverse_row_norm, ws->unit_scaled_y,
                ws->m, false);
        }
        if (ws->num_AT_rows_short > 0) {
            u32_u16_dictionary_update_x_rows_short_kernel<<<
                (ws->num_AT_rows_short + kFusedThreads - 1) / kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output,
                active_packed_state_plan,
                ws->coefficient_dictionary, ws->AT_dictionary_indices_u32,
                ws->AT_dictionary_codes_u16, ws->AT->rowPtr,
                ws->Halpern_params, ws->halpern_factors,
                ws->AT_rows_short, ws->num_AT_rows_short);
        }
        if (ws->num_AT_rows_medium > 0) {
            u32_u16_dictionary_update_x_rows_warp_kernel<<<
                (ws->num_AT_rows_medium + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output,
                active_packed_state_plan,
                ws->coefficient_dictionary, ws->AT_dictionary_indices_u32,
                ws->AT_dictionary_codes_u16, ws->AT->rowPtr,
                ws->Halpern_params, ws->halpern_factors,
                ws->AT_rows_medium, ws->num_AT_rows_medium);
        }
        if (ws->num_AT_rows_long > 0) {
            u32_u16_dictionary_update_x_rows_block_kernel<<<
                ws->num_AT_rows_long, kFusedThreads, 0, ws->stream>>>(
                ws->x, normal_x_hat_output, ws->l, ws->u,
                ws->x_bound_type, ws->c,
                ws->last_x, ws->unit_scaled_y, ws->inverse_col_norm,
                scaled_x_hat_output,
                active_packed_state_plan,
                ws->coefficient_dictionary, ws->AT_dictionary_indices_u32,
                ws->AT_dictionary_codes_u16, ws->AT->rowPtr,
                ws->Halpern_params, ws->halpern_factors,
                ws->AT_rows_long, ws->num_AT_rows_long);
        }
}

} // namespace
