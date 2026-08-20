namespace {

void update_y_normal_fixed_degree_packed_dictionary_gpu(
    HPRLP_workspace_gpu *ws) {
    const bool paired_dictionary_x =
        is_dictionary_x_backend(ws->x_backend) &&
        ws->dictionary_operator_x_ready;
    const HPRLPPackedStatePlan active_packed_state_plan =
        paired_dictionary_x ? ws->packed_state_plan
                            : HPRLPPackedStatePlan{};
    HPRLP_FLOAT *scaled_y_output =
        paired_dictionary_x ? ws->unit_scaled_y : nullptr;
    if (!paired_dictionary_x) {
        vector_dot_product_kernel<<<numBlocks(ws->n), numThreads, 0,
                                    ws->stream>>>(
            ws->x_hat, ws->inverse_col_norm, ws->unit_scaled_x_hat,
            ws->n, false);
    }

    const HPRLPFixedDegreeRunPlan &plan = ws->fixed_degree_A_plan;
    for (int run_index = 0; run_index < plan.run_count; ++run_index) {
        const HPRLPFixedDegreeRun &run = plan.runs[run_index];
        packed_dictionary_update_y_fixed_degree_run_kernel<<<
            (run.row_count + plan.threads - 1) / plan.threads,
            plan.threads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            scaled_y_output, active_packed_state_plan,
            ws->A_coefficient_dictionary,
            run.packed_entries_soa,
            ws->A_packed_dictionary_code_bits, ws->Halpern_params,
            ws->halpern_factors, run.row_begin, run.row_count,
            run.degree);
    }
    if (plan.fallback_short_count > 0) {
        packed_dictionary_update_y_rows_short_kernel<<<
            (plan.fallback_short_count + kFusedThreads - 1) /
                kFusedThreads,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            scaled_y_output, active_packed_state_plan,
            ws->A_coefficient_dictionary,
            ws->A_dictionary_packed_u32, ws->A->rowPtr,
            ws->A_packed_dictionary_code_bits, ws->Halpern_params,
            ws->halpern_factors, plan.fallback_short_rows,
            plan.fallback_short_count);
    }
    if (plan.fallback_warp_count > 0) {
        packed_dictionary_update_y_rows_warp_kernel<<<
            (plan.fallback_warp_count + kWarpsPerBlock - 1) /
                kWarpsPerBlock,
            kFusedThreads, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            scaled_y_output, active_packed_state_plan,
            ws->A_coefficient_dictionary,
            ws->A_dictionary_packed_u32, ws->A->rowPtr,
            ws->A_packed_dictionary_code_bits, ws->Halpern_params,
            ws->halpern_factors, plan.fallback_warp_rows,
            plan.fallback_warp_count);
    }
    if (plan.fallback_block_count > 0) {
        packed_dictionary_update_y_rows_block_kernel<<<
            plan.fallback_block_count, kFusedThreads, 0,
            ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            scaled_y_output, active_packed_state_plan,
            ws->A_coefficient_dictionary,
            ws->A_dictionary_packed_u32, ws->A->rowPtr,
            ws->A_packed_dictionary_code_bits, ws->Halpern_params,
            ws->halpern_factors, plan.fallback_block_rows,
            plan.fallback_block_count);
    }
}

void update_y_normal_packed_dictionary_gpu(HPRLP_workspace_gpu *ws) {
    const bool paired_dictionary_x =
        is_dictionary_x_backend(ws->x_backend) &&
        ws->dictionary_operator_x_ready;
    const HPRLPPackedStatePlan active_packed_state_plan =
        paired_dictionary_x ? ws->packed_state_plan
                            : HPRLPPackedStatePlan{};
    HPRLP_FLOAT *scaled_y_output =
        paired_dictionary_x ? ws->unit_scaled_y : nullptr;
    if (!paired_dictionary_x) {
        vector_dot_product_kernel<<<numBlocks(ws->n), numThreads, 0,
                                    ws->stream>>>(
            ws->x_hat, ws->inverse_col_norm, ws->unit_scaled_x_hat,
            ws->n, false);
    }

    if (ws->A_packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32) {
        const HPRLP_packed_dictionary_y_view_gpu view{
            ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
            ws->unit_scaled_x_hat, ws->inverse_row_norm,
            scaled_y_output, nullptr, active_packed_state_plan,
            ws->A_coefficient_dictionary, ws->A_dictionary_packed_u32,
            ws->A->rowPtr, ws->A_packed_dictionary_code_bits,
            ws->A_rows_short, ws->num_A_rows_short,
            ws->A_rows_medium, ws->num_A_rows_medium,
            nullptr, 0, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, false};
        hprlp_enqueue_packed_dictionary_y(
            view, ws->Halpern_params, ws->halpern_factors,
            kFusedThreads, ws->stream);
        if (ws->segmented_A_ready) {
            if (ws->segmented_A_fallback_long_count > 0) {
                packed_dictionary_update_y_rows_block_kernel<<<
                    ws->segmented_A_fallback_long_count,
                    kFusedThreads, 0, ws->stream>>>(
                    ws->y, ws->AL, ws->AU, ws->y_bound_type,
                    ws->last_y, ws->unit_scaled_x_hat,
                    ws->inverse_row_norm, scaled_y_output,
                    active_packed_state_plan,
                    ws->A_coefficient_dictionary,
                    ws->A_dictionary_packed_u32, ws->A->rowPtr,
                    ws->A_packed_dictionary_code_bits,
                    ws->Halpern_params, ws->halpern_factors,
                    ws->segmented_A_fallback_long_rows,
                    ws->segmented_A_fallback_long_count);
            }
            packed_dictionary_segmented_y_partial_kernel<<<
                ws->segmented_A_tile_count, ws->segmented_A_threads, 0,
                ws->stream>>>(
                ws->segmented_A_partials, ws->unit_scaled_x_hat,
                ws->A_coefficient_dictionary,
                ws->A_dictionary_packed_u32,
                ws->segmented_A_tile_begin,
                ws->segmented_A_tile_end,
                ws->A_packed_dictionary_code_bits,
                ws->segmented_A_tile_count);
            packed_dictionary_segmented_y_finalize_kernel<<<
                ws->segmented_A_row_count, ws->segmented_A_threads, 0,
                ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type,
                ws->last_y, ws->inverse_row_norm, scaled_y_output,
                active_packed_state_plan, ws->segmented_A_partials,
                ws->segmented_A_rows, ws->segmented_A_row_tile_ptr,
                ws->Halpern_params, ws->halpern_factors,
                ws->segmented_A_row_count);
        } else if (ws->num_A_rows_long > 0) {
            packed_dictionary_update_y_rows_block_kernel<<<
                ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                scaled_y_output, active_packed_state_plan,
                ws->A_coefficient_dictionary,
                ws->A_dictionary_packed_u32, ws->A->rowPtr,
                ws->A_packed_dictionary_code_bits,
                ws->Halpern_params, ws->halpern_factors,
                ws->A_rows_long, ws->num_A_rows_long);
        }
    } else if (ws->A_packed_dictionary_storage ==
                   HPRLPPackedDictionaryStorage::SeparateU32U16) {
        if (ws->num_A_rows_short > 0) {
            u32_u16_dictionary_update_y_rows_short_kernel<<<
                (ws->num_A_rows_short + kFusedThreads - 1) /
                    kFusedThreads,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                scaled_y_output,
                active_packed_state_plan,
                ws->A_coefficient_dictionary,
                ws->A_dictionary_indices_u32,
                ws->A_dictionary_codes_u16, ws->A->rowPtr,
                ws->Halpern_params, ws->halpern_factors,
                ws->A_rows_short, ws->num_A_rows_short);
        }
        if (ws->num_A_rows_medium > 0) {
            u32_u16_dictionary_update_y_rows_warp_kernel<<<
                (ws->num_A_rows_medium + kWarpsPerBlock - 1) /
                    kWarpsPerBlock,
                kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                scaled_y_output,
                active_packed_state_plan,
                ws->A_coefficient_dictionary,
                ws->A_dictionary_indices_u32,
                ws->A_dictionary_codes_u16, ws->A->rowPtr,
                ws->Halpern_params, ws->halpern_factors,
                ws->A_rows_medium, ws->num_A_rows_medium);
        }
        if (ws->num_A_rows_long > 0) {
            u32_u16_dictionary_update_y_rows_block_kernel<<<
                ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                scaled_y_output,
                active_packed_state_plan,
                ws->A_coefficient_dictionary,
                ws->A_dictionary_indices_u32,
                ws->A_dictionary_codes_u16, ws->A->rowPtr,
                ws->Halpern_params, ws->halpern_factors,
                ws->A_rows_long, ws->num_A_rows_long);
        }
    }
}

} // namespace
