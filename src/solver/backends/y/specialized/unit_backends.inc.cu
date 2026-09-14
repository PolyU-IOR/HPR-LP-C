namespace {

// Signed-unit is one specialized normal Y backend family.

void update_y_normal_signed_unit_gpu(HPRLP_workspace_gpu *ws) {
        HPRLP_signed_unit_operator_gpu *op = ws->signed_unit_operator;
        const bool use_zero_skip =
            ws->signed_zero_skip_enabled &&
            ws->unit_scaled_x_hat_nonzero != nullptr;
        // In a signed/signed pair, the preceding signed X update produced
        // this cache. Mixed pairs retain the standalone scaling fallback.
        if (!is_signed_x_backend(ws->x_backend)) {
            vector_dot_product_kernel<<<HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0,
                                        ws->stream>>>(
                ws->x_hat, ws->inverse_col_norm, ws->unit_scaled_x_hat,
                ws->n, false);
        }
        if (ws->y_backend == HPRLPYBackend::SignedUnitPackedCombined) {
            const HPRLP_signed_unit_y_combined_view_gpu view{
                ws->m, ws->y, ws->AL, ws->AU, ws->y_bound_type,
                ws->last_y, ws->unit_scaled_x_hat,
                use_zero_skip ? ws->unit_scaled_x_hat_nonzero : nullptr,
                ws->unit_scaled_y, nullptr, ws->inverse_row_norm,
                ws->A->rowPtr,
                op->A_uses_u16 ? op->A_entries_u16 : nullptr,
                op->A_uses_u16 ? nullptr : op->A_entries_u32,
                ws->A_rows_medium, ws->num_A_rows_medium,
                ws->A_rows_long, ws->num_A_rows_long};
            hprlp_enqueue_signed_unit_y_combined(
                view, ws->Halpern_params, ws->halpern_factors,
                kFusedThreads, ws->stream);
            return;
        }
        if (hprlp_all_rows_fit_scalar(
                ws->m, ws->max_A_row_nnz,
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ)) {
            const HPRLP_signed_unit_y_scalar_view_gpu view{
                ws->m, ws->y, ws->AL, ws->AU, ws->y_bound_type,
                ws->last_y, ws->unit_scaled_x_hat,
                use_zero_skip
                    ? ws->unit_scaled_x_hat_nonzero : nullptr,
                ws->unit_scaled_y, nullptr, ws->inverse_row_norm,
                ws->A->rowPtr,
                op->A_uses_u16 ? op->A_entries_u16 : nullptr,
                op->A_uses_u16 ? nullptr : op->A_entries_u32};
            hprlp_enqueue_signed_unit_y_scalar(
                view, ws->Halpern_params, ws->halpern_factors,
                kSignedScalarThreads, ws->stream);
            return;
        }
        const bool use_direct_short =
            ws->num_A_rows_short > 0 &&
            static_cast<long long>(ws->num_A_rows_short) * 10 >=
                static_cast<long long>(ws->m) * 9;
        if (!use_direct_short) {
            const HPRLP_signed_unit_y_bucket_view_gpu view{
                {ws->m, ws->y, ws->AL, ws->AU, ws->y_bound_type,
                 ws->last_y, ws->unit_scaled_x_hat,
                 use_zero_skip
                     ? ws->unit_scaled_x_hat_nonzero : nullptr,
                 ws->unit_scaled_y, nullptr, ws->inverse_row_norm,
                 ws->A->rowPtr,
                 op->A_uses_u16 ? op->A_entries_u16 : nullptr,
                 op->A_uses_u16 ? nullptr : op->A_entries_u32},
                ws->A_rows_short, ws->num_A_rows_short,
                ws->A_rows_medium, ws->num_A_rows_medium,
                ws->A_rows_long, ws->num_A_rows_long};
            hprlp_enqueue_signed_unit_y_bucketed(
                view, ws->Halpern_params, ws->halpern_factors,
                kSignedScalarThreads, kFusedThreads, ws->stream);
            return;
        }
        if (op->A_uses_u16) {
            if (use_direct_short) {
                if (use_zero_skip) {
                    signed_unit_update_y_direct_short_skip_zero_u16_kernel<<<
                        (ws->m + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->m);
                } else {
                    signed_unit_update_y_direct_short_u16_kernel<<<
                        (ws->m + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->m);
                }
            } else if (ws->num_A_rows_short > 0) {
                if (use_zero_skip) {
                    signed_unit_update_y_rows_short_skip_zero_u16_kernel<<<
                        (ws->num_A_rows_short + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_short,
                        ws->num_A_rows_short);
                } else {
                    signed_unit_update_y_rows_short_u16_kernel<<<
                        (ws->num_A_rows_short + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_short,
                        ws->num_A_rows_short);
                }
            }
            if (ws->num_A_rows_medium > 0) {
                if (use_zero_skip) {
                    signed_unit_update_y_rows_warp_skip_zero_u16_kernel<<<
                        (ws->num_A_rows_medium + kWarpsPerBlock - 1) /
                            kWarpsPerBlock,
                        kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_medium,
                        ws->num_A_rows_medium);
                } else {
                    signed_unit_update_y_rows_warp_u16_kernel<<<
                        (ws->num_A_rows_medium + kWarpsPerBlock - 1) /
                            kWarpsPerBlock,
                        kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_medium,
                        ws->num_A_rows_medium);
                }
            }
            if (ws->num_A_rows_long > 0) {
                if (use_zero_skip) {
                    signed_unit_update_y_rows_block_skip_zero_u16_kernel<<<
                        ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_long,
                        ws->num_A_rows_long);
                } else {
                    signed_unit_update_y_rows_block_u16_kernel<<<
                        ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u16, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_long,
                        ws->num_A_rows_long);
                }
            }
        } else {
            if (use_direct_short) {
                if (use_zero_skip) {
                    if (op->A_degree2_run_row_count > 0) {
                        if (ws->signed_state_specialization_enabled) {
                            signed_unit_update_y_direct_short_degree2_run_state_specialized_skip_zero_u32_kernel<<<
                                (ws->m + kSignedScalarThreads - 1) /
                                    kSignedScalarThreads,
                                kSignedScalarThreads, 0, ws->stream>>>(
                                ws->y, ws->AL, ws->AU,
                                ws->y_bound_type, ws->last_y,
                                ws->unit_scaled_x_hat,
                                ws->unit_scaled_x_hat_nonzero,
                                ws->unit_scaled_y, ws->inverse_row_norm,
                                ws->A->rowPtr, op->A_entries_u32,
                                ws->Halpern_params, ws->halpern_factors,
                                ws->m,
                                ws->signed_y_upper_zero_run_begin,
                                ws->signed_y_upper_zero_run_count,
                                op->A_degree2_run_row_begin,
                                op->A_degree2_run_row_count,
                                op->A_degree2_run_entry_begin);
                        } else {
                            signed_unit_update_y_direct_short_degree2_run_skip_zero_u32_kernel<<<
                                (ws->m + kSignedScalarThreads - 1) /
                                    kSignedScalarThreads,
                                kSignedScalarThreads, 0, ws->stream>>>(
                                ws->y, ws->AL, ws->AU,
                                ws->y_bound_type, ws->last_y,
                                ws->unit_scaled_x_hat,
                                ws->unit_scaled_x_hat_nonzero,
                                ws->unit_scaled_y, ws->inverse_row_norm,
                                ws->A->rowPtr, op->A_entries_u32,
                                ws->Halpern_params, ws->halpern_factors,
                                ws->m, op->A_degree2_run_row_begin,
                                op->A_degree2_run_row_count,
                                op->A_degree2_run_entry_begin);
                        }
                    } else {
                        signed_unit_update_y_direct_short_skip_zero_u32_kernel<<<
                            (ws->m + kSignedScalarThreads - 1) /
                                kSignedScalarThreads,
                            kSignedScalarThreads, 0, ws->stream>>>(
                            ws->y, ws->AL, ws->AU, ws->y_bound_type,
                            ws->last_y, ws->unit_scaled_x_hat,
                            ws->unit_scaled_x_hat_nonzero,
                            ws->unit_scaled_y, ws->inverse_row_norm,
                            ws->A->rowPtr, op->A_entries_u32,
                            ws->Halpern_params, ws->halpern_factors,
                            ws->m);
                    }
                } else {
                    if (op->A_degree2_run_row_count > 0) {
                        if (ws->signed_state_specialization_enabled) {
                            signed_unit_update_y_direct_short_degree2_run_state_specialized_u32_kernel<<<
                                (ws->m + kSignedScalarThreads - 1) /
                                    kSignedScalarThreads,
                                kSignedScalarThreads, 0, ws->stream>>>(
                                ws->y, ws->AL, ws->AU,
                                ws->y_bound_type, ws->last_y,
                                ws->unit_scaled_x_hat,
                                ws->unit_scaled_y, ws->inverse_row_norm,
                                ws->A->rowPtr, op->A_entries_u32,
                                ws->Halpern_params, ws->halpern_factors,
                                ws->m,
                                ws->signed_y_upper_zero_run_begin,
                                ws->signed_y_upper_zero_run_count,
                                op->A_degree2_run_row_begin,
                                op->A_degree2_run_row_count,
                                op->A_degree2_run_entry_begin);
                        } else {
                            signed_unit_update_y_direct_short_degree2_run_u32_kernel<<<
                                (ws->m + kSignedScalarThreads - 1) /
                                    kSignedScalarThreads,
                                kSignedScalarThreads, 0, ws->stream>>>(
                                ws->y, ws->AL, ws->AU,
                                ws->y_bound_type, ws->last_y,
                                ws->unit_scaled_x_hat,
                                ws->unit_scaled_y, ws->inverse_row_norm,
                                ws->A->rowPtr, op->A_entries_u32,
                                ws->Halpern_params, ws->halpern_factors,
                                ws->m, op->A_degree2_run_row_begin,
                                op->A_degree2_run_row_count,
                                op->A_degree2_run_entry_begin);
                        }
                    } else {
                        signed_unit_update_y_direct_short_u32_kernel<<<
                            (ws->m + kSignedScalarThreads - 1) /
                                kSignedScalarThreads,
                            kSignedScalarThreads, 0, ws->stream>>>(
                            ws->y, ws->AL, ws->AU, ws->y_bound_type,
                            ws->last_y, ws->unit_scaled_x_hat,
                            ws->unit_scaled_y,
                            ws->inverse_row_norm, ws->A->rowPtr,
                            op->A_entries_u32, ws->Halpern_params,
                            ws->halpern_factors, ws->m);
                    }
                }
            } else if (ws->num_A_rows_short > 0) {
                if (use_zero_skip) {
                    signed_unit_update_y_rows_short_skip_zero_u32_kernel<<<
                        (ws->num_A_rows_short + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_short,
                        ws->num_A_rows_short);
                } else {
                    signed_unit_update_y_rows_short_u32_kernel<<<
                        (ws->num_A_rows_short + kSignedScalarThreads - 1) /
                            kSignedScalarThreads,
                        kSignedScalarThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_short,
                        ws->num_A_rows_short);
                }
            }
            if (ws->num_A_rows_medium > 0) {
                if (use_zero_skip) {
                    signed_unit_update_y_rows_warp_skip_zero_u32_kernel<<<
                        (ws->num_A_rows_medium + kWarpsPerBlock - 1) /
                            kWarpsPerBlock,
                        kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_medium,
                        ws->num_A_rows_medium);
                } else {
                    signed_unit_update_y_rows_warp_u32_kernel<<<
                        (ws->num_A_rows_medium + kWarpsPerBlock - 1) /
                            kWarpsPerBlock,
                        kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_medium,
                        ws->num_A_rows_medium);
                }
            }
            if (ws->num_A_rows_long > 0) {
                if (use_zero_skip) {
                    signed_unit_update_y_rows_block_skip_zero_u32_kernel<<<
                        ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_x_hat_nonzero, ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_long,
                        ws->num_A_rows_long);
                } else {
                    signed_unit_update_y_rows_block_u32_kernel<<<
                        ws->num_A_rows_long, kFusedThreads, 0, ws->stream>>>(
                        ws->y, ws->AL, ws->AU, ws->y_bound_type,
                        ws->last_y, ws->unit_scaled_x_hat,
                        ws->unit_scaled_y,
                        ws->inverse_row_norm, ws->A->rowPtr,
                        op->A_entries_u32, ws->Halpern_params,
                        ws->halpern_factors, ws->A_rows_long,
                        ws->num_A_rows_long);
                }
            }
        }
}

} // namespace
namespace {

void update_y_normal_unit_active_scatter_gpu(HPRLP_workspace_gpu *ws) {
        update_y_normal_unit_kernel<<<
            HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->Ax,
            ws->inverse_row_norm, ws->last_y, ws->Halpern_params,
            ws->halpern_factors, ws->uniform_unit_sign, ws->m);
}

} // namespace
namespace {

void update_y_normal_unit_coltile_gpu(HPRLP_workspace_gpu *ws) {
        HPRLP_unit_coltile_gpu *op = ws->unit_coltile;
        if (ws->y_backend == HPRLPYBackend::UnitColTileZeroBitset) {
            vector_dot_product_zero_bitset_kernel<<<
                HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0, ws->stream>>>(
                ws->x_hat, ws->inverse_col_norm,
                ws->unit_scaled_x_hat, ws->unit_scaled_x_zero_bits,
                ws->n);
            unit_coltile_update_y_zero_bitset_kernel<<<
                op->rows, kUnitColTileThreads, 0, ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->unit_scaled_x_zero_bits,
                ws->inverse_row_norm, op->row_tile_offsets,
                op->local_cols, ws->Halpern_params,
                ws->halpern_factors, op->tile_count, op->tile_cols,
                ws->uniform_unit_sign, ws->m);
        } else {
            const bool paired_unit_x =
                ws->x_backend == HPRLPXBackend::UnitFactorized &&
                ws->unit_operator_x_ready;
            if (!paired_unit_x) {
                vector_dot_product_kernel<<<HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0,
                                            ws->stream>>>(
                    ws->x_hat, ws->inverse_col_norm,
                    ws->unit_scaled_x_hat, ws->n, false);
            }
            unit_coltile_update_y_kernel<<<op->rows, kUnitColTileThreads, 0,
                                           ws->stream>>>(
                ws->y, ws->AL, ws->AU, ws->y_bound_type, ws->last_y,
                ws->unit_scaled_x_hat, ws->inverse_row_norm,
                paired_unit_x ? ws->unit_scaled_y : nullptr,
                op->row_tile_offsets, op->local_cols,
                ws->Halpern_params, ws->halpern_factors,
                op->tile_count, op->tile_cols,
                ws->uniform_unit_sign, ws->m);
        }
}

} // namespace
namespace {

void update_y_normal_unit_factorized_gpu(HPRLP_workspace_gpu *ws) {
        vector_dot_product_kernel<<<HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0, ws->stream>>>(
            ws->x_hat, ws->inverse_col_norm, ws->unit_scaled_x_hat, ws->n, false);
        CUSPARSE_CHECK(hprlp_run_spmvop(
            ws->spmv_A->cusparseHandle, ws->spmv_A->unit_operation,
            &ws->spmv_A->alpha, &ws->spmv_A->beta,
            ws->spmv_A->unit_x_hat_cusparseDescr,
            ws->spmv_A->Ax_cusparseDescr,
            ws->spmv_A->Ax_cusparseDescr));
        update_y_normal_unit_kernel<<<HPRLP_NUM_BLOCKS(ws->m), HPRLP_NUM_THREADS, 0, ws->stream>>>(
            ws->y, ws->AL, ws->AU, ws->Ax, ws->inverse_row_norm,
            ws->last_y, ws->Halpern_params, ws->halpern_factors,
            ws->uniform_unit_sign, ws->m);
}

} // namespace
