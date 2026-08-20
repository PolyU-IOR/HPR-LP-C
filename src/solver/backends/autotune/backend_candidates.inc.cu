namespace {

std::vector<BackendCandidate> collect_eligible_backend_pairs(HPRLP_workspace_gpu *ws) {
    std::vector<HPRLPXBackend> x_backends{
        HPRLPXBackend::ScaledCusparse,
        HPRLPXBackend::GenericFused};
    std::vector<HPRLPYBackend> y_backends{
        HPRLPYBackend::ScaledCusparse,
        HPRLPYBackend::GenericFused};
    if (ws->segmented_A_ready) {
        y_backends.push_back(HPRLPYBackend::SegmentedFused);
    }
    if (ws->unit_operator_x_ready) {
        x_backends.push_back(HPRLPXBackend::UnitFactorized);
    }
    if (ws->unit_operator_y_ready) {
        y_backends.push_back(HPRLPYBackend::UnitFactorized);
    }
    if (ws->signed_unit_operator_ready) {
        x_backends.push_back(HPRLPXBackend::SignedUnitPacked);
        y_backends.push_back(HPRLPYBackend::SignedUnitPacked);
        const bool split_signed_x_eligible =
            !ws->signed_state_specialization_enabled &&
            ws->signed_unit_operator != nullptr &&
            ws->signed_unit_operator->AT_split_u16_ready &&
            ws->signed_unit_operator->AT_split_indices_u16 != nullptr &&
            ws->signed_unit_operator->AT_split_negative_u8 != nullptr &&
            hprlp_all_rows_fit_scalar(
                ws->n, ws->max_AT_row_nnz,
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ);
        if (split_signed_x_eligible) {
            x_backends.push_back(HPRLPXBackend::SignedUnitSplitU16);
        }
        const bool combined_signed_y_eligible =
            !ws->signed_state_specialization_enabled &&
            ws->num_A_rows_short > 0 &&
            static_cast<long long>(ws->num_A_rows_short) * 10 >=
                static_cast<long long>(ws->m) * 9 &&
            (ws->num_A_rows_medium > 0 || ws->num_A_rows_long > 0);
        if (combined_signed_y_eligible) {
            y_backends.push_back(
                HPRLPYBackend::SignedUnitPackedCombined);
        }
    }
    if (ws->dictionary_operator_x_ready &&
        (ws->packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::PackedU32 ||
         ws->packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::SeparateU32U16)) {
        x_backends.push_back(HPRLPXBackend::PackedDictionary);
    }
    if (ws->dictionary_operator_y_ready &&
        (ws->A_packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::PackedU32 ||
         ws->A_packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::SeparateU32U16)) {
        y_backends.push_back(HPRLPYBackend::PackedDictionary);
    }
    if (ws->unit_coltile_ready) {
        y_backends.push_back(HPRLPYBackend::UnitColTile);
        if (ws->unit_scaled_x_zero_bits != nullptr) {
            y_backends.push_back(HPRLPYBackend::UnitColTileZeroBitset);
        }
        if (ws->unit_operator_x_ready && ws->unit_operator_y_ready &&
            ws->unit_AT_col_index_u16 != nullptr &&
            hprlp_all_rows_fit_scalar(
                ws->n, ws->max_AT_row_nnz,
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ)) {
            y_backends.push_back(HPRLPYBackend::UnitActiveScatter);
        }
    }
    if (ws->structured_operator_ready) {
        x_backends.push_back(HPRLPXBackend::StructuredOriginal);
        y_backends.push_back(HPRLPYBackend::StructuredOriginal);
    }
    if (ws->grid_slack_laplacian_operator_ready) {
        x_backends.push_back(HPRLPXBackend::GridSlackLaplacian);
        y_backends.push_back(HPRLPYBackend::GridSlackLaplacian);
    }
    std::vector<BackendCandidate> candidates;

        candidates.reserve(x_backends.size() * y_backends.size());
        for (std::size_t y_index = 0; y_index < y_backends.size(); ++y_index) {
            for (std::size_t x_index = 0; x_index < x_backends.size(); ++x_index) {
                if (y_backends[y_index] ==
                        HPRLPYBackend::UnitActiveScatter &&
                    x_backends[x_index] !=
                        HPRLPXBackend::UnitFactorized) {
                    continue;
                }
                candidates.push_back(
                    BackendCandidate{x_backends[x_index], y_backends[y_index]});
            }
        }
        const bool fixed_degree_x_ready =
            ws->fixed_degree_AT_plan.ready &&
            ws->dictionary_operator_x_ready &&
            ws->packed_dictionary_storage ==
                HPRLPPackedDictionaryStorage::PackedU32;
        const bool fixed_degree_y_ready =
            ws->fixed_degree_A_plan.ready &&
            ws->dictionary_operator_y_ready &&
            ws->A_packed_dictionary_storage ==
                HPRLPPackedDictionaryStorage::PackedU32;
        if (fixed_degree_x_ready) {
            if (ws->dictionary_operator_y_ready) {
                candidates.push_back(BackendCandidate{
                    HPRLPXBackend::FixedDegreePackedDictionary,
                    HPRLPYBackend::PackedDictionary});
            } else {
                for (HPRLPYBackend y_backend : y_backends) {
                    candidates.push_back(BackendCandidate{
                        HPRLPXBackend::FixedDegreePackedDictionary,
                        y_backend});
                }
            }
        }
        if (fixed_degree_y_ready) {
            if (ws->dictionary_operator_x_ready) {
                candidates.push_back(BackendCandidate{
                    HPRLPXBackend::PackedDictionary,
                    HPRLPYBackend::FixedDegreePackedDictionary});
            } else {
                for (HPRLPXBackend x_backend : x_backends) {
                    candidates.push_back(BackendCandidate{
                        x_backend,
                        HPRLPYBackend::FixedDegreePackedDictionary});
                }
            }
        }
        if (fixed_degree_x_ready && fixed_degree_y_ready) {
            candidates.push_back(BackendCandidate{
                HPRLPXBackend::FixedDegreePackedDictionary,
                HPRLPYBackend::FixedDegreePackedDictionary});
        }
        if (ws->row_template_operator_ready &&
            ws->row_template_operator != nullptr &&
            ws->inverse_row_norm != nullptr &&
            ws->inverse_col_norm != nullptr &&
            ws->unit_scaled_y != nullptr &&
            ws->unit_scaled_x_hat != nullptr) {
            candidates.push_back(BackendCandidate{
                HPRLPXBackend::RowTemplate,
                HPRLPYBackend::RowTemplate});
        }
        if (ws->affine_block_operator_ready &&
            ws->affine_block_operator != nullptr &&
            ws->inverse_row_norm != nullptr &&
            ws->inverse_col_norm != nullptr &&
            ws->unit_scaled_y != nullptr &&
            ws->unit_scaled_x_hat != nullptr) {
            candidates.push_back(BackendCandidate{
                HPRLPXBackend::AffineBlock,
                HPRLPYBackend::AffineBlock});
        }
        if (ws->windowed_stencil_operator_ready &&
            ws->factorized_static_records_ready &&
            ws->inverse_row_norm != nullptr &&
            ws->inverse_col_norm != nullptr &&
            ws->unit_scaled_y != nullptr &&
            ws->unit_scaled_x_hat != nullptr &&
            ws->factorized_stencil_dense_partials != nullptr &&
            ws->factorized_stencil_dense_counter != nullptr) {
            candidates.push_back(BackendCandidate{
                HPRLPXBackend::FactorizedStencil,
                HPRLPYBackend::FactorizedStencil});
        }

    return candidates;
}

} // namespace
