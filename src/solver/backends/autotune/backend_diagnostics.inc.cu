namespace {

void print_backend_autotune_diagnostics(HPRLP_workspace_gpu *ws,
                                        int bench_iters,
                                        int warmup_iters) {
        std::cout << "AUTO-SELECT custom backends (" << bench_iters
                  << " timed iterations, " << warmup_iters
                  << " warmup iterations per candidate) ..." << std::endl;
        int total_A = ws->num_A_rows_short + ws->num_A_rows_medium + ws->num_A_rows_long;
        int total_AT = ws->num_AT_rows_short + ws->num_AT_rows_medium + ws->num_AT_rows_long;
        std::cout << "  A row buckets: short=" << ws->num_A_rows_short << ", medium=" << ws->num_A_rows_medium
                  << ", long=" << ws->num_A_rows_long
                  << ", total=" << total_A
                  << ", max=" << ws->max_A_row_nnz << std::endl;
        std::cout << "  AT row buckets: short=" << ws->num_AT_rows_short << ", medium=" << ws->num_AT_rows_medium
                  << ", long=" << ws->num_AT_rows_long
                  << ", total=" << total_AT
                  << ", max=" << ws->max_AT_row_nnz << std::endl;
        std::cout << "  unit-operator x: " << (ws->unit_operator_x_ready ? "enabled" : "disabled") << std::endl;
        std::cout << "  unit-operator y: " << (ws->unit_operator_y_ready ? "enabled" : "disabled") << std::endl;
        std::cout << "  unit col-tile y: "
                  << (ws->unit_coltile_ready ? "enabled" : "disabled");
        if (ws->unit_coltile_ready && ws->unit_coltile != nullptr) {
            std::cout << " (tiles=" << ws->unit_coltile->tile_count
                      << ", tile_cols=" << ws->unit_coltile->tile_cols
                      << ")";
        }
        std::cout << std::endl;
        std::cout << "  uniform unit sign: "
                  << static_cast<int>(ws->uniform_unit_sign) << std::endl;
        std::cout << "  signed packed operator: "
                  << (ws->signed_unit_operator_ready ? "enabled" : "disabled");
        if (ws->signed_unit_operator_ready &&
            ws->signed_unit_operator != nullptr) {
            std::cout << " (A="
                      << (ws->signed_unit_operator->A_uses_u16
                              ? "u16"
                              : "u32")
                      << ", AT="
                      << (ws->signed_unit_operator->AT_uses_u16
                              ? "u16"
                              : "u32")
                      << ")";
        }
        std::cout << std::endl;
        if (ws->signed_unit_operator_ready &&
            ws->signed_unit_operator != nullptr) {
            std::cout << "  signed fixed-degree runs: A(degree=2, row_begin="
                      << ws->signed_unit_operator->A_degree2_run_row_begin
                      << ", rows="
                      << ws->signed_unit_operator->A_degree2_run_row_count
                      << ", entry_begin="
                      << ws->signed_unit_operator->A_degree2_run_entry_begin
                      << "), AT(degree=3, row_begin="
                      << ws->signed_unit_operator->AT_degree3_run_row_begin
                      << ", rows="
                      << ws->signed_unit_operator->AT_degree3_run_row_count
                      << ", entry_begin="
                      << ws->signed_unit_operator->AT_degree3_run_entry_begin
                      << ")" << std::endl;
        }
        std::cout << "  dictionary-operator x: "
                  << (ws->dictionary_operator_x_ready ? "enabled" : "disabled")
                  << " (values=" << ws->coefficient_dictionary_size
                  << ", storage=" << hprlp_packed_dictionary_storage_name(
                         ws->packed_dictionary_storage)
                  << ", index_bits=" << ws->packed_dictionary_index_bits
                  << ", code_bits=" << ws->packed_dictionary_code_bits
                  << ")"
                  << std::endl;
        std::cout << "  dictionary-operator y: "
                  << (ws->dictionary_operator_y_ready
                          ? "enabled" : "disabled")
                  << " (values=" << ws->A_coefficient_dictionary_size
                  << ", storage=" << hprlp_packed_dictionary_storage_name(
                         ws->A_packed_dictionary_storage)
                  << ", code_bits="
                  << ws->A_packed_dictionary_code_bits << ")"
                  << std::endl;
        std::cout << "  fixed-degree packed runs: A="
                  << (ws->fixed_degree_A_plan.ready ? "enabled" : "disabled")
                  << " (runs=" << ws->fixed_degree_A_plan.run_count
                  << ", threads=" << ws->fixed_degree_A_plan.threads
                  << ", rows=" << ws->fixed_degree_A_plan.covered_rows
                  << ", nnz=" << ws->fixed_degree_A_plan.covered_nnz
                  << "), AT="
                  << (ws->fixed_degree_AT_plan.ready ? "enabled" : "disabled")
                  << " (runs=" << ws->fixed_degree_AT_plan.run_count
                  << ", threads=" << ws->fixed_degree_AT_plan.threads
                  << ", rows=" << ws->fixed_degree_AT_plan.covered_rows
                  << ", nnz=" << ws->fixed_degree_AT_plan.covered_nnz
                  << ")" << std::endl;
        std::cout << "  packed state runs: x-lower-boxed="
                  << ws->packed_state_plan.x_zero_lower_boxed_begin << ":"
                  << ws->packed_state_plan.x_zero_lower_boxed_count
                  << ", x-c-zero="
                  << ws->packed_state_plan.x_objective_zero_begin << ":"
                  << ws->packed_state_plan.x_objective_zero_count
                  << ", y-lower="
                  << ws->packed_state_plan.y_lower_only_begin << ":"
                  << ws->packed_state_plan.y_lower_only_count
                  << ", y-upper="
                  << ws->packed_state_plan.y_upper_only_begin << ":"
                  << ws->packed_state_plan.y_upper_only_count
                  << ", y-equality="
                  << ws->packed_state_plan.y_equality_begin << ":"
                  << ws->packed_state_plan.y_equality_count
                  << std::endl;
        std::cout << "  structured original operator: "
                  << (ws->structured_operator_ready ? "enabled" : "disabled");
        if (ws->structured_operator_ready) {
            std::cout << " (dense_rows="
                      << ws->structured_operator->dense_row_count
                      << ", dense_cols="
                      << ws->structured_operator->dense_col_count << ")";
        }
        std::cout << std::endl;
        std::cout << "  factorized static records: "
                  << (ws->factorized_static_records_ready
                          ? "enabled"
                          : "disabled");
        if (ws->factorized_static_records_ready) {
            std::cout << " (x=" << ws->factorized_x_static_record_count
                      << ", y="
                      << ws->factorized_y_static_record_count << ")";
        }
        std::cout << std::endl;
        std::cout << "  parameterized grid-slack-laplacian pair: "
                  << (ws->grid_slack_laplacian_operator_ready
                          ? "enabled"
                          : "disabled");
        if (ws->grid_slack_laplacian_operator_ready) {
            const HPRLPGridSlackLaplacianShape &shape =
                ws->grid_slack_laplacian_shape;
            std::cout << " (grid=" << shape.grid_side << "x"
                      << shape.grid_side << ", interior="
                      << shape.interior_side << "x"
                      << shape.interior_side << ", grid_variables="
                      << shape.grid_count << ", slacks="
                      << shape.slack_count << ")";
        }
        std::cout << std::endl;
        std::cout << "  repeated-row template pair: "
                  << (ws->row_template_operator_ready &&
                              ws->row_template_operator != nullptr
                          ? "enabled"
                          : "disabled");
        if (ws->row_template_operator_ready &&
            ws->row_template_operator != nullptr) {
            std::cout << " (A="
                      << ws->row_template_operator->A.encoded_row_count
                      << "/" << ws->m << ", templates="
                      << ws->row_template_operator->A.template_count
                      << ", fallback="
                      << ws->row_template_operator->A.fallback_short_count
                      << "/"
                      << ws->row_template_operator->A.fallback_warp_count
                      << "/"
                      << ws->row_template_operator->A.fallback_block_count
                      << "; AT="
                      << ws->row_template_operator->AT.encoded_row_count
                      << "/" << ws->n << ", templates="
                      << ws->row_template_operator->AT.template_count
                      << ", fallback="
                      << ws->row_template_operator->AT.fallback_short_count
                      << "/"
                      << ws->row_template_operator->AT.fallback_warp_count
                      << "/"
                      << ws->row_template_operator->AT.fallback_block_count
                      << ")";
        }
        std::cout << std::endl;
        std::cout << "  affine-block pair: "
                  << (ws->affine_block_operator_ready &&
                              ws->affine_block_operator != nullptr
                          ? "enabled"
                          : "disabled");
        if (ws->affine_block_operator_ready &&
            ws->affine_block_operator != nullptr) {
            std::cout << " (A="
                      << ws->affine_block_operator->A.encoded_row_count
                      << "/" << ws->m << ", blocks="
                      << ws->affine_block_operator->A.block_count
                      << ", fallback="
                      << ws->affine_block_operator->A.fallback_short_count
                      << "/"
                      << ws->affine_block_operator->A.fallback_warp_count
                      << "/"
                      << ws->affine_block_operator->A.fallback_block_count
                      << "; AT="
                      << ws->affine_block_operator->AT.encoded_row_count
                      << "/" << ws->n << ", blocks="
                      << ws->affine_block_operator->AT.block_count
                      << ", fallback="
                      << ws->affine_block_operator->AT.fallback_short_count
                      << "/"
                      << ws->affine_block_operator->AT.fallback_warp_count
                      << "/"
                      << ws->affine_block_operator->AT.fallback_block_count
                      << ")";
        }
        std::cout << std::endl;
        std::cout << "  parameterized windowed-stencil pair: "
                  << (ws->windowed_stencil_operator_ready
                          ? "enabled"
                          : "disabled");
        if (ws->windowed_stencil_operator_ready) {
            const HPRLPWindowedStencilShape &shape =
                ws->windowed_stencil_shape;
            std::cout << " (grid=" << shape.grid_height << "x"
                      << shape.grid_width << ", observations="
                      << shape.observation_height << "x"
                      << shape.observation_width << ", offset="
                      << shape.observation_y0 << ","
                      << shape.observation_x0 << ")";
        }
        std::cout << std::endl;
        std::cout << "  windowed-stencil state specialization: "
                  << (ws->windowed_stencil_state_ready
                          ? "enabled"
                          : "disabled")
                  << std::endl;
        std::cout << "  nonnegative unit-x: "
                  << (ws->unit_operator_x_ready && ws->all_zero_lower_unbounded_variables &&
                      ws->num_AT_rows_short == ws->n ? "enabled" : "disabled")
                  << std::endl;

}

} // namespace
