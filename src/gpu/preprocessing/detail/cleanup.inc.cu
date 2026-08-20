void free_workspace(HPRLP_workspace_gpu *workspace) {
    /*
     * Free all GPU memory allocated in allocate_memory and prepare_spmv.
     * This prevents memory leaks. Note: When called from Python ctypes, the
     * process may still segfault during Python interpreter shutdown due to
     * CUDA/ctypes interaction, but this is harmless (happens after results returned).
    */
    if (!workspace) return;

    // All objects below may still be referenced by work queued on this stream.
    // Drain it before destroying descriptors, graphs, handles, or VMM-backed
    // allocations.
    if (workspace->stream) {
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    }
    // Destroy cuBLAS handles FIRST (before freeing vectors they might reference)
    if (workspace->cublasHandle_device) {
        cublasDestroy(workspace->cublasHandle_device);
        workspace->cublasHandle_device = nullptr;
    }
    if (workspace->cublasHandle) {
        cublasDestroy(workspace->cublasHandle);
        workspace->cublasHandle = nullptr;
    }

    // Free reduction scalar staging buffers
    if (workspace->reduction_scalars) {
        cudaFree(workspace->reduction_scalars);
        workspace->reduction_scalars = nullptr;
    }
    if (workspace->reduction_scalars_host) {
        cudaFreeHost(workspace->reduction_scalars_host);
        workspace->reduction_scalars_host = nullptr;
    }

    // Destroy SpMVOp plans before their matrix/vector descriptors and shared
    // cuSPARSE handle.
    cusparseHandle_t spmvop_handle = workspace->spmv_AT
        ? workspace->spmv_AT->cusparseHandle
        : (workspace->spmv_A ? workspace->spmv_A->cusparseHandle : nullptr);

    // Destroy CUSPARSE descriptors BEFORE freeing the underlying memory
    // Free CUSPARSE resources for AT matrix operations
    if (workspace->spmv_AT) {
        hprlp_destroy_spmvop(&workspace->spmv_AT->operation);
        if (workspace->spmv_AT->y_bar_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_AT->y_bar_cusparseDescr);
        if (workspace->spmv_AT->y_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_AT->y_cusparseDescr);
        if (workspace->spmv_AT->ATy_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_AT->ATy_cusparseDescr);
        if (workspace->spmv_AT->AT_cusparseDescr) cusparseDestroySpMat(workspace->spmv_AT->AT_cusparseDescr);
        delete workspace->spmv_AT;
        workspace->spmv_AT = nullptr;
    }

    // Free CUSPARSE resources for A matrix operations
    if (workspace->spmv_A) {
        hprlp_destroy_spmvop(&workspace->spmv_A->unit_operation);
        hprlp_destroy_spmvop(&workspace->spmv_A->operation);
        if (workspace->spmv_A->unit_x_hat_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->unit_x_hat_cusparseDescr);
        if (workspace->spmv_A->unit_A_cusparseDescr) cusparseDestroySpMat(workspace->spmv_A->unit_A_cusparseDescr);
        if (workspace->spmv_A->x_bar_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->x_bar_cusparseDescr);
        if (workspace->spmv_A->x_hat_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->x_hat_cusparseDescr);
        if (workspace->spmv_A->x_temp_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->x_temp_cusparseDescr);
        if (workspace->spmv_A->Ax_cusparseDescr) cusparseDestroyDnVec(workspace->spmv_A->Ax_cusparseDescr);
        if (workspace->spmv_A->A_cusparseDescr) cusparseDestroySpMat(workspace->spmv_A->A_cusparseDescr);
        delete workspace->spmv_A;
        workspace->spmv_A = nullptr;
    }
    if (spmvop_handle) {
        cusparseDestroy(spmvop_handle);
    }

    // Destroy CUDA Graph resources
    if (workspace->graph_exec) {
        cudaGraphExecDestroy(workspace->graph_exec);
        workspace->graph_exec = nullptr;
    }
    if (workspace->graph) {
        cudaGraphDestroy(workspace->graph);
        workspace->graph = nullptr;
    }
    if (workspace->graph_exec_batch) {
        cudaGraphExecDestroy(workspace->graph_exec_batch);
        workspace->graph_exec_batch = nullptr;
    }
    if (workspace->graph_batch) {
        cudaGraphDestroy(workspace->graph_batch);
        workspace->graph_batch = nullptr;
    }
    if (workspace->check_graph_exec) {
        cudaGraphExecDestroy(workspace->check_graph_exec);
        workspace->check_graph_exec = nullptr;
    }
    if (workspace->check_graph) {
        cudaGraphDestroy(workspace->check_graph);
        workspace->check_graph = nullptr;
    }
    // Free Halpern runtime parameter buffers
    if (workspace->Halpern_params) {
        cudaFree(workspace->Halpern_params);
        workspace->Halpern_params = nullptr;
    }
    if (workspace->halpern_inner) {
        cudaFree(workspace->halpern_inner);
        workspace->halpern_inner = nullptr;
    }
    if (workspace->halpern_factors) {
        cudaFree(workspace->halpern_factors);
        workspace->halpern_factors = nullptr;
    }
    if (workspace->halpern_factor_batch) {
        cudaFree(workspace->halpern_factor_batch);
        workspace->halpern_factor_batch = nullptr;
    }
    if (workspace->iter_params_host) {
        cudaFreeHost(workspace->iter_params_host);
        workspace->iter_params_host = nullptr;
    }
    if (workspace->halpern_inner_host) {
        cudaFreeHost(workspace->halpern_inner_host);
        workspace->halpern_inner_host = nullptr;
    }
    if (workspace->halpern_factors_host) {
        cudaFreeHost(workspace->halpern_factors_host);
        workspace->halpern_factors_host = nullptr;
    }

    free_device_allocation(workspace->x_bound_type);
    free_device_allocation(workspace->y_bound_type);
    free_device_allocation(workspace->A_rows_short);
    free_device_allocation(workspace->A_rows_medium);
    free_device_allocation(workspace->A_rows_long);
    free_device_allocation(workspace->segmented_A_rows);
    free_device_allocation(workspace->segmented_A_row_tile_ptr);
    free_device_allocation(workspace->segmented_A_tile_begin);
    free_device_allocation(workspace->segmented_A_tile_end);
    free_device_allocation(workspace->segmented_A_fallback_long_rows);
    free_device_allocation(workspace->segmented_A_partials);
    free_device_allocation(workspace->segmented_AT_rows);
    free_device_allocation(workspace->segmented_AT_row_tile_ptr);
    free_device_allocation(workspace->segmented_AT_tile_begin);
    free_device_allocation(workspace->segmented_AT_tile_end);
    free_device_allocation(workspace->segmented_AT_fallback_long_rows);
    free_device_allocation(workspace->segmented_AT_partials);
    free_device_allocation(
        workspace->fixed_degree_A_plan.fallback_short_rows);
    free_device_allocation(
        workspace->fixed_degree_A_plan.fallback_warp_rows);
    free_device_allocation(
        workspace->fixed_degree_A_plan.fallback_block_rows);
    free_device_allocation(
        workspace->fixed_degree_AT_plan.fallback_short_rows);
    free_device_allocation(
        workspace->fixed_degree_AT_plan.fallback_warp_rows);
    free_device_allocation(
        workspace->fixed_degree_AT_plan.fallback_block_rows);
    for (int run_index = 0;
         run_index < workspace->fixed_degree_A_plan.run_count;
         ++run_index) {
        free_device_allocation(
            workspace->fixed_degree_A_plan.runs[run_index]
                .packed_entries_soa);
    }
    for (int run_index = 0;
         run_index < workspace->fixed_degree_AT_plan.run_count;
         ++run_index) {
        free_device_allocation(
            workspace->fixed_degree_AT_plan.runs[run_index]
                .packed_entries_soa);
    }
    free_device_allocation(workspace->AT_rows_short);
    free_device_allocation(workspace->AT_rows_medium);
    free_device_allocation(workspace->AT_rows_long);
    free_device_allocation(workspace->unit_AT_col_index_u16);
    free_device_allocation(workspace->A_col_index_u16);
    free_device_allocation(workspace->inverse_row_norm);
    free_device_allocation(workspace->inverse_col_norm);
    free_device_allocation(workspace->unit_scaled_y);
    free_device_allocation(workspace->unit_A_values);
    free_device_allocation(workspace->unit_scaled_x_hat);
    free_device_allocation(workspace->factorized_stencil_dense_partials);
    free_device_allocation(workspace->factorized_stencil_dense_counter);
    free_device_allocation(workspace->factorized_x_static_codes);
    free_device_allocation(workspace->factorized_y_static_codes);
    free_device_allocation(workspace->factorized_x_static_records);
    free_device_allocation(workspace->factorized_y_static_records);
    free_device_allocation(workspace->unit_scaled_x_zero_bits);
    free_device_allocation(workspace->unit_scaled_x_positive_zero_count);
    free_device_allocation(workspace->unit_scaled_x_hat_nonzero);
    free_device_allocation(workspace->signed_xhat_positive_zero_count);
    if (workspace->signed_xhat_positive_zero_count_host) {
        cudaFreeHost(workspace->signed_xhat_positive_zero_count_host);
        workspace->signed_xhat_positive_zero_count_host = nullptr;
    }
    workspace->unit_operator_x_ready = false;
    workspace->unit_operator_y_ready = false;
    workspace->signed_unit_operator_ready = false;
    workspace->signed_state_plan_ready = false;
    workspace->signed_state_specialization_enabled = false;
    workspace->signed_x_zero_objective_run_begin = 0;
    workspace->signed_x_zero_objective_run_count = 0;
    workspace->signed_y_upper_zero_run_begin = 0;
    workspace->signed_y_upper_zero_run_count = 0;
    workspace->signed_zero_skip_monitor_enabled = false;
    workspace->signed_zero_skip_enabled = false;
    workspace->unit_coltile_ready = false;
    workspace->max_A_row_nnz = 0;
    workspace->max_AT_row_nnz = 0;
    workspace->segmented_A_ready = false;
    workspace->segmented_A_threshold = 0;
    workspace->segmented_A_tile_entries = 0;
    workspace->segmented_A_threads = 0;
    workspace->segmented_A_row_count = 0;
    workspace->segmented_A_tile_count = 0;
    workspace->segmented_A_fallback_long_count = 0;
    workspace->segmented_A_nnz = 0;
    workspace->segmented_AT_ready = false;
    workspace->segmented_AT_threshold = 0;
    workspace->segmented_AT_tile_entries = 0;
    workspace->segmented_AT_threads = 0;
    workspace->segmented_AT_row_count = 0;
    workspace->segmented_AT_tile_count = 0;
    workspace->segmented_AT_fallback_long_count = 0;
    workspace->segmented_AT_nnz = 0;
    workspace->fixed_degree_A_plan = HPRLPFixedDegreeRunPlan{};
    workspace->fixed_degree_AT_plan = HPRLPFixedDegreeRunPlan{};
    workspace->uniform_unit_sign = 0;
    workspace->dictionary_operator_x_ready = false;
    workspace->dictionary_operator_y_ready = false;
    workspace->structured_operator_ready = false;
    workspace->factorized_static_records_ready = false;
    workspace->factorized_last_block_enabled = false;
    workspace->factorized_x_static_record_count = 0;
    workspace->factorized_y_static_record_count = 0;
    workspace->row_template_operator_ready = false;
    workspace->affine_block_operator_ready = false;
    workspace->windowed_stencil_operator_ready = false;
    workspace->windowed_stencil_shape = HPRLPWindowedStencilShape{};
    workspace->windowed_stencil_state_ready = false;
    workspace->windowed_observation_first_bound_type = 0;
    workspace->windowed_observation_second_bound_type = 0;
    workspace->grid_slack_laplacian_operator_ready = false;
    workspace->grid_slack_laplacian_shape =
        HPRLPGridSlackLaplacianShape{};
    workspace->structured_operator = nullptr;
    workspace->row_template_operator = nullptr;
    workspace->affine_block_operator = nullptr;
    workspace->unit_coltile = nullptr;
    workspace->signed_unit_operator = nullptr;
    workspace->factor_row_norm = nullptr;
    workspace->factor_col_norm = nullptr;
    workspace->x_backend = HPRLPXBackend::ScaledCusparse;
    workspace->y_backend = HPRLPYBackend::ScaledCusparse;
    workspace->reduced_use_fused_x = false;
    workspace->reduced_use_fused_y = false;
    workspace->reduced_backend_autotune_done = false;

    // NOW free device vectors (after descriptors are destroyed)
    free_device_allocation(workspace->x);
    free_device_allocation(workspace->last_x);
    free_device_allocation(workspace->x_temp);
    free_device_allocation(workspace->x_hat);
    free_device_allocation(workspace->x_bar);
    free_device_allocation(workspace->y);
    free_device_allocation(workspace->last_y);
    free_device_allocation(workspace->y_temp);
    free_device_allocation(workspace->y_bar);
    free_device_allocation(workspace->y_hat);
    free_device_allocation(workspace->y_obj);
    free_device_allocation(workspace->z_bar);
    free_device_allocation(workspace->Rd);
    free_device_allocation(workspace->Rp);
    free_device_allocation(workspace->ATy);
    free_device_allocation(workspace->Ax);

    if (workspace->stream) {
        CUDA_CHECK(cudaStreamDestroy(workspace->stream));
        workspace->stream = nullptr;
    }
    // Note: A, AT, AL, AU, c, l, u are just pointers to lp_info_gpu data.
    // They should NOT be freed here - they will be freed in free_lp_info().
}


void free_lp_info(LP_info_gpu *lp_info) {
    /*
     * Free GPU memory allocated in copy_lpinfo_to_device.
     */
    if (!lp_info) return;

    // Free sparse matrices A and AT
    if (lp_info->A) {
        free_device_allocation(lp_info->A->rowPtr);
        free_device_allocation(lp_info->A->colIndex);
        free_device_allocation(lp_info->A->value);
        delete lp_info->A;
        lp_info->A = nullptr;
    }

    if (lp_info->AT) {
        free_device_allocation(lp_info->AT->rowPtr);
        free_device_allocation(lp_info->AT->colIndex);
        free_device_allocation(lp_info->AT->value);
        delete lp_info->AT;
        lp_info->AT = nullptr;
    }

    if (lp_info->structured_operator) {
        HPRLP_structured_operator_gpu *op = lp_info->structured_operator;
        free_device_allocation(op->dense_rows);
        free_device_allocation(op->dense_cols);
        free_device_allocation(op->dense_local_cols);
        free_device_allocation(op->dense_A_values_u16);
        free_device_allocation(op->dense_AT_values_u16);
        free_device_allocation(op->sparse_rows);
        free_device_allocation(op->sparse_col0);
        free_device_allocation(op->sparse_col1);
        free_device_allocation(op->sparse_second_sign);
        free_device_allocation(op->short_AT_output_cols);
        free_device_allocation(op->short_AT_row_ptr);
        free_device_allocation(op->short_AT_rows);
        free_device_allocation(op->short_AT_values_u16);
        delete op;
        lp_info->structured_operator = nullptr;
    }

    if (lp_info->row_template_operator) {
        HPRLP_row_template_operator_gpu *op =
            lp_info->row_template_operator;
        free_device_allocation(op->A.row_template_ids);
        free_device_allocation(op->A.row_bases);
        free_device_allocation(op->A.template_ptr);
        free_device_allocation(op->A.template_offsets);
        free_device_allocation(op->A.template_values);
        free_device_allocation(op->A.fallback_short_rows);
        free_device_allocation(op->A.fallback_warp_rows);
        free_device_allocation(op->A.fallback_block_rows);
        free_device_allocation(op->A.fallback_row_ptr);
        free_device_allocation(op->A.fallback_col_indices);
        free_device_allocation(op->A.fallback_values);
        free_device_allocation(op->AT.row_template_ids);
        free_device_allocation(op->AT.row_bases);
        free_device_allocation(op->AT.template_ptr);
        free_device_allocation(op->AT.template_offsets);
        free_device_allocation(op->AT.template_values);
        free_device_allocation(op->AT.fallback_short_rows);
        free_device_allocation(op->AT.fallback_warp_rows);
        free_device_allocation(op->AT.fallback_block_rows);
        free_device_allocation(op->AT.fallback_row_ptr);
        free_device_allocation(op->AT.fallback_col_indices);
        free_device_allocation(op->AT.fallback_values);
        delete op;
        lp_info->row_template_operator = nullptr;
    }

    if (lp_info->affine_block_operator) {
        HPRLP_affine_block_operator_gpu *op =
            lp_info->affine_block_operator;
        free_affine_block_matrix_device(&op->A);
        free_affine_block_matrix_device(&op->AT);
        delete op;
        lp_info->affine_block_operator = nullptr;
    }

    if (lp_info->unit_coltile) {
        HPRLP_unit_coltile_gpu *op = lp_info->unit_coltile;
        free_device_allocation(op->row_tile_offsets);
        free_device_allocation(op->local_cols);
        delete op;
        lp_info->unit_coltile = nullptr;
    }

    if (lp_info->signed_unit_operator) {
        HPRLP_signed_unit_operator_gpu *op =
            lp_info->signed_unit_operator;
        free_device_allocation(op->A_entries_u16);
        free_device_allocation(op->A_entries_u32);
        free_device_allocation(op->AT_entries_u16);
        free_device_allocation(op->AT_entries_u32);
        free_device_allocation(op->A_split_indices_u16);
        free_device_allocation(op->A_split_negative_u8);
        free_device_allocation(op->AT_split_indices_u16);
        free_device_allocation(op->AT_split_negative_u8);
        delete op;
        lp_info->signed_unit_operator = nullptr;
    }

    // Free constraint and variable bound vectors
    free_device_allocation(lp_info->AL);
    free_device_allocation(lp_info->AU);
    free_device_allocation(lp_info->l);
    free_device_allocation(lp_info->u);
    free_device_allocation(lp_info->c);
    free_device_allocation(lp_info->coefficient_dictionary);
    free_device_allocation(lp_info->AT_value_codes);
    free_device_allocation(lp_info->AT_dictionary_packed_u32);
    free_device_allocation(lp_info->AT_dictionary_indices_u16);
    free_device_allocation(lp_info->AT_dictionary_indices_u32);
    free_device_allocation(lp_info->AT_dictionary_codes_u8);
    free_device_allocation(lp_info->AT_dictionary_codes_u16);
    free_device_allocation(lp_info->A_coefficient_dictionary);
    free_device_allocation(lp_info->A_dictionary_packed_u32);
    free_device_allocation(lp_info->A_dictionary_indices_u32);
    free_device_allocation(lp_info->A_dictionary_codes_u16);
    lp_info->has_original_coefficient_dictionary = false;
    lp_info->coefficient_dictionary_size = 0;
    lp_info->packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    lp_info->packed_dictionary_index_bits = 0;
    lp_info->packed_dictionary_code_bits = 0;
    lp_info->A_coefficient_dictionary_size = 0;
    lp_info->A_packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    lp_info->A_packed_dictionary_code_bits = 0;
    lp_info->packed_state_plan = HPRLPPackedStatePlan{};
}

void free_lp_info_cpu(LP_info_cpu *lp_info) {
    /*
     * Free CPU memory allocated in build_model_from_arrays() or build_model_from_mps().
     * This is used to clean up LP_info_cpu structures after solving.
     * Note: AT is no longer stored in LP_info_cpu; it's generated on-the-fly
     */
    if (!lp_info) return;

    // Free sparse matrix A
    if (lp_info->A) {
        if (lp_info->A->rowPtr) free(lp_info->A->rowPtr);
        if (lp_info->A->colIndex) free(lp_info->A->colIndex);
        if (lp_info->A->value) free(lp_info->A->value);
        free(lp_info->A);
        lp_info->A = nullptr;
    }

    // Free constraint and variable bound vectors
    if (lp_info->AL) {
        free(lp_info->AL);
        lp_info->AL = nullptr;
    }
    if (lp_info->AU) {
        free(lp_info->AU);
        lp_info->AU = nullptr;
    }
    if (lp_info->l) {
        free(lp_info->l);
        lp_info->l = nullptr;
    }
    if (lp_info->u) {
        free(lp_info->u);
        lp_info->u = nullptr;
    }
    if (lp_info->c) {
        free(lp_info->c);
        lp_info->c = nullptr;
    }
}
