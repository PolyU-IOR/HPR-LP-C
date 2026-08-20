void allocate_memory(HPRLP_workspace_gpu *workspace, LP_info_gpu *lp_info_gpu) {
    // allocate memory for the workspace
    int m = workspace->m;
    int n = workspace->n;
    cudaStreamCreate(&workspace->stream);
    CUDA_CHECK(cudaMalloc((void**)&workspace->Halpern_params, 4 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemset(workspace->Halpern_params, 0, 4 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc((void**)&workspace->halpern_inner, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&workspace->halpern_factors, 2 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(
        (void**)&workspace->halpern_factor_batch,
        2 * HPRLP_NORMAL_GRAPH_BATCH_SIZE * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMallocHost(&workspace->iter_params_host, 4 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMallocHost(&workspace->halpern_inner_host, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&workspace->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT)));
    workspace->halpern_inner_host[0] = 0;
    workspace->halpern_factors_host[0] = 0.5;
    workspace->halpern_factors_host[1] = 0.5;
    memset(workspace->iter_params_host, 0, 4 * sizeof(HPRLP_FLOAT));
    CUDA_CHECK(cudaMemcpyAsync(workspace->halpern_inner, workspace->halpern_inner_host, sizeof(int),
                               cudaMemcpyHostToDevice, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace->halpern_factors, workspace->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, workspace->stream));

    create_zero_vector_device_compressible(workspace->x, n);
    create_zero_vector_device_compressible(workspace->last_x, n);
    create_zero_vector_device_compressible(workspace->x_temp, n);
    create_zero_vector_device_compressible(workspace->x_hat, n);
    create_zero_vector_device_compressible(workspace->x_bar, n);
    create_zero_vector_device_compressible(workspace->y, m);
    create_zero_vector_device_compressible(workspace->last_y, m);
    create_zero_vector_device_compressible(workspace->y_temp, m);
    create_zero_vector_device_compressible(workspace->y_bar, m);
    create_zero_vector_device_compressible(workspace->y_hat, m);
    create_zero_vector_device_compressible(workspace->y_obj, m);
    create_zero_vector_device_compressible(workspace->z_bar, n);

    workspace->A = lp_info_gpu->A;
    workspace->AT = lp_info_gpu->AT;
    workspace->AL = lp_info_gpu->AL;
    workspace->AU = lp_info_gpu->AU;
    workspace->c = lp_info_gpu->c;
    workspace->l = lp_info_gpu->l;
    workspace->u = lp_info_gpu->u;

    create_zero_vector_device_compressible(workspace->Rd, n);
    create_zero_vector_device_compressible(workspace->Rp, m);
    create_zero_vector_device_compressible(workspace->ATy, n);
    create_zero_vector_device_compressible(workspace->Ax, m);

    workspace->check = false;

    workspace->graph = nullptr;  // Initialize CUDA Graph pointer
    workspace->graph_exec = nullptr;
    workspace->graph_batch = nullptr;
    workspace->graph_exec_batch = nullptr;
    workspace->graph_initialized = false;
    workspace->check_graph = nullptr;
    workspace->check_graph_exec = nullptr;
    workspace->check_graph_initialized = false;
    workspace->x_backend = HPRLPXBackend::ScaledCusparse;
    workspace->y_backend = HPRLPYBackend::ScaledCusparse;
    workspace->reduced_use_fused_x = false;
    workspace->reduced_use_fused_y = false;
    workspace->reduced_backend_autotune_done = false;
    workspace->all_positive_unit_coefficients = lp_info_gpu->all_positive_unit_coefficients;
    workspace->uniform_unit_sign = lp_info_gpu->uniform_unit_sign;
    workspace->all_zero_lower_unbounded_variables =
        lp_info_gpu->all_zero_lower_unbounded_variables;
    workspace->unit_operator_x_ready = false;
    workspace->unit_operator_y_ready = false;
    workspace->signed_unit_operator_ready = false;
    workspace->signed_state_plan_ready =
        lp_info_gpu->signed_state_plan_ready;
    workspace->signed_x_zero_objective_run_begin =
        lp_info_gpu->signed_x_zero_objective_run_begin;
    workspace->signed_x_zero_objective_run_count =
        lp_info_gpu->signed_x_zero_objective_run_count;
    workspace->signed_y_upper_zero_run_begin =
        lp_info_gpu->signed_y_upper_zero_run_begin;
    workspace->signed_y_upper_zero_run_count =
        lp_info_gpu->signed_y_upper_zero_run_count;
    workspace->signed_state_specialization_enabled =
        workspace->signed_state_plan_ready;
    workspace->unit_coltile_ready = lp_info_gpu->unit_coltile != nullptr;
    workspace->dictionary_operator_x_ready = false;
    workspace->dictionary_operator_y_ready =
        lp_info_gpu->A_coefficient_dictionary != nullptr &&
        (lp_info_gpu->A_packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::PackedU32 ||
         lp_info_gpu->A_packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::SeparateU32U16);
    workspace->structured_operator_ready = false;
    workspace->row_template_operator_ready =
        lp_info_gpu->row_template_operator != nullptr;
    workspace->affine_block_operator_ready =
        lp_info_gpu->affine_block_operator != nullptr;
    workspace->windowed_stencil_operator_ready =
        lp_info_gpu->windowed_stencil_operator;
    workspace->windowed_stencil_shape =
        lp_info_gpu->windowed_stencil_shape;
    workspace->windowed_stencil_state_ready = false;
    workspace->windowed_observation_first_bound_type = 0;
    workspace->windowed_observation_second_bound_type = 0;
    workspace->grid_slack_laplacian_operator_ready =
        lp_info_gpu->grid_slack_laplacian_operator;
    workspace->grid_slack_laplacian_shape =
        lp_info_gpu->grid_slack_laplacian_shape;
    workspace->structured_operator = lp_info_gpu->structured_operator;
    workspace->row_template_operator = lp_info_gpu->row_template_operator;
    workspace->affine_block_operator = lp_info_gpu->affine_block_operator;
    workspace->unit_coltile = lp_info_gpu->unit_coltile;
    workspace->signed_unit_operator = lp_info_gpu->signed_unit_operator;
    workspace->coefficient_dictionary_size =
        lp_info_gpu->coefficient_dictionary_size;
    workspace->coefficient_dictionary = lp_info_gpu->coefficient_dictionary;
    workspace->AT_value_codes = lp_info_gpu->AT_value_codes;
    workspace->packed_dictionary_storage =
        lp_info_gpu->packed_dictionary_storage;
    workspace->packed_dictionary_index_bits =
        lp_info_gpu->packed_dictionary_index_bits;
    workspace->packed_dictionary_code_bits =
        lp_info_gpu->packed_dictionary_code_bits;
    workspace->AT_dictionary_packed_u32 =
        lp_info_gpu->AT_dictionary_packed_u32;
    workspace->AT_dictionary_indices_u16 =
        lp_info_gpu->AT_dictionary_indices_u16;
    workspace->AT_dictionary_indices_u32 =
        lp_info_gpu->AT_dictionary_indices_u32;
    workspace->AT_dictionary_codes_u8 =
        lp_info_gpu->AT_dictionary_codes_u8;
    workspace->AT_dictionary_codes_u16 =
        lp_info_gpu->AT_dictionary_codes_u16;
    workspace->A_coefficient_dictionary_size =
        lp_info_gpu->A_coefficient_dictionary_size;
    workspace->A_coefficient_dictionary =
        lp_info_gpu->A_coefficient_dictionary;
    workspace->A_packed_dictionary_storage =
        lp_info_gpu->A_packed_dictionary_storage;
    workspace->A_packed_dictionary_code_bits =
        lp_info_gpu->A_packed_dictionary_code_bits;
    workspace->A_dictionary_packed_u32 =
        lp_info_gpu->A_dictionary_packed_u32;
    workspace->A_dictionary_indices_u32 =
        lp_info_gpu->A_dictionary_indices_u32;
    workspace->A_dictionary_codes_u16 =
        lp_info_gpu->A_dictionary_codes_u16;
    workspace->packed_state_plan = lp_info_gpu->packed_state_plan;
    workspace->unit_AT_col_index_u16 = nullptr;
    workspace->A_col_index_u16 = nullptr;
    workspace->inverse_row_norm = nullptr;
    workspace->inverse_col_norm = nullptr;
    workspace->factor_row_norm = nullptr;
    workspace->factor_col_norm = nullptr;
    workspace->unit_scaled_y = nullptr;
    workspace->unit_A_values = nullptr;
    workspace->unit_scaled_x_hat = nullptr;
    workspace->factorized_stencil_dense_partials = nullptr;
    workspace->factorized_stencil_dense_counter = nullptr;
    workspace->factorized_last_block_enabled =
        workspace->windowed_stencil_operator_ready;
    workspace->factorized_static_records_ready = false;
    workspace->factorized_x_static_record_count = 0;
    workspace->factorized_y_static_record_count = 0;
    workspace->factorized_x_static_codes = nullptr;
    workspace->factorized_y_static_codes = nullptr;
    workspace->factorized_x_static_records = nullptr;
    workspace->factorized_y_static_records = nullptr;
    workspace->unit_scaled_x_zero_bits = nullptr;
    workspace->unit_scaled_x_positive_zero_count = nullptr;
    workspace->unit_scaled_x_hat_nonzero = nullptr;
    workspace->signed_xhat_positive_zero_count = nullptr;
    workspace->signed_xhat_positive_zero_count_host = nullptr;

    CUBLAS_CHECK(cublasCreate(&workspace->cublasHandle));
    CUBLAS_CHECK(cublasSetStream(workspace->cublasHandle, workspace->stream));

    // device-mode CUBLAS handle for queued (async) reductions
    CUBLAS_CHECK(cublasCreate(&workspace->cublasHandle_device));
    CUBLAS_CHECK(cublasSetStream(workspace->cublasHandle_device,
                                workspace->stream));
    CUBLAS_CHECK(cublasSetPointerMode(workspace->cublasHandle_device,
                                     CUBLAS_POINTER_MODE_DEVICE));

    // 14-slot reduction scalar staging buffers
    CUDA_CHECK(cudaMalloc(&workspace->reduction_scalars, 14 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemset(workspace->reduction_scalars, 0, 14 * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMallocHost(&workspace->reduction_scalars_host, 14 * sizeof(HPRLP_FLOAT)));
    memset(workspace->reduction_scalars_host, 0, 14 * sizeof(HPRLP_FLOAT));

}
