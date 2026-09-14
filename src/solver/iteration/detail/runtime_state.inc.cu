void reset_halpern_runtime_params(HPRLP_workspace_gpu *ws) {
    HPRLP_FLOAT fact1 = ws->lambda_max * ws->sigma;
    ws->iter_params_host[0] = ws->sigma;
    ws->iter_params_host[1] = fact1;
    ws->iter_params_host[2] = 1.0 / fact1;
    ws->iter_params_host[3] = 1.0 / ws->sigma;
    ws->halpern_inner_host[0] = 0;
    ws->halpern_factors_host[0] = 0.5;
    ws->halpern_factors_host[1] = 0.5;

    CUDA_CHECK(cudaMemcpyAsync(ws->Halpern_params, ws->iter_params_host, 4 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_inner, ws->halpern_inner_host, sizeof(int),
                               cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_factors, ws->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));

    ws->uploaded_sigma = ws->sigma;
    ws->uploaded_lambda_max = ws->lambda_max;
}

void upload_halpern_iter_params_if_needed(HPRLP_workspace_gpu *ws) {
    if (ws->sigma == ws->uploaded_sigma && ws->lambda_max == ws->uploaded_lambda_max) {
        return;
    }

    HPRLP_FLOAT y_fact1 = ws->lambda_max * ws->sigma;
    ws->iter_params_host[0] = ws->sigma;
    ws->iter_params_host[1] = y_fact1;
    ws->iter_params_host[2] = 1.0 / y_fact1;
    ws->iter_params_host[3] = 1.0 / ws->sigma;
    CUDA_CHECK(cudaMemcpyAsync(ws->Halpern_params, ws->iter_params_host, 4 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));
    ws->uploaded_sigma = ws->sigma;
    ws->uploaded_lambda_max = ws->lambda_max;
}

void upload_halpern_restart_params(HPRLP_workspace_gpu *ws, HPRLP_restart *restart_info) {
    if (restart_info->restart_flag <= 0) {
        return;
    }

    ws->halpern_inner_host[0] = 0.0;
    ws->halpern_factors_host[0] = 0.5;
    ws->halpern_factors_host[1] = 1.0 - ws->halpern_factors_host[0];
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_inner, ws->halpern_inner_host, sizeof(int),
                               cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(ws->halpern_factors, ws->halpern_factors_host, 2 * sizeof(HPRLP_FLOAT),
                               cudaMemcpyHostToDevice, ws->stream));
}

void advance_halpern_factors(HPRLP_workspace_gpu *ws) {
    advance_halpern_factors_kernel<<<1, 1, 0, ws->stream>>>(ws->halpern_inner, ws->halpern_factors);
}

void monitor_signed_zero_skip_backend(HPRLP_workspace_gpu *ws) {
    if (!ws->signed_zero_skip_monitor_enabled || ws->n <= 0 ||
        ws->unit_scaled_x_hat_nonzero == nullptr ||
        ws->signed_xhat_positive_zero_count == nullptr ||
        ws->signed_xhat_positive_zero_count_host == nullptr) {
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(ws->signed_xhat_positive_zero_count, 0,
                               sizeof(unsigned long long), ws->stream));
    count_raw_positive_zero_flags_kernel<<<
        HPRLP_NUM_BLOCKS(ws->n), HPRLP_NUM_THREADS, 0, ws->stream>>>(
        ws->unit_scaled_x_hat_nonzero, ws->n,
        ws->signed_xhat_positive_zero_count);
    CUDA_CHECK(cudaMemcpyAsync(ws->signed_xhat_positive_zero_count_host,
                               ws->signed_xhat_positive_zero_count,
                               sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    const double positive_zero_fraction =
        static_cast<double>(*ws->signed_xhat_positive_zero_count_host) /
        static_cast<double>(ws->n);
    const bool next_enabled = ws->signed_zero_skip_enabled
        ? positive_zero_fraction >= kSignedZeroSkipDisableFraction
        : positive_zero_fraction >= kSignedZeroSkipEnableFraction;
    if (next_enabled == ws->signed_zero_skip_enabled) {
        return;
    }

    ws->signed_zero_skip_enabled = next_enabled;
    // Kernel identities are captured in both normal-update graphs.  Force a
    // rebuild exactly when hysteresis changes the selected Y implementation.
    ws->graph_initialized = false;
}
