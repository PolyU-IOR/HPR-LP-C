void autotune_reduced_update_backends(
        HPRLP_workspace_gpu *ws,
        LP_info_gpu *lp,
        Scaling_info *scaling,
        const HPRLP_parameters *param) {
    if (ws == nullptr || lp == nullptr || scaling == nullptr ||
        param == nullptr || ws->reduced_backend_autotune_done) {
        return;
    }
    if (param->CUSPARSE_spmv) {
        ws->reduced_use_fused_x = false;
        ws->reduced_use_fused_y = false;
        ws->reduced_backend_autotune_done = true;
        return;
    }

    const HPRLPXBackend saved_x_backend = ws->x_backend;
    const HPRLPYBackend saved_y_backend = ws->y_backend;
    const bool saved_signed_single_state = ws->signed_single_state_enabled;
    const bool saved_zero_skip_monitor =
        ws->signed_zero_skip_monitor_enabled;
    const bool saved_zero_skip = ws->signed_zero_skip_enabled;
    const HPRLP_FLOAT saved_uploaded_sigma = ws->uploaded_sigma;
    const HPRLP_FLOAT saved_uploaded_lambda_max = ws->uploaded_lambda_max;
    HPRLP_FLOAT saved_iter_params[4]{};
    HPRLP_FLOAT saved_halpern_factors[2]{};
    int saved_halpern_inner = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        saved_iter_params, ws->Halpern_params,
        sizeof(saved_iter_params), cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &saved_halpern_inner, ws->halpern_inner,
        sizeof(saved_halpern_inner), cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        saved_halpern_factors, ws->halpern_factors,
        sizeof(saved_halpern_factors), cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    HPRLP_FLOAT *x_save = nullptr, *x_hat_save = nullptr;
    HPRLP_FLOAT *x_bar_save = nullptr, *y_save = nullptr;
    HPRLP_FLOAT *y_hat_save = nullptr, *y_bar_save = nullptr;
    HPRLP_FLOAT *y_obj_save = nullptr, *z_bar_save = nullptr;
    HPRLP_FLOAT *Ax_save = nullptr, *ATy_save = nullptr;
    save_device_state(
        ws, &x_save, &x_hat_save, &x_bar_save, &y_save, &y_hat_save,
        &y_bar_save, &y_obj_save, &z_bar_save, &Ax_save, &ATy_save);

    const int julia_probe_iters = std::min(param->max_iter, param->check_iter);
    auto independent_probe = [&](bool probe_x, bool use_fused) {
        const HPRLPXBackend x_backend =
            probe_x && use_fused
                ? HPRLPXBackend::GenericFused
                : HPRLPXBackend::ScaledCusparse;
        const HPRLPYBackend y_backend =
            !probe_x && use_fused
                ? HPRLPYBackend::GenericFused
                : HPRLPYBackend::ScaledCusparse;

        auto prepare = [&]() {
            restore_device_state(
                ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save,
                y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
            reset_halpern_runtime_params(ws);
            ws->x_backend = x_backend;
            ws->y_backend = y_backend;
            refresh_paired_scaled_cache(ws);
            CUDA_CHECK(cudaStreamSynchronize(ws->stream));
        };
        auto execute = [&]() {
            for (int i = 0; i < julia_probe_iters; ++i) {
                if (probe_x) {
                    update_zx_normal_gpu(ws);
                } else {
                    update_y_normal_gpu(ws);
                    // Julia advances Halpern factors inside every y update.
                    advance_halpern_factors(ws);
                }
            }
            if (probe_x) {
                update_zx_check_gpu(ws);
            } else {
                update_y_check_gpu(ws);
                advance_halpern_factors(ws);
            }
            HPRLP_residuals residual{};
            compute_residuals(
                ws, lp, scaling, &residual, julia_probe_iters);
            CUDA_CHECK(cudaStreamSynchronize(ws->stream));
            return residual.KKTx_and_gap_org_bar;
        };

        prepare();
        execute();
        prepare();
        const auto start = std::chrono::steady_clock::now();
        const HPRLP_FLOAT merit = execute();
        const long long elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
        return std::make_pair(merit, elapsed);
    };

    const auto x_ref = independent_probe(true, false);
    const auto x_fused = independent_probe(true, true);
    const auto y_ref = independent_probe(false, false);
    const auto y_fused = independent_probe(false, true);
    auto choose_fused = [](
        const std::pair<HPRLP_FLOAT, long long> &reference,
        const std::pair<HPRLP_FLOAT, long long> &candidate) {
        if (!std::isfinite(reference.first) || reference.second <= 0 ||
            !std::isfinite(candidate.first) || candidate.second <= 0) {
            return false;
        }
        const HPRLP_FLOAT allowed_metric = reference.first +
            std::max(static_cast<HPRLP_FLOAT>(1e-12),
                     std::abs(reference.first) *
                         static_cast<HPRLP_FLOAT>(0.01));
        return candidate.first <= allowed_metric &&
            candidate.second <= static_cast<long long>(
                reference.second * 0.95);
    };
    ws->reduced_use_fused_x = choose_fused(x_ref, x_fused);
    ws->reduced_use_fused_y = choose_fused(y_ref, y_fused);

    if (param->autotune_verbose) {
        const auto print_probe = [](
            const char *axis, const char *backend,
            const std::pair<HPRLP_FLOAT, long long> &probe) {
            std::cout << "  Julia-aligned " << axis
                      << " backend=" << backend << " -> "
                      << (static_cast<double>(probe.second) / 1e6)
                      << " ms, merit=" << std::setprecision(17)
                      << probe.first << std::endl;
        };
        print_probe("x", "cusparse", x_ref);
        print_probe("x", "fused", x_fused);
        print_probe("y", "cusparse", y_ref);
        print_probe("y", "fused", y_fused);
        std::cout << "  reduced selected x="
                  << (ws->reduced_use_fused_x ? "fused" : "cusparse")
                  << ", y="
                  << (ws->reduced_use_fused_y ? "fused" : "cusparse")
                  << std::endl;
    }

    restore_device_state(
        ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save,
        y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
    CUDA_CHECK(cudaMemcpyAsync(
        ws->Halpern_params, saved_iter_params,
        sizeof(saved_iter_params), cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        ws->halpern_inner, &saved_halpern_inner,
        sizeof(saved_halpern_inner), cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        ws->halpern_factors, saved_halpern_factors,
        sizeof(saved_halpern_factors), cudaMemcpyHostToDevice, ws->stream));
    for (int i = 0; i < 4; ++i) {
        ws->iter_params_host[i] = saved_iter_params[i];
    }
    ws->halpern_inner_host[0] = saved_halpern_inner;
    for (int i = 0; i < 2; ++i) {
        ws->halpern_factors_host[i] = saved_halpern_factors[i];
    }
    ws->uploaded_sigma = saved_uploaded_sigma;
    ws->uploaded_lambda_max = saved_uploaded_lambda_max;
    ws->x_backend = saved_x_backend;
    ws->y_backend = saved_y_backend;
    ws->signed_single_state_enabled = saved_signed_single_state;
    ws->signed_zero_skip_monitor_enabled = saved_zero_skip_monitor;
    ws->signed_zero_skip_enabled = saved_zero_skip;
    refresh_paired_scaled_cache(ws);
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    free_saved_device_state(
        x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save,
        y_obj_save, z_bar_save, Ax_save, ATy_save);
    ws->reduced_backend_autotune_done = true;
}

void autotune_custom_update_backends(HPRLP_workspace_gpu *ws, LP_info_gpu *lp, Scaling_info *scaling, const HPRLP_parameters *param) {
    // Canonical autotune probes always materialize x_hat.  Single-state mode
    // and never emit or consume zero flags.  These modes are activated only
    // after the backend pair has been selected.
    ws->signed_single_state_enabled = false;
    ws->signed_zero_skip_monitor_enabled = false;
    ws->signed_zero_skip_enabled = false;
    ws->reduced_use_fused_x = false;
    ws->reduced_use_fused_y = false;
    if (param->CUSPARSE_spmv) {
        ws->x_backend = HPRLPXBackend::ScaledCusparse;
        ws->y_backend = HPRLPYBackend::ScaledCusparse;
        return;
    }

    std::vector<BackendCandidate> candidates =
        collect_eligible_backend_pairs(ws);

    const int bench_iters = hprlp_autotune_probe_iterations(
        param->max_iter, param->check_iter);
    const int warmup_iters =
        hprlp_autotune_warmup_iterations(bench_iters);

    if (param->autotune_verbose) {
        print_backend_autotune_diagnostics(ws, bench_iters, warmup_iters);
    }

    HPRLP_FLOAT *x_save = nullptr, *x_hat_save = nullptr, *x_bar_save = nullptr;
    HPRLP_FLOAT *y_save = nullptr, *y_hat_save = nullptr, *y_bar_save = nullptr, *y_obj_save = nullptr;
    HPRLP_FLOAT *z_bar_save = nullptr, *Ax_save = nullptr, *ATy_save = nullptr;
    save_device_state(ws, &x_save, &x_hat_save, &x_bar_save, &y_save, &y_hat_save, &y_bar_save, &y_obj_save, &z_bar_save, &Ax_save, &ATy_save);

    std::vector<std::pair<BackendCandidate, std::pair<HPRLP_FLOAT, long long>>> results;
    results.reserve(candidates.size());
    auto run_updates = [&](int iterations) {
        for (int i = 0; i < iterations; ++i) {
            update_zx_normal_gpu(ws);
            update_y_normal_gpu(ws);
            advance_halpern_factors(ws);
        }
    };

    auto evaluate_probe = [&]() {
        update_zx_check_gpu(ws);
        update_y_check_gpu(ws);
        advance_halpern_factors(ws);
        HPRLP_residuals residual{};
        compute_residuals(ws, lp, scaling, &residual, bench_iters);
        CUDA_CHECK(cudaStreamSynchronize(ws->stream));
        return residual.KKTx_and_gap_org_bar;
    };

    for (const auto &candidate : candidates) {
        restore_device_state(ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
        reset_halpern_runtime_params(ws);
        ws->x_backend = candidate.x_backend;
        ws->y_backend = candidate.y_backend;
        // Device-state restore does not include derived caches.
        refresh_paired_scaled_cache(ws);
        run_updates(warmup_iters);
        CUDA_CHECK(cudaStreamSynchronize(ws->stream));

        restore_device_state(ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
        reset_halpern_runtime_params(ws);
        // Refresh after the timed-state restore as well as the warmup restore.
        refresh_paired_scaled_cache(ws);
        CUDA_CHECK(cudaStreamSynchronize(ws->stream));
        auto start = std::chrono::steady_clock::now();
        run_updates(bench_iters);
        CUDA_CHECK(cudaStreamSynchronize(ws->stream));
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        HPRLP_FLOAT merit = evaluate_probe();
        results.push_back({candidate, {merit, elapsed}});

        if (param->autotune_verbose) {
            const std::streamsize saved_precision = std::cout.precision();
            std::cout << "  candidate x="
                      << x_backend_name(candidate.x_backend)
                      << ", y=" << y_backend_name(candidate.y_backend)
                      << " -> " << (static_cast<double>(elapsed) / 1e6)
                      << " ms (" << elapsed << " ns), merit="
                      << std::setprecision(17) << merit << std::endl;
            std::cout.precision(saved_precision);
        }
    }

    BackendCandidate selected =
        choose_backend_with_fixed_degree_gate(results);

    restore_device_state(ws, x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
    reset_halpern_runtime_params(ws);
    ws->x_backend = selected.x_backend;
    ws->y_backend = selected.y_backend;
    // Seed the selected pair before the first captured normal iteration.
    refresh_paired_scaled_cache(ws);
    // Reduced workspace construction gathers canonical x_hat.  Signed
    // single-state mode deliberately omits that vector, so retain the signed
    // specialized kernels but make them materialize x_hat while reduced mode
    // is available.
    ws->signed_single_state_enabled = !param->use_reduced_matrix &&
        is_signed_x_backend(selected.x_backend) &&
        is_signed_y_backend(selected.y_backend);
    ws->signed_zero_skip_monitor_enabled =
        ws->signed_single_state_enabled &&
        ws->unit_scaled_x_hat_nonzero != nullptr;
    ws->signed_zero_skip_enabled = false;

    if (param->autotune_verbose) {
        std::cout << "AUTO-SELECT selected x="
                  << x_backend_name(selected.x_backend)
                  << ", y=" << y_backend_name(selected.y_backend) << std::endl;
        std::cout << "  selected signed single-state x_hat omission: "
                  << (ws->signed_single_state_enabled ? "enabled" : "disabled")
                  << std::endl;
        std::cout << "  selected signed positive-zero monitor: "
                  << (ws->signed_zero_skip_monitor_enabled
                          ? "enabled"
                          : "disabled")
                  << std::endl;
    }

    free_saved_device_state(x_save, x_hat_save, x_bar_save, y_save, y_hat_save, y_bar_save, y_obj_save, z_bar_save, Ax_save, ATy_save);
}
