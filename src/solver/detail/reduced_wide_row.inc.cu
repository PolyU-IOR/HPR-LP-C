__global__ void gather_wide_reduced_row_bound_type_kernel(
    std::uint8_t *compact, const std::uint8_t *full,
    const int *active_to_original, int count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < count) compact[row] = full[active_to_original[row]];
}

// Row reduction for a wide source. The selected matrix chooses its own CSR
// offset width. Packed/delta backends remain int32-only, while the direct
// fused update consumes either width; mask changes rebuild this CSR.
bool build_reduced_row_workspace_from_wide_source(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters, int iteration,
    std::chrono::steady_clock::time_point started) {
    build_selected_csr_from_wide_source(
        workspace, state, workspace->A,
        state->row_active_to_original, state->row_active_count,
        &state->row_A, &state->row_AT);

    state->use_compact_row_y = true;
    state->row_use_fused_x = parameters != nullptr &&
        !parameters->CUSPARSE_spmv;
    state->row_use_fused_y = state->row_use_fused_x;
    state->row_use_signed_x = false;
    state->row_use_signed_y = false;
    state->row_use_packed_dictionary_x = false;
    state->row_use_packed_dictionary_y = false;
    state->row_delta_count = 0;
    state->row_backend_profile_done = true;

    allocate_device(&state->row_Ax, state->row_active_count);
    allocate_device(&state->row_y, state->row_active_count);
    allocate_device(&state->row_last_y, state->row_active_count);
    allocate_device(&state->row_lower, state->row_active_count);
    allocate_device(&state->row_upper, state->row_active_count);
    if (state->row_active_count > 0) {
        gather_reduced_row_state_kernel<<<
            HPRLP_NUM_BLOCKS(state->row_active_count),
            HPRLP_NUM_THREADS, 0, workspace->stream>>>(
            state->row_y, state->row_last_y, state->row_lower,
            state->row_upper, workspace->y, workspace->last_y,
            workspace->AL, workspace->AU,
            state->row_active_to_original, state->row_active_count);
        CUDA_CHECK(cudaGetLastError());
        if (state->row_use_fused_y) {
            allocate_device(&state->row_bound_type,
                            state->row_active_count);
            gather_wide_reduced_row_bound_type_kernel<<<
                HPRLP_NUM_BLOCKS(state->row_active_count),
                HPRLP_NUM_THREADS, 0, workspace->stream>>>(
                state->row_bound_type, workspace->y_bound_type,
                state->row_active_to_original, state->row_active_count);
        }
    }
    zero_inactive_y_state_kernel<<<
        HPRLP_NUM_BLOCKS(workspace->m), HPRLP_NUM_THREADS,
        0, workspace->stream>>>(
        workspace->y, workspace->last_y,
        state->y_bar_mask, workspace->m);
    CUDA_CHECK(cudaGetLastError());
    prepare_reduced_row_spmv(workspace, state, false);
    CUDA_CHECK(cudaMemcpyAsync(
        state->row_base_mask, state->y_bar_mask,
        static_cast<std::size_t>(workspace->m) * sizeof(std::uint8_t),
        cudaMemcpyDeviceToDevice, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->row_built = true;
    state->row_rebuilds += 1;
    state->row_last_rebuild_iteration = iteration;
    state->row_build_time += std::chrono::duration<HPRLP_FLOAT>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "  row-reduced adaptive CSR: source_nnz="
              << workspace->A->numElements
              << " reduced_nnz=" << state->row_A.numElements
              << " offset_bits="
              << (state->row_A.rowPtr64 ? 64 : 32)
              << " backend="
              << (state->row_use_fused_x ? "wide-fused" : "cusparse")
              << std::endl;
    return true;
}
