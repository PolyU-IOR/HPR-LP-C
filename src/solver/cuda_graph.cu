#include "HPRLP.h"
#include "solver/graph/graph_batch_policy.h"
#include "solver/internal/solver_cuda_graph.h"
#include "gpu/preprocessing/policies/row_bucket_policy.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

/** Rebuild the normal-iteration CUDA graphs for the selected backends. */
void rebuild_cuda_graph(HPRLP_workspace_gpu *ws) {
    if (ws->graph_exec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(ws->graph_exec));
        ws->graph_exec = nullptr;
    }
    if (ws->graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(ws->graph));
        ws->graph = nullptr;
    }
    if (ws->graph_exec_batch != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(ws->graph_exec_batch));
        ws->graph_exec_batch = nullptr;
    }
    if (ws->graph_batch != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(ws->graph_batch));
        ws->graph_batch = nullptr;
    }
    CUDA_CHECK(cudaStreamBeginCapture(ws->stream, cudaStreamCaptureModeGlobal));

    update_zx_normal_gpu(ws);
    update_y_normal_gpu(ws);
    advance_halpern_factors(ws);

    CUDA_CHECK(cudaStreamEndCapture(ws->stream, &ws->graph));

    CUDA_CHECK(cudaGraphInstantiate(&ws->graph_exec, ws->graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaStreamBeginCapture(
        ws->stream, cudaStreamCaptureModeGlobal));
    HPRLP_FLOAT *const canonical_halpern_factors =
        ws->halpern_factors;
    prepare_halpern_factor_batch_kernel<<<1, 1, 0, ws->stream>>>(
        ws->halpern_inner, canonical_halpern_factors,
        ws->halpern_factor_batch, HPRLP_NORMAL_GRAPH_BATCH_SIZE);
    for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
        ws->halpern_factors = ws->halpern_factor_batch + 2 * i;
        update_zx_normal_gpu(ws);
        update_y_normal_gpu(ws);
    }
    ws->halpern_factors = canonical_halpern_factors;
    CUDA_CHECK(cudaStreamEndCapture(
        ws->stream, &ws->graph_batch));
    CUDA_CHECK(cudaGraphInstantiate(
        &ws->graph_exec_batch, ws->graph_batch, nullptr, nullptr, 0));
    ws->graph_initialized = true;
}

void launch_check_cuda_graph(HPRLP_workspace_gpu *ws) {
    if (!ws->check_graph_initialized) {
        CUDA_CHECK(cudaStreamBeginCapture(
            ws->stream, cudaStreamCaptureModeGlobal));
        update_zx_check_gpu(ws);
        update_y_check_gpu(ws);
        advance_halpern_factors(ws);
        CUDA_CHECK(cudaStreamEndCapture(ws->stream, &ws->check_graph));
        CUDA_CHECK(cudaGraphInstantiate(
            &ws->check_graph_exec, ws->check_graph, nullptr, nullptr, 0));
        ws->check_graph_initialized = true;
    }
    CUDA_CHECK(cudaGraphLaunch(ws->check_graph_exec, ws->stream));
}

void update_unit_coltile_backend_from_zero_density(
    HPRLP_workspace_gpu *ws, const HPRLP_parameters *param,
    int completed_iteration) {
    if (param->check_iter <= 0 ||
        completed_iteration % param->check_iter != 0 ||
        (ws->y_backend != HPRLPYBackend::UnitColTile &&
         ws->y_backend != HPRLPYBackend::UnitColTileZeroBitset &&
         ws->y_backend != HPRLPYBackend::UnitActiveScatter) ||
        ws->unit_scaled_x_zero_bits == nullptr ||
        ws->unit_scaled_x_positive_zero_count == nullptr) {
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(
        ws->unit_scaled_x_positive_zero_count, 0,
        sizeof(unsigned long long), ws->stream));
    pack_positive_zero_bitset_count_kernel<<<
        numBlocks(ws->n), numThreads, 0, ws->stream>>>(
        ws->x_hat, ws->unit_scaled_x_zero_bits,
        ws->unit_scaled_x_positive_zero_count, ws->n);
    unsigned long long positive_zero_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &positive_zero_count, ws->unit_scaled_x_positive_zero_count,
        sizeof(positive_zero_count), cudaMemcpyDeviceToHost,
        ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    const double zero_rate = static_cast<double>(positive_zero_count) /
                             static_cast<double>(ws->n);
    constexpr double kEnableZeroBitsetRate = 0.50;
    constexpr double kDisableZeroBitsetRate = 0.35;
    constexpr double kEnableActiveScatterRate = 0.95;
    constexpr double kDisableActiveScatterRate = 0.90;
    const bool active_scatter_ready =
        ws->x_backend == HPRLPXBackend::UnitFactorized &&
        ws->unit_operator_x_ready && ws->unit_operator_y_ready &&
        ws->unit_AT_col_index_u16 != nullptr &&
        hprlp_all_rows_fit_scalar(
            ws->n, ws->max_AT_row_nnz,
            HPRLP_UNIT_SCALAR_ROW_MAX_NNZ);
    const HPRLPYBackend previous_backend = ws->y_backend;
    const bool was_active_scatter =
        previous_backend == HPRLPYBackend::UnitActiveScatter;
    const bool was_bitset =
        previous_backend == HPRLPYBackend::UnitColTileZeroBitset;
    bool use_bitset = was_bitset;
    bool use_active_scatter = was_active_scatter;
    if (was_active_scatter && zero_rate < kDisableActiveScatterRate) {
        use_active_scatter = false;
        use_bitset = zero_rate >= kEnableZeroBitsetRate;
    } else if (!was_active_scatter && active_scatter_ready &&
               zero_rate >= kEnableActiveScatterRate) {
        use_active_scatter = true;
        use_bitset = false;
    } else if (!was_active_scatter && !was_bitset &&
               zero_rate >= kEnableZeroBitsetRate) {
        use_bitset = true;
    } else if (was_bitset && zero_rate < kDisableZeroBitsetRate) {
        use_bitset = false;
    }
    const HPRLPYBackend selected_backend = use_active_scatter
        ? HPRLPYBackend::UnitActiveScatter
        : (use_bitset ? HPRLPYBackend::UnitColTileZeroBitset
                      : HPRLPYBackend::UnitColTile);
    const auto backend_label = [](HPRLPYBackend backend) {
        switch (backend) {
            case HPRLPYBackend::UnitActiveScatter:
                return "active-scatter";
            case HPRLPYBackend::UnitColTileZeroBitset:
                return "zero-bitset";
            default:
                return "ordinary";
        }
    };

    if (param->autotune_verbose) {
        const std::streamsize saved_precision = std::cout.precision();
        std::cout << "UNIT_COLTILE_ZERO_DENSITY iter="
                  << completed_iteration
                  << " positive_zero=" << positive_zero_count
                  << " count=" << ws->n
                  << " rate=" << std::setprecision(8) << zero_rate
                  << " backend="
                  << backend_label(selected_backend)
                  << std::endl;
        std::cout.precision(saved_precision);
    }
    if (selected_backend == previous_backend) {
        return;
    }

    ws->y_backend = selected_backend;
    rebuild_cuda_graph(ws);
    if (param->autotune_verbose) {
        std::cout << "UNIT_COLTILE_DYNAMIC_SWITCH iter="
                  << completed_iteration << " from="
                  << backend_label(previous_backend)
                  << " to="
                  << backend_label(selected_backend)
                  << std::endl;
    }
}
