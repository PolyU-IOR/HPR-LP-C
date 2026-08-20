#include "HPRLP.h"
#include "solver/graph/graph_batch_policy.h"
#include "solver/internal/solver_cuda_graph.h"
#include "solver/internal/solver_output.h"
#include "solver/progress_monitor.cuh"
#include "solver/restart_control.h"
#include "solver/reduced_matrix.cuh"
#include "gpu/memory/compressible_memory.h"
#include "presolve/pslp_integration.h"
#ifdef HPRLP_HAS_GPU_PRESOLVER
#include "presolve/gpu_presolver_integration.h"
#endif

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

static void update_phase2_control(
    HPRLP_control_state *control,
    const HPRLP_progress_metrics &progress,
    const HPRLP_residuals &residuals,
    int iter,
    const HPRLP_parameters *param);

static void apply_periodic_control(
    HPRLP_control_state *control,
    const HPRLP_progress_metrics &progress,
    HPRLP_restart *restart,
    HPRLP_workspace_gpu *workspace,
    const HPRLP_residuals &residuals,
    int iter,
    const HPRLP_parameters *param,
    const std::string &status);

struct HPRLP_experiment_stage_timing {
    long long calls = 0;
    double seconds = 0.0;
};

struct HPRLP_experiment_phase_timing {
    HPRLP_experiment_stage_timing flush;
    HPRLP_experiment_stage_timing residual;
    HPRLP_experiment_stage_timing mask;
    HPRLP_experiment_stage_timing progress;
    HPRLP_experiment_stage_timing mode_maintenance;
    HPRLP_experiment_stage_timing full_check_update;
};

static bool hprlp_environment_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) return false;
    return std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0;
}
constexpr int HPRLP_AUTO_REDUCED_MIN_COLUMNS = 2000000;
constexpr int HPRLP_AUTO_REDUCED_COLUMN_ROW_RATIO = 2;

class HPRLP_compressible_memory_mode_guard {
public:
    explicit HPRLP_compressible_memory_mode_guard(
            HPRLP_compressible_memory_mode mode)
        : previous_(hprlp_get_compressible_memory_mode()) {
        hprlp_set_compressible_memory_mode(mode);
    }

    ~HPRLP_compressible_memory_mode_guard() {
        hprlp_set_compressible_memory_mode(previous_);
    }

    HPRLP_compressible_memory_mode_guard(
        const HPRLP_compressible_memory_mode_guard &) = delete;
    HPRLP_compressible_memory_mode_guard &operator=(
        const HPRLP_compressible_memory_mode_guard &) = delete;

private:
    HPRLP_compressible_memory_mode previous_;
};

static bool hprlp_auto_policy_uses_reduced_matrix(
        const LP_info_cpu *model) {
    if (model == nullptr) return false;
    const long long rows = static_cast<long long>(model->m);
    const long long columns = static_cast<long long>(model->n);
    return columns > HPRLP_AUTO_REDUCED_MIN_COLUMNS &&
        columns > HPRLP_AUTO_REDUCED_COLUMN_ROW_RATIO * rows;
}

static HPRLP_parameters hprlp_effective_parameters(
        const HPRLP_parameters *parameters,
        const LP_info_cpu *solve_model) {
    HPRLP_parameters effective =
        parameters != nullptr ? *parameters : HPRLP_parameters{};
    if (effective.auto_reduced_compression_policy) {
        effective.use_reduced_matrix =
            hprlp_auto_policy_uses_reduced_matrix(solve_model);
    }
    return effective;
}

static HPRLP_compressible_memory_mode hprlp_compression_mode(
        const HPRLP_parameters &parameters) {
    if (!parameters.auto_reduced_compression_policy) {
        return HPRLP_COMPRESSIBLE_MEMORY_FROM_ENVIRONMENT;
    }
    return parameters.use_reduced_matrix
        ? HPRLP_COMPRESSIBLE_MEMORY_DISABLED
        : HPRLP_COMPRESSIBLE_MEMORY_ENABLED;
}

static void hprlp_print_automatic_memory_policy(
        const LP_info_cpu *model,
        const HPRLP_parameters &parameters) {
    if (!parameters.auto_reduced_compression_policy || model == nullptr) {
        return;
    }
    std::cout << "Automatic reduced/compression policy: nRows="
              << model->m << " nCols=" << model->n
              << " rule=(nCols > " << HPRLP_AUTO_REDUCED_MIN_COLUMNS
              << " && nCols > "
              << HPRLP_AUTO_REDUCED_COLUMN_ROW_RATIO << " * nRows)"
              << " reduced_matrix="
              << (parameters.use_reduced_matrix ? "enabled" : "disabled")
              << " compressible_memory="
              << (parameters.use_reduced_matrix ? "disabled" : "enabled")
              << std::endl;
}


static void print_experiment_phase_timing(
    const char *name,
    const HPRLP_experiment_phase_timing &timing) {
    const auto print_stage = [](const char *stage,
                                const HPRLP_experiment_stage_timing &value) {
        const double total_ms = 1000.0 * value.seconds;
        const double mean_ms = value.calls > 0
            ? total_ms / static_cast<double>(value.calls) : 0.0;
        std::cout << "    " << stage << ": calls=" << value.calls
                  << " total_ms=" << std::fixed << std::setprecision(3)
                  << total_ms << " mean_ms=" << mean_ms << "\n";
    };
    std::cout << "  " << name << ":\n";
    print_stage("flush", timing.flush);
    print_stage("residual", timing.residual);
    print_stage("mask", timing.mask);
    print_stage("progress", timing.progress);
    print_stage("mode_maintenance", timing.mode_maintenance);
    print_stage("full_check_update", timing.full_check_update);
}

/** Select the CUDA device used by this solve. */
static int initialize_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "[error] Failed to set CUDA device " << device_id << ": "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Note: cudaDeviceReset() removed to avoid segfaults during Python exit
    // The CUDA runtime will clean up automatically when the process exits

    return 0;
}

static HPRLP_results make_error_result(const char *status) {
    HPRLP_results result;
    std::strncpy(result.status, status, sizeof(result.status) - 1);
    result.status[sizeof(result.status) - 1] = '\0';
    result.iter = 0;
    result.time = 0.0;
    result.primal_obj = 0.0;
    result.residuals = 0.0;
    result.gap = 0.0;
    result.x = nullptr;
    result.y = nullptr;
    result.z = nullptr;
    return result;
}
struct HPRLP_device_finite_abs_range {
    unsigned long long minimum_bits;
    unsigned long long maximum_bits;
    unsigned int flags;
};

constexpr unsigned int HPRLP_RANGE_HAS_VALUE = 1u;
constexpr unsigned int HPRLP_RANGE_HAS_NEGATIVE_INFINITY = 2u;
constexpr unsigned int HPRLP_RANGE_HAS_POSITIVE_INFINITY = 4u;
constexpr int HPRLP_RANGE_THREADS = 256;
constexpr int HPRLP_RANGE_MAX_BLOCKS = 65535;

__global__ void hprlp_finite_abs_range_kernel(
    const HPRLP_FLOAT *values,
    std::size_t count,
    HPRLP_device_finite_abs_range *result) {
    __shared__ HPRLP_FLOAT minimums[HPRLP_RANGE_THREADS];
    __shared__ HPRLP_FLOAT maximums[HPRLP_RANGE_THREADS];
    __shared__ unsigned int flags[HPRLP_RANGE_THREADS];

    HPRLP_FLOAT local_minimum = DBL_MAX;
    HPRLP_FLOAT local_maximum = 0.0;
    unsigned int local_flags = 0u;
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = first; index < count; index += stride) {
        const HPRLP_FLOAT value = values[index];
        if (isinf(value)) {
            local_flags |= signbit(value)
                ? HPRLP_RANGE_HAS_NEGATIVE_INFINITY
                : HPRLP_RANGE_HAS_POSITIVE_INFINITY;
        } else if (isfinite(value)) {
            const HPRLP_FLOAT magnitude = fabs(value);
            local_minimum = fmin(local_minimum, magnitude);
            local_maximum = fmax(local_maximum, magnitude);
            local_flags |= HPRLP_RANGE_HAS_VALUE;
        }
    }

    const int thread = threadIdx.x;
    minimums[thread] = local_minimum;
    maximums[thread] = local_maximum;
    flags[thread] = local_flags;
    __syncthreads();
    for (int offset = HPRLP_RANGE_THREADS / 2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            minimums[thread] = fmin(minimums[thread], minimums[thread + offset]);
            maximums[thread] = fmax(maximums[thread], maximums[thread + offset]);
            flags[thread] |= flags[thread + offset];
        }
        __syncthreads();
    }
    if (thread == 0) {
        if ((flags[0] & HPRLP_RANGE_HAS_VALUE) != 0u) {
            atomicMin(
                &result->minimum_bits,
                static_cast<unsigned long long>(
                    __double_as_longlong(minimums[0])));
            atomicMax(
                &result->maximum_bits,
                static_cast<unsigned long long>(
                    __double_as_longlong(maximums[0])));
        }
        atomicOr(&result->flags, flags[0]);
    }
}

static unsigned long long hprlp_range_double_bits(HPRLP_FLOAT value) {
    unsigned long long bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected floating type");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static HPRLP_FLOAT hprlp_range_double_from_bits(unsigned long long bits) {
    HPRLP_FLOAT value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void hprlp_launch_finite_abs_range(
    const HPRLP_FLOAT *values,
    std::size_t count,
    HPRLP_device_finite_abs_range *result,
    cudaStream_t stream) {
    if (values == nullptr || count == 0) return;
    const std::size_t required_blocks =
        (count + HPRLP_RANGE_THREADS - 1) / HPRLP_RANGE_THREADS;
    const int blocks = static_cast<int>(
        required_blocks < HPRLP_RANGE_MAX_BLOCKS
            ? required_blocks : HPRLP_RANGE_MAX_BLOCKS);
    hprlp_finite_abs_range_kernel<<<
        blocks, HPRLP_RANGE_THREADS, 0, stream>>>(values, count, result);
    CUDA_CHECK(cudaGetLastError());
}

static HPRLP_finite_abs_range hprlp_host_finite_abs_range(
    const HPRLP_device_finite_abs_range &device_range) {
    HPRLP_finite_abs_range range;
    range.has_value =
        (device_range.flags & HPRLP_RANGE_HAS_VALUE) != 0u;
    range.has_negative_infinity =
        (device_range.flags & HPRLP_RANGE_HAS_NEGATIVE_INFINITY) != 0u;
    range.has_positive_infinity =
        (device_range.flags & HPRLP_RANGE_HAS_POSITIVE_INFINITY) != 0u;
    if (range.has_value) {
        range.minimum = hprlp_range_double_from_bits(
            device_range.minimum_bits);
        range.maximum = hprlp_range_double_from_bits(
            device_range.maximum_bits);
    }
    return range;
}

static void print_device_numerical_ranges(
    const LP_info_gpu *model,
    cudaStream_t stream,
    const char *stage) {
    if (model == nullptr || model->A == nullptr) {
        HPRLP_numerical_ranges unavailable;
        print_numerical_ranges(unavailable, stage);
        return;
    }

    HPRLP_device_finite_abs_range initial_ranges[6];
    for (int index = 0; index < 6; ++index) {
        initial_ranges[index].minimum_bits =
            hprlp_range_double_bits(DBL_MAX);
        initial_ranges[index].maximum_bits =
            hprlp_range_double_bits(0.0);
        initial_ranges[index].flags = 0u;
    }
    HPRLP_device_finite_abs_range *device_ranges = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_ranges), sizeof(initial_ranges)));
    CUDA_CHECK(cudaMemcpyAsync(
        device_ranges, initial_ranges, sizeof(initial_ranges),
        cudaMemcpyHostToDevice, stream));

    hprlp_launch_finite_abs_range(
        model->A->value, static_cast<std::size_t>(model->A->numElements),
        device_ranges + 0, stream);
    hprlp_launch_finite_abs_range(
        model->AL, static_cast<std::size_t>(model->m),
        device_ranges + 1, stream);
    hprlp_launch_finite_abs_range(
        model->AU, static_cast<std::size_t>(model->m),
        device_ranges + 2, stream);
    hprlp_launch_finite_abs_range(
        model->l, static_cast<std::size_t>(model->n),
        device_ranges + 3, stream);
    hprlp_launch_finite_abs_range(
        model->u, static_cast<std::size_t>(model->n),
        device_ranges + 4, stream);
    hprlp_launch_finite_abs_range(
        model->c, static_cast<std::size_t>(model->n),
        device_ranges + 5, stream);

    HPRLP_device_finite_abs_range host_ranges[6];
    CUDA_CHECK(cudaMemcpyAsync(
        host_ranges, device_ranges, sizeof(host_ranges),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(device_ranges);

    HPRLP_numerical_ranges ranges;
    ranges.A = hprlp_host_finite_abs_range(host_ranges[0]);
    ranges.AL = hprlp_host_finite_abs_range(host_ranges[1]);
    ranges.AU = hprlp_host_finite_abs_range(host_ranges[2]);
    ranges.l = hprlp_host_finite_abs_range(host_ranges[3]);
    ranges.u = hprlp_host_finite_abs_range(host_ranges[4]);
    ranges.c = hprlp_host_finite_abs_range(host_ranges[5]);
    print_numerical_ranges(ranges, stage);
}

static void compute_maximum_eigenvalue(HPRLP_workspace_gpu *ws) {
    double lambda_max = power_method_cusparse(ws, 5000, 1e-4) * 1.01;
    ws->lambda_max = lambda_max;
}
static HPRLP_results HPRLP_main_solve_impl(
        const LP_info_cpu *lp_info_cpu,
        LP_info_gpu *preloaded_lp_info_gpu,
        const HPRLP_parameters *param,
        std::chrono::steady_clock::time_point first_device_ready,
        HPRLP_FLOAT presolve_time,
        HPRLP_FLOAT folding_time) {

    // Exactly one model representation must be supplied.  The ordinary path
    // uploads an LP_info_cpu here; the GPU-presolver path transfers ownership
    // of an already-device-resident reduced model.
    if ((lp_info_cpu == nullptr) == (preloaded_lp_info_gpu == nullptr)) {
        std::cerr << "[error] Internal model handoff must provide exactly one "
                  << "host or device model" << std::endl;
        return make_error_result("ERROR");
    }

    // Print startup banner
    print_solver_banner();

    // Print parameters
    print_solver_parameters(param);

    // Device selection is outside the setup phase.
    initialize_device(param->device_number);

    HPRLP_time phase_times;
    phase_times.presolve_time = presolve_time;

    // Setup is bounded to model transfer and workspace allocation.
    const auto t_start_setup = time_now();
    LP_info_gpu uploaded_lp_info_gpu{};
    LP_info_gpu &lp_info_gpu = preloaded_lp_info_gpu != nullptr
        ? *preloaded_lp_info_gpu : uploaded_lp_info_gpu;
    if (preloaded_lp_info_gpu == nullptr) {
        copy_lpinfo_to_device(lp_info_cpu, &lp_info_gpu);
    }
    // Start total_time after the first successful CPU-to-GPU model transfer.
    // GPU presolve supplies an earlier boundary; ordinary and CPU-presolved
    // paths establish it here after their only model upload.
    if (first_device_ready == std::chrono::steady_clock::time_point{}) {
        first_device_ready = time_now();
    }
    const HPRLP_FLOAT total_time_offset =
        param->use_presolve && param->presolver == HPRLP_PRESOLVER_PSLP
        ? presolve_time : 0.0;
    auto reported_total_time = [&]() {
        return total_time_offset + time_since(first_device_ready);
    };

    HPRLP_workspace_gpu workspace;
    workspace.m = lp_info_gpu.m;
    workspace.n = lp_info_gpu.n;
    allocate_memory(&workspace, &lp_info_gpu);
    cudaDeviceSynchronize();
    phase_times.setup_time = time_since(t_start_setup);

    const auto t_start_scaling = time_now();
    Scaling_info scaling_info;
    scaling(&lp_info_gpu, &scaling_info, param, workspace.cublasHandle);
    cudaDeviceSynchronize();
    phase_times.scaling_time = time_since(t_start_scaling);
    print_device_numerical_ranges(
        &lp_info_gpu, workspace.stream, "after scaling");

    // solve_time is the enclosing interval for analysis, eigenvalue
    // estimation, initialization, and the main iteration loop.
    const auto t_start_solve = time_now();

    const auto t_start_spmv_analysis = time_now();
    analyze_spmv_pattern(&workspace, param);
    if (!param->CUSPARSE_spmv) {
        prepare_unit_operators(&workspace, &scaling_info);
    }
    cudaDeviceSynchronize();
    phase_times.analyze_time += time_since(t_start_spmv_analysis);

    const auto t_start_power_iteration = time_now();
    compute_maximum_eigenvalue(&workspace);
    cudaDeviceSynchronize();
    phase_times.power_iteration_time = time_since(t_start_power_iteration);

    // ### Initialization ###
    HPRLP_residuals residuals;
    if (scaling_info.norm_b > 1e-8 && scaling_info.norm_c > 1e-8) {
        workspace.sigma = scaling_info.norm_b / scaling_info.norm_c;
    } else {
        workspace.sigma = 1.0;
    }
    if (std::isfinite(param->fixed_sigma) && param->fixed_sigma > 0.0) {
        workspace.sigma = param->fixed_sigma;
    }
    HPRLP_restart restart_info = {};
    restart_info.best_sigma = workspace.sigma;
    restart_info.restart_flag = 0;
    restart_info.first_restart = true;
    HPRLP_control_state control_state =
        hprlp_initialize_control_state(param->check_iter);
    HPRLP_progress_monitor_gpu progress_monitor;
    const bool progress_monitor_enabled = param->enable_progress_monitor ||
        param->use_progress_restart_guard ||
        param->enable_sigma_rebalance_restart;

    reset_halpern_runtime_params(&workspace);

    const auto t_start_backend_analysis = time_now();
    autotune_custom_update_backends(
        &workspace, &lp_info_gpu, &scaling_info, param);
    cudaDeviceSynchronize();
    if (progress_monitor_enabled) {
        hprlp_initialize_progress_monitor(&progress_monitor, &workspace);
    }
    if (param->use_reduced_matrix) {
        hprlp_initialize_reduced_matrix_state(&workspace);
    }
    phase_times.analyze_time += time_since(t_start_backend_analysis);

    const bool compressible_memory_requested =
        hprlp_compressible_memory_requested();
    std::cout << "Compressible GPU memory: requested="
              << (compressible_memory_requested ? "yes" : "no")
              << " active_allocations="
              << hprlp_compressible_allocation_count()
              << " device_supported="
              << (hprlp_device_supports_compression() ? "yes" : "no")
              << " (automatic policy or manual HPRLP_ENABLE_COMPRESSIBLE_MEMORY=1)"
              << std::endl;

    std::cout << "Setup (copy and allocation) time = " << std::fixed
              << std::setprecision(2) << phase_times.setup_time
              << " seconds" << std::endl;
    std::cout << "Scaling time = " << std::fixed << std::setprecision(2)
              << phase_times.scaling_time << " seconds" << std::endl;
    std::cout << "Analyze time = " << std::fixed << std::setprecision(2)
              << phase_times.analyze_time << " seconds" << std::endl;
    std::cout << "ESTIMATING MAXIMUM EIGENVALUE time = " << std::fixed
              << std::setprecision(2) << phase_times.power_iteration_time
              << " seconds" << std::endl;

    HPRLP_results output;
    output.timing = phase_times;
    output.presolve_time = presolve_time;
    output.folding_time = folding_time;
    bool first_4 = true;
    bool first_6 = true;
    bool first_8 = true;
    const bool experiment_timing_enabled =
        param->use_reduced_matrix &&
        hprlp_environment_flag_enabled("HPRLP_PROFILE_REDUCED");
    HPRLP_experiment_phase_timing pre_reduced_timing;
    HPRLP_experiment_phase_timing reduced_timing;
    int last_experiment_timing_print = -1;
    auto print_experiment_timing = [&](int iteration) {
        if (!experiment_timing_enabled ||
            iteration == last_experiment_timing_print) return;
        last_experiment_timing_print = iteration;
        std::cout << "Reduced experimental timing through iteration "
                  << iteration << ":\n";
        print_experiment_phase_timing("pre-reduced", pre_reduced_timing);
        print_experiment_phase_timing("reduced-active", reduced_timing);
        hprlp_print_reduced_matrix_profile(&workspace);
        std::cout << std::flush;
    };
    auto experiment_stage_start = [&]() {
        if (experiment_timing_enabled) {
            CUDA_CHECK(cudaStreamSynchronize(workspace.stream));
        }
        return time_now();
    };
    auto experiment_stage_finish = [&](
        HPRLP_experiment_stage_timing &stage,
        std::chrono::steady_clock::time_point started) {
        if (!experiment_timing_enabled) return;
        CUDA_CHECK(cudaStreamSynchronize(workspace.stream));
        stage.calls += 1;
        stage.seconds += time_since(started);
    };

    std::cout << " iter     errRp        errRd         p_obj            d_obj          gap         sigma       time\n" << std::flush;

    for (int iter = 0 ; iter < param->max_iter ; iter ++) {

        const bool reduced_active_at_loop_start =
            param->use_reduced_matrix &&
            hprlp_reduced_matrix_active(&workspace);
        HPRLP_experiment_phase_timing &loop_timing =
            reduced_active_at_loop_start
            ? reduced_timing : pre_reduced_timing;

        bool periodic_check = (iter % param->check_iter == 0);
        const bool progress_due = progress_monitor_enabled && iter > 0 &&
            iter % HPRLP_PROGRESS_MONITOR_INTERVAL == 0;
        bool print_flag = ((iter % step(iter) == 0) ||
                            (iter == param->max_iter) ||
                            time_since(t_start_solve) > param->time_limit);

        // Julia flushes compact state before every operation that observes
        // the canonical full vectors: residuals, progress samples, and logs.
        if (param->use_reduced_matrix &&
            (periodic_check || print_flag || progress_due)) {
            const auto stage_started = experiment_stage_start();
            hprlp_flush_reduced_matrix_state(&workspace);
            experiment_stage_finish(loop_timing.flush, stage_started);
        }
        if (progress_due) {
            const auto stage_started = experiment_stage_start();
            hprlp_queue_progress_sample(&progress_monitor, &workspace);
            experiment_stage_finish(loop_timing.progress, stage_started);
        }
        if (periodic_check) {
            monitor_signed_zero_skip_backend(&workspace);
        }
        // Batch the restart-gap computation with the residual fetch (iter > 0 only).
        bool compute_gap = (periodic_check && iter > 0);

        if (periodic_check || print_flag || progress_due) {
            const auto stage_started = experiment_stage_start();
            compute_residuals(&workspace, &lp_info_gpu, &scaling_info, &residuals, iter,
                              &restart_info, compute_gap);    // KKT residuals + optional gap
            experiment_stage_finish(loop_timing.residual, stage_started);
            residuals.is_updated = true;
            if (param->use_reduced_matrix) {
                const auto mask_started = experiment_stage_start();
                hprlp_refresh_reduced_matrix_mask(
                    &workspace, param, &residuals);
                experiment_stage_finish(loop_timing.mask, mask_started);
            }
        }
        else {
            residuals.is_updated = false;
        }

        if (progress_due) {
            const auto stage_started = experiment_stage_start();
            hprlp_finalize_progress_sample(&progress_monitor, &workspace);
            experiment_stage_finish(loop_timing.progress, stage_started);
        }
        if (periodic_check && progress_due) {
            update_phase2_control(&control_state, progress_monitor.metrics,
                                  residuals, iter, param);
        } else if (iter == 0 && std::isfinite(residuals.err_Rp_org_bar)) {
            control_state.phase2_primal_reference =
                std::max(std::abs(residuals.err_Rp_org_bar), 1e-12);
        }

        // check stopping criterion
        std::string status = check_stopping(&residuals, iter, t_start_solve, param);

        if (periodic_check) {
            const HPRLP_progress_metrics progress = progress_due
                ? progress_monitor.metrics : HPRLP_progress_metrics{};
            apply_periodic_control(&control_state, progress, &restart_info,
                                   &workspace, residuals, iter, param, status);
        } else {
            restart_info.restart_flag = 0;
        }

        if (print_flag || status != "CONTINUE") {
            std::cout << std::setw(5) << iter << "    "
                      << std::scientific << std::setprecision(2)
                      << residuals.err_Rp_org_bar << "    "
                      << residuals.err_Rd_org_bar << "    "
                      << std::scientific << std::setprecision(6) << std::showpos << residuals.primal_obj_bar << "    "
                      << residuals.dual_obj_bar << "    "
                      << std::scientific << std::setprecision(2) << std::noshowpos << residuals.rel_gap_bar << "    "
                      << workspace.sigma << "      "
                      << std::fixed << std::setprecision(2)
                      << reported_total_time() << "\n" << std::flush;
        }

        if (first_4 && residuals.KKTx_and_gap_org_bar < 1e-4) {
            output.iter4 = iter;
            output.time4 = time_since(t_start_solve);
            first_4 = false;
            std::cout << "Residual < 1e-4 at iter = " << iter << "\n" << std::flush;
        }
        if (first_6 && residuals.KKTx_and_gap_org_bar < 1e-6) {
            output.iter6 = iter;
            output.time6 = time_since(t_start_solve);
            first_6 = false;
            std::cout << "Residual < 1e-6 at iter = " << iter << "\n" << std::flush;
        }
        if (first_8 && residuals.KKTx_and_gap_org_bar < 1e-8) {
            output.iter8 = iter;
            output.time8 = time_since(t_start_solve);
            first_8 = false;
            std::cout << "Residual < 1e-8 at iter = " << iter << "\n" << std::flush;
        }

        if (status != "CONTINUE") {
            if (param->use_reduced_matrix) {
                // A ratio-ineligible mask may skip synchronized recounts.
                // Do one final recount so reported reduction statistics
                // retain their original end-of-solve meaning.
                hprlp_refresh_reduced_matrix_mask(
                    &workspace, param, &residuals, true);
            }
            print_experiment_timing(iter);
            strncpy(output.status, status.c_str(), sizeof(output.status) - 1);
            output.status[sizeof(output.status) - 1] = '\0';  // Ensure null termination
            output.iter = iter;
            output.gap = residuals.rel_gap_bar;
            output.residuals = residuals.KKTx_and_gap_org_bar;
            output.primal_obj = residuals.primal_obj_bar;
            cudaDeviceSynchronize();
            output.timing.solve_time = time_since(t_start_solve);
            output.time = output.timing.solve_time;

            output.time4 = (output.time4 == 0.0) ? output.time : output.time4;
            output.time6 = (output.time6 == 0.0) ? output.time : output.time6;
            output.time8 = (output.time8 == 0.0) ? output.time : output.time8;
            output.iter4 = (output.iter4 == 0) ? output.iter : output.iter4;
            output.iter6 = (output.iter6 == 0) ? output.iter : output.iter6;
            output.iter8 = (output.iter8 == 0) ? output.iter : output.iter8;

            // The GPU interval starts after the solver model upload. GPU
            // presolve is already inside that interval because its boundary
            // is recorded after the original upload and before presolve.
            // PSLP runs on the CPU before the reduced-model upload, so add its
            // measured wall time explicitly without charging the upload.
            output.timing.total_time = reported_total_time();

            if (param->use_reduced_matrix) {
                hprlp_collect_reduced_matrix_results(&workspace, &output);
            }

            // Collect solution vectors from GPU workspace
            collect_solution(&workspace, &scaling_info, &output);

            print_solution_summary(
                output, residuals.primal_obj_bar, residuals.dual_obj_bar,
                residuals.rel_gap_bar, residuals.err_Rp_org_bar,
                residuals.err_Rd_org_bar);

            // Clean up GPU resources to prevent memory leaks
            // Synchronize to ensure all CUDA operations complete before cleanup
            cudaDeviceSynchronize();

            if (progress_monitor_enabled) {
                hprlp_free_progress_monitor(&progress_monitor);
            }
            if (param->use_reduced_matrix) {
                hprlp_free_reduced_matrix_state(&workspace);
            }
            free_workspace(&workspace);
            free_scaling_info(&scaling_info);
            free_lp_info(&lp_info_gpu);

            // Final synchronization to ensure cleanup is complete
            cudaDeviceSynchronize();

            return output;
        }


        // update sigma
        update_sigma(&restart_info, &workspace, &residuals, param);

        do_restart(&workspace, &restart_info);
        if (param->use_reduced_matrix && restart_info.restart_flag > 0) {
            hprlp_maybe_reset_reduced_matrix_mask_on_restart(
                &workspace, param, &residuals, iter,
                restart_info.restart_flag);
        }
        if (restart_info.restart_flag == HPRLP_SIGMA_REBALANCE_RESTART_FLAG) {
            control_state.phase2_identified = false;
            control_state.phase2_confirmations = 0;
            control_state.phase2_age_checks = 0;
            control_state.phase2_entry_primal =
                std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
        }


        upload_halpern_iter_params_if_needed(&workspace);
        upload_halpern_restart_params(&workspace, &restart_info);

        workspace.check = ((iter + 1) % param->check_iter == 0 ||
                           restart_info.restart_flag > 0);
        workspace.check = workspace.check ||
            (iter + 1) % step(iter + 1) == 0;
        workspace.check = workspace.check || (progress_monitor_enabled &&
            (iter + 1) % HPRLP_PROGRESS_MONITOR_INTERVAL == 0);

        if (param->use_reduced_matrix) {
            if (residuals.is_updated) {
                const auto stage_started = experiment_stage_start();
                hprlp_update_reduced_matrix_mode(
                    &workspace, &lp_info_gpu, &scaling_info,
                    param, &residuals, iter,
                    true, workspace.check);
                experiment_stage_finish(
                    loop_timing.mode_maintenance, stage_started);
            } else {
                hprlp_update_reduced_matrix_mode(
                    &workspace, &lp_info_gpu, &scaling_info,
                    param, &residuals, iter,
                    false, workspace.check);
            }
        }
        const bool reduced_active = param->use_reduced_matrix &&
            hprlp_reduced_matrix_active(&workspace);

        int iterations_advanced = 1;
        const bool can_batch_updates =
            !workspace.check &&
            static_cast<long long>(iter) +
                    HPRLP_NORMAL_GRAPH_BATCH_SIZE <
                param->max_iter &&
            hprlp_can_batch_normal_updates(
                static_cast<long long>(iter),
                HPRLP_NORMAL_GRAPH_BATCH_SIZE,
                param->check_iter) &&
            (!progress_monitor_enabled || hprlp_can_batch_normal_updates(
                static_cast<long long>(iter),
                HPRLP_NORMAL_GRAPH_BATCH_SIZE,
                HPRLP_PROGRESS_MONITOR_INTERVAL)) &&
            time_since(t_start_solve) + 0.01 < param->time_limit;
        if (workspace.check) {
            const auto stage_started = experiment_stage_start();
            // A check iteration runs through the canonical full-state kernels.
            // Reduced iterations leave those vectors stale between observation
            // points, so flush before advancing the full state and gather the
            // resulting state back into compact storage below.
            if (reduced_active) {
                hprlp_flush_reduced_matrix_state(&workspace);
            }
            launch_check_cuda_graph(&workspace);
            if (reduced_active) {
                hprlp_sync_reduced_matrix_state_from_full(&workspace);
            }
            experiment_stage_finish(
                loop_timing.full_check_update, stage_started);
        } else if (reduced_active) {
            iterations_advanced = hprlp_launch_reduced_matrix_iterations(
                &workspace, iter, restart_info.restart_flag,
                can_batch_updates ? HPRLP_NORMAL_GRAPH_BATCH_SIZE : 1);
        } else {
            if (!workspace.graph_initialized) {
                rebuild_cuda_graph(&workspace);
            }
            if (workspace.graph_exec_batch != nullptr &&
                can_batch_updates) {
                CUDA_CHECK(cudaGraphLaunch(
                    workspace.graph_exec_batch, workspace.stream));
                iterations_advanced = HPRLP_NORMAL_GRAPH_BATCH_SIZE;
            } else {
                CUDA_CHECK(cudaGraphLaunch(
                    workspace.graph_exec, workspace.stream));
            }
        }

        update_unit_coltile_backend_from_zero_density(
            &workspace, param, iter + iterations_advanced);

        if (experiment_timing_enabled &&
            ((reduced_active && !reduced_active_at_loop_start) ||
             (iter > 0 && iter % 100000 == 0))) {
            print_experiment_timing(iter);
        }

        if (restart_info.restart_flag > 0) {
            restart_info.last_gap = compute_weighted_norm(&workspace);
        }

        restart_info.inner += iterations_advanced;
        iter += iterations_advanced - 1;
    }
}

HPRLP_results HPRLP_main_solve(const LP_info_cpu *lp_info_cpu,
                               const HPRLP_parameters *param) {
    const HPRLP_parameters effective =
        hprlp_effective_parameters(param, lp_info_cpu);
    HPRLP_scoped_output_filter output_filter(effective.print_debug_info);
    if (!effective.print_debug_info && lp_info_cpu && lp_info_cpu->A) {
        std::cout << "Original Model: nRow = " << lp_info_cpu->m
                  << ", nCol = " << lp_info_cpu->n
                  << ", nnz = " << lp_info_cpu->A->numElements << std::endl;
    }
    hprlp_print_automatic_memory_policy(lp_info_cpu, effective);
    HPRLP_compressible_memory_mode_guard compression_guard(
        hprlp_compression_mode(effective));
    return HPRLP_main_solve_impl(
        lp_info_cpu, nullptr, &effective, {}, 0.0, 0.0);
}

/* ============================================================================
 * Model-Based API Implementation
 * ============================================================================
 */

/**
 * Create an LP model from raw arrays
 */
HPRLP_results solve(const LP_info_cpu *model, const HPRLP_parameters *param) {
    // Validate input
    if (!model) {
        std::cerr << "[error] Null model pointer" << std::endl;
        return make_error_result("ERROR");
    }
    // Use default parameters if param is NULL
    HPRLP_parameters default_param;
    const HPRLP_parameters *actual_param = param ? param : &default_param;
    HPRLP_scoped_output_filter output_filter(actual_param->print_debug_info);
    if (!actual_param->print_debug_info && model->A) {
        std::cout << "Original Model: nRow = " << model->m
                  << ", nCol = " << model->n
                  << ", nnz = " << model->A->numElements << std::endl;
    }

    if (!actual_param->use_presolve ||
        actual_param->presolver == HPRLP_PRESOLVER_NONE) {
        const HPRLP_parameters effective =
            hprlp_effective_parameters(actual_param, model);
        hprlp_print_automatic_memory_policy(model, effective);
        HPRLP_compressible_memory_mode_guard compression_guard(
            hprlp_compression_mode(effective));
        return HPRLP_main_solve_impl(
            model, nullptr, &effective, {}, 0.0, 0.0);
    }

    print_numerical_ranges(model, "before presolve");

    LP_info_cpu reduced_model{};
    LP_info_gpu reduced_device_model{};
    void *presolver_handle = nullptr;
    const bool use_gpu_presolver =
        actual_param->presolver == HPRLP_PRESOLVER_GPU;
    HPRLP_FLOAT presolve_time = 0.0;
    HPRLP_FLOAT folding_time = 0.0;
    std::chrono::steady_clock::time_point first_device_ready{};
    bool presolve_ok = false;
    bool device_handoff_active = false;
    if (use_gpu_presolver) {
#ifdef HPRLP_HAS_GPU_PRESOLVER
        presolve_ok = run_embedded_gpu_presolve_device(
            model, actual_param, &reduced_device_model, &presolver_handle,
            &presolve_time, &folding_time, &first_device_ready);
        device_handoff_active = presolve_ok;
#else
        std::cerr << "[error] GPU-Presolver-C support was not compiled into this build"
                  << std::endl;
        return make_error_result("UNSUPPORTED_PRESOLVER");
#endif
    } else {
        const auto presolve_start = std::chrono::steady_clock::now();
        presolve_ok = run_embedded_pslp_presolve(
            model, actual_param, &reduced_model, &presolver_handle);
        presolve_time = static_cast<HPRLP_FLOAT>(
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - presolve_start).count());
    }
    const LP_info_cpu *solve_model =
        presolve_ok && !device_handoff_active ? &reduced_model : model;
    if (!actual_param->print_debug_info && presolve_ok) {
        const char *presolver_label = use_gpu_presolver
            ? "GPU-presolver" : "PSLP";
        const int reduced_m = device_handoff_active
            ? reduced_device_model.m : solve_model->m;
        const int reduced_n = device_handoff_active
            ? reduced_device_model.n : solve_model->n;
        const int reduced_nnz = device_handoff_active
            ? (reduced_device_model.A
                ? reduced_device_model.A->numElements : 0)
            : (solve_model->A ? solve_model->A->numElements : 0);
        std::cout << "Presolved (" << presolver_label
                  << ") Model: nRow = " << reduced_m
                  << ", nCol = " << reduced_n
                  << ", nnz = " << reduced_nnz << std::endl;
    } else if (!actual_param->print_debug_info) {
        const char *presolver_label = use_gpu_presolver
            ? "GPU-presolver" : "PSLP";
        std::cout << "Presolved (" << presolver_label
                  << ") Model: original-model fallback, nRow = "
                  << model->m << ", nCol = " << model->n
                  << ", nnz = " << (model->A ? model->A->numElements : 0)
                  << std::endl;
    }
    if (device_handoff_active) {
        std::cout << "GPU-resident presolve-to-solver handoff: active"
                  << std::endl;
        print_device_numerical_ranges(
            &reduced_device_model, nullptr, "after presolve");
    } else {
        if (use_gpu_presolver) {
            std::cout << "GPU-resident presolve-to-solver handoff: "
                      << "original-model fallback" << std::endl;
        }
        print_numerical_ranges(
            solve_model,
            presolve_ok ? "after presolve"
                        : "after presolve (original-model fallback)");
    }

    LP_info_cpu device_model_dimensions{};
    if (device_handoff_active) {
        device_model_dimensions.m = reduced_device_model.m;
        device_model_dimensions.n = reduced_device_model.n;
    }
    const LP_info_cpu *policy_model = device_handoff_active
        ? &device_model_dimensions : solve_model;
    const HPRLP_parameters effective =
        hprlp_effective_parameters(actual_param, policy_model);
    hprlp_print_automatic_memory_policy(policy_model, effective);
    HPRLP_compressible_memory_mode_guard compression_guard(
        hprlp_compression_mode(effective));

    HPRLP_results result = HPRLP_main_solve_impl(
        device_handoff_active ? nullptr : solve_model,
        device_handoff_active ? &reduced_device_model : nullptr,
        &effective, first_device_ready, presolve_time,
        use_gpu_presolver ? folding_time : 0.0);
    // Record attempted presolve time even when presolve falls back to the
    // original model. This keeps the timing breakdown faithful to work done.
    result.timing.presolve_time = presolve_time;
    result.presolve_time = presolve_time;
    result.folding_time = use_gpu_presolver ? folding_time : 0.0;

    if (presolve_ok) {
        if (result.x && result.y && result.z) {
            if (use_gpu_presolver) {
#ifdef HPRLP_HAS_GPU_PRESOLVER
                if (!gpu_postsolve_and_validate_original_kkt(
                        &result, model, presolver_handle, &effective)) {
                    std::free(result.x);
                    std::free(result.y);
                    std::free(result.z);
                    result.x = nullptr;
                    result.y = nullptr;
                    result.z = nullptr;
                    std::strncpy(result.status, "POSTSOLVE_FAILED",
                                 sizeof(result.status) - 1);
                    result.status[sizeof(result.status) - 1] = '\0';
                }
#endif
            } else {
                postsolve_and_validate_original_kkt(
                    &result, model, presolver_handle, &effective);
            }
        }
        if (use_gpu_presolver) {
#ifdef HPRLP_HAS_GPU_PRESOLVER
            free_embedded_gpu_presolver(presolver_handle);
#endif
        } else {
            free_embedded_pslp_presolver(presolver_handle);
        }
        if (!device_handoff_active) {
            free_lp_info_cpu(&reduced_model);
        }
    }

    return result;
}

/**
 * Progress-aware phase and sigma-trial orchestration for one solve.
 */
static void update_phase2_control(
        HPRLP_control_state *control,
        const HPRLP_progress_metrics &progress,
        const HPRLP_residuals &residuals,
        int iter,
        const HPRLP_parameters *param) {
    if (!std::isfinite(control->phase2_primal_reference) &&
        std::isfinite(residuals.err_Rp_org_bar)) {
        control->phase2_primal_reference =
            std::max(std::abs(residuals.err_Rp_org_bar), 1e-12);
    }
    if (!param->enable_progress_control || iter == 0) return;
    const HPRLP_phase_update update = hprlp_update_phase2_state(
        control->phase2_identified, control->phase2_confirmations,
        residuals.err_Rp_org_bar, control->phase2_primal_reference, progress);
    control->phase2_identified = update.identified;
    control->phase2_confirmations = update.confirmations;
    control->phase2_event = update.event;
    if (update.event != HPRLPPhaseEvent::None) {
        control->phase2_age_checks = 0;
        if (update.event == HPRLPPhaseEvent::Entered) {
            control->phase2_entry_primal = std::abs(residuals.err_Rp_org_bar);
        } else {
            control->phase2_entry_primal =
                std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
            if (hprlp_sigma_settled_block_should_clear(update.event))
                control->sigma_phase_settled_blocked = false;
        }
    } else if (control->phase2_identified) {
        ++control->phase2_age_checks;
    } else {
        control->phase2_age_checks = 0;
    }
    if ((param->debug_restart || param->debug_sigma) &&
        update.event != HPRLPPhaseEvent::None) {
        std::cout << "Phase 2 event at iteration " << iter
                  << ": reason=" << hprlp_phase_event_name(update.event)
                  << ", primal_residual=" << std::abs(residuals.err_Rp_org_bar)
                  << ", threshold=" << update.primal_threshold
                  << ", active_set_change_ratio="
                  << progress.active_set_change_ratio
                  << ", step_direction_cosine="
                  << progress.step_direction_cosine << std::endl;
    }
}

static void apply_periodic_control(
        HPRLP_control_state *control,
        const HPRLP_progress_metrics &progress,
        HPRLP_restart *restart,
        HPRLP_workspace_gpu *workspace,
        const HPRLP_residuals &residuals,
        int iter,
        const HPRLP_parameters *param,
        const std::string &status) {
    check_restart(restart, iter, param, workspace->sigma, progress);
    const bool revoked = control->phase2_event != HPRLPPhaseEvent::None &&
        control->phase2_event != HPRLPPhaseEvent::Entered;
    if (control->sigma_trial_active &&
        (revoked || iter - control->sigma_trial_iter >= 5 * param->check_iter)) {
        const HPRLP_FLOAT rp = std::abs(residuals.err_Rp_org_bar);
        const HPRLP_FLOAT rd = std::abs(residuals.err_Rd_org_bar);
        const HPRLP_FLOAT kkt = std::abs(residuals.KKTx_and_gap_org_bar);
        HPRLP_trial_assessment assessment = revoked
            ? HPRLP_trial_assessment{false, HPRLPTrialReason::Phase2Revoked}
            : hprlp_assess_sigma_trial(
                control->sigma_trial_rp_before, control->sigma_trial_rd_before,
                control->sigma_trial_kkt_before, rp, rd, kkt, param->stop_tol);
        control->sigma_trial_active = false;
        control->sigma_trial_last_resolution_iter = iter;
        const bool terminal_keep = !assessment.accepted &&
            control->sigma_trial_accepted_once && !revoked &&
            hprlp_sigma_trial_should_keep_terminal(
                control->sigma_trial_rp_before, control->sigma_trial_rd_before,
                control->sigma_trial_kkt_before, rp, rd, kkt);
        if (assessment.accepted) {
            control->sigma_trial_blocked = false;
            control->sigma_trial_accepted_once = true;
        } else if (terminal_keep) {
            control->sigma_trial_blocked = true;
            restart->best_sigma = workspace->sigma;
        } else {
            workspace->sigma = control->sigma_trial_sigma_before;
            restart->restart_flag = HPRLP_SIGMA_TRIAL_ROLLBACK_FLAG;
            restart->best_sigma = workspace->sigma;
            control->sigma_trial_blocked = true;
            control->sigma_trial_rollback_blocked = true;
        }
        if (param->debug_restart || param->debug_sigma)
            std::cout << "Sigma trial resolved at iteration " << iter
                      << ": reason=" << hprlp_trial_reason_name(assessment.reason)
                      << ", accepted=" << (assessment.accepted || terminal_keep)
                      << ", sigma=" << workspace->sigma << std::endl;
    }
    if (status == "CONTINUE" &&
        control->phase2_event == HPRLPPhaseEvent::ExitedPrimalRebound &&
        !control->sigma_trial_active &&
        restart->restart_flag != HPRLP_SIGMA_TRIAL_ROLLBACK_FLAG &&
        !(std::isfinite(param->fixed_sigma) && param->fixed_sigma > 0.0)) {
        if (std::isfinite(restart->best_sigma) && restart->best_sigma > 0.0)
            workspace->sigma = restart->best_sigma;
        restart->restart_flag = HPRLP_PHASE2_REBOUND_RESTART_FLAG;
    }
    if (hprlp_sigma_trial_floor_recovery_ready(
            *control, iter, param->check_iter, progress,
            residuals.err_Rp_org_bar, residuals.err_Rd_org_bar,
            param->stop_tol)) {
        control->sigma_trial_blocked = false;
        control->sigma_trial_rollback_blocked = false;
        control->sigma_trial_floor_recovery_used = true;
        control->sigma_trial_last_resolution_iter = iter;
    }
    if (!control->sigma_trial_active && !control->sigma_trial_blocked &&
        !control->sigma_phase_settled_blocked &&
        !control->sigma_trial_accepted_once && control->phase2_identified &&
        control->phase2_age_checks >= 5 &&
        std::isfinite(progress.active_set_change_ratio) &&
        progress.active_set_change_ratio <= 1e-3 &&
        std::isfinite(progress.step_direction_cosine) &&
        hprlp_sigma_path_should_block(progress, residuals.err_Rp_org_bar,
                                      residuals.err_Rd_org_bar))
        control->sigma_phase_settled_blocked = true;
    if (param->enable_sigma_rebalance_restart && status == "CONTINUE" &&
        !(std::isfinite(param->fixed_sigma) && param->fixed_sigma > 0.0) &&
        !control->sigma_trial_active && !control->sigma_trial_blocked &&
        !control->sigma_phase_settled_blocked &&
        iter - control->sigma_trial_last_resolution_iter >= 5 * param->check_iter &&
        check_sigma_rebalance_restart(
            restart, &residuals, iter, param, *control, progress,
            control->sigma_trial_accepted_once ? 4.4 : 4.0)) {
        control->sigma_trial_active = true;
        control->sigma_trial_iter = iter;
        control->sigma_trial_sigma_before = workspace->sigma;
        control->sigma_trial_rp_before = std::abs(residuals.err_Rp_org_bar);
        control->sigma_trial_rd_before = std::abs(residuals.err_Rd_org_bar);
        control->sigma_trial_kkt_before =
            std::abs(residuals.KKTx_and_gap_org_bar);
    }
    if (param->enable_progress_control && status == "CONTINUE" &&
        !control->sigma_trial_active &&
        !(std::isfinite(param->fixed_sigma) && param->fixed_sigma > 0.0)) {
        const int stall_window_iterations =
            HPRLP_ONE_SIDED_STALL_CHECKS * param->check_iter;
        if (control->one_sided_stall_trial_active &&
            iter - control->one_sided_stall_trial_iter >=
                stall_window_iterations) {
            const bool rollback =
                hprlp_one_sided_stall_trial_should_rollback(
                    control->one_sided_stall_trial_side,
                    control->one_sided_stall_trial_target_before,
                    residuals.err_Rp_org_bar, residuals.err_Rd_org_bar,
                    param->stop_tol);
            control->one_sided_stall_trial_active = false;
            if (rollback) {
                workspace->sigma =
                    control->one_sided_stall_trial_sigma_before;
                restart->restart_flag = HPRLP_SIGMA_TRIAL_ROLLBACK_FLAG;
                restart->best_sigma = workspace->sigma;
                control->one_sided_stall_blocked = true;
                control->one_sided_stall_active = false;
                if (param->debug_restart || param->debug_sigma)
                    std::cout << std::scientific << std::setprecision(16)
                              << "One-sided feasibility stall trial rolled back at iteration "
                              << iter
                              << ": target_before="
                              << control->one_sided_stall_trial_target_before
                              << ", residual_primal="
                              << residuals.err_Rp_org_bar
                              << ", residual_dual="
                              << residuals.err_Rd_org_bar
                              << ", restored_sigma=" << workspace->sigma
                              << std::defaultfloat << std::setprecision(2)
                              << std::endl;
            }
        }
        const HPRLP_one_sided_stall_update stall =
            hprlp_update_one_sided_stall(
                control->one_sided_stall_active,
                control->one_sided_stall_side,
                control->one_sided_stall_best_iter,
                control->one_sided_stall_best_target, iter,
                residuals.err_Rp_org_bar, residuals.err_Rd_org_bar,
                param->stop_tol,
                stall_window_iterations);
        control->one_sided_stall_active = stall.active;
        control->one_sided_stall_side = stall.side;
        control->one_sided_stall_best_iter = stall.best_iter;
        control->one_sided_stall_best_target = stall.best_target;
        if (stall.trigger && !control->one_sided_stall_blocked) {
            const bool changes_direction =
                control->one_sided_stall_last_correction_side != 0 &&
                stall.side != control->one_sided_stall_last_correction_side;
            if (changes_direction &&
                control->one_sided_stall_direction_transitions >= 1) {
                control->one_sided_stall_blocked = true;
                if (param->debug_restart || param->debug_sigma)
                    std::cout
                        << "One-sided feasibility stall correction blocked at iteration "
                        << iter
                        << ": reason=repeated_direction_transition"
                        << ", side=" << stall.side << std::endl;
            }
        }
        if (stall.trigger && !control->one_sided_stall_blocked) {
            const HPRLP_FLOAT old_sigma = workspace->sigma;
            const HPRLP_sigma_update_result correction =
                hprlp_one_sided_stall_sigma_update(
                    old_sigma, residuals.err_Rp_org_bar,
                    residuals.err_Rd_org_bar);
            if (correction.valid) {
                const bool changes_direction =
                    control->one_sided_stall_last_correction_side != 0 &&
                    stall.side != control->one_sided_stall_last_correction_side;
                if (changes_direction)
                    ++control->one_sided_stall_direction_transitions;
                control->one_sided_stall_last_correction_side = stall.side;
                control->one_sided_stall_trial_active = true;
                control->one_sided_stall_trial_iter = iter;
                control->one_sided_stall_trial_side = stall.side;
                control->one_sided_stall_trial_sigma_before = old_sigma;
                control->one_sided_stall_trial_target_before = std::max(
                    std::abs(residuals.err_Rp_org_bar),
                    std::abs(residuals.err_Rd_org_bar));
                workspace->sigma = correction.sigma;
                restart->restart_flag =
                    HPRLP_ONE_SIDED_STALL_RESTART_FLAG;
                restart->best_sigma = workspace->sigma;
                control->sigma_trial_last_resolution_iter = iter;
                if (param->debug_restart || param->debug_sigma)
                    std::cout << std::scientific << std::setprecision(16)
                              << "One-sided feasibility stall correction at iteration "
                              << iter
                              << ": old_sigma=" << old_sigma
                              << ", residual_primal="
                              << residuals.err_Rp_org_bar
                              << ", residual_dual="
                              << residuals.err_Rd_org_bar
                              << ", side=" << stall.side
                              << ", direction_transitions="
                              << control->one_sided_stall_direction_transitions
                              << ", new_sigma=" << workspace->sigma
                              << std::defaultfloat << std::setprecision(2)
                              << std::endl;
            }
        }
    }
    if (restart->restart_flag > 0 && !control->phase2_identified)
        control->phase2_confirmations = 0;
}
