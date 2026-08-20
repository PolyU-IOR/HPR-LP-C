#include "solver/progress_monitor.cuh"

#include "cuda_kernels/cuda_check.h"
#include "support/utils.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr std::uint8_t kInactive = 0;
constexpr std::uint8_t kLower = 1;
constexpr std::uint8_t kUpper = 2;
constexpr std::uint8_t kExcluded = 255;

__device__ std::uint8_t primal_state(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper) {
    const bool has_lower = lower > -1e90;
    const bool has_upper = upper < 1e90;
    if ((!has_lower && !has_upper) ||
        (has_lower && has_upper && lower == upper)) return kExcluded;
    if (has_lower && value == lower) return kLower;
    if (has_upper && value == upper) return kUpper;
    return kInactive;
}

__device__ std::uint8_t dual_state(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper) {
    const bool has_lower = lower > -1e90;
    const bool has_upper = upper < 1e90;
    if ((!has_lower && !has_upper) ||
        (has_lower && has_upper && lower == upper)) return kExcluded;
    if (value > 0.0) return kLower;
    if (value < 0.0) return kUpper;
    return kInactive;
}

__global__ void baseline_kernel(
    int n, int m, const HPRLP_FLOAT *x, const HPRLP_FLOAT *y,
    const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const HPRLP_FLOAT *al, const HPRLP_FLOAT *au,
    std::uint8_t *x_state, std::uint8_t *y_state) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x_state[i] = primal_state(x[i], l[i], u[i]);
    if (i < m) y_state[i] = dual_state(y[i], al[i], au[i]);
}

__global__ void sample_kernel(
    int n, int m, const HPRLP_FLOAT *x, const HPRLP_FLOAT *y,
    const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
    const HPRLP_FLOAT *al, const HPRLP_FLOAT *au,
    std::uint8_t *x_state, std::uint8_t *y_state,
    HPRLP_FLOAT *previous_x, HPRLP_FLOAT *previous_y,
    HPRLP_FLOAT *current_step_x, HPRLP_FLOAT *current_step_y,
    unsigned long long *counters) {
    extern __shared__ unsigned long long shared[];
    unsigned long long *changed_shared = shared;
    unsigned long long *eligible_shared = shared + blockDim.x;
    unsigned long long *zero_shared = shared + 2 * blockDim.x;
    unsigned long long *primal_eligible_shared =
        shared + 3 * blockDim.x;
    unsigned long long *stationary_bound_shared =
        shared + 4 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long changed = 0;
    unsigned long long eligible = 0;
    unsigned long long primal_zero_move = 0;
    unsigned long long primal_eligible = 0;
    unsigned long long stationary_bound = 0;
    if (i < n) {
        const std::uint8_t state = primal_state(x[i], l[i], u[i]);
        if (state != kExcluded) {
            changed += static_cast<unsigned long long>(state != x_state[i]);
            eligible += 1;
            primal_eligible += 1;
            const HPRLP_FLOAT old_value = previous_x[i];
            const bool zero_move = isfinite(x[i]) && isfinite(old_value) &&
                fabs(x[i] - old_value) <= 1e-12 + 1e-12 *
                    fmax(1.0, fmax(fabs(x[i]), fabs(old_value)));
            primal_zero_move += static_cast<unsigned long long>(zero_move);
            stationary_bound += static_cast<unsigned long long>(
                zero_move && (state == kLower || state == kUpper));
        }
        x_state[i] = state;
        current_step_x[i] = x[i] - previous_x[i];
        previous_x[i] = x[i];
    }
    if (i < m) {
        const std::uint8_t state = dual_state(y[i], al[i], au[i]);
        if (state != kExcluded) {
            changed += static_cast<unsigned long long>(state != y_state[i]);
            eligible += 1;
        }
        y_state[i] = state;
        current_step_y[i] = y[i] - previous_y[i];
        previous_y[i] = y[i];
    }

    changed_shared[threadIdx.x] = changed;
    eligible_shared[threadIdx.x] = eligible;
    zero_shared[threadIdx.x] = primal_zero_move;
    primal_eligible_shared[threadIdx.x] = primal_eligible;
    stationary_bound_shared[threadIdx.x] = stationary_bound;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            changed_shared[threadIdx.x] +=
                changed_shared[threadIdx.x + offset];
            eligible_shared[threadIdx.x] +=
                eligible_shared[threadIdx.x + offset];
            zero_shared[threadIdx.x] += zero_shared[threadIdx.x + offset];
            primal_eligible_shared[threadIdx.x] +=
                primal_eligible_shared[threadIdx.x + offset];
            stationary_bound_shared[threadIdx.x] +=
                stationary_bound_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(counters + 0, changed_shared[0]);
        atomicAdd(counters + 1, eligible_shared[0]);
        atomicAdd(counters + 2, zero_shared[0]);
        atomicAdd(counters + 3, primal_eligible_shared[0]);
        atomicAdd(counters + 4, stationary_bound_shared[0]);
    }
}
}

void hprlp_initialize_progress_monitor(
    HPRLP_progress_monitor_gpu *monitor, HPRLP_workspace_gpu *ws) {
    CUDA_CHECK(cudaMalloc(&monitor->previous_x_state,
                          ws->n * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&monitor->previous_y_state,
                          ws->m * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&monitor->previous_x_bar,
                          ws->n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&monitor->previous_y_bar,
                          ws->m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&monitor->previous_step_x,
                          ws->n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&monitor->previous_step_y,
                          ws->m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&monitor->current_step_x,
                          ws->n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&monitor->current_step_y,
                          ws->m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&monitor->counters, 5 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&monitor->counters_host,
                              5 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(monitor->previous_step_x, 0,
                               ws->n * sizeof(HPRLP_FLOAT), ws->stream));
    CUDA_CHECK(cudaMemsetAsync(monitor->previous_step_y, 0,
                               ws->m * sizeof(HPRLP_FLOAT), ws->stream));
    const int count = std::max(ws->n, ws->m);
    baseline_kernel<<<numBlocks(count), numThreads, 0, ws->stream>>>(
        ws->n, ws->m, ws->x_bar, ws->y_bar, ws->l, ws->u, ws->AL, ws->AU,
        monitor->previous_x_state, monitor->previous_y_state);
    CUDA_CHECK(cudaMemcpyAsync(monitor->previous_x_bar, ws->x_bar,
        ws->n * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(monitor->previous_y_bar, ws->y_bar,
        ws->m * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice, ws->stream));
}

void hprlp_queue_progress_sample(
    HPRLP_progress_monitor_gpu *monitor, HPRLP_workspace_gpu *ws) {
    CUDA_CHECK(cudaMemsetAsync(monitor->counters, 0,
        5 * sizeof(unsigned long long), ws->stream));
    const int count = std::max(ws->n, ws->m);
    const std::size_t shared_bytes =
        5 * static_cast<std::size_t>(numThreads) *
        sizeof(unsigned long long);
    sample_kernel<<<numBlocks(count), numThreads, shared_bytes, ws->stream>>>(
        ws->n, ws->m, ws->x_bar, ws->y_bar, ws->l, ws->u, ws->AL, ws->AU,
        monitor->previous_x_state, monitor->previous_y_state,
        monitor->previous_x_bar, monitor->previous_y_bar,
        monitor->current_step_x, monitor->current_step_y,
        monitor->counters);
    queue_dot(ws->reduction_scalars, 10, monitor->current_step_x,
              monitor->previous_step_x, ws->n, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 11, monitor->current_step_y,
              monitor->previous_step_y, ws->m, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 12, monitor->current_step_x,
              monitor->current_step_x, ws->n, ws->cublasHandle_device);
    queue_dot(ws->reduction_scalars, 13, monitor->current_step_y,
              monitor->current_step_y, ws->m, ws->cublasHandle_device);
    CUDA_CHECK(cudaMemcpyAsync(monitor->counters_host, monitor->counters,
        5 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, ws->stream));
    std::swap(monitor->current_step_x, monitor->previous_step_x);
    std::swap(monitor->current_step_y, monitor->previous_step_y);
}

void hprlp_finalize_progress_sample(
    HPRLP_progress_monitor_gpu *monitor, const HPRLP_workspace_gpu *ws) {
    const unsigned long long eligible = monitor->counters_host[1];
    const unsigned long long primal_eligible = monitor->counters_host[3];
    monitor->metrics.active_set_change_ratio = eligible == 0 ? 0.0 :
        static_cast<HPRLP_FLOAT>(monitor->counters_host[0]) / eligible;
    monitor->metrics.primal_zero_move_ratio = primal_eligible == 0 ? 0.0 :
        static_cast<HPRLP_FLOAT>(monitor->counters_host[2]) / primal_eligible;
    monitor->metrics.stationary_bound_ratio = primal_eligible == 0 ? 0.0 :
        static_cast<HPRLP_FLOAT>(monitor->counters_host[4]) / primal_eligible;
    const HPRLP_FLOAT dx_norm_sq = ws->reduction_scalars_host[12];
    const HPRLP_FLOAT dy_norm_sq = ws->reduction_scalars_host[13];
    if (monitor->direction_initialized) {
        const HPRLP_FLOAT numerator = ws->reduction_scalars_host[10] / ws->sigma +
            ws->lambda_max * ws->sigma * ws->reduction_scalars_host[11];
        const HPRLP_FLOAT current_norm_sq = dx_norm_sq / ws->sigma +
            ws->lambda_max * ws->sigma * dy_norm_sq;
        const HPRLP_FLOAT previous_norm_sq =
            monitor->previous_dx_norm_sq / ws->sigma +
            ws->lambda_max * ws->sigma * monitor->previous_dy_norm_sq;
        monitor->metrics.step_direction_cosine =
            hprlp_bounded_direction_cosine(
                numerator, current_norm_sq, previous_norm_sq);
    } else {
        monitor->metrics.step_direction_cosine =
            std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    }
    monitor->previous_dx_norm_sq = dx_norm_sq;
    monitor->previous_dy_norm_sq = dy_norm_sq;
    monitor->direction_initialized = true;
}

void hprlp_free_progress_monitor(HPRLP_progress_monitor_gpu *monitor) {
    if (!monitor) return;
    cudaFree(monitor->previous_x_state);
    cudaFree(monitor->previous_y_state);
    cudaFree(monitor->previous_x_bar);
    cudaFree(monitor->previous_y_bar);
    cudaFree(monitor->previous_step_x);
    cudaFree(monitor->previous_step_y);
    cudaFree(monitor->current_step_x);
    cudaFree(monitor->current_step_y);
    cudaFree(monitor->counters);
    if (monitor->counters_host) cudaFreeHost(monitor->counters_host);
    *monitor = HPRLP_progress_monitor_gpu{};
}
