#include "solver/reduced_matrix.cuh"

#include "HPRLP.h"
#include "cuda_kernels/backends/simple/simple_update_kernels.cuh"
#include "cuda_kernels/backends/detail/update_device_helpers.cuh"
#include "cuda_kernels/cuda_check.h"
#include "cuda_kernels/shared/vector_kernels.cuh"
#include "gpu/preprocessing/policies/row_bucket_policy.h"
#include "solver/graph/graph_batch_policy.h"
#include "solver/backends/signed_unit_launcher.cuh"
#include "solver/backends/packed_dictionary_launcher.cuh"
#include "solver/backends/unit_factorized_launcher.cuh"
#include "solver/iteration/main_iterate.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using hprlp::cuda_kernels::detail::project_x_with_bounds;
using hprlp::cuda_kernels::detail::project_y_delta;

namespace {

constexpr int kReducedVectorThreads = 512;
constexpr int kReducedYThreads = 1024;
constexpr int kReducedUnitYThreads = 256;
constexpr int kReducedSignedXThreads = 256;
constexpr int kReducedSignedYThreads = 512;
constexpr int kReducedSignedEmptyBatchMinimumRows = 1000000;
constexpr int kReducedDictionaryEmptyBatchMinimumPercent = 10;
constexpr int kReducedCusparseEmptyBatchMinimumPercent = 50;
constexpr int kReducedParallelDeltaMinimumColumns = 10000;
constexpr int kReducedParallelDeltaMinimumNnz = 30000;
constexpr int kReducedPrecomputedDeltaInputMinimumNnz = 256;

enum class HPRLP_reduced_mode : std::uint8_t {
    None,
    Columns,
    Rows
};

void check_cusparse(cusparseStatus_t status, const char *context) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(context) + ": cuSPARSE status " +
            std::to_string(static_cast<int>(status)));
    }
}

HPRLP_FLOAT reduced_enter_residual() {
    static const HPRLP_FLOAT threshold = []() {
        const char *value =
            std::getenv("HPRLP_REDUCED_ENTER_RESIDUAL_OVERRIDE");
        if (value == nullptr || *value == '\0') {
            return HPRLP_REDUCED_ENTER_RESIDUAL;
        }
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed) ||
            parsed <= 0.0) {
            std::cerr
                << "Ignoring invalid HPRLP_REDUCED_ENTER_RESIDUAL_OVERRIDE='"
                << value << "'; using " << HPRLP_REDUCED_ENTER_RESIDUAL
                << std::endl;
            return HPRLP_REDUCED_ENTER_RESIDUAL;
        }
        std::cout << "Reduced matrix entry residual override: "
                  << parsed << " (default "
                  << HPRLP_REDUCED_ENTER_RESIDUAL << ")" << std::endl;
        return static_cast<HPRLP_FLOAT>(parsed);
    }();
    return threshold;
}

bool recompute_reduced_prebuild_mask() {
    static const bool enabled = []() {
        const char *value =
            std::getenv("HPRLP_REDUCED_RECOMPUTE_PREBUILD_MASK");
        if (value == nullptr) return false;
        const std::string setting(value);
        const bool result = setting == "1" || setting == "true" ||
            setting == "TRUE" || setting == "yes" || setting == "YES";
        if (result) {
            std::cout << "Reduced matrix prebuild mask recomputation: enabled"
                      << std::endl;
        }
        return result;
    }();
    return enabled;
}

HPRLP_FLOAT reduced_prebuild_enter_ratio() {
    static const HPRLP_FLOAT ratio = []() {
        const char *value =
            std::getenv("HPRLP_REDUCED_PREBUILD_ENTER_RATIO_OVERRIDE");
        if (value == nullptr || *value == '\0') {
            return HPRLP_REDUCED_ENTER_RATIO;
        }
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed) ||
            parsed <= 0.0 || parsed >= 1.0) {
            std::cerr
                << "Ignoring invalid "
                   "HPRLP_REDUCED_PREBUILD_ENTER_RATIO_OVERRIDE='"
                << value << "'; using " << HPRLP_REDUCED_ENTER_RATIO
                << std::endl;
            return HPRLP_REDUCED_ENTER_RATIO;
        }
        std::cout << "Reduced matrix prebuild entry ratio override: "
                  << parsed << " (default " << HPRLP_REDUCED_ENTER_RATIO
                  << ")" << std::endl;
        return static_cast<HPRLP_FLOAT>(parsed);
    }();
    return ratio;
}

bool reduced_restart_mask_reset_enabled() {
    static const bool enabled = []() {
        const char *value =
            std::getenv("HPRLP_REDUCED_RESET_MASK_ON_RESTART");
        if (value == nullptr) return false;
        const std::string setting(value);
        const bool result = setting == "1" || setting == "true" ||
            setting == "TRUE" || setting == "yes" || setting == "YES";
        if (result) {
            std::cout << "Reduced matrix restart mask reset: enabled"
                      << std::endl;
        }
        return result;
    }();
    return enabled;
}

HPRLP_FLOAT reduced_restart_mask_min_recovery() {
    static const HPRLP_FLOAT recovery = []() {
        constexpr HPRLP_FLOAT default_recovery = 0.25;
        const char *value =
            std::getenv("HPRLP_REDUCED_RESTART_MASK_MIN_RECOVERY");
        if (value == nullptr || *value == '\0') return default_recovery;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed) ||
            parsed < 0.0 || parsed >= 1.0) {
            std::cerr
                << "Ignoring invalid "
                   "HPRLP_REDUCED_RESTART_MASK_MIN_RECOVERY='"
                << value << "'; using " << default_recovery << std::endl;
            return default_recovery;
        }
        return static_cast<HPRLP_FLOAT>(parsed);
    }();
    return recovery;
}

long long reduced_restart_mask_min_saved_columns() {
    static const long long min_saved_columns = []() {
        constexpr long long default_min_saved_columns = 25000;
        const char *value = std::getenv(
            "HPRLP_REDUCED_RESTART_MASK_MIN_SAVED_COLUMNS");
        if (value == nullptr || *value == '\0') {
            return default_min_saved_columns;
        }
        char *end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0) {
            std::cerr
                << "Ignoring invalid "
                   "HPRLP_REDUCED_RESTART_MASK_MIN_SAVED_COLUMNS='"
                << value << "'; using " << default_min_saved_columns
                << std::endl;
            return default_min_saved_columns;
        }
        return parsed;
    }();
    return min_saved_columns;
}

long long reduced_restart_mask_min_current_columns() {
    static const long long min_current_columns = []() {
        constexpr long long default_min_current_columns = 95000;
        const char *value = std::getenv(
            "HPRLP_REDUCED_RESTART_MASK_MIN_CURRENT_COLUMNS");
        if (value == nullptr || *value == '\0') {
            return default_min_current_columns;
        }
        char *end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0) {
            std::cerr
                << "Ignoring invalid "
                   "HPRLP_REDUCED_RESTART_MASK_MIN_CURRENT_COLUMNS='"
                << value << "'; using " << default_min_current_columns
                << std::endl;
            return default_min_current_columns;
        }
        return parsed;
    }();
    return min_current_columns;
}

bool reduced_rows_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("HPRLP_USE_ROW_REDUCTION");
        if (value == nullptr || *value == '\0') {
            std::cout << "Adaptive row/column reduction: enabled"
                      << std::endl;
            return true;
        }
        const std::string setting(value);
        if (setting == "0" || setting == "false" || setting == "FALSE" ||
            setting == "no" || setting == "NO") {
            std::cout << "Adaptive row/column reduction: row candidate "
                         "disabled by HPRLP_USE_ROW_REDUCTION"
                      << std::endl;
            return false;
        }
        if (setting == "1" || setting == "true" || setting == "TRUE" ||
            setting == "yes" || setting == "YES") {
            std::cout << "Adaptive row/column reduction: enabled"
                      << std::endl;
            return true;
        }
        std::cerr << "Ignoring invalid HPRLP_USE_ROW_REDUCTION='" << value
                  << "'; adaptive row/column reduction remains enabled"
                  << std::endl;
        return true;
    }();
    return enabled;
}

HPRLP_FLOAT reduced_row_enter_ratio() {
    static const HPRLP_FLOAT ratio = []() {
        constexpr HPRLP_FLOAT default_ratio = 0.40;
        const char *value = std::getenv(
            "HPRLP_REDUCED_ROW_ENTER_RATIO_OVERRIDE");
        if (value == nullptr || *value == '\0') return default_ratio;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed) ||
            parsed <= 0.0 || parsed >= 1.0) {
            std::cerr << "Ignoring invalid "
                "HPRLP_REDUCED_ROW_ENTER_RATIO_OVERRIDE='" << value
                      << "'; using " << default_ratio << std::endl;
            return default_ratio;
        }
        std::cout << "Row reduction entry ratio override: " << parsed
                  << " (default " << default_ratio << ")" << std::endl;
        return static_cast<HPRLP_FLOAT>(parsed);
    }();
    return ratio;
}

HPRLP_FLOAT reduced_compact_row_y_nnz_ratio() {
    static const HPRLP_FLOAT ratio = []() {
        constexpr HPRLP_FLOAT default_ratio = 0.50;
        const char *value = std::getenv(
            "HPRLP_REDUCED_COMPACT_ROW_Y_NNZ_RATIO");
        if (value == nullptr || *value == '\0') return default_ratio;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed) ||
            parsed < 0.0 || parsed > 1.0) {
            std::cerr << "Ignoring invalid "
                "HPRLP_REDUCED_COMPACT_ROW_Y_NNZ_RATIO='" << value
                      << "'; using " << default_ratio << std::endl;
            return default_ratio;
        }
        return static_cast<HPRLP_FLOAT>(parsed);
    }();
    return ratio;
}

bool reduced_adaptive_row_rebase_enabled() {
    const char *value = std::getenv("HPRLP_DISABLE_ADAPTIVE_ROW_REBASE");
    return value == nullptr || std::string(value) != "1";
}

bool reduced_row_backend_profile_enabled() {
    const char *value = std::getenv("HPRLP_PROFILE_ROW_BACKEND_AUTOTUNE");
    return value != nullptr && std::string(value) != "0";
}

bool reduced_row_compressed_autotune_enabled() {
    const char *value =
        std::getenv("HPRLP_USE_ROW_COMPRESSED_AUTOTUNE");
    if (value == nullptr) return false;
    const std::string setting(value);
    return setting != "0" && setting != "false" && setting != "FALSE" &&
        setting != "no" && setting != "NO";
}

__global__ void initialize_y_bar_mask_kernel(
    std::uint8_t *mask,
    const HPRLP_FLOAT *y_bar,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    int *changed,
    int *active_count_by_warp,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = false;
    if (row < m) {
        const bool freezable = bound_type[row] != 0 &&
            lower[row] != upper[row];
        active = !freezable || y_bar[row] != 0.0;
        const std::uint8_t new_mask = active ? 1 : 0;
        if (mask[row] != new_mask) atomicExch(changed, 1);
        mask[row] = new_mask;
    }
    const unsigned selected = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && row < m) {
        active_count_by_warp[row >> 5] = __popc(selected);
    }
}

__global__ void count_current_active_rows_by_warp_kernel(
    const HPRLP_FLOAT *y_bar,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    int *active_count_by_warp,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = false;
    if (row < m) {
        const bool freezable = bound_type[row] != 0 &&
            lower[row] != upper[row];
        active = !freezable || y_bar[row] != 0.0;
    }
    const unsigned selected = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && row < m) {
        active_count_by_warp[row >> 5] = __popc(selected);
    }
}

__global__ void update_y_bar_mask_kernel(
    std::uint8_t *mask,
    const HPRLP_FLOAT *y_bar,
    int *changed,
    int *active_count_by_warp,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = false;
    if (row < m) {
        active = mask[row] != 0;
        if (!active && y_bar[row] != 0.0) {
            mask[row] = 1;
            active = true;
            atomicExch(changed, 1);
        }
    }
    const unsigned selected = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && row < m) {
        active_count_by_warp[row >> 5] = __popc(selected);
    }
}

__global__ void scatter_active_row_indices_kernel(
    const std::uint8_t *mask,
    const int *offsets_by_warp,
    int *selected_indices,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = row < m && mask[row] != 0;
    const unsigned selected = __ballot_sync(0xffffffffu, active);
    if (!active) return;
    const int lane = threadIdx.x & 31;
    const unsigned lanes_before = lane == 0 ? 0u : ((1u << lane) - 1u);
    selected_indices[offsets_by_warp[row >> 5] +
                     __popc(selected & lanes_before)] = row;
}

__global__ void count_delta_rows_by_warp_kernel(
    const std::uint8_t *mask,
    const std::uint8_t *base_mask,
    int *counts_by_warp,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const bool selected = row < m && mask[row] != 0 &&
        base_mask[row] == 0;
    const unsigned ballot = __ballot_sync(0xffffffffu, selected);
    if ((threadIdx.x & 31) == 0 && row < m) {
        counts_by_warp[row >> 5] = __popc(ballot);
    }
}

__global__ void scatter_delta_row_indices_kernel(
    const std::uint8_t *mask,
    const std::uint8_t *base_mask,
    const int *offsets_by_warp,
    int *selected_indices,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const bool selected = row < m && mask[row] != 0 &&
        base_mask[row] == 0;
    const unsigned ballot = __ballot_sync(0xffffffffu, selected);
    if (!selected) return;
    const int lane = threadIdx.x & 31;
    const unsigned lanes_before = lane == 0 ? 0u : ((1u << lane) - 1u);
    selected_indices[offsets_by_warp[row >> 5] +
                     __popc(ballot & lanes_before)] = row;
}

__global__ void remap_compact_row_columns_kernel(
    int *columns,
    const int *active_to_original,
    int nnz) {
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < nnz) columns[entry] = active_to_original[columns[entry]];
}

__global__ void zero_inactive_y_state_kernel(
    HPRLP_FLOAT *y,
    HPRLP_FLOAT *last_y,
    const std::uint8_t *mask,
    int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && mask[row] == 0) {
        y[row] = 0.0;
        last_y[row] = 0.0;
    }
}

__global__ void gather_reduced_row_state_kernel(
    HPRLP_FLOAT *compact_y,
    HPRLP_FLOAT *compact_last_y,
    HPRLP_FLOAT *compact_lower,
    HPRLP_FLOAT *compact_upper,
    const HPRLP_FLOAT *y,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const int *active_to_original,
    int active_count) {
    const int compact_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_row >= active_count) return;
    const int row = active_to_original[compact_row];
    compact_y[compact_row] = y[row];
    compact_last_y[compact_row] = last_y[row];
    compact_lower[compact_row] = lower[row];
    compact_upper[compact_row] = upper[row];
}

__global__ void scatter_reduced_row_state_kernel(
    HPRLP_FLOAT *y,
    HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *compact_y,
    const HPRLP_FLOAT *compact_last_y,
    const int *active_to_original,
    int active_count) {
    const int compact_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_row >= active_count) return;
    const int row = active_to_original[compact_row];
    y[row] = compact_y[compact_row];
    last_y[row] = compact_last_y[compact_row];
}

__global__ void update_reduced_row_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *compact_ax,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int active_count) {
    const int compact_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_row >= active_count) return;
    const HPRLP_FLOAT current = y[compact_row];
    const HPRLP_FLOAT value = compact_ax[compact_row] -
        sigma_params[1] * current;
    const HPRLP_FLOAT delta = fmax(
        lower[compact_row] - value,
        fmin(upper[compact_row] - value, 0.0));
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    y[compact_row] = halpern_factors[1] * (2.0 * y_bar - current) +
        halpern_factors[0] * last_y[compact_row];
}

__global__ void update_reduced_row_y_full_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *compact_ax,
    const HPRLP_FLOAT *last_y,
    const int *active_to_original,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int active_count) {
    const int compact_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_row >= active_count) return;
    const int row = active_to_original[compact_row];
    const HPRLP_FLOAT current = y[row];
    const HPRLP_FLOAT value = compact_ax[compact_row] -
        sigma_params[1] * current;
    const HPRLP_FLOAT delta = fmax(
        lower[row] - value, fmin(upper[row] - value, 0.0));
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    y[row] = halpern_factors[1] * (2.0 * y_bar - current) +
        halpern_factors[0] * last_y[row];
}

__global__ void update_reduced_row_x_with_delta_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *base_aty,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= n) return;
    HPRLP_FLOAT correction = 0.0;
    for (int entry = delta_row_ptr[column];
         entry < delta_row_ptr[column + 1]; ++entry) {
        correction = fma(
            delta_values[entry], y[delta_columns[entry]], correction);
    }
    const HPRLP_FLOAT current = x[column];
    const HPRLP_FLOAT projected = fmin(
        upper[column], fmax(
            lower[column], current + sigma_params[0] *
                (base_aty[column] + correction - objective[column])));
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[column] = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_x[column]);
    x_hat[column] = reflected;
}

__global__ void fused_reduced_row_x_short_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *base_y,
    const int *base_row_ptr,
    const int *base_columns,
    const HPRLP_FLOAT *base_values,
    const HPRLP_FLOAT *delta_y,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int row = row_ids[item];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = base_row_ptr[row];
         entry < base_row_ptr[row + 1]; ++entry) {
        sum = fma(base_values[entry], base_y[base_columns[entry]], sum);
    }
    if (delta_row_ptr != nullptr) {
        for (int entry = delta_row_ptr[row];
             entry < delta_row_ptr[row + 1]; ++entry) {
            sum = fma(
                delta_values[entry], delta_y[delta_columns[entry]], sum);
        }
    }
    const HPRLP_FLOAT current = x[row];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        fma(sigma_params[0], sum - objective[row], current),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[row] = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
}

__global__ void fused_reduced_row_x_warp_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *base_y,
    const int *base_row_ptr,
    const int *base_columns,
    const HPRLP_FLOAT *base_values,
    const HPRLP_FLOAT *delta_y,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= row_count) return;
    const int row = row_ids[warp];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = base_row_ptr[row] + lane;
         entry < base_row_ptr[row + 1]; entry += 32) {
        sum = fma(base_values[entry], base_y[base_columns[entry]], sum);
    }
    if (delta_row_ptr != nullptr) {
        for (int entry = delta_row_ptr[row] + lane;
             entry < delta_row_ptr[row + 1]; entry += 32) {
            sum = fma(
                delta_values[entry], delta_y[delta_columns[entry]], sum);
        }
    }
    constexpr unsigned mask = 0xffffffffu;
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (lane != 0) return;
    const HPRLP_FLOAT current = x[row];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        fma(sigma_params[0], sum - objective[row], current),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[row] = fma(
        halpern_factors[1], reflected,
        halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
}

__device__ __forceinline__ HPRLP_FLOAT reduced_row_y_update(
    HPRLP_FLOAT current,
    HPRLP_FLOAT activity,
    HPRLP_FLOAT lower,
    HPRLP_FLOAT upper,
    HPRLP_FLOAT last,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT value = fma(-sigma_params[1], current, activity);
    const HPRLP_FLOAT delta = fmax(
        lower - value, fmin(upper - value, 0.0));
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    return fma(
        halpern_factors[1], 2.0 * y_bar - current,
        halpern_factors[0] * last);
}

__global__ void fused_reduced_row_y_short_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const int *active_to_original,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int reduced_row = row_ids[item];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[reduced_row];
         entry < row_ptr[reduced_row + 1]; ++entry) {
        sum = fma(values[entry], x_hat[columns[entry]], sum);
    }
    const int state_row = active_to_original != nullptr
        ? active_to_original[reduced_row] : reduced_row;
    y[state_row] = reduced_row_y_update(
        y[state_row], sum, lower[state_row], upper[state_row],
        last_y[state_row], sigma_params, halpern_factors);
}

__global__ void fused_reduced_row_y_warp_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const int *active_to_original,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= row_count) return;
    const int reduced_row = row_ids[warp];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[reduced_row] + lane;
         entry < row_ptr[reduced_row + 1]; entry += 32) {
        sum = fma(values[entry], x_hat[columns[entry]], sum);
    }
    constexpr unsigned mask = 0xffffffffu;
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (lane != 0) return;
    const int state_row = active_to_original != nullptr
        ? active_to_original[reduced_row] : reduced_row;
    y[state_row] = reduced_row_y_update(
        y[state_row], sum, lower[state_row], upper[state_row],
        last_y[state_row], sigma_params, halpern_factors);
}

__global__ void update_reduced_row_delta_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int delta_count) {
    const int delta_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (delta_row >= delta_count) return;
    HPRLP_FLOAT activity = 0.0;
    for (int entry = row_ptr[delta_row];
         entry < row_ptr[delta_row + 1]; ++entry) {
        activity = fma(values[entry], x_hat[columns[entry]], activity);
    }
    const HPRLP_FLOAT current = y[delta_row];
    const HPRLP_FLOAT value = activity - sigma_params[1] * current;
    const HPRLP_FLOAT delta = fmax(
        lower[delta_row] - value,
        fmin(upper[delta_row] - value, 0.0));
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    y[delta_row] = fma(
        halpern_factors[1], 2.0 * y_bar - current,
        halpern_factors[0] * last_y[delta_row]);
}

__global__ void update_reduced_row_delta_y_full_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const int *delta_to_original,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int delta_count) {
    const int delta_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (delta_row >= delta_count) return;
    HPRLP_FLOAT activity = 0.0;
    for (int entry = row_ptr[delta_row];
         entry < row_ptr[delta_row + 1]; ++entry) {
        activity = fma(values[entry], x_hat[columns[entry]], activity);
    }
    const int row = delta_to_original[delta_row];
    const HPRLP_FLOAT current = y[row];
    const HPRLP_FLOAT value = activity - sigma_params[1] * current;
    const HPRLP_FLOAT delta = fmax(
        lower[row] - value, fmin(upper[row] - value, 0.0));
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    y[row] = fma(
        halpern_factors[1], 2.0 * y_bar - current,
        halpern_factors[0] * last_y[row]);
}

__global__ void initialize_x_bar_mask_kernel(
    std::uint8_t *mask,
    const HPRLP_FLOAT *x_bar,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int *changed,
    int *free_count_by_warp,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int free_count = 0;
    if (index < n) {
        const std::uint8_t old_mask = mask[index];
        const HPRLP_FLOAT value = x_bar[index];
        const std::uint8_t new_mask =
            value == lower[index] ? HPRLP_XBAR_AT_LOWER :
            (value == upper[index] ? HPRLP_XBAR_AT_UPPER :
                                     HPRLP_XBAR_INTERIOR);
        mask[index] = new_mask;
        if (new_mask != old_mask) atomicExch(changed, 1);
        free_count = new_mask == HPRLP_XBAR_INTERIOR ? 1 : 0;
    }
    constexpr unsigned full_mask = 0xffffffffu;
    free_count += __shfl_down_sync(full_mask, free_count, 16);
    free_count += __shfl_down_sync(full_mask, free_count, 8);
    free_count += __shfl_down_sync(full_mask, free_count, 4);
    free_count += __shfl_down_sync(full_mask, free_count, 2);
    free_count += __shfl_down_sync(full_mask, free_count, 1);
    const int lane = threadIdx.x & 31;
    if (lane == 0 && index < n) {
        const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        free_count_by_warp[warp] = free_count;
    }
}

__global__ void count_current_interior_by_warp_kernel(
    const HPRLP_FLOAT *x_bar,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int *free_count_by_warp,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool interior = index < n &&
        x_bar[index] != lower[index] && x_bar[index] != upper[index];
    const unsigned selected = __ballot_sync(0xffffffffu, interior);
    if ((threadIdx.x & 31) == 0 && index < n) {
        free_count_by_warp[index >> 5] = __popc(selected);
    }
}

__global__ void update_x_bar_mask_kernel(
    std::uint8_t *mask,
    const HPRLP_FLOAT *x_bar,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int *changed,
    int *changed_count,
    int *free_count_by_warp,
    int *delta_count,
    int *delta_indices,
    std::uint8_t *delta_old_mask,
    bool record_delta,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int free_count = 0;
    if (index < n) {
        const std::uint8_t old_mask = mask[index];
        std::uint8_t new_mask = old_mask;
        if (old_mask != HPRLP_XBAR_INTERIOR) {
            const HPRLP_FLOAT value = x_bar[index];
            new_mask =
                value == lower[index] ? HPRLP_XBAR_AT_LOWER :
                (value == upper[index] ? HPRLP_XBAR_AT_UPPER :
                                         HPRLP_XBAR_INTERIOR);
            mask[index] = new_mask;
            if (new_mask != old_mask) {
                atomicExch(changed, 1);
                if (record_delta) {
                    atomicAdd(changed_count, 1);
                    if (new_mask == HPRLP_XBAR_INTERIOR) {
                        const int slot = atomicAdd(delta_count, 1);
                        delta_indices[slot] = index;
                        delta_old_mask[slot] = old_mask;
                    }
                }
            }
        }
        free_count = new_mask == HPRLP_XBAR_INTERIOR ? 1 : 0;
    }
    constexpr unsigned full_mask = 0xffffffffu;
    free_count += __shfl_down_sync(full_mask, free_count, 16);
    free_count += __shfl_down_sync(full_mask, free_count, 8);
    free_count += __shfl_down_sync(full_mask, free_count, 4);
    free_count += __shfl_down_sync(full_mask, free_count, 2);
    free_count += __shfl_down_sync(full_mask, free_count, 1);
    const int lane = threadIdx.x & 31;
    if (lane == 0 && index < n) {
        const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        free_count_by_warp[warp] = free_count;
    }
}

__global__ void release_x_bar_mask_kernel(
    std::uint8_t *mask,
    const HPRLP_FLOAT *x_bar,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n || mask[index] == HPRLP_XBAR_INTERIOR) return;
    const HPRLP_FLOAT value = x_bar[index];
    if (value != lower[index] && value != upper[index]) {
        mask[index] = HPRLP_XBAR_INTERIOR;
    }
}

__global__ void selected_csr_row_lengths_kernel(
    int *selected_row_ptr,
    const int *selected_to_original,
    const int *original_row_ptr,
    int selected_count) {
    const int selected_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (selected_row >= selected_count) return;
    const int original_row = selected_to_original[selected_row];
    selected_row_ptr[selected_row + 1] =
        original_row_ptr[original_row + 1] -
        original_row_ptr[original_row];
}

__global__ void build_compact_row_buckets_kernel(
    const int *row_ptr,
    int *row_buckets,
    int *bucket_counts,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const int row_nnz = row_ptr[row + 1] - row_ptr[row];
    if (row_nnz <= HPRLP_SCALAR_ROW_MAX_NNZ) {
        const int slot = atomicAdd(bucket_counts, 1);
        row_buckets[slot] = row;
    } else {
        const int slot = atomicAdd(bucket_counts + 1, 1);
        row_buckets[row_count - 1 - slot] = row;
    }
}

constexpr int kReducedRowAnalysisFields = 9;
constexpr int kReducedRowShortCount = 0;
constexpr int kReducedRowWarpCount = 1;
constexpr int kReducedRowMaxNnz = 2;
constexpr int kReducedRowFirstDegree = 3;
constexpr int kReducedRowNonuniform = 4;
constexpr int kReducedRowEmptyCount = 5;
constexpr int kReducedRowShortNonemptyCount = 6;
constexpr int kReducedRowMediumCount = 7;
constexpr int kReducedRowLongCount = 8;

__global__ void analyze_compact_rows_kernel(
    const int *row_ptr,
    int *row_buckets,
    int *analysis,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const int row_nnz = row_ptr[row + 1] - row_ptr[row];
    const int first_degree = row_ptr[1] - row_ptr[0];
    if (row == 0) analysis[kReducedRowFirstDegree] = row_nnz;
    if (row_nnz != first_degree) {
        atomicExch(analysis + kReducedRowNonuniform, 1);
    }
    atomicMax(analysis + kReducedRowMaxNnz, row_nnz);
    if (row_nnz <= HPRLP_SCALAR_ROW_MAX_NNZ) {
        const int slot = atomicAdd(analysis + kReducedRowShortCount, 1);
        row_buckets[slot] = row;
        if (row_nnz == 0) {
            atomicAdd(analysis + kReducedRowEmptyCount, 1);
        } else {
            atomicAdd(analysis + kReducedRowShortNonemptyCount, 1);
        }
    } else {
        const int slot = atomicAdd(analysis + kReducedRowWarpCount, 1);
        row_buckets[row_count - 1 - slot] = row;
        if (row_nnz <= HPRLP_WARP_ROW_MAX_NNZ) {
            atomicAdd(analysis + kReducedRowMediumCount, 1);
        } else {
            atomicAdd(analysis + kReducedRowLongCount, 1);
        }
    }
}

__global__ void collect_compact_medium_long_rows_kernel(
    const int *row_ptr,
    int *row_buckets,
    int *bucket_counts,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const int row_nnz = row_ptr[row + 1] - row_ptr[row];
    if (row_nnz > HPRLP_SCALAR_ROW_MAX_NNZ &&
        row_nnz <= HPRLP_WARP_ROW_MAX_NNZ) {
        const int slot = atomicAdd(bucket_counts, 1);
        row_buckets[slot] = row;
    } else if (row_nnz > HPRLP_WARP_ROW_MAX_NNZ) {
        const int slot = atomicAdd(bucket_counts + 1, 1);
        row_buckets[row_count - 1 - slot] = row;
    }
}

__global__ void collect_compact_empty_short_rows_kernel(
    const int *row_ptr,
    int *row_buckets,
    int *bucket_counts,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const int row_nnz = row_ptr[row + 1] - row_ptr[row];
    if (row_nnz > 0 && row_nnz <= HPRLP_SCALAR_ROW_MAX_NNZ) {
        const int slot = atomicAdd(bucket_counts, 1);
        row_buckets[slot] = row;
    } else if (row_nnz == 0) {
        const int slot = atomicAdd(bucket_counts + 1, 1);
        row_buckets[row_count - 1 - slot] = row;
    }
}

__global__ void mark_nonempty_csr_rows_kernel(
    const int *row_ptr,
    std::uint32_t *nonempty_words,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || row_ptr[row] == row_ptr[row + 1]) return;
    atomicOr(nonempty_words + (row >> 5), 1u << (row & 31));
}

__global__ void build_delta_only_row_list_kernel(
    const int *base_row_ptr,
    const int *delta_row_ptr,
    int *delta_only_rows,
    int *delta_only_count,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count ||
        base_row_ptr[row] != base_row_ptr[row + 1] ||
        delta_row_ptr[row] == delta_row_ptr[row + 1]) {
        return;
    }
    const int slot = atomicAdd(delta_only_count, 1);
    delta_only_rows[slot] = row;
}

__global__ void count_interior_columns_by_warp_kernel(
    const std::uint8_t *mask,
    int *counts_by_warp,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool interior =
        index < n && mask[index] == HPRLP_XBAR_INTERIOR;
    const unsigned selected = __ballot_sync(0xffffffffu, interior);
    if ((threadIdx.x & 31) == 0 && index < n) {
        counts_by_warp[index >> 5] = __popc(selected);
    }
}

__global__ void scatter_interior_column_indices_kernel(
    const std::uint8_t *mask,
    const int *offsets_by_warp,
    int *selected_indices,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool interior =
        index < n && mask[index] == HPRLP_XBAR_INTERIOR;
    const unsigned selected = __ballot_sync(0xffffffffu, interior);
    if (!interior) return;
    const int lane = threadIdx.x & 31;
    const unsigned lanes_before = lane == 0 ? 0u : ((1u << lane) - 1u);
    const int rank = __popc(selected & lanes_before);
    selected_indices[offsets_by_warp[index >> 5] + rank] = index;
}

__global__ void gather_reduced_state_kernel(
    HPRLP_FLOAT *x_reduced,
    HPRLP_FLOAT *x_bar_reduced,
    HPRLP_FLOAT *x_hat_reduced,
    HPRLP_FLOAT *last_x_reduced,
    HPRLP_FLOAT *lower_reduced,
    HPRLP_FLOAT *upper_reduced,
    HPRLP_FLOAT *objective_reduced,
    std::uint8_t *bound_type_reduced,
    const HPRLP_FLOAT *x_full,
    const HPRLP_FLOAT *x_bar_full,
    const HPRLP_FLOAT *x_hat_full,
    const HPRLP_FLOAT *last_x_full,
    const HPRLP_FLOAT *lower_full,
    const HPRLP_FLOAT *upper_full,
    const HPRLP_FLOAT *objective_full,
    const std::uint8_t *bound_type_full,
    const int *free_to_original,
    int free_count) {
    const int reduced_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (reduced_index >= free_count) return;
    const int original_index = free_to_original[reduced_index];
    x_reduced[reduced_index] = x_full[original_index];
    x_bar_reduced[reduced_index] = x_bar_full[original_index];
    x_hat_reduced[reduced_index] = x_hat_full[original_index];
    last_x_reduced[reduced_index] = last_x_full[original_index];
    lower_reduced[reduced_index] = lower_full[original_index];
    upper_reduced[reduced_index] = upper_full[original_index];
    objective_reduced[reduced_index] = objective_full[original_index];
    bound_type_reduced[reduced_index] = bound_type_full[original_index];
}

__global__ void scatter_reduced_state_kernel(
    HPRLP_FLOAT *x_full,
    HPRLP_FLOAT *x_bar_full,
    HPRLP_FLOAT *x_hat_full,
    const HPRLP_FLOAT *x_reduced,
    const HPRLP_FLOAT *x_bar_reduced,
    const HPRLP_FLOAT *x_hat_reduced,
    const int *free_to_original,
    int free_count) {
    const int reduced_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (reduced_index >= free_count) return;
    const int original_index = free_to_original[reduced_index];
    x_full[original_index] = x_reduced[reduced_index];
    x_bar_full[original_index] = x_bar_reduced[reduced_index];
    x_hat_full[original_index] = x_hat_reduced[reduced_index];
}

__global__ void scatter_fixed_bounds_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_bar,
    HPRLP_FLOAT *x_hat,
    HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *mask,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    HPRLP_FLOAT value;
    if (mask[index] == HPRLP_XBAR_AT_LOWER) {
        value = lower[index];
    } else if (mask[index] == HPRLP_XBAR_AT_UPPER) {
        value = upper[index];
    } else {
        return;
    }
    x[index] = value;
    x_bar[index] = value;
    x_hat[index] = value;
    last_x[index] = value;
}

__global__ void fixed_values_from_mask_kernel(
    HPRLP_FLOAT *fixed_values,
    const std::uint8_t *mask,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    fixed_values[index] =
        mask[index] == HPRLP_XBAR_AT_LOWER ? lower[index] :
        (mask[index] == HPRLP_XBAR_AT_UPPER ? upper[index] : 0.0);
}

__global__ void copy_selected_csr_rows_kernel(
    int *reduced_columns,
    HPRLP_FLOAT *reduced_values,
    const int *reduced_row_ptr,
    const int *free_to_original,
    const int *original_row_ptr,
    const int *original_columns,
    const HPRLP_FLOAT *original_values,
    int free_count) {
    const int reduced_row = blockIdx.x;
    if (reduced_row >= free_count) return;
    const int original_row = free_to_original[reduced_row];
    const int original_begin = original_row_ptr[original_row];
    const int original_end = original_row_ptr[original_row + 1];
    const int reduced_begin = reduced_row_ptr[reduced_row];
    for (int offset = threadIdx.x;
         original_begin + offset < original_end;
         offset += blockDim.x) {
        reduced_columns[reduced_begin + offset] =
            original_columns[original_begin + offset];
        reduced_values[reduced_begin + offset] =
            original_values[original_begin + offset];
    }
}

__global__ void compact_u16_indices_kernel(
    std::uint16_t *compact_indices,
    const int *csr_indices,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        compact_indices[index] =
            static_cast<std::uint16_t>(csr_indices[index]);
    }
}

__global__ void pack_signed_split_u16_kernel(
    std::uint16_t *compact_indices,
    std::uint8_t *compact_negative,
    const int *csr_indices,
    const HPRLP_FLOAT *csr_values,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        compact_indices[index] =
            static_cast<std::uint16_t>(csr_indices[index]);
        compact_negative[index] = csr_values[index] < 0.0 ? 1 : 0;
    }
}

__global__ void pack_signed_entries_u16_kernel(
    std::uint16_t *packed_entries,
    const int *csr_indices,
    const HPRLP_FLOAT *csr_values,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        packed_entries[index] =
            static_cast<std::uint16_t>(csr_indices[index]) |
            (csr_values[index] < 0.0 ? 0x8000u : 0u);
    }
}

__global__ void pack_signed_entries_u32_kernel(
    std::uint32_t *packed_entries,
    const int *csr_indices,
    const HPRLP_FLOAT *csr_values,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        packed_entries[index] =
            static_cast<std::uint32_t>(csr_indices[index]) |
            (csr_values[index] < 0.0 ? 0x80000000u : 0u);
    }
}

__global__ void copy_selected_packed_dictionary_rows_kernel(
    std::uint32_t *compact_entries,
    const int *compact_row_ptr,
    const int *selected_to_original,
    const int *original_row_ptr,
    const std::uint32_t *original_entries,
    int selected_row_count) {
    const int selected_row = blockIdx.x;
    if (selected_row >= selected_row_count) return;
    const int original_row = selected_to_original[selected_row];
    const int compact_begin = compact_row_ptr[selected_row];
    const int original_begin = original_row_ptr[original_row];
    const int count = compact_row_ptr[selected_row + 1] - compact_begin;
    for (int offset = threadIdx.x; offset < count; offset += blockDim.x) {
        compact_entries[compact_begin + offset] =
            original_entries[original_begin + offset];
    }
}

__global__ void scatter_original_to_selected_rows_kernel(
    int *original_to_selected,
    const int *selected_to_original,
    int selected_count) {
    const int selected_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (selected_row < selected_count) {
        original_to_selected[selected_to_original[selected_row]] =
            selected_row;
    }
}

// Row reduction filters the constraint index of AT while leaving every x
// row intact.  Preserve the full operator's dictionary code exactly and
// rewrite only the packed constraint index into compact-row coordinates.
__global__ void filter_packed_dictionary_AT_rows_kernel(
    std::uint32_t *compact_entries,
    const int *compact_row_ptr,
    const int *original_row_ptr,
    const std::uint32_t *original_entries,
    const int *original_to_selected,
    unsigned code_bits,
    int row_count) {
    const int row = blockIdx.x;
    if (row >= row_count || threadIdx.x != 0) return;
    const std::uint32_t code_mask = code_bits == 0
        ? 0u : ((UINT32_C(1) << code_bits) - 1u);
    int output = compact_row_ptr[row];
    for (int entry = original_row_ptr[row];
         entry < original_row_ptr[row + 1]; ++entry) {
        const std::uint32_t packed = original_entries[entry];
        const int original_constraint =
            static_cast<int>(packed >> code_bits);
        const int selected_constraint =
            original_to_selected[original_constraint];
        if (selected_constraint >= 0) {
            compact_entries[output++] =
                (static_cast<std::uint32_t>(selected_constraint)
                 << code_bits) |
                (packed & code_mask);
        }
    }
}

__global__ void gather_reduced_row_backend_state_kernel(
    HPRLP_FLOAT *compact_inverse_row_norm,
    HPRLP_FLOAT *compact_scaled_y,
    std::uint8_t *compact_bound_type,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *y,
    const std::uint8_t *bound_type,
    const int *active_to_original,
    int active_count) {
    const int compact_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_row >= active_count) return;
    const int original_row = active_to_original[compact_row];
    const HPRLP_FLOAT inverse_norm = inverse_row_norm[original_row];
    if (compact_inverse_row_norm != nullptr) {
        compact_inverse_row_norm[compact_row] = inverse_norm;
    }
    if (compact_scaled_y != nullptr) {
        compact_scaled_y[compact_row] = y[original_row] * inverse_norm;
    }
    if (compact_bound_type != nullptr) {
        compact_bound_type[compact_row] = bound_type[original_row];
    }
}

__global__ void repack_uniform_degree_dictionary_rows_soa_kernel(
    const std::uint32_t *source,
    std::uint32_t *destination,
    int row_count,
    int degree,
    int entry_count) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= entry_count) return;
    const int position = output_index / row_count;
    const int row = output_index - position * row_count;
    destination[output_index] = source[row * degree + position];
}

__global__ void build_dictionary_code_translation_kernel(
    std::uint32_t *translation,
    const HPRLP_FLOAT *source_dictionary,
    int source_dictionary_size,
    const HPRLP_FLOAT *target_dictionary,
    int target_dictionary_size) {
    const int source_code = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_code >= source_dictionary_size) return;
    const unsigned long long source_bits =
        __double_as_longlong(source_dictionary[source_code]);
    std::uint32_t target_code = UINT32_MAX;
    for (int candidate = 0; candidate < target_dictionary_size;
         ++candidate) {
        if (__double_as_longlong(target_dictionary[candidate]) ==
            source_bits) {
            target_code = static_cast<std::uint32_t>(candidate);
            break;
        }
    }
    translation[source_code] = target_code;
}

__global__ void pack_compact_dictionary_A_from_AT_kernel(
    std::uint32_t *compact_entries,
    const int *compact_A_row_ptr,
    const int *compact_A_col_index,
    const int *compact_AT_row_ptr,
    const int *compact_AT_col_index,
    const std::uint32_t *compact_AT_entries,
    const std::uint32_t *code_translation,
    unsigned AT_code_bits,
    unsigned A_code_bits,
    int row_count) {
    const int row = blockIdx.x;
    if (row >= row_count) return;
    const std::uint32_t AT_code_mask = AT_code_bits == 0
        ? 0u
        : ((UINT32_C(1) << AT_code_bits) - 1u);
    const std::uint32_t A_code_mask = A_code_bits == 0
        ? 0u
        : ((UINT32_C(1) << A_code_bits) - 1u);
    for (int compact_entry = compact_A_row_ptr[row] + threadIdx.x;
         compact_entry < compact_A_row_ptr[row + 1];
         compact_entry += blockDim.x) {
        const int compact_col = compact_A_col_index[compact_entry];
        int left = compact_AT_row_ptr[compact_col];
        int right = compact_AT_row_ptr[compact_col + 1];
        const int source_end = right;
        while (left < right) {
            const int middle = left + (right - left) / 2;
            if (compact_AT_col_index[middle] < row) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        const std::uint32_t source_code =
            left < source_end && compact_AT_col_index[left] == row
            ? compact_AT_entries[left] & AT_code_mask : UINT32_MAX;
        const std::uint32_t translated_code = source_code != UINT32_MAX
            ? code_translation[source_code] : UINT32_MAX;
        const std::uint32_t code = translated_code != UINT32_MAX
            ? translated_code & A_code_mask : 0u;
        compact_entries[compact_entry] =
            (static_cast<std::uint32_t>(compact_col) << A_code_bits) | code;
    }
}

__global__ void gather_selected_values_kernel(
    HPRLP_FLOAT *selected_values,
    const HPRLP_FLOAT *full_values,
    const int *selected_to_original,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        selected_values[index] = full_values[selected_to_original[index]];
    }
}

__global__ void build_scaled_fixed_delta_kernel(
    HPRLP_FLOAT *scaled_fixed,
    const HPRLP_FLOAT *inverse_col_norm,
    const std::uint8_t *old_mask,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const HPRLP_FLOAT fixed =
        old_mask[index] == HPRLP_XBAR_AT_LOWER ? lower[index] :
        (old_mask[index] == HPRLP_XBAR_AT_UPPER ? upper[index] : 0.0);
    scaled_fixed[index] = fixed * inverse_col_norm[index];
}

__global__ void build_unscaled_delta_input_kernel(
    HPRLP_FLOAT *input,
    const HPRLP_FLOAT *x_hat,
    const std::uint8_t *old_mask,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const HPRLP_FLOAT fixed =
        old_mask[index] == HPRLP_XBAR_AT_LOWER ? lower[index] :
        (old_mask[index] == HPRLP_XBAR_AT_UPPER ? upper[index] : 0.0);
    input[index] = x_hat[index] - fixed;
}

__global__ void add_fixed_shift_kernel(
    HPRLP_FLOAT *values,
    const HPRLP_FLOAT *shift,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] += shift[index];
}


__global__ void update_reduced_x_from_aty_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *aty,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const HPRLP_FLOAT current = x[index];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        current + sigma_params[0] * (aty[index] - objective[index]),
        lower[index], upper[index], bound_type[index]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[index] = fma(halpern_factors[1], reflected,
                   halpern_factors[0] * last_x[index]);
    x_hat[index] = reflected;
}


__global__ void fused_reduced_x_short_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int row = row_ids[item];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum = fma(values[entry], y[columns[entry]], sum);
    }
    const HPRLP_FLOAT current = x[row];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        fma(sigma_params[0], sum - objective[row], current),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
}

__global__ void fused_reduced_x_warp_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= row_count) return;
    const int row = row_ids[warp];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + lane;
         entry < row_ptr[row + 1]; entry += 32) {
        sum = fma(values[entry], y[columns[entry]], sum);
    }
    constexpr unsigned mask = 0xffffffffu;
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (lane != 0) return;
    const HPRLP_FLOAT current = x[row];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        fma(sigma_params[0], sum - objective[row], current),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
}

__global__ void fused_reduced_delta_x_short_with_input_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    HPRLP_FLOAT *delta_input,
    const HPRLP_FLOAT *delta_fixed,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int row = row_ids[item];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum = fma(values[entry], y[columns[entry]], sum);
    }
    const HPRLP_FLOAT current = x[row];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        fma(sigma_params[0], sum - objective[row], current),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
    delta_input[row] = reflected - delta_fixed[row];
}

__global__ void fused_reduced_delta_x_warp_with_input_kernel(
    HPRLP_FLOAT *x,
    HPRLP_FLOAT *x_hat,
    HPRLP_FLOAT *delta_input,
    const HPRLP_FLOAT *delta_fixed,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective,
    const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= row_count) return;
    const int row = row_ids[warp];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + lane;
         entry < row_ptr[row + 1]; entry += 32) {
        sum = fma(values[entry], y[columns[entry]], sum);
    }
    constexpr unsigned mask = 0xffffffffu;
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (lane != 0) return;
    const HPRLP_FLOAT current = x[row];
    const HPRLP_FLOAT projected = project_x_with_bounds(
        fma(sigma_params[0], sum - objective[row], current),
        lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT reflected = 2.0 * projected - current;
    x[row] = fma(halpern_factors[1], reflected,
                 halpern_factors[0] * last_x[row]);
    x_hat[row] = reflected;
    delta_input[row] = reflected - delta_fixed[row];
}

__device__ __forceinline__ HPRLP_FLOAT reduced_y_update(
    HPRLP_FLOAT current,
    HPRLP_FLOAT activity,
    HPRLP_FLOAT lower,
    HPRLP_FLOAT upper,
    std::uint8_t bound_type,
    HPRLP_FLOAT last,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const HPRLP_FLOAT value = fma(-sigma_params[1], current, activity);
    const HPRLP_FLOAT delta =
        project_y_delta(value, lower, upper, bound_type);
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    return fma(halpern_factors[1], 2.0 * y_bar - current,
               halpern_factors[0] * last);
}

__global__ void update_reduced_full_order_y_from_compact_ax_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *compact_ax,
    const int *original_to_nonempty,
    const HPRLP_FLOAT *fixed_shift,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *delta_input,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_fixed,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const HPRLP_FLOAT *delta_ax,
    const std::uint32_t *delta_nonempty_words,
    bool use_delta,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const int compact_position = original_to_nonempty[row];
    const bool delta_nonempty = use_delta &&
        (delta_nonempty_words[row >> 5] & (1u << (row & 31))) != 0;
    HPRLP_FLOAT activity = fixed_shift[row];
    if (compact_position >= 0) activity += compact_ax[compact_position];
    if (delta_ax != nullptr && delta_nonempty) {
        activity += delta_ax[row];
    } else if (delta_nonempty) {
        for (int entry = delta_row_ptr[row];
             entry < delta_row_ptr[row + 1]; ++entry) {
            const int column = delta_columns[entry];
            const HPRLP_FLOAT input = delta_input != nullptr
                ? delta_input[column]
                : delta_x_hat[column] - delta_fixed[column];
            activity = fma(
                delta_values[entry], input, activity);
        }
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    if (scaled_y != nullptr) {
        scaled_y[row] = updated * inverse_row_norm[row];
    }
}

__global__ void update_reduced_nonempty_y_from_compact_ax_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *compact_ax,
    const int *nonempty_to_original,
    const HPRLP_FLOAT *fixed_shift,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *delta_input,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_fixed,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const HPRLP_FLOAT *delta_ax,
    const std::uint32_t *delta_nonempty_words,
    bool use_delta,
    int nonempty_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int compact_position = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_position >= nonempty_count) return;
    const int row = nonempty_to_original[compact_position];
    const bool delta_nonempty = use_delta &&
        (delta_nonempty_words[row >> 5] & (1u << (row & 31))) != 0;
    HPRLP_FLOAT activity =
        fixed_shift[row] + compact_ax[compact_position];
    if (delta_ax != nullptr && delta_nonempty) {
        activity += delta_ax[row];
    } else if (delta_nonempty) {
        for (int entry = delta_row_ptr[row];
             entry < delta_row_ptr[row + 1]; ++entry) {
            const int column = delta_columns[entry];
            const HPRLP_FLOAT input = delta_input != nullptr
                ? delta_input[column]
                : delta_x_hat[column] - delta_fixed[column];
            activity = fma(delta_values[entry], input, activity);
        }
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    if (scaled_y != nullptr) {
        scaled_y[row] = updated * inverse_row_norm[row];
    }
}

__global__ void update_reduced_delta_only_y_for_nonempty_cusparse_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *fixed_shift,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *delta_input,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_fixed,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const HPRLP_FLOAT *delta_ax,
    const int *row_ids,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= row_count) return;
    const int row = row_ids[position];
    HPRLP_FLOAT activity = fixed_shift[row];
    if (delta_ax != nullptr) {
        activity += delta_ax[row];
    } else {
        for (int entry = delta_row_ptr[row];
             entry < delta_row_ptr[row + 1]; ++entry) {
            const int column = delta_columns[entry];
            const HPRLP_FLOAT input = delta_input != nullptr
                ? delta_input[column]
                : delta_x_hat[column] - delta_fixed[column];
            activity = fma(delta_values[entry], input, activity);
        }
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    if (scaled_y != nullptr) {
        scaled_y[row] = updated * inverse_row_norm[row];
    }
}

__device__ __forceinline__ HPRLP_FLOAT reduced_signed_delta_input(
    std::uint32_t entry,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed) {
    const int column = static_cast<int>(entry & 0x7fffffffu);
    const HPRLP_FLOAT value =
        delta_scaled_x_hat[column] - delta_scaled_fixed[column];
    return (entry & 0x80000000u) != 0 ? -value : value;
}

__global__ void update_reduced_signed_delta_only_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *fixed_shift,
    const HPRLP_FLOAT *delta_scaled_x_hat,
    const HPRLP_FLOAT *delta_scaled_fixed,
    const int *delta_row_ptr,
    const std::uint32_t *delta_entries,
    const int *row_ids,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= row_count) return;
    const int row = row_ids[position];
    HPRLP_FLOAT delta_sum = 0.0;
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        delta_sum += reduced_signed_delta_input(
            delta_entries[entry], delta_scaled_x_hat, delta_scaled_fixed);
    }
    HPRLP_FLOAT activity = fixed_shift[row];
    activity += delta_sum * inverse_row_norm[row];
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    scaled_y[row] = updated * inverse_row_norm[row];
}

__global__ void update_reduced_dictionary_delta_only_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *fixed_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    const int *row_ids,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= row_count) return;
    const int row = row_ids[position];
    HPRLP_FLOAT activity = fixed_shift[row];
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        const int column = delta_columns[entry];
        const HPRLP_FLOAT fixed =
            delta_old_mask[column] == HPRLP_XBAR_AT_LOWER
                ? delta_lower[column]
                : (delta_old_mask[column] == HPRLP_XBAR_AT_UPPER
                       ? delta_upper[column]
                       : 0.0);
        activity = fma(
            delta_values[entry], delta_x_hat[column] - fixed, activity);
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    if (scaled_y != nullptr) {
        scaled_y[row] = updated * inverse_row_norm[row];
    }
}

template <int BatchSize>
__global__ void update_reduced_signed_empty_y_batch_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *fixed_shift,
    const std::uint32_t *delta_nonempty_words,
    const int *row_ids,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factor_batch) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= row_count) return;
    const int row = row_ids[position];
    if (delta_nonempty_words != nullptr &&
        (delta_nonempty_words[row >> 5] & (1u << (row & 31))) != 0) {
        return;
    }
    HPRLP_FLOAT current = y[row];
    const HPRLP_FLOAT activity = fixed_shift[row];
    const HPRLP_FLOAT row_lower = lower[row];
    const HPRLP_FLOAT row_upper = upper[row];
    const std::uint8_t row_bound_type = bound_type[row];
    const HPRLP_FLOAT anchor = last_y[row];
    #pragma unroll
    for (int iteration = 0; iteration < BatchSize; ++iteration) {
        current = reduced_y_update(
            current, activity, row_lower, row_upper, row_bound_type, anchor,
            sigma_params, halpern_factor_batch + 2 * iteration);
    }
    y[row] = current;
    scaled_y[row] = current * inverse_row_norm[row];
}

template <int BatchSize>
__global__ void update_reduced_dictionary_empty_y_batch_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *fixed_shift,
    const int *base_row_ptr,
    const std::uint32_t *delta_nonempty_words,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factor_batch) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || base_row_ptr[row] != base_row_ptr[row + 1]) {
        return;
    }
    if (delta_nonempty_words != nullptr &&
        (delta_nonempty_words[row >> 5] & (1u << (row & 31))) != 0) {
        return;
    }
    HPRLP_FLOAT current = y[row];
    const HPRLP_FLOAT activity = fixed_shift[row];
    const HPRLP_FLOAT row_lower = lower[row];
    const HPRLP_FLOAT row_upper = upper[row];
    const std::uint8_t row_bound_type = bound_type[row];
    const HPRLP_FLOAT anchor = last_y[row];
    #pragma unroll
    for (int iteration = 0; iteration < BatchSize; ++iteration) {
        current = reduced_y_update(
            current, activity, row_lower, row_upper, row_bound_type, anchor,
            sigma_params, halpern_factor_batch + 2 * iteration);
    }
    y[row] = current;
    if (scaled_y != nullptr) {
        scaled_y[row] = current * inverse_row_norm[row];
    }
}

__global__ void update_reduced_deferred_empty_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *fixed_shift,
    const std::uint32_t *delta_nonempty_words,
    const int *row_ids,
    int row_count,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *deferred_halpern_factors,
    int deferred_iterations) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= row_count || deferred_iterations <= 0) return;
    const int row = row_ids[position];
    if (delta_nonempty_words != nullptr &&
        (delta_nonempty_words[row >> 5] & (1u << (row & 31))) != 0) {
        return;
    }

    HPRLP_FLOAT current = y[row];
    const HPRLP_FLOAT activity = fixed_shift[row];
    const HPRLP_FLOAT row_lower = lower[row];
    const HPRLP_FLOAT row_upper = upper[row];
    const std::uint8_t row_bound_type = bound_type[row];
    const HPRLP_FLOAT anchor = last_y[row];
    for (int offset = 0; offset < deferred_iterations; ++offset) {
        const HPRLP_FLOAT value =
            fma(-sigma_params[1], current, activity);
        const HPRLP_FLOAT delta = project_y_delta(
            value, row_lower, row_upper, row_bound_type);
        const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
        const HPRLP_FLOAT factor1 =
            deferred_halpern_factors[2 * offset];
        current = fma(
            deferred_halpern_factors[2 * offset + 1],
            2.0 * y_bar - current, factor1 * anchor);
    }
    y[row] = current;
    if (scaled_y != nullptr) {
        scaled_y[row] = current * inverse_row_norm[row];
    }
}

__global__ void prepare_reduced_deferred_halpern_factors_kernel(
    const int *halpern_inner_after,
    HPRLP_FLOAT *deferred_halpern_factors,
    int deferred_iterations) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int first_inner = halpern_inner_after[0] - deferred_iterations;
    for (int offset = 0; offset < deferred_iterations; ++offset) {
        const HPRLP_FLOAT factor1 =
            1.0 / (static_cast<HPRLP_FLOAT>(first_inner + offset) + 2.0);
        deferred_halpern_factors[2 * offset] = factor1;
        deferred_halpern_factors[2 * offset + 1] = 1.0 - factor1;
    }
}

__device__ __forceinline__ HPRLP_FLOAT reduced_fixed_value(
    std::uint8_t old_mask, HPRLP_FLOAT lower, HPRLP_FLOAT upper) {
    return old_mask == HPRLP_XBAR_AT_LOWER ? lower :
        (old_mask == HPRLP_XBAR_AT_UPPER ? upper : 0.0);
}

__device__ __forceinline__ HPRLP_FLOAT add_reduced_delta_activity(
    int row, HPRLP_FLOAT activity,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values) {
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        const int column = delta_columns[entry];
        const HPRLP_FLOAT fixed = reduced_fixed_value(
            delta_old_mask[column], delta_lower[column],
            delta_upper[column]);
        activity = fma(delta_values[entry],
                       delta_x_hat[column] - fixed, activity);
    }
    return activity;
}

__global__ void reduced_unit_factorized_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scaled_x_hat,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *fixed_shift,
    const int *row_ptr,
    const int *column_indices,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    bool use_delta,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign,
    int row_count) {
    const int row = blockIdx.x;
    if (row >= row_count) return;

    constexpr int kWarpSize = 32;
    constexpr int kMaxWarpsPerBlock = 32;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        sum += scaled_x_hat[column_indices[entry]];
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    __shared__ HPRLP_FLOAT warp_sums[kMaxWarpsPerBlock];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp != 0) return;

    sum = lane < warps_per_block ? warp_sums[lane] : 0.0;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane != 0) return;

    HPRLP_FLOAT activity = sum * inverse_row_norm[row];
    if (uniform_unit_sign < 0) activity = -activity;
    activity += fixed_shift[row];
    if (use_delta) {
        activity = add_reduced_delta_activity(
            row, activity, delta_x_hat, delta_lower, delta_upper,
            delta_old_mask, delta_row_ptr, delta_columns, delta_values);
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    scaled_y[row] = updated * inverse_row_norm[row];
}

__global__ void reduced_unit_active_scatter_y_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *scattered_activity,
    const HPRLP_FLOAT *inverse_row_norm,
    HPRLP_FLOAT *scaled_y,
    const HPRLP_FLOAT *fixed_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    bool use_delta,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int uniform_unit_sign,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    HPRLP_FLOAT activity =
        scattered_activity[row] * inverse_row_norm[row];
    if (uniform_unit_sign < 0) activity = -activity;
    activity += fixed_shift[row];
    if (use_delta) {
        activity = add_reduced_delta_activity(
            row, activity, delta_x_hat, delta_lower, delta_upper,
            delta_old_mask, delta_row_ptr, delta_columns, delta_values);
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    scaled_y[row] = updated * inverse_row_norm[row];
}

__global__ void update_reduced_y_delta_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *base_activity,
    const HPRLP_FLOAT *fixed_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    HPRLP_FLOAT *scaled_y_output,
    const HPRLP_FLOAT *inverse_row_norm,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    HPRLP_FLOAT activity = base_activity[row] + fixed_shift[row];
    for (int entry = delta_row_ptr[row];
         entry < delta_row_ptr[row + 1]; ++entry) {
        const int column = delta_columns[entry];
        const HPRLP_FLOAT fixed = reduced_fixed_value(
            delta_old_mask[column], delta_lower[column],
            delta_upper[column]);
        activity = fma(delta_values[entry],
                       delta_x_hat[column] - fixed, activity);
    }
    const HPRLP_FLOAT updated = reduced_y_update(
        y[row], activity, lower[row], upper[row], bound_type[row],
        last_y[row], sigma_params, halpern_factors);
    y[row] = updated;
    if (scaled_y_output != nullptr) {
        scaled_y_output[row] = updated * inverse_row_norm[row];
    }
}

__global__ void fused_reduced_y_short_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *fixed_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    bool use_delta,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= row_count) return;
    const int row = row_ids[item];
    HPRLP_FLOAT sum = fixed_shift[row];
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        sum = fma(values[entry], x_hat[columns[entry]], sum);
    }
    if (use_delta) {
        for (int entry = delta_row_ptr[row];
             entry < delta_row_ptr[row + 1]; ++entry) {
            const int column = delta_columns[entry];
            const HPRLP_FLOAT fixed = reduced_fixed_value(
                delta_old_mask[column], delta_lower[column],
                delta_upper[column]);
            sum = fma(delta_values[entry],
                      delta_x_hat[column] - fixed, sum);
        }
    }
    y[row] = reduced_y_update(y[row], sum, lower[row], upper[row],
                              bound_type[row], last_y[row], sigma_params,
                              halpern_factors);
}

__global__ void fused_reduced_y_warp_kernel(
    HPRLP_FLOAT *y,
    const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type,
    const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat,
    const int *row_ptr,
    const int *columns,
    const HPRLP_FLOAT *values,
    const HPRLP_FLOAT *fixed_shift,
    const HPRLP_FLOAT *delta_x_hat,
    const HPRLP_FLOAT *delta_lower,
    const HPRLP_FLOAT *delta_upper,
    const std::uint8_t *delta_old_mask,
    const int *delta_row_ptr,
    const int *delta_columns,
    const HPRLP_FLOAT *delta_values,
    bool use_delta,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors,
    const int *row_ids,
    int row_count) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= row_count) return;
    const int row = row_ids[warp];
    HPRLP_FLOAT sum = 0.0;
    for (int entry = row_ptr[row] + lane;
         entry < row_ptr[row + 1]; entry += 32) {
        sum = fma(values[entry], x_hat[columns[entry]], sum);
    }
    if (use_delta) {
        for (int entry = delta_row_ptr[row] + lane;
             entry < delta_row_ptr[row + 1]; entry += 32) {
            const int column = delta_columns[entry];
            const HPRLP_FLOAT fixed = reduced_fixed_value(
                delta_old_mask[column], delta_lower[column],
                delta_upper[column]);
            sum = fma(delta_values[entry],
                      delta_x_hat[column] - fixed, sum);
        }
    }
    constexpr unsigned mask = 0xffffffffu;
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if (lane != 0) return;
    sum += fixed_shift[row];
    y[row] = reduced_y_update(y[row], sum, lower[row], upper[row],
                              bound_type[row], last_y[row], sigma_params,
                              halpern_factors);
}

template <typename T>
void allocate_device(T **pointer, std::size_t count) {
    *pointer = nullptr;
    if (count > 0) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(pointer), count * sizeof(T)));
    }
}

template <typename T>
void allocate_device_async(
    T **pointer, std::size_t count, cudaStream_t stream) {
    *pointer = nullptr;
    if (count > 0) {
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void **>(pointer), count * sizeof(T), stream));
    }
}

void ensure_device_byte_capacity(
    void **buffer,
    std::size_t *capacity,
    std::size_t required_bytes) {
    if (required_bytes <= *capacity) return;
    cudaFree(*buffer);
    *buffer = nullptr;
    *capacity = 0;
    if (required_bytes > 0) {
        CUDA_CHECK(cudaMalloc(buffer, required_bytes));
        *capacity = required_bytes;
    }
}

void free_sparse_matrix(sparseMatrix *matrix) {
    if (matrix == nullptr) return;
    cudaFree(matrix->rowPtr);
    cudaFree(matrix->colIndex);
    cudaFree(matrix->value);
    *matrix = sparseMatrix{};
}

void free_sparse_matrix_async(
    sparseMatrix *matrix, cudaStream_t stream) {
    if (matrix == nullptr) return;
    if (matrix->rowPtr) CUDA_CHECK(cudaFreeAsync(matrix->rowPtr, stream));
    if (matrix->colIndex) {
        CUDA_CHECK(cudaFreeAsync(matrix->colIndex, stream));
    }
    if (matrix->value) CUDA_CHECK(cudaFreeAsync(matrix->value, stream));
    *matrix = sparseMatrix{};
}

void transpose_csr(
    const sparseMatrix &source,
    sparseMatrix *transpose,
    cudaStream_t stream,
    cusparseHandle_t handle,
    void **construction_temp,
    std::size_t *construction_temp_capacity,
    bool stream_ordered_allocations = false) {
    *transpose = sparseMatrix{};
    transpose->row = source.col;
    transpose->col = source.row;
    transpose->numElements = source.numElements;
    if (stream_ordered_allocations) {
        allocate_device_async(
            &transpose->rowPtr,
            static_cast<std::size_t>(transpose->row) + 1, stream);
    } else {
        allocate_device(&transpose->rowPtr,
                        static_cast<std::size_t>(transpose->row) + 1);
    }
    if (source.numElements == 0) {
        CUDA_CHECK(cudaMemsetAsync(
            transpose->rowPtr, 0,
            (static_cast<std::size_t>(transpose->row) + 1) * sizeof(int),
            stream));
        return;
    }
    if (stream_ordered_allocations) {
        allocate_device_async(
            &transpose->colIndex, source.numElements, stream);
        allocate_device_async(&transpose->value, source.numElements, stream);
    } else {
        allocate_device(&transpose->colIndex, source.numElements);
        allocate_device(&transpose->value, source.numElements);
    }

    check_cusparse(cusparseSetStream(handle, stream),
                   "cusparseSetStream transpose");
    std::size_t buffer_size = 0;
    constexpr cusparseCsr2CscAlg_t algorithm = CUSPARSE_CSR2CSC_ALG1;
    check_cusparse(cusparseCsr2cscEx2_bufferSize(
        handle, source.row, source.col, source.numElements,
        source.value, source.rowPtr, source.colIndex,
        transpose->value, transpose->rowPtr, transpose->colIndex,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        algorithm, &buffer_size), "transpose buffer size");
    ensure_device_byte_capacity(
        construction_temp, construction_temp_capacity, buffer_size);
    check_cusparse(cusparseCsr2cscEx2(
        handle, source.row, source.col, source.numElements,
        source.value, source.rowPtr, source.colIndex,
        transpose->value, transpose->rowPtr, transpose->colIndex,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        algorithm, *construction_temp), "transpose csr2csc");
}

void destroy_spmv(CUSPARSE_spmvop_A *a, CUSPARSE_spmvop_AT *at) {
    cusparseHandle_t handle = at != nullptr && at->cusparseHandle != nullptr
        ? at->cusparseHandle
        : (a != nullptr ? a->cusparseHandle : nullptr);
    if (at != nullptr) {
        hprlp_destroy_spmvop(&at->operation);
        if (at->y_bar_cusparseDescr)
            cusparseDestroyDnVec(at->y_bar_cusparseDescr);
        if (at->y_cusparseDescr)
            cusparseDestroyDnVec(at->y_cusparseDescr);
        if (at->ATy_cusparseDescr)
            cusparseDestroyDnVec(at->ATy_cusparseDescr);
        if (at->AT_cusparseDescr)
            cusparseDestroySpMat(at->AT_cusparseDescr);
        *at = CUSPARSE_spmvop_AT{};
    }
    if (a != nullptr) {
        hprlp_destroy_spmvop(&a->unit_operation);
        hprlp_destroy_spmvop(&a->operation);
        if (a->x_bar_cusparseDescr)
            cusparseDestroyDnVec(a->x_bar_cusparseDescr);
        if (a->x_hat_cusparseDescr)
            cusparseDestroyDnVec(a->x_hat_cusparseDescr);
        if (a->x_temp_cusparseDescr)
            cusparseDestroyDnVec(a->x_temp_cusparseDescr);
        if (a->Ax_cusparseDescr)
            cusparseDestroyDnVec(a->Ax_cusparseDescr);
        if (a->A_cusparseDescr)
            cusparseDestroySpMat(a->A_cusparseDescr);
        *a = CUSPARSE_spmvop_A{};
    }
    if (handle != nullptr) cusparseDestroy(handle);
}

}  // namespace

struct HPRLP_reduced_stage_profile {
    long long calls = 0;
    double seconds = 0.0;
};

struct HPRLP_reduced_matrix_state {
    HPRLP_reduced_mode mode = HPRLP_reduced_mode::None;
    std::uint8_t *x_bar_mask = nullptr;
    int *changed_device = nullptr;
    int *changed_count_device = nullptr;
    int *free_count_by_warp_device = nullptr;
    void *free_count_reduce_temp = nullptr;
    std::size_t free_count_reduce_temp_bytes = 0;
    int free_count_warp_count = 0;
    int *free_count_device = nullptr;
    int *delta_count_device = nullptr;
    int *delta_indices_device = nullptr;
    std::uint8_t *delta_old_mask_device = nullptr;
    int *delta_row_bucket_counts_device = nullptr;
    int *base_row_analysis_device = nullptr;
    void *construction_temp = nullptr;
    std::size_t construction_temp_capacity = 0;
    cusparseHandle_t transpose_handle = nullptr;
    int recorded_changed_count = 0;
    int recorded_delta_count = 0;
    bool mask_started = false;
    bool mask_changed = false;
    bool mask_count_stale = false;
    bool built = false;
    bool active = false;
    bool full_dirty = false;
    bool profile_done = false;
    bool profile_enabled = false;
    int profile_count = 0;
    int free_count = 0;
    int base_count = 0;
    int stable_count = 0;
    long long base_reduced_nnz = 0;
    long long delta_nnz = 0;
    long long delta_work_accum = 0;
    int delta_count = 0;

    int activation_checks = 0;
    int active_iterations = 0;
    long double active_column_iteration_sum = 0.0L;
    long double active_nnz_iteration_sum = 0.0L;
    long long minimum_active_columns = std::numeric_limits<long long>::max();
    long long minimum_active_nnz = std::numeric_limits<long long>::max();
    int first_iteration = -1;
    int rebuilds = 0;
    HPRLP_FLOAT build_time = 0.0;
    int last_trigger_iteration = -1;
    HPRLP_FLOAT last_free_ratio = 1.0;
    HPRLP_FLOAT last_trigger_residual =
        std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT last_trigger_sigma =
        std::numeric_limits<HPRLP_FLOAT>::quiet_NaN();
    int last_free_columns = 0;
    int last_rebuild_iteration = -1;
    int restart_mask_checks = 0;
    int restart_mask_resets = 0;
    int deferred_empty_pending_iterations = 0;
    long long deferred_empty_total_iterations = 0;
    long long deferred_empty_flushes = 0;
    int row_restart_mask_checks = 0;
    int row_restart_mask_resets = 0;

    HPRLP_reduced_stage_profile profile_backend_autotune{};
    HPRLP_reduced_stage_profile profile_full_build{};
    HPRLP_reduced_stage_profile profile_delta_extend{};
    HPRLP_reduced_stage_profile profile_workspace_destroy{};
    HPRLP_reduced_stage_profile profile_graph_capture{};
    HPRLP_reduced_stage_profile profile_flush_to_full{};
    HPRLP_reduced_stage_profile profile_deferred_empty_flush{};
    HPRLP_reduced_stage_profile profile_gather_from_full{};

    int *free_to_original = nullptr;
    sparseMatrix A{};
    sparseMatrix AT{};
    sparseMatrix delta_A{};
    sparseMatrix delta_AT{};
    int *delta_free_to_original = nullptr;
    std::uint8_t *delta_old_mask = nullptr;
    // Each compact buffer stores short rows from the front and warp rows
    // from the back.  The two row-list pointers below are non-owning aliases.
    int *delta_A_row_buckets = nullptr;
    int *delta_AT_row_buckets = nullptr;
    int *delta_A_rows_short = nullptr;
    int *delta_A_rows_warp = nullptr;
    int delta_A_short_count = 0;
    int delta_A_warp_count = 0;
    int *delta_AT_rows_short = nullptr;
    int *delta_AT_rows_warp = nullptr;
    int delta_AT_short_count = 0;
    int delta_AT_warp_count = 0;
    HPRLP_FLOAT *delta_x = nullptr;
    HPRLP_FLOAT *delta_x_bar = nullptr;
    HPRLP_FLOAT *delta_x_hat = nullptr;
    HPRLP_FLOAT *delta_last_x = nullptr;
    HPRLP_FLOAT *delta_lower = nullptr;
    HPRLP_FLOAT *delta_upper = nullptr;
    HPRLP_FLOAT *delta_objective = nullptr;
    std::uint8_t *delta_bound_type = nullptr;
    std::uint32_t *delta_signed_AT_entries_u32 = nullptr;
    std::uint32_t *delta_signed_A_entries_u32 = nullptr;
    std::uint32_t *delta_signed_A_nonempty_words = nullptr;
    int *signed_delta_only_rows = nullptr;
    int signed_delta_only_count = 0;
    HPRLP_FLOAT *delta_inverse_col_norm = nullptr;
    HPRLP_FLOAT *delta_scaled_x_hat = nullptr;
    HPRLP_FLOAT *delta_scaled_fixed = nullptr;
    int *A_row_buckets = nullptr;
    int *AT_row_buckets = nullptr;
    int *A_rows_short = nullptr;
    int *A_rows_warp = nullptr;
    int A_short_count = 0;
    int A_warp_count = 0;
    int *AT_rows_short = nullptr;
    int *AT_rows_warp = nullptr;
    int AT_short_count = 0;
    int AT_warp_count = 0;
    bool use_unit_factorized = false;
    bool use_unit_active_scatter = false;
    bool use_signed_factorized = false;
    bool use_packed_dictionary_x = false;
    bool use_packed_dictionary = false;
    bool use_fixed_degree_dictionary_x = false;
    int fixed_degree_dictionary_x_degree = 0;
    bool signed_x_uses_split_u16 = false;
    bool signed_x_uses_scalar = false;
    bool signed_y_uses_combined = false;
    bool signed_y_uses_scalar = false;
    bool signed_AT_uses_u16 = false;
    bool signed_A_uses_u16 = false;
    bool use_fused_x = false;
    bool use_fused_y = false;
    std::uint16_t *unit_AT_constraint_index = nullptr;
    std::uint16_t *signed_AT_constraint_index = nullptr;
    std::uint8_t *signed_AT_negative = nullptr;
    std::uint16_t *signed_AT_entries_u16 = nullptr;
    std::uint32_t *signed_AT_entries_u32 = nullptr;
    std::uint16_t *signed_A_entries_u16 = nullptr;
    std::uint32_t *signed_A_entries_u32 = nullptr;
    std::uint32_t *dictionary_AT_entries_u32 = nullptr;
    std::uint32_t *dictionary_AT_entries_soa = nullptr;
    std::uint32_t *dictionary_A_entries_u32 = nullptr;
    std::uint32_t *dictionary_AT_to_A_code = nullptr;
    int *signed_AT_row_buckets = nullptr;
    int *signed_A_row_buckets = nullptr;
    int *signed_AT_rows_medium = nullptr;
    int signed_AT_medium_count = 0;
    int *signed_AT_rows_long = nullptr;
    int signed_AT_long_count = 0;
    int *signed_A_rows_medium = nullptr;
    int signed_A_medium_count = 0;
    int *signed_A_rows_long = nullptr;
    int signed_A_long_count = 0;
    int *signed_A_empty_row_buckets = nullptr;
    int *signed_A_rows_short_nonempty = nullptr;
    int signed_A_short_nonempty_count = 0;
    int *signed_A_rows_empty = nullptr;
    int signed_A_empty_count = 0;
    bool use_signed_empty_row_batch = false;
    bool use_dictionary_empty_row_batch = false;
    bool use_cusparse_empty_row_batch = false;
    bool defer_empty_rows_to_observation = false;
    // Preserve the autotuner's implementation choice across compact
    // workspace rebuilds without exposing reduced internals in the API.
    bool defer_empty_rows_selected = false;
    bool use_nonempty_cusparse_y = false;
    int nonempty_A_count = 0;
    int *nonempty_A_row_ptr = nullptr;
    int *original_to_nonempty_A = nullptr;
    int *nonempty_to_original_A = nullptr;
    HPRLP_FLOAT *nonempty_Ax = nullptr;
    cusparseSpMatDescr_t nonempty_A_descr = nullptr;
    cusparseDnVecDescr_t nonempty_x_hat_descr = nullptr;
    cusparseDnVecDescr_t nonempty_Ax_descr = nullptr;
    HPRLP_spmvop nonempty_A_operation{};
    bool use_parallel_delta_cusparse_y = false;
    HPRLP_FLOAT *parallel_delta_input = nullptr;
    HPRLP_FLOAT *parallel_delta_ax = nullptr;
    HPRLP_FLOAT *delta_fixed = nullptr;
    HPRLP_FLOAT *delta_input = nullptr;
    cusparseSpMatDescr_t parallel_delta_A_descr = nullptr;
    cusparseDnVecDescr_t parallel_delta_input_descr = nullptr;
    cusparseDnVecDescr_t parallel_delta_ax_descr = nullptr;
    HPRLP_spmvop parallel_delta_operation{};
    HPRLP_FLOAT *compact_inverse_col_norm = nullptr;
    HPRLP_FLOAT *compact_scaled_x_hat = nullptr;
    HPRLP_FLOAT *row_fixed_shift = nullptr;
    HPRLP_FLOAT *deferred_empty_halpern_factors = nullptr;
    int deferred_empty_halpern_capacity = 0;
    HPRLP_FLOAT *x = nullptr;
    HPRLP_FLOAT *x_bar = nullptr;
    HPRLP_FLOAT *x_hat = nullptr;
    HPRLP_FLOAT *last_x = nullptr;
    HPRLP_FLOAT *lower = nullptr;
    HPRLP_FLOAT *upper = nullptr;
    HPRLP_FLOAT *objective = nullptr;
    std::uint8_t *bound_type = nullptr;
    HPRLP_FLOAT *ATy = nullptr;
    HPRLP_FLOAT *Ax = nullptr;
    CUSPARSE_spmvop_A spmv_A{};
    CUSPARSE_spmvop_AT spmv_AT{};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraph_t graph_batch = nullptr;
    cudaGraphExec_t graph_exec_batch = nullptr;

    // Row reduction keeps x in canonical storage and compacts the active y
    // state so both the sparse operators and vector updates scale with the
    // number of active rows.
    std::uint8_t *y_bar_mask = nullptr;
    int *row_changed_device = nullptr;
    int *row_active_count_device = nullptr;
    int *row_active_count_by_warp_device = nullptr;
    void *row_count_reduce_temp = nullptr;
    std::size_t row_count_reduce_temp_bytes = 0;
    int row_active_count_warp_count = 0;
    int row_active_count = 0;
    int row_stable_count = 0;
    bool row_mask_started = false;
    bool row_mask_changed = false;
    bool row_built = false;
    bool use_compact_row_y = false;
    std::uint8_t *row_base_mask = nullptr;
    int row_base_count = 0;
    int *row_active_to_original = nullptr;
    HPRLP_FLOAT *row_y = nullptr;
    HPRLP_FLOAT *row_last_y = nullptr;
    HPRLP_FLOAT *row_lower = nullptr;
    HPRLP_FLOAT *row_upper = nullptr;
    sparseMatrix row_A{};
    sparseMatrix row_AT{};
    HPRLP_FLOAT *row_Ax = nullptr;
    CUSPARSE_spmvop_A row_spmv_A{};
    CUSPARSE_spmvop_AT row_spmv_AT{};
    bool row_use_fused_x = false;
    bool row_use_fused_y = false;
    bool row_use_signed_x = false;
    bool row_use_signed_y = false;
    bool row_signed_y_uses_combined = false;
    bool row_signed_y_uses_scalar = false;
    bool row_use_packed_dictionary_x = false;
    bool row_use_packed_dictionary_y = false;
    std::uint32_t *row_AT_packed_entries_u32 = nullptr;
    std::uint32_t *row_A_packed_entries_u32 = nullptr;
    int *row_original_to_active = nullptr;
    HPRLP_FLOAT *row_inverse_row_norm = nullptr;
    HPRLP_FLOAT *row_scaled_y = nullptr;
    std::uint8_t *row_bound_type = nullptr;
    int *row_AT_special_row_buckets = nullptr;
    int *row_A_special_row_buckets = nullptr;
    int *row_AT_rows_medium = nullptr;
    int row_AT_medium_count = 0;
    int *row_AT_rows_long = nullptr;
    int row_AT_long_count = 0;
    int *row_A_rows_medium = nullptr;
    int row_A_medium_count = 0;
    int *row_A_rows_long = nullptr;
    int row_A_long_count = 0;
    int *row_A_row_buckets = nullptr;
    int *row_AT_row_buckets = nullptr;
    int *row_A_rows_short = nullptr;
    int *row_A_rows_warp = nullptr;
    int row_A_short_count = 0;
    int row_A_warp_count = 0;
    int *row_AT_rows_short = nullptr;
    int *row_AT_rows_warp = nullptr;
    int row_AT_short_count = 0;
    int row_AT_warp_count = 0;
    int row_delta_count = 0;
    int *row_delta_to_original = nullptr;
    HPRLP_FLOAT *row_delta_y = nullptr;
    HPRLP_FLOAT *row_delta_last_y = nullptr;
    HPRLP_FLOAT *row_delta_lower = nullptr;
    HPRLP_FLOAT *row_delta_upper = nullptr;
    sparseMatrix row_delta_A{};
    sparseMatrix row_delta_AT{};
    long long row_delta_work_accum = 0;
    cudaGraph_t row_graph = nullptr;
    cudaGraphExec_t row_graph_exec = nullptr;
    cudaGraph_t row_graph_batch = nullptr;
    cudaGraphExec_t row_graph_exec_batch = nullptr;
    int row_first_iteration = -1;
    int row_last_rebuild_iteration = -1;
    int row_rebuilds = 0;
    bool row_backend_profile_done = false;
    bool row_selected_use_fused_x = false;
    bool row_selected_use_fused_y = false;
    bool row_selected_use_signed_x = false;
    bool row_selected_use_signed_y = false;
    bool row_selected_use_packed_dictionary_x = false;
    bool row_selected_use_packed_dictionary_y = false;
    int row_active_iterations = 0;
    HPRLP_FLOAT row_build_time = 0.0;
    HPRLP_FLOAT row_last_active_ratio = 1.0;
    long double row_active_iteration_sum = 0.0L;
    long double row_nnz_iteration_sum = 0.0L;
    long long row_minimum_active = std::numeric_limits<long long>::max();
    long long row_minimum_nnz = std::numeric_limits<long long>::max();
};

namespace {

bool reduced_profile_enabled();

struct HPRLP_reduced_profile_token {
    bool enabled = false;
    std::chrono::steady_clock::time_point started{};
};

HPRLP_reduced_profile_token begin_reduced_stage_profile(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_reduced_matrix_state *state) {
    HPRLP_reduced_profile_token token;
    token.enabled = workspace != nullptr && state != nullptr &&
        state->profile_enabled;
    if (token.enabled) {
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        token.started = std::chrono::steady_clock::now();
    }
    return token;
}

void finish_reduced_stage_profile(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_reduced_profile_token &token,
    HPRLP_reduced_stage_profile *profile) {
    if (!token.enabled || workspace == nullptr || profile == nullptr) return;
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    profile->calls += 1;
    profile->seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - token.started).count();
}

void destroy_reduced_graph(HPRLP_reduced_matrix_state *state) {
    if (state->graph_exec_batch) {
        cudaGraphExecDestroy(state->graph_exec_batch);
        state->graph_exec_batch = nullptr;
    }
    if (state->graph_batch) {
        cudaGraphDestroy(state->graph_batch);
        state->graph_batch = nullptr;
    }
    if (state->graph_exec) {
        cudaGraphExecDestroy(state->graph_exec);
        state->graph_exec = nullptr;
    }
    if (state->graph) {
        cudaGraphDestroy(state->graph);
        state->graph = nullptr;
    }
}

void destroy_parallel_delta_cusparse_y(
    HPRLP_reduced_matrix_state *state) {
    hprlp_destroy_spmvop(&state->parallel_delta_operation);
    if (state->parallel_delta_ax_descr != nullptr) {
        cusparseDestroyDnVec(state->parallel_delta_ax_descr);
    }
    if (state->parallel_delta_input_descr != nullptr) {
        cusparseDestroyDnVec(state->parallel_delta_input_descr);
    }
    if (state->parallel_delta_A_descr != nullptr) {
        cusparseDestroySpMat(state->parallel_delta_A_descr);
    }
    cudaFree(state->parallel_delta_ax);
    cudaFree(state->parallel_delta_input);
    state->use_parallel_delta_cusparse_y = false;
    state->parallel_delta_input = nullptr;
    state->parallel_delta_ax = nullptr;
    state->parallel_delta_A_descr = nullptr;
    state->parallel_delta_input_descr = nullptr;
    state->parallel_delta_ax_descr = nullptr;
    state->parallel_delta_operation = HPRLP_spmvop{};
}

void destroy_reduced_nonempty_cusparse_y(
    HPRLP_reduced_matrix_state *state) {
    if (state == nullptr) return;
    hprlp_destroy_spmvop(&state->nonempty_A_operation);
    if (state->nonempty_Ax_descr != nullptr) {
        cusparseDestroyDnVec(state->nonempty_Ax_descr);
    }
    if (state->nonempty_x_hat_descr != nullptr) {
        cusparseDestroyDnVec(state->nonempty_x_hat_descr);
    }
    if (state->nonempty_A_descr != nullptr) {
        cusparseDestroySpMat(state->nonempty_A_descr);
    }
    cudaFree(state->nonempty_Ax);
    cudaFree(state->nonempty_A_row_ptr);
    cudaFree(state->original_to_nonempty_A);
    cudaFree(state->nonempty_to_original_A);
    state->use_nonempty_cusparse_y = false;
    state->nonempty_A_count = 0;
    state->nonempty_A_row_ptr = nullptr;
    state->original_to_nonempty_A = nullptr;
    state->nonempty_to_original_A = nullptr;
    state->nonempty_Ax = nullptr;
    state->nonempty_A_descr = nullptr;
    state->nonempty_x_hat_descr = nullptr;
    state->nonempty_Ax_descr = nullptr;
    state->nonempty_A_operation = HPRLP_spmvop{};
}

void reset_delta_workspace_metadata(HPRLP_reduced_matrix_state *state) {
    state->delta_free_to_original = nullptr;
    state->delta_old_mask = nullptr;
    state->delta_A_row_buckets = nullptr;
    state->delta_AT_row_buckets = nullptr;
    state->delta_A_rows_short = nullptr;
    state->delta_A_rows_warp = nullptr;
    state->delta_AT_rows_short = nullptr;
    state->delta_AT_rows_warp = nullptr;
    state->delta_A_short_count = 0;
    state->delta_A_warp_count = 0;
    state->delta_AT_short_count = 0;
    state->delta_AT_warp_count = 0;
    state->delta_x = nullptr;
    state->delta_x_bar = nullptr;
    state->delta_x_hat = nullptr;
    state->delta_last_x = nullptr;
    state->delta_lower = nullptr;
    state->delta_upper = nullptr;
    state->delta_objective = nullptr;
    state->delta_bound_type = nullptr;
    state->delta_signed_AT_entries_u32 = nullptr;
    state->delta_signed_A_entries_u32 = nullptr;
    state->delta_signed_A_nonempty_words = nullptr;
    state->signed_delta_only_rows = nullptr;
    state->signed_delta_only_count = 0;
    state->delta_inverse_col_norm = nullptr;
    state->delta_scaled_x_hat = nullptr;
    state->delta_scaled_fixed = nullptr;
    state->delta_fixed = nullptr;
    state->delta_input = nullptr;
    state->use_parallel_delta_cusparse_y = false;
    state->parallel_delta_input = nullptr;
    state->parallel_delta_ax = nullptr;
    state->parallel_delta_A_descr = nullptr;
    state->parallel_delta_input_descr = nullptr;
    state->parallel_delta_ax_descr = nullptr;
    state->parallel_delta_operation = HPRLP_spmvop{};
    state->delta_nnz = 0;
    state->delta_count = 0;
}

void destroy_delta_workspace(HPRLP_reduced_matrix_state *state) {
    if (state == nullptr) return;
    destroy_parallel_delta_cusparse_y(state);
    free_sparse_matrix(&state->delta_A);
    free_sparse_matrix(&state->delta_AT);
    cudaFree(state->delta_free_to_original);
    cudaFree(state->delta_old_mask);
    cudaFree(state->delta_A_row_buckets);
    cudaFree(state->delta_AT_row_buckets);
    cudaFree(state->delta_x);
    cudaFree(state->delta_x_bar);
    cudaFree(state->delta_x_hat);
    cudaFree(state->delta_last_x);
    cudaFree(state->delta_lower);
    cudaFree(state->delta_upper);
    cudaFree(state->delta_objective);
    cudaFree(state->delta_bound_type);
    cudaFree(state->delta_signed_AT_entries_u32);
    cudaFree(state->delta_signed_A_entries_u32);
    cudaFree(state->delta_signed_A_nonempty_words);
    cudaFree(state->delta_inverse_col_norm);
    cudaFree(state->delta_scaled_x_hat);
    cudaFree(state->delta_scaled_fixed);
    cudaFree(state->delta_fixed);
    cudaFree(state->delta_input);
    reset_delta_workspace_metadata(state);
}

void destroy_delta_workspace_async(
    HPRLP_reduced_matrix_state *state, cudaStream_t stream) {
    if (state == nullptr) return;
    destroy_parallel_delta_cusparse_y(state);
    free_sparse_matrix_async(&state->delta_A, stream);
    free_sparse_matrix_async(&state->delta_AT, stream);
    auto free_async = [stream](void *pointer) {
        if (pointer) CUDA_CHECK(cudaFreeAsync(pointer, stream));
    };
    free_async(state->delta_free_to_original);
    free_async(state->delta_old_mask);
    free_async(state->delta_A_row_buckets);
    free_async(state->delta_AT_row_buckets);
    free_async(state->delta_x);
    free_async(state->delta_x_bar);
    free_async(state->delta_x_hat);
    free_async(state->delta_last_x);
    free_async(state->delta_lower);
    free_async(state->delta_upper);
    free_async(state->delta_objective);
    free_async(state->delta_bound_type);
    free_async(state->delta_signed_AT_entries_u32);
    free_async(state->delta_signed_A_entries_u32);
    free_async(state->delta_signed_A_nonempty_words);
    free_async(state->delta_inverse_col_norm);
    free_async(state->delta_scaled_x_hat);
    free_async(state->delta_scaled_fixed);
    free_async(state->delta_fixed);
    free_async(state->delta_input);
    reset_delta_workspace_metadata(state);
}

void destroy_reduced_workspace(HPRLP_reduced_matrix_state *state) {
    if (state == nullptr) return;
    destroy_reduced_graph(state);
    destroy_delta_workspace(state);
    destroy_reduced_nonempty_cusparse_y(state);
    destroy_spmv(&state->spmv_A, &state->spmv_AT);
    free_sparse_matrix(&state->A);
    free_sparse_matrix(&state->AT);
    cudaFree(state->A_row_buckets);
    cudaFree(state->AT_row_buckets);
    cudaFree(state->free_to_original);
    cudaFree(state->unit_AT_constraint_index);
    cudaFree(state->signed_AT_constraint_index);
    cudaFree(state->signed_AT_negative);
    cudaFree(state->signed_AT_entries_u16);
    cudaFree(state->signed_AT_entries_u32);
    cudaFree(state->signed_A_entries_u16);
    cudaFree(state->signed_A_entries_u32);
    cudaFree(state->dictionary_AT_entries_u32);
    cudaFree(state->dictionary_AT_entries_soa);
    cudaFree(state->dictionary_A_entries_u32);
    cudaFree(state->dictionary_AT_to_A_code);
    cudaFree(state->signed_AT_row_buckets);
    cudaFree(state->signed_A_row_buckets);
    cudaFree(state->signed_A_empty_row_buckets);
    cudaFree(state->compact_inverse_col_norm);
    cudaFree(state->compact_scaled_x_hat);
    cudaFree(state->row_fixed_shift);
    cudaFree(state->deferred_empty_halpern_factors);
    cudaFree(state->x);
    cudaFree(state->x_bar);
    cudaFree(state->x_hat);
    cudaFree(state->last_x);
    cudaFree(state->lower);
    cudaFree(state->upper);
    cudaFree(state->objective);
    cudaFree(state->bound_type);
    cudaFree(state->ATy);
    cudaFree(state->Ax);
    state->free_to_original = nullptr;
    state->A_row_buckets = nullptr;
    state->AT_row_buckets = nullptr;
    state->A_rows_short = nullptr;
    state->A_rows_warp = nullptr;
    state->AT_rows_short = nullptr;
    state->AT_rows_warp = nullptr;
    state->A_short_count = 0;
    state->A_warp_count = 0;
    state->AT_short_count = 0;
    state->AT_warp_count = 0;
    state->use_unit_factorized = false;
    state->use_unit_active_scatter = false;
    state->use_signed_factorized = false;
    state->use_packed_dictionary_x = false;
    state->use_packed_dictionary = false;
    state->use_fixed_degree_dictionary_x = false;
    state->fixed_degree_dictionary_x_degree = 0;
    state->signed_x_uses_split_u16 = false;
    state->signed_x_uses_scalar = false;
    state->signed_y_uses_combined = false;
    state->signed_y_uses_scalar = false;
    state->signed_AT_uses_u16 = false;
    state->signed_A_uses_u16 = false;
    state->unit_AT_constraint_index = nullptr;
    state->signed_AT_constraint_index = nullptr;
    state->signed_AT_negative = nullptr;
    state->signed_AT_entries_u16 = nullptr;
    state->signed_AT_entries_u32 = nullptr;
    state->signed_A_entries_u16 = nullptr;
    state->signed_A_entries_u32 = nullptr;
    state->dictionary_AT_entries_u32 = nullptr;
    state->dictionary_AT_entries_soa = nullptr;
    state->dictionary_A_entries_u32 = nullptr;
    state->dictionary_AT_to_A_code = nullptr;
    state->signed_AT_row_buckets = nullptr;
    state->signed_A_row_buckets = nullptr;
    state->signed_AT_rows_medium = nullptr;
    state->signed_AT_medium_count = 0;
    state->signed_AT_rows_long = nullptr;
    state->signed_AT_long_count = 0;
    state->signed_A_rows_medium = nullptr;
    state->signed_A_medium_count = 0;
    state->signed_A_rows_long = nullptr;
    state->signed_A_long_count = 0;
    state->signed_A_empty_row_buckets = nullptr;
    state->signed_A_rows_short_nonempty = nullptr;
    state->signed_A_short_nonempty_count = 0;
    state->signed_A_rows_empty = nullptr;
    state->signed_A_empty_count = 0;
    state->use_signed_empty_row_batch = false;
    state->use_dictionary_empty_row_batch = false;
    state->use_cusparse_empty_row_batch = false;
    state->defer_empty_rows_to_observation = false;
    state->deferred_empty_pending_iterations = 0;
    state->compact_inverse_col_norm = nullptr;
    state->compact_scaled_x_hat = nullptr;
    state->row_fixed_shift = nullptr;
    state->deferred_empty_halpern_factors = nullptr;
    state->deferred_empty_halpern_capacity = 0;
    state->x = nullptr;
    state->x_bar = nullptr;
    state->x_hat = nullptr;
    state->last_x = nullptr;
    state->lower = nullptr;
    state->upper = nullptr;
    state->objective = nullptr;
    state->bound_type = nullptr;
    state->ATy = nullptr;
    state->Ax = nullptr;
    state->active = false;
    if (state->mode == HPRLP_reduced_mode::Columns) {
        state->mode = HPRLP_reduced_mode::None;
    }
    state->built = false;
    state->full_dirty = false;
    state->base_count = 0;
    state->base_reduced_nnz = 0;
    state->delta_work_accum = 0;
}

void destroy_reduced_row_graph(HPRLP_reduced_matrix_state *state) {
    if (state->row_graph_exec_batch) {
        cudaGraphExecDestroy(state->row_graph_exec_batch);
        state->row_graph_exec_batch = nullptr;
    }
    if (state->row_graph_batch) {
        cudaGraphDestroy(state->row_graph_batch);
        state->row_graph_batch = nullptr;
    }
    if (state->row_graph_exec) {
        cudaGraphExecDestroy(state->row_graph_exec);
        state->row_graph_exec = nullptr;
    }
    if (state->row_graph) {
        cudaGraphDestroy(state->row_graph);
        state->row_graph = nullptr;
    }
}

void destroy_reduced_row_delta_workspace(
    HPRLP_reduced_matrix_state *state) {
    if (state == nullptr) return;
    destroy_reduced_row_graph(state);
    free_sparse_matrix(&state->row_delta_A);
    free_sparse_matrix(&state->row_delta_AT);
    cudaFree(state->row_delta_to_original);
    cudaFree(state->row_delta_y);
    cudaFree(state->row_delta_last_y);
    cudaFree(state->row_delta_lower);
    cudaFree(state->row_delta_upper);
    state->row_delta_to_original = nullptr;
    state->row_delta_y = nullptr;
    state->row_delta_last_y = nullptr;
    state->row_delta_lower = nullptr;
    state->row_delta_upper = nullptr;
    state->row_delta_count = 0;
}

void destroy_reduced_row_workspace(HPRLP_reduced_matrix_state *state) {
    if (state == nullptr) return;
    destroy_reduced_row_delta_workspace(state);
    destroy_spmv(&state->row_spmv_A, &state->row_spmv_AT);
    free_sparse_matrix(&state->row_A);
    free_sparse_matrix(&state->row_AT);
    cudaFree(state->row_active_to_original);
    cudaFree(state->row_Ax);
    cudaFree(state->row_A_row_buckets);
    cudaFree(state->row_AT_row_buckets);
    cudaFree(state->row_AT_packed_entries_u32);
    cudaFree(state->row_A_packed_entries_u32);
    cudaFree(state->row_original_to_active);
    cudaFree(state->row_inverse_row_norm);
    cudaFree(state->row_scaled_y);
    cudaFree(state->row_bound_type);
    cudaFree(state->row_AT_special_row_buckets);
    cudaFree(state->row_A_special_row_buckets);
    cudaFree(state->row_y);
    cudaFree(state->row_last_y);
    cudaFree(state->row_lower);
    cudaFree(state->row_upper);
    state->row_active_to_original = nullptr;
    state->row_Ax = nullptr;
    state->row_A_row_buckets = nullptr;
    state->row_AT_row_buckets = nullptr;
    state->row_A_rows_short = nullptr;
    state->row_A_rows_warp = nullptr;
    state->row_A_short_count = 0;
    state->row_A_warp_count = 0;
    state->row_AT_rows_short = nullptr;
    state->row_AT_rows_warp = nullptr;
    state->row_AT_short_count = 0;
    state->row_AT_warp_count = 0;
    state->row_use_fused_x = false;
    state->row_use_fused_y = false;
    state->row_use_signed_x = false;
    state->row_use_signed_y = false;
    state->row_signed_y_uses_combined = false;
    state->row_signed_y_uses_scalar = false;
    state->row_use_packed_dictionary_x = false;
    state->row_use_packed_dictionary_y = false;
    state->row_AT_packed_entries_u32 = nullptr;
    state->row_A_packed_entries_u32 = nullptr;
    state->row_original_to_active = nullptr;
    state->row_inverse_row_norm = nullptr;
    state->row_scaled_y = nullptr;
    state->row_bound_type = nullptr;
    state->row_AT_special_row_buckets = nullptr;
    state->row_A_special_row_buckets = nullptr;
    state->row_AT_rows_medium = nullptr;
    state->row_AT_medium_count = 0;
    state->row_AT_rows_long = nullptr;
    state->row_AT_long_count = 0;
    state->row_A_rows_medium = nullptr;
    state->row_A_medium_count = 0;
    state->row_A_rows_long = nullptr;
    state->row_A_long_count = 0;
    state->row_y = nullptr;
    state->row_last_y = nullptr;
    state->row_lower = nullptr;
    state->row_upper = nullptr;
    state->use_compact_row_y = false;
    state->row_base_count = 0;
    state->row_delta_work_accum = 0;
    state->row_built = false;
    if (state->mode == HPRLP_reduced_mode::Rows) {
        state->mode = HPRLP_reduced_mode::None;
        state->active = false;
    }
}

bool refresh_reduced_row_mask(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (!reduced_rows_enabled() || workspace->m <= 0) return false;
    CUDA_CHECK(cudaMemsetAsync(
        state->row_changed_device, 0, sizeof(int), workspace->stream));
    if (state->row_mask_started) {
        update_y_bar_mask_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            state->y_bar_mask, workspace->y_bar,
            state->row_changed_device,
            state->row_active_count_by_warp_device, workspace->m);
    } else {
        initialize_y_bar_mask_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            state->y_bar_mask, workspace->y_bar, workspace->AL,
            workspace->AU, workspace->y_bound_type,
            state->row_changed_device,
            state->row_active_count_by_warp_device, workspace->m);
        state->row_mask_started = true;
    }
    CUDA_CHECK(cub::DeviceReduce::Sum(
        state->row_count_reduce_temp,
        state->row_count_reduce_temp_bytes,
        state->row_active_count_by_warp_device,
        state->row_active_count_device,
        state->row_active_count_warp_count, workspace->stream));
    int changed = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &changed, state->row_changed_device, sizeof(int),
        cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &state->row_active_count, state->row_active_count_device,
        sizeof(int), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->row_mask_changed = changed != 0;
    return state->row_mask_changed;
}

void build_device_medium_long_buckets(
    const int *row_ptr,
    int row_count,
    int medium_count,
    int long_count,
    int **row_buckets,
    int **medium_rows,
    int **long_rows,
    int *scratch_counts,
    cudaStream_t stream);

void enqueue_reduced_row_iteration_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state);

void profile_reduced_row_backend_candidates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters,
    int iteration);

void prepare_reduced_row_spmv_pair(
    HPRLP_workspace_gpu *workspace,
    const sparseMatrix &row_A,
    const sparseMatrix &row_AT,
    HPRLP_FLOAT *row_Ax,
    HPRLP_FLOAT *row_y,
    int row_y_count,
    int active_count,
    HPRLP_FLOAT at_beta,
    bool prepare_a,
    bool prepare_at,
    CUSPARSE_spmvop_A *a_pointer,
    CUSPARSE_spmvop_AT *at_pointer) {
    if (!prepare_a && !prepare_at) return;
    CUSPARSE_spmvop_A &a = *a_pointer;
    CUSPARSE_spmvop_AT &at = *at_pointer;
    check_cusparse(cusparseCreate(&a.cusparseHandle),
                   "cusparseCreate reduced rows");
    at.cusparseHandle = a.cusparseHandle;
    check_cusparse(cusparseSetStream(a.cusparseHandle, workspace->stream),
                   "cusparseSetStream reduced rows");
    a.alpha = at.alpha = 1.0;
    a.beta = 0.0;
    at.beta = at_beta;
    a.computeType = at.computeType = CUDA_R_64F;

    if (prepare_a) {
        check_cusparse(cusparseCreateDnVec(
            &a.x_hat_cusparseDescr, workspace->n, workspace->x_hat,
            CUDA_R_64F), "create row-reduced xhat");
        check_cusparse(cusparseCreateDnVec(
            &a.Ax_cusparseDescr, active_count, row_Ax,
            CUDA_R_64F), "create row-reduced Ax");
        check_cusparse(cusparseCreateCsr(
            &a.A_cusparseDescr, row_A.row, row_A.col,
            row_A.numElements, row_A.rowPtr, row_A.colIndex, row_A.value,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "create row-reduced A");
        check_cusparse(hprlp_prepare_spmvop(
            a.cusparseHandle, a.A_cusparseDescr, a.x_hat_cusparseDescr,
            a.Ax_cusparseDescr, a.Ax_cusparseDescr, a.computeType,
            &a.operation), "prepare row-reduced A SpMVOp");
    }

    if (prepare_at) {
        check_cusparse(cusparseCreateDnVec(
            &at.y_cusparseDescr, row_y_count, row_y, CUDA_R_64F),
            "create row-reduced y");
        check_cusparse(cusparseCreateDnVec(
            &at.ATy_cusparseDescr, workspace->n, workspace->ATy,
            CUDA_R_64F), "create row-reduced ATy");
        check_cusparse(cusparseCreateCsr(
            &at.AT_cusparseDescr, row_AT.row, row_AT.col,
            row_AT.numElements, row_AT.rowPtr, row_AT.colIndex,
            row_AT.value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "create row-reduced AT");
        check_cusparse(hprlp_prepare_spmvop(
            at.cusparseHandle, at.AT_cusparseDescr,
            at.y_cusparseDescr, at.ATy_cusparseDescr,
            at.ATy_cusparseDescr, at.computeType, &at.operation),
            "prepare row-reduced AT SpMVOp");
    }
}

void prepare_reduced_row_spmv(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool prepare_all_candidates) {
    prepare_reduced_row_spmv_pair(
        workspace, state->row_A, state->row_AT, state->row_Ax,
        state->use_compact_row_y ? state->row_y : workspace->y,
        state->use_compact_row_y ? state->row_base_count : workspace->m,
        state->row_base_count, 0.0,
        prepare_all_candidates ||
            (!state->row_use_fused_y && !state->row_use_signed_y &&
             !state->row_use_packed_dictionary_y),
        prepare_all_candidates ||
            (!state->row_use_fused_x && !state->row_use_signed_x &&
             !state->row_use_packed_dictionary_x),
        &state->row_spmv_A, &state->row_spmv_AT);
}

bool build_reduced_row_workspace(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters,
    int iteration) {
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    const auto start = std::chrono::steady_clock::now();
    state->row_base_count = state->row_active_count;
    state->row_delta_work_accum = 0;
    if (workspace->m > 0) {
        std::size_t scan_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, scan_temp_bytes,
            state->row_active_count_by_warp_device,
            state->row_active_count_by_warp_device,
            state->row_active_count_warp_count, workspace->stream));
        ensure_device_byte_capacity(
            &state->construction_temp, &state->construction_temp_capacity,
            scan_temp_bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            state->construction_temp, scan_temp_bytes,
            state->row_active_count_by_warp_device,
            state->row_active_count_by_warp_device,
            state->row_active_count_warp_count, workspace->stream));
    }
    allocate_device(
        &state->row_active_to_original, state->row_active_count);
    if (workspace->m > 0) {
        scatter_active_row_indices_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            state->y_bar_mask, state->row_active_count_by_warp_device,
            state->row_active_to_original, workspace->m);
    }

    state->row_A.row = state->row_active_count;
    state->row_A.col = workspace->n;
    allocate_device(
        &state->row_A.rowPtr,
        static_cast<std::size_t>(state->row_active_count) + 1);
    CUDA_CHECK(cudaMemsetAsync(
        state->row_A.rowPtr, 0, sizeof(int), workspace->stream));
    if (state->row_active_count > 0) {
        selected_csr_row_lengths_kernel<<<
            numBlocks(state->row_active_count), numThreads, 0,
            workspace->stream>>>(
            state->row_A.rowPtr, state->row_active_to_original,
            workspace->A->rowPtr, state->row_active_count);
        std::size_t scan_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr, scan_temp_bytes, state->row_A.rowPtr + 1,
            state->row_A.rowPtr + 1, state->row_active_count,
            workspace->stream));
        ensure_device_byte_capacity(
            &state->construction_temp, &state->construction_temp_capacity,
            scan_temp_bytes);
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            state->construction_temp, scan_temp_bytes,
            state->row_A.rowPtr + 1, state->row_A.rowPtr + 1,
            state->row_active_count, workspace->stream));
    }
    int reduced_nnz = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &reduced_nnz, state->row_A.rowPtr + state->row_active_count,
        sizeof(int), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->row_A.numElements = reduced_nnz;
    const HPRLP_FLOAT retained_nnz_ratio = workspace->A->numElements > 0
        ? static_cast<HPRLP_FLOAT>(reduced_nnz) /
            workspace->A->numElements
        : 1.0;
    const auto packed_indices_fit = [](int index_count,
                                       unsigned code_bits) {
        if (index_count < 0 || code_bits >= 32) return false;
        const unsigned index_bits = 32 - code_bits;
        const std::uint64_t capacity = UINT64_C(1) << index_bits;
        return static_cast<std::uint64_t>(index_count) <= capacity;
    };
    state->row_use_signed_x =
        workspace->x_backend == HPRLPXBackend::SignedUnitPacked &&
        workspace->signed_unit_operator_ready &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_x_hat != nullptr;
    state->row_use_signed_y =
        (workspace->y_backend == HPRLPYBackend::SignedUnitPacked ||
         workspace->y_backend ==
             HPRLPYBackend::SignedUnitPackedCombined) &&
        workspace->signed_unit_operator_ready &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_x_hat != nullptr;
    state->row_signed_y_uses_combined = state->row_use_signed_y &&
        workspace->y_backend == HPRLPYBackend::SignedUnitPackedCombined;
    const bool selected_dictionary_x =
        workspace->x_backend == HPRLPXBackend::PackedDictionary ||
        workspace->x_backend ==
            HPRLPXBackend::FixedDegreePackedDictionary;
    state->row_use_packed_dictionary_x = selected_dictionary_x &&
        workspace->dictionary_operator_x_ready &&
        workspace->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32 &&
        workspace->coefficient_dictionary != nullptr &&
        workspace->AT_dictionary_packed_u32 != nullptr &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_x_hat != nullptr &&
        packed_indices_fit(
            state->row_active_count,
            workspace->packed_dictionary_code_bits);
    state->row_use_packed_dictionary_y =
        workspace->y_backend == HPRLPYBackend::PackedDictionary &&
        workspace->dictionary_operator_y_ready &&
        workspace->A_packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32 &&
        workspace->A_coefficient_dictionary != nullptr &&
        workspace->A_dictionary_packed_u32 != nullptr &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_x_hat != nullptr &&
        packed_indices_fit(
            workspace->n,
            workspace->A_packed_dictionary_code_bits);
    if (state->row_backend_profile_done) {
        state->row_use_fused_x = state->row_selected_use_fused_x;
        state->row_use_fused_y = state->row_selected_use_fused_y;
        state->row_use_signed_x = state->row_selected_use_signed_x;
        state->row_use_signed_y = state->row_selected_use_signed_y;
        state->row_use_packed_dictionary_x =
            state->row_selected_use_packed_dictionary_x;
        state->row_use_packed_dictionary_y =
            state->row_selected_use_packed_dictionary_y;
    }
    const bool row_has_special_x = state->row_use_signed_x ||
        state->row_use_packed_dictionary_x;
    const bool row_has_special_y = state->row_use_signed_y ||
        state->row_use_packed_dictionary_y;
    const bool profile_row_backends =
        (reduced_row_compressed_autotune_enabled() ||
         reduced_row_backend_profile_enabled()) &&
        !state->row_backend_profile_done;
    state->use_compact_row_y =
        row_has_special_x || row_has_special_y ||
        retained_nnz_ratio <= reduced_compact_row_y_nnz_ratio();
    allocate_device(&state->row_A.colIndex, reduced_nnz);
    allocate_device(&state->row_A.value, reduced_nnz);
    if (state->row_active_count > 0 && reduced_nnz > 0) {
        copy_selected_csr_rows_kernel<<<
            state->row_active_count, 256, 0, workspace->stream>>>(
            state->row_A.colIndex, state->row_A.value,
            state->row_A.rowPtr, state->row_active_to_original,
            workspace->A->rowPtr, workspace->A->colIndex,
            workspace->A->value, state->row_active_count);
    }
    transpose_csr(
        state->row_A, &state->row_AT, workspace->stream,
        state->transpose_handle, &state->construction_temp,
        &state->construction_temp_capacity);
    if (!state->use_compact_row_y) {
        if (reduced_nnz > 0) {
            remap_compact_row_columns_kernel<<<
                numBlocks(reduced_nnz), numThreads, 0,
                workspace->stream>>>(
                state->row_AT.colIndex, state->row_active_to_original,
                reduced_nnz);
        }
        state->row_AT.col = workspace->m;
    }
    if (!state->row_backend_profile_done) {
        state->row_use_fused_x =
            !row_has_special_x && workspace->reduced_use_fused_x;
        state->row_use_fused_y =
            !row_has_special_y && workspace->reduced_use_fused_y;
    }
    CUDA_CHECK(cudaMemsetAsync(
        state->base_row_analysis_device, 0,
        2 * kReducedRowAnalysisFields * sizeof(int), workspace->stream));
    if (state->row_use_fused_x || row_has_special_x ||
        profile_row_backends) {
        allocate_device(&state->row_AT_row_buckets, workspace->n);
        if (workspace->n > 0) {
            analyze_compact_rows_kernel<<<
                numBlocks(workspace->n), numThreads, 0,
                workspace->stream>>>(
                state->row_AT.rowPtr, state->row_AT_row_buckets,
                state->base_row_analysis_device, workspace->n);
        }
    }
    if (state->row_use_fused_y || row_has_special_y ||
        profile_row_backends) {
        allocate_device(
            &state->row_A_row_buckets, state->row_base_count);
        if (state->row_base_count > 0) {
            analyze_compact_rows_kernel<<<
                numBlocks(state->row_base_count), numThreads, 0,
                workspace->stream>>>(
                state->row_A.rowPtr, state->row_A_row_buckets,
                state->base_row_analysis_device + kReducedRowAnalysisFields,
                state->row_base_count);
        }
    }
    int row_analysis[2 * kReducedRowAnalysisFields] = {};
    CUDA_CHECK(cudaMemcpyAsync(
        row_analysis, state->base_row_analysis_device,
        sizeof(row_analysis), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->row_AT_short_count = row_analysis[kReducedRowShortCount];
    state->row_AT_warp_count = row_analysis[kReducedRowWarpCount];
    state->row_AT_rows_short = state->row_AT_short_count > 0
        ? state->row_AT_row_buckets : nullptr;
    state->row_AT_rows_warp = state->row_AT_warp_count > 0
        ? state->row_AT_row_buckets + workspace->n -
            state->row_AT_warp_count
        : nullptr;
    const int *const a_analysis =
        row_analysis + kReducedRowAnalysisFields;
    state->row_A_short_count = a_analysis[kReducedRowShortCount];
    state->row_A_warp_count = a_analysis[kReducedRowWarpCount];
    state->row_A_rows_short = state->row_A_short_count > 0
        ? state->row_A_row_buckets : nullptr;
    state->row_A_rows_warp = state->row_A_warp_count > 0
        ? state->row_A_row_buckets + state->row_base_count -
            state->row_A_warp_count
        : nullptr;
    if (row_has_special_x) {
        state->row_AT_medium_count =
            row_analysis[kReducedRowMediumCount];
        state->row_AT_long_count =
            row_analysis[kReducedRowLongCount];
        build_device_medium_long_buckets(
            state->row_AT.rowPtr, workspace->n,
            state->row_AT_medium_count, state->row_AT_long_count,
            &state->row_AT_special_row_buckets,
            &state->row_AT_rows_medium, &state->row_AT_rows_long,
            state->base_row_analysis_device, workspace->stream);
    }
    if (row_has_special_y) {
        state->row_A_medium_count =
            a_analysis[kReducedRowMediumCount];
        state->row_A_long_count =
            a_analysis[kReducedRowLongCount];
        build_device_medium_long_buckets(
            state->row_A.rowPtr, state->row_base_count,
            state->row_A_medium_count, state->row_A_long_count,
            &state->row_A_special_row_buckets,
            &state->row_A_rows_medium, &state->row_A_rows_long,
            state->base_row_analysis_device, workspace->stream);
        state->row_signed_y_uses_scalar = state->row_use_signed_y &&
            hprlp_all_rows_fit_scalar(
                state->row_base_count,
                a_analysis[kReducedRowMaxNnz],
                HPRLP_UNIT_SCALAR_ROW_MAX_NNZ) &&
            !state->row_signed_y_uses_combined;
    }
    if (profile_row_backends ||
        (!state->row_use_fused_y && !row_has_special_y)) {
        allocate_device(&state->row_Ax, state->row_active_count);
    }
    if (state->use_compact_row_y) {
        allocate_device(&state->row_y, state->row_active_count);
        allocate_device(&state->row_last_y, state->row_active_count);
        allocate_device(&state->row_lower, state->row_active_count);
        allocate_device(&state->row_upper, state->row_active_count);
        gather_reduced_row_state_kernel<<<
            numBlocks(state->row_active_count), numThreads, 0,
            workspace->stream>>>(
            state->row_y, state->row_last_y, state->row_lower,
            state->row_upper, workspace->y, workspace->last_y,
            workspace->AL, workspace->AU,
            state->row_active_to_original, state->row_active_count);
    }
    if (row_has_special_x || row_has_special_y) {
        allocate_device(
            &state->row_inverse_row_norm, state->row_active_count);
        if (row_has_special_x || state->row_use_signed_y) {
            allocate_device(&state->row_scaled_y, state->row_active_count);
        }
        if (row_has_special_y) {
            allocate_device(&state->row_bound_type, state->row_active_count);
        }
        if (state->row_active_count > 0) {
            gather_reduced_row_backend_state_kernel<<<
                numBlocks(state->row_active_count), numThreads, 0,
                workspace->stream>>>(
                state->row_inverse_row_norm, state->row_scaled_y,
                state->row_bound_type, workspace->inverse_row_norm,
                workspace->y, workspace->y_bound_type,
                state->row_active_to_original, state->row_active_count);
        }
    }
    if (state->row_use_signed_x) {
        allocate_device(
            &state->row_AT_packed_entries_u32,
            state->row_AT.numElements);
        if (state->row_AT.numElements > 0) {
            pack_signed_entries_u32_kernel<<<
                numBlocks(state->row_AT.numElements), numThreads, 0,
                workspace->stream>>>(
                state->row_AT_packed_entries_u32,
                state->row_AT.colIndex, state->row_AT.value,
                state->row_AT.numElements);
        }
    } else if (state->row_use_packed_dictionary_x) {
        allocate_device(
            &state->row_AT_packed_entries_u32,
            state->row_AT.numElements);
        allocate_device(&state->row_original_to_active, workspace->m);
        CUDA_CHECK(cudaMemsetAsync(
            state->row_original_to_active, 0xff,
            static_cast<std::size_t>(workspace->m) * sizeof(int),
            workspace->stream));
        if (state->row_active_count > 0) {
            scatter_original_to_selected_rows_kernel<<<
                numBlocks(state->row_active_count), numThreads, 0,
                workspace->stream>>>(
                state->row_original_to_active,
                state->row_active_to_original,
                state->row_active_count);
        }
        if (workspace->n > 0 && state->row_AT.numElements > 0) {
            filter_packed_dictionary_AT_rows_kernel<<<
                workspace->n, 32, 0, workspace->stream>>>(
                state->row_AT_packed_entries_u32,
                state->row_AT.rowPtr, workspace->AT->rowPtr,
                workspace->AT_dictionary_packed_u32,
                state->row_original_to_active,
                workspace->packed_dictionary_code_bits, workspace->n);
        }
    }
    if (state->row_use_signed_y) {
        allocate_device(
            &state->row_A_packed_entries_u32,
            state->row_A.numElements);
        if (state->row_A.numElements > 0) {
            pack_signed_entries_u32_kernel<<<
                numBlocks(state->row_A.numElements), numThreads, 0,
                workspace->stream>>>(
                state->row_A_packed_entries_u32,
                state->row_A.colIndex, state->row_A.value,
                state->row_A.numElements);
        }
    } else if (state->row_use_packed_dictionary_y) {
        allocate_device(
            &state->row_A_packed_entries_u32,
            state->row_A.numElements);
        if (state->row_base_count > 0 && state->row_A.numElements > 0) {
            copy_selected_packed_dictionary_rows_kernel<<<
                state->row_base_count, 256, 0, workspace->stream>>>(
                state->row_A_packed_entries_u32,
                state->row_A.rowPtr, state->row_active_to_original,
                workspace->A->rowPtr,
                workspace->A_dictionary_packed_u32,
                state->row_base_count);
        }
    }
    zero_inactive_y_state_kernel<<<
        numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
        workspace->y, workspace->last_y, state->y_bar_mask, workspace->m);
    const auto row_backend_profile_started =
        std::chrono::steady_clock::now();
    prepare_reduced_row_spmv(workspace, state, profile_row_backends);
    if (profile_row_backends) {
        profile_reduced_row_backend_candidates(
            workspace, state, parameters, iteration);
        // Do not let autotune calls share cuSPARSE descriptor/preprocess
        // state with production iterations.  Recreate only the descriptors
        // selected for production after tuning.
        destroy_spmv(&state->row_spmv_A, &state->row_spmv_AT);
        prepare_reduced_row_spmv(workspace, state, false);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        const double isolated_total_ms =
            1000.0 * std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                row_backend_profile_started).count();
        std::cout << "  row backend autotune isolated_total_ms="
                  << std::fixed << std::setprecision(3)
                  << isolated_total_ms
                  << " (includes candidate descriptors and production "
                     "descriptor rebuild)"
                  << std::defaultfloat << std::setprecision(2)
                  << std::endl;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        state->row_base_mask, state->y_bar_mask,
        static_cast<std::size_t>(workspace->m) * sizeof(std::uint8_t),
        cudaMemcpyDeviceToDevice, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->row_built = true;
    state->row_rebuilds += 1;
    state->row_last_rebuild_iteration = iteration;
    state->row_build_time += std::chrono::duration<HPRLP_FLOAT>(
        std::chrono::steady_clock::now() - start).count();
    if (std::getenv("HPRLP_TRACE_ROW_REDUCTION") != nullptr) {
        std::cout << "  row-reduced build: iteration=" << iteration
                  << ", active_rows=" << state->row_active_count
                  << "/" << workspace->m << ", nnz=" << reduced_nnz
                  << "/" << workspace->A->numElements
                  << ", storage="
                  << (state->use_compact_row_y ? "compact" : "full")
                  << ", x_backend="
                  << (state->row_use_packed_dictionary_x
                      ? "packed-dictionary"
                      : (state->row_use_signed_x ? "signed-packed"
                         : (state->row_use_fused_x
                            ? "fused" : "cusparse")))
                  << ", y_backend="
                  << (state->row_use_packed_dictionary_y
                      ? "packed-dictionary"
                      : (state->row_use_signed_y
                         ? (state->row_signed_y_uses_combined
                            ? "signed-packed-combined"
                            : "signed-packed")
                         : (state->row_use_fused_y
                            ? "fused" : "cusparse")))
                  << std::endl;
    }
    return true;
}

bool rebuild_reduced_row_delta_workspace(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (!state->row_built ||
        state->row_active_count < state->row_base_count) {
        return false;
    }
    const int delta_count =
        state->row_active_count - state->row_base_count;
    destroy_reduced_row_delta_workspace(state);
    if (delta_count == 0) return true;
    const auto start = std::chrono::steady_clock::now();

    count_delta_rows_by_warp_kernel<<<
        numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
        state->y_bar_mask, state->row_base_mask,
        state->row_active_count_by_warp_device, workspace->m);
    std::size_t select_scan_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, select_scan_temp_bytes,
        state->row_active_count_by_warp_device,
        state->row_active_count_by_warp_device,
        state->row_active_count_warp_count, workspace->stream));
    ensure_device_byte_capacity(
        &state->construction_temp, &state->construction_temp_capacity,
        select_scan_temp_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        state->construction_temp, select_scan_temp_bytes,
        state->row_active_count_by_warp_device,
        state->row_active_count_by_warp_device,
        state->row_active_count_warp_count, workspace->stream));
    allocate_device(&state->row_delta_to_original, delta_count);
    scatter_delta_row_indices_kernel<<<
        numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
        state->y_bar_mask, state->row_base_mask,
        state->row_active_count_by_warp_device,
        state->row_delta_to_original, workspace->m);

    sparseMatrix &delta_A = state->row_delta_A;
    delta_A.row = delta_count;
    delta_A.col = workspace->n;
    allocate_device(
        &delta_A.rowPtr, static_cast<std::size_t>(delta_count) + 1);
    CUDA_CHECK(cudaMemsetAsync(
        delta_A.rowPtr, 0, sizeof(int), workspace->stream));
    selected_csr_row_lengths_kernel<<<
        numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
        delta_A.rowPtr, state->row_delta_to_original,
        workspace->A->rowPtr, delta_count);
    std::size_t row_scan_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, row_scan_temp_bytes, delta_A.rowPtr + 1,
        delta_A.rowPtr + 1, delta_count, workspace->stream));
    ensure_device_byte_capacity(
        &state->construction_temp, &state->construction_temp_capacity,
        row_scan_temp_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        state->construction_temp, row_scan_temp_bytes,
        delta_A.rowPtr + 1, delta_A.rowPtr + 1,
        delta_count, workspace->stream));
    int delta_nnz = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &delta_nnz, delta_A.rowPtr + delta_count, sizeof(int),
        cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    delta_A.numElements = delta_nnz;
    allocate_device(&delta_A.colIndex, delta_nnz);
    allocate_device(&delta_A.value, delta_nnz);
    if (delta_nnz > 0) {
        copy_selected_csr_rows_kernel<<<
            delta_count, 256, 0, workspace->stream>>>(
            delta_A.colIndex, delta_A.value, delta_A.rowPtr,
            state->row_delta_to_original, workspace->A->rowPtr,
            workspace->A->colIndex, workspace->A->value, delta_count);
    }
    transpose_csr(
        delta_A, &state->row_delta_AT, workspace->stream,
        state->transpose_handle, &state->construction_temp,
        &state->construction_temp_capacity);
    if (!state->use_compact_row_y) {
        if (delta_nnz > 0) {
            remap_compact_row_columns_kernel<<<
                numBlocks(delta_nnz), numThreads, 0,
                workspace->stream>>>(
                state->row_delta_AT.colIndex,
                state->row_delta_to_original, delta_nnz);
        }
        state->row_delta_AT.col = workspace->m;
    }
    state->row_delta_count = delta_count;
    if (state->use_compact_row_y) {
        allocate_device(&state->row_delta_y, delta_count);
        allocate_device(&state->row_delta_last_y, delta_count);
        allocate_device(&state->row_delta_lower, delta_count);
        allocate_device(&state->row_delta_upper, delta_count);
        gather_reduced_row_state_kernel<<<
            numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
            state->row_delta_y, state->row_delta_last_y,
            state->row_delta_lower, state->row_delta_upper,
            workspace->y, workspace->last_y,
            workspace->AL, workspace->AU,
            state->row_delta_to_original, delta_count);
    }
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->row_build_time += std::chrono::duration<HPRLP_FLOAT>(
        std::chrono::steady_clock::now() - start).count();
    return true;
}

void enqueue_reduced_row_iteration_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (state->row_use_packed_dictionary_x) {
        const HPRLP_packed_dictionary_x_view_gpu view{
            workspace->x, workspace->x_hat,
            workspace->l, workspace->u, workspace->x_bound_type,
            workspace->c, workspace->last_x, state->row_scaled_y,
            workspace->inverse_col_norm,
            workspace->unit_scaled_x_hat, workspace->packed_state_plan,
            workspace->coefficient_dictionary,
            state->row_AT_packed_entries_u32, state->row_AT.rowPtr,
            workspace->packed_dictionary_code_bits,
            state->row_AT_rows_short, state->row_AT_short_count,
            state->row_AT_rows_medium, state->row_AT_medium_count,
            state->row_AT_rows_long, state->row_AT_long_count, true};
        hprlp_enqueue_packed_dictionary_x(
            view, workspace->Halpern_params,
            workspace->halpern_factors, kReducedSignedYThreads,
            workspace->stream);
    } else if (state->row_use_signed_x) {
        const HPRLP_signed_unit_x_packed_bucket_view_gpu view{
            {workspace->n, workspace->x, workspace->x_hat,
             workspace->l, workspace->u, workspace->x_bound_type,
             workspace->c, workspace->last_x, state->row_scaled_y,
             workspace->unit_scaled_x_hat, nullptr,
             workspace->inverse_col_norm, state->row_AT.rowPtr,
             nullptr, state->row_AT_packed_entries_u32},
            state->row_AT_rows_short, state->row_AT_short_count,
            state->row_AT_rows_medium, state->row_AT_medium_count,
            state->row_AT_rows_long, state->row_AT_long_count};
        hprlp_enqueue_signed_unit_x_packed_bucketed(
            view, workspace->Halpern_params, workspace->halpern_factors,
            kReducedSignedXThreads, kReducedSignedYThreads,
            workspace->stream);
    } else if (state->row_use_fused_x) {
        const HPRLP_FLOAT *const base_y = state->use_compact_row_y
            ? state->row_y : workspace->y;
        const HPRLP_FLOAT *const delta_y = state->use_compact_row_y
            ? state->row_delta_y : workspace->y;
        const int *const delta_row_ptr = state->row_delta_count > 0
            ? state->row_delta_AT.rowPtr : nullptr;
        if (state->row_AT_short_count > 0) {
            fused_reduced_row_x_short_kernel<<<
                numBlocks(state->row_AT_short_count), numThreads, 0,
                workspace->stream>>>(
                workspace->x, workspace->x_hat,
                workspace->l, workspace->u, workspace->x_bound_type,
                workspace->c, workspace->last_x, base_y,
                state->row_AT.rowPtr, state->row_AT.colIndex,
                state->row_AT.value, delta_y, delta_row_ptr,
                state->row_delta_AT.colIndex,
                state->row_delta_AT.value, workspace->Halpern_params,
                workspace->halpern_factors, state->row_AT_rows_short,
                state->row_AT_short_count);
        }
        if (state->row_AT_warp_count > 0) {
            fused_reduced_row_x_warp_kernel<<<
                (state->row_AT_warp_count + 15) / 16, 512, 0,
                workspace->stream>>>(
                workspace->x, workspace->x_hat,
                workspace->l, workspace->u, workspace->x_bound_type,
                workspace->c, workspace->last_x, base_y,
                state->row_AT.rowPtr, state->row_AT.colIndex,
                state->row_AT.value, delta_y, delta_row_ptr,
                state->row_delta_AT.colIndex,
                state->row_delta_AT.value, workspace->Halpern_params,
                workspace->halpern_factors, state->row_AT_rows_warp,
                state->row_AT_warp_count);
        }
    } else {
        CUSPARSE_spmvop_AT &at = state->row_spmv_AT;
        check_cusparse(hprlp_run_spmvop(
            at.cusparseHandle, at.operation, &at.alpha, &at.beta,
            at.y_cusparseDescr, at.ATy_cusparseDescr,
            at.ATy_cusparseDescr), "row-reduced AT SpMVOp");
        if (state->row_delta_count > 0) {
            update_reduced_row_x_with_delta_kernel<<<
                numBlocks(workspace->n), numThreads, 0,
                workspace->stream>>>(
                workspace->x, workspace->x_hat,
                workspace->l, workspace->u, workspace->ATy,
                workspace->c, workspace->last_x,
                state->use_compact_row_y
                    ? state->row_delta_y : workspace->y,
                state->row_delta_AT.rowPtr,
                state->row_delta_AT.colIndex,
                state->row_delta_AT.value, workspace->Halpern_params,
                workspace->halpern_factors, workspace->n);
        } else {
            update_zx_normal_kernel<<<
                numBlocks(workspace->n), numThreads, 0,
                workspace->stream>>>(
                workspace->x, workspace->x_hat,
                workspace->l, workspace->u, workspace->ATy,
                workspace->c, workspace->last_x,
                workspace->Halpern_params, workspace->halpern_factors,
                workspace->n);
        }
    }

    if ((state->row_use_signed_y ||
         state->row_use_packed_dictionary_y) &&
        !state->row_use_signed_x &&
        !state->row_use_packed_dictionary_x) {
        vector_dot_product_kernel<<<
            numBlocks(workspace->n), numThreads, 0,
            workspace->stream>>>(
            workspace->x_hat, workspace->inverse_col_norm,
            workspace->unit_scaled_x_hat, workspace->n, false);
    }

    if (state->row_use_packed_dictionary_y) {
        const HPRLP_packed_dictionary_y_view_gpu view{
            state->row_y, state->row_lower, state->row_upper,
            state->row_bound_type, state->row_last_y,
            workspace->unit_scaled_x_hat, state->row_inverse_row_norm,
            state->row_use_packed_dictionary_x
                ? state->row_scaled_y : nullptr,
            nullptr, HPRLPPackedStatePlan{},
            workspace->A_coefficient_dictionary,
            state->row_A_packed_entries_u32, state->row_A.rowPtr,
            workspace->A_packed_dictionary_code_bits,
            state->row_A_rows_short, state->row_A_short_count,
            state->row_A_rows_medium, state->row_A_medium_count,
            state->row_A_rows_long, state->row_A_long_count,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            true};
        hprlp_enqueue_packed_dictionary_y(
            view, workspace->Halpern_params,
            workspace->halpern_factors, kReducedSignedYThreads,
            workspace->stream);
    } else if (state->row_use_signed_y) {
        // Signed Y kernels always materialize their scaled-y cache, even
        // when the current X backend does not consume it.
        HPRLP_FLOAT *const scaled_y_output = state->row_scaled_y;
        if (state->row_signed_y_uses_combined) {
            const HPRLP_signed_unit_y_combined_view_gpu view{
                state->row_base_count, state->row_y,
                state->row_lower, state->row_upper,
                state->row_bound_type, state->row_last_y,
                workspace->unit_scaled_x_hat, nullptr, scaled_y_output,
                nullptr, state->row_inverse_row_norm,
                state->row_A.rowPtr, nullptr,
                state->row_A_packed_entries_u32,
                state->row_A_rows_medium, state->row_A_medium_count,
                state->row_A_rows_long, state->row_A_long_count,
                nullptr, nullptr, nullptr, nullptr, nullptr};
            hprlp_enqueue_signed_unit_y_combined(
                view, workspace->Halpern_params,
                workspace->halpern_factors, kReducedSignedYThreads,
                workspace->stream);
        } else if (state->row_signed_y_uses_scalar) {
            const HPRLP_signed_unit_y_scalar_view_gpu view{
                state->row_base_count, state->row_y,
                state->row_lower, state->row_upper,
                state->row_bound_type, state->row_last_y,
                workspace->unit_scaled_x_hat, nullptr, scaled_y_output,
                nullptr, state->row_inverse_row_norm,
                state->row_A.rowPtr, nullptr,
                state->row_A_packed_entries_u32};
            hprlp_enqueue_signed_unit_y_scalar(
                view, workspace->Halpern_params,
                workspace->halpern_factors, kReducedSignedXThreads,
                workspace->stream);
        } else {
            const HPRLP_signed_unit_y_bucket_view_gpu view{
                {state->row_base_count, state->row_y,
                 state->row_lower, state->row_upper,
                 state->row_bound_type, state->row_last_y,
                 workspace->unit_scaled_x_hat, nullptr, scaled_y_output,
                 nullptr, state->row_inverse_row_norm,
                 state->row_A.rowPtr, nullptr,
                 state->row_A_packed_entries_u32},
                state->row_A_rows_short, state->row_A_short_count,
                state->row_A_rows_medium, state->row_A_medium_count,
                state->row_A_rows_long, state->row_A_long_count,
                nullptr, nullptr, nullptr, nullptr};
            hprlp_enqueue_signed_unit_y_bucketed(
                view, workspace->Halpern_params,
                workspace->halpern_factors, kReducedSignedXThreads,
                kReducedSignedYThreads, workspace->stream);
        }
    } else if (state->row_use_fused_y) {
        HPRLP_FLOAT *const y = state->use_compact_row_y
            ? state->row_y : workspace->y;
        const HPRLP_FLOAT *const lower = state->use_compact_row_y
            ? state->row_lower : workspace->AL;
        const HPRLP_FLOAT *const upper = state->use_compact_row_y
            ? state->row_upper : workspace->AU;
        const HPRLP_FLOAT *const last_y = state->use_compact_row_y
            ? state->row_last_y : workspace->last_y;
        const int *const active_to_original = state->use_compact_row_y
            ? nullptr : state->row_active_to_original;
        if (state->row_A_short_count > 0) {
            fused_reduced_row_y_short_kernel<<<
                numBlocks(state->row_A_short_count), numThreads, 0,
                workspace->stream>>>(
                y, lower, upper, last_y, workspace->x_hat,
                state->row_A.rowPtr, state->row_A.colIndex,
                state->row_A.value, active_to_original,
                workspace->Halpern_params, workspace->halpern_factors,
                state->row_A_rows_short, state->row_A_short_count);
        }
        if (state->row_A_warp_count > 0) {
            fused_reduced_row_y_warp_kernel<<<
                (state->row_A_warp_count + 15) / 16, 512, 0,
                workspace->stream>>>(
                y, lower, upper, last_y, workspace->x_hat,
                state->row_A.rowPtr, state->row_A.colIndex,
                state->row_A.value, active_to_original,
                workspace->Halpern_params, workspace->halpern_factors,
                state->row_A_rows_warp, state->row_A_warp_count);
        }
    } else {
        CUSPARSE_spmvop_A &a = state->row_spmv_A;
        check_cusparse(hprlp_run_spmvop(
            a.cusparseHandle, a.operation, &a.alpha, &a.beta,
            a.x_hat_cusparseDescr, a.Ax_cusparseDescr,
            a.Ax_cusparseDescr), "row-reduced A SpMVOp");
        if (state->use_compact_row_y) {
            update_reduced_row_y_kernel<<<
                numBlocks(state->row_base_count), numThreads, 0,
                workspace->stream>>>(
                state->row_y, state->row_lower, state->row_upper,
                state->row_Ax, state->row_last_y,
                workspace->Halpern_params, workspace->halpern_factors,
                state->row_base_count);
        } else {
            update_reduced_row_y_full_kernel<<<
                numBlocks(state->row_base_count), numThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU,
                state->row_Ax, workspace->last_y,
                state->row_active_to_original,
                workspace->Halpern_params, workspace->halpern_factors,
                state->row_base_count);
        }
    }
    if (state->row_delta_count > 0) {
        if (state->use_compact_row_y) {
            update_reduced_row_delta_y_kernel<<<
                numBlocks(state->row_delta_count), numThreads, 0,
                workspace->stream>>>(
                state->row_delta_y, state->row_delta_lower,
                state->row_delta_upper, state->row_delta_last_y,
                workspace->x_hat,
                state->row_delta_A.rowPtr, state->row_delta_A.colIndex,
                state->row_delta_A.value,
                workspace->Halpern_params, workspace->halpern_factors,
                state->row_delta_count);
        } else {
            update_reduced_row_delta_y_full_kernel<<<
                numBlocks(state->row_delta_count), numThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU,
                workspace->last_y, workspace->x_hat,
                state->row_delta_A.rowPtr, state->row_delta_A.colIndex,
                state->row_delta_A.value,
                state->row_delta_to_original,
                workspace->Halpern_params, workspace->halpern_factors,
                state->row_delta_count);
        }
    }
    if ((state->row_use_signed_x ||
         state->row_use_packed_dictionary_x) &&
        !state->row_use_signed_y &&
        !state->row_use_packed_dictionary_y) {
        vector_dot_product_kernel<<<
            numBlocks(state->row_base_count), numThreads, 0,
            workspace->stream>>>(
            state->row_y, state->row_inverse_row_norm,
            state->row_scaled_y, state->row_base_count, false);
    }
}

void profile_reduced_row_backend_candidates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters,
    int iteration) {
    if (workspace == nullptr || state == nullptr ||
        state->row_backend_profile_done || state->row_delta_count != 0) {
        return;
    }
    constexpr int short_probe_iterations = 3;
    constexpr int finalist_probe_iterations = 8;
    constexpr double minimum_improvement = 0.05;
    constexpr double early_abort_ratio = 1.50;
    const auto total_started = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

    HPRLP_FLOAT *saved_x = nullptr;
    HPRLP_FLOAT *saved_x_hat = nullptr;
    HPRLP_FLOAT *saved_y = nullptr;
    const int y_count = state->use_compact_row_y
        ? state->row_base_count : workspace->m;
    HPRLP_FLOAT *const active_y = state->use_compact_row_y
        ? state->row_y : workspace->y;
    allocate_device(&saved_x, workspace->n);
    allocate_device(&saved_x_hat, workspace->n);
    allocate_device(&saved_y, y_count);
    if (workspace->n > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            saved_x, workspace->x,
            static_cast<std::size_t>(workspace->n) * sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            saved_x_hat, workspace->x_hat,
            static_cast<std::size_t>(workspace->n) * sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice, workspace->stream));
    }
    if (y_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            saved_y, active_y,
            static_cast<std::size_t>(y_count) * sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice, workspace->stream));
    }

    const bool saved_fused_x = state->row_use_fused_x;
    const bool saved_fused_y = state->row_use_fused_y;
    const bool saved_signed_x = state->row_use_signed_x;
    const bool saved_signed_y = state->row_use_signed_y;
    const bool saved_dictionary_x =
        state->row_use_packed_dictionary_x;
    const bool saved_dictionary_y =
        state->row_use_packed_dictionary_y;

    auto restore_vectors = [&]() {
        if (workspace->n > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                workspace->x, saved_x,
                static_cast<std::size_t>(workspace->n) *
                    sizeof(HPRLP_FLOAT),
                cudaMemcpyDeviceToDevice, workspace->stream));
            CUDA_CHECK(cudaMemcpyAsync(
                workspace->x_hat, saved_x_hat,
                static_cast<std::size_t>(workspace->n) *
                    sizeof(HPRLP_FLOAT),
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        if (y_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                active_y, saved_y,
                static_cast<std::size_t>(y_count) * sizeof(HPRLP_FLOAT),
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        if (state->row_scaled_y != nullptr &&
            state->row_inverse_row_norm != nullptr &&
            state->row_base_count > 0) {
            vector_dot_product_kernel<<<
                numBlocks(state->row_base_count), numThreads, 0,
                workspace->stream>>>(
                state->row_y, state->row_inverse_row_norm,
                state->row_scaled_y, state->row_base_count, false);
        }
    };
    auto configure_generic = [&](bool fused_x, bool fused_y) {
        state->row_use_fused_x = fused_x;
        state->row_use_fused_y = fused_y;
        state->row_use_signed_x = false;
        state->row_use_signed_y = false;
        state->row_use_packed_dictionary_x = false;
        state->row_use_packed_dictionary_y = false;
    };
    auto restore_backend = [&]() {
        state->row_use_fused_x = saved_fused_x;
        state->row_use_fused_y = saved_fused_y;
        state->row_use_signed_x = saved_signed_x;
        state->row_use_signed_y = saved_signed_y;
        state->row_use_packed_dictionary_x = saved_dictionary_x;
        state->row_use_packed_dictionary_y = saved_dictionary_y;
    };

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));
    struct RowBackendCandidate {
        bool incumbent = false;
        bool fused_x = false;
        bool fused_y = false;
        double short_ms = std::numeric_limits<double>::infinity();
        double final_ms = std::numeric_limits<double>::infinity();
        bool finalist = false;
    };
    std::vector<RowBackendCandidate> candidates;
    candidates.reserve(5);
    candidates.push_back(RowBackendCandidate{
        true, saved_fused_x, saved_fused_y});
    auto add_generic = [&](bool fused_x, bool fused_y) {
        const bool incumbent_is_generic =
            !saved_signed_x && !saved_signed_y &&
            !saved_dictionary_x && !saved_dictionary_y;
        if (incumbent_is_generic && saved_fused_x == fused_x &&
            saved_fused_y == fused_y) {
            return;
        }
        candidates.push_back(RowBackendCandidate{
            false, fused_x, fused_y});
    };
    add_generic(false, false);
    add_generic(true, false);
    add_generic(false, true);
    add_generic(true, true);

    auto configure_candidate = [&](const RowBackendCandidate &candidate) {
        if (candidate.incumbent) {
            restore_backend();
        } else {
            configure_generic(candidate.fused_x, candidate.fused_y);
        }
    };
    auto time_candidate = [&](RowBackendCandidate &candidate,
                              int probe_iterations) {
        configure_candidate(candidate);
        restore_vectors();
        enqueue_reduced_row_iteration_updates(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        restore_vectors();
        CUDA_CHECK(cudaEventRecord(event_start, workspace->stream));
        for (int probe = 0; probe < probe_iterations; ++probe) {
            enqueue_reduced_row_iteration_updates(workspace, state);
        }
        CUDA_CHECK(cudaEventRecord(event_stop, workspace->stream));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, event_start, event_stop));
        return static_cast<double>(elapsed_ms) / probe_iterations;
    };
    for (RowBackendCandidate &candidate : candidates) {
        candidate.short_ms =
            time_candidate(candidate, short_probe_iterations);
    }
    const double incumbent_short_ms = candidates.front().short_ms;
    int best_alternate = -1;
    for (int index = 1; index < static_cast<int>(candidates.size());
         ++index) {
        if (candidates[index].short_ms <
            (best_alternate >= 0
                ? candidates[best_alternate].short_ms
                : std::numeric_limits<double>::infinity())) {
            best_alternate = index;
        }
    }
    candidates.front().finalist = true;
    candidates.front().final_ms = time_candidate(
        candidates.front(), finalist_probe_iterations);
    if (best_alternate >= 0 &&
        candidates[best_alternate].short_ms <=
            incumbent_short_ms * early_abort_ratio) {
        candidates[best_alternate].finalist = true;
        candidates[best_alternate].final_ms = time_candidate(
            candidates[best_alternate], finalist_probe_iterations);
    }
    auto combined_time = [&](const RowBackendCandidate &candidate) {
        if (!candidate.finalist || !std::isfinite(candidate.final_ms)) {
            return candidate.short_ms;
        }
        return (candidate.short_ms * short_probe_iterations +
                candidate.final_ms * finalist_probe_iterations) /
            (short_probe_iterations + finalist_probe_iterations);
    };
    const double incumbent_ms = combined_time(candidates.front());
    int selected = 0;
    double selected_ms = incumbent_ms;
    if (best_alternate >= 0 && candidates[best_alternate].finalist) {
        const double alternate_ms =
            combined_time(candidates[best_alternate]);
        if (std::isfinite(alternate_ms) && alternate_ms < selected_ms) {
            selected = best_alternate;
            selected_ms = alternate_ms;
        }
    }
    const double probe_total_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_started).count();
    const double saved_ms = incumbent_ms - selected_ms;
    const double improvement = incumbent_ms > 0.0
        ? saved_ms / incumbent_ms : 0.0;
    const double payback_iterations = saved_ms > 0.0
        ? probe_total_ms / saved_ms
        : std::numeric_limits<double>::infinity();
    const int remaining_iterations = parameters != nullptr
        ? std::max(0, parameters->max_iter - iteration) : 0;
    if (!reduced_row_compressed_autotune_enabled() ||
        (selected != 0 &&
         (improvement < minimum_improvement ||
          payback_iterations > remaining_iterations))) {
        selected = 0;
        selected_ms = incumbent_ms;
    }

    const RowBackendCandidate &choice = candidates[selected];
    configure_candidate(choice);
    state->row_selected_use_fused_x = state->row_use_fused_x;
    state->row_selected_use_fused_y = state->row_use_fused_y;
    state->row_selected_use_signed_x = state->row_use_signed_x;
    state->row_selected_use_signed_y = state->row_use_signed_y;
    state->row_selected_use_packed_dictionary_x =
        state->row_use_packed_dictionary_x;
    state->row_selected_use_packed_dictionary_y =
        state->row_use_packed_dictionary_y;
    restore_vectors();
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));
    CUDA_CHECK(cudaFree(saved_x));
    CUDA_CHECK(cudaFree(saved_x_hat));
    CUDA_CHECK(cudaFree(saved_y));
    auto generic_backend_name = [](bool fused_x, bool fused_y) {
        if (fused_x && fused_y) return std::string("fused/fused");
        if (fused_x) return std::string("fused/cusparse");
        if (fused_y) return std::string("cusparse/fused");
        return std::string("cusparse/cusparse");
    };
    auto incumbent_backend_name = [&]() {
        const char *const x = saved_dictionary_x
            ? "packed-dictionary"
            : (saved_signed_x ? "signed-packed"
               : (saved_fused_x ? "fused" : "cusparse"));
        const char *const y = saved_dictionary_y
            ? "packed-dictionary"
            : (saved_signed_y
                ? (state->row_signed_y_uses_combined
                    ? "signed-packed-combined" : "signed-packed")
                : (saved_fused_y ? "fused" : "cusparse"));
        return std::string(x) + "/" + y;
    };
    auto backend_name = [&](const RowBackendCandidate &candidate) {
        return candidate.incumbent
            ? incumbent_backend_name()
            : generic_backend_name(candidate.fused_x, candidate.fused_y);
    };
    std::cout << "  row compressed autotune: iteration=" << iteration
              << ", active_rows="
              << state->row_base_count << "/" << workspace->m
              << ", nnz=" << state->row_A.numElements << "/"
              << workspace->A->numElements
              << ", short_probes=" << short_probe_iterations
              << ", finalist_probes=" << finalist_probe_iterations
              << ", probe_total_ms=" << std::fixed
              << std::setprecision(3) << probe_total_ms << std::endl;
    for (const RowBackendCandidate &candidate : candidates) {
        std::cout << "    candidate=" << backend_name(candidate)
                  << " short_ms=" << std::fixed
                  << std::setprecision(6) << candidate.short_ms;
        if (candidate.finalist) {
            std::cout << " combined_ms=" << combined_time(candidate);
        } else {
            std::cout << " early_rejected=1";
        }
        std::cout << std::endl;
    }
    std::cout << "    selected=" << backend_name(choice)
              << " incumbent=" << backend_name(candidates.front())
              << " measured_improvement=" << std::fixed
              << std::setprecision(4) << improvement
              << " payback_iterations=" << std::setprecision(1)
              << payback_iterations
                  << std::endl;
    std::cout << std::defaultfloat << std::setprecision(2);
    state->row_backend_profile_done = true;
}

void capture_reduced_row_graphs(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    destroy_reduced_row_graph(state);
    CUDA_CHECK(cudaStreamBeginCapture(
        workspace->stream, cudaStreamCaptureModeGlobal));
    enqueue_reduced_row_iteration_updates(workspace, state);
    advance_halpern_factors(workspace);
    CUDA_CHECK(cudaStreamEndCapture(workspace->stream, &state->row_graph));
    CUDA_CHECK(cudaGraphInstantiate(
        &state->row_graph_exec, state->row_graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaStreamBeginCapture(
        workspace->stream, cudaStreamCaptureModeGlobal));
    HPRLP_FLOAT *const canonical_halpern_factors =
        workspace->halpern_factors;
    prepare_halpern_factor_batch_kernel<<<1, 1, 0, workspace->stream>>>(
        workspace->halpern_inner, canonical_halpern_factors,
        workspace->halpern_factor_batch, HPRLP_NORMAL_GRAPH_BATCH_SIZE);
    for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
        workspace->halpern_factors = workspace->halpern_factor_batch + 2 * i;
        enqueue_reduced_row_iteration_updates(workspace, state);
    }
    workspace->halpern_factors = canonical_halpern_factors;
    CUDA_CHECK(cudaStreamEndCapture(
        workspace->stream, &state->row_graph_batch));
    CUDA_CHECK(cudaGraphInstantiate(
        &state->row_graph_exec_batch, state->row_graph_batch,
        nullptr, nullptr, 0));
}

void refresh_reduced_factor_scaled_y(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_reduced_matrix_state *state) {
    if (workspace == nullptr || state == nullptr ||
        (!state->use_unit_factorized && !state->use_signed_factorized &&
         !state->use_packed_dictionary_x) ||
        workspace->m <= 0) {
        return;
    }
    vector_dot_product_kernel<<<
        numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
        workspace->y, workspace->inverse_row_norm,
        workspace->unit_scaled_y, workspace->m, false);
}

void refresh_reduced_factor_scaled_x_hat(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_reduced_matrix_state *state) {
    if (workspace == nullptr || state == nullptr ||
        (!state->use_unit_factorized && !state->use_signed_factorized &&
         !state->use_packed_dictionary) ||
        state->base_count <= 0 || state->compact_scaled_x_hat == nullptr) {
        return;
    }
    vector_dot_product_kernel<<<
        numBlocks(state->base_count), numThreads, 0,
        workspace->stream>>>(
        state->x_hat, state->compact_inverse_col_norm,
        state->compact_scaled_x_hat, state->base_count, false);
}

bool reduced_uses_cusparse_x(const HPRLP_reduced_matrix_state *state) {
    return state != nullptr && state->base_count > 0 &&
        !state->use_packed_dictionary_x &&
        !state->use_signed_factorized &&
        !state->use_unit_factorized && !state->use_fused_x;
}

bool reduced_uses_generic_y(const HPRLP_reduced_matrix_state *state) {
    return state != nullptr && !state->use_packed_dictionary &&
        !state->use_signed_factorized &&
        !state->use_unit_factorized && !state->use_fused_y;
}

void prepare_reduced_spmv(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool prepare_all_candidates = false) {
    if (state->base_count <= 0) return;
    const bool prepare_a = prepare_all_candidates ||
        reduced_uses_generic_y(state);
    const bool prepare_at = prepare_all_candidates ||
        reduced_uses_cusparse_x(state);
    if (!prepare_a && !prepare_at) return;

    CUSPARSE_spmvop_A &a = state->spmv_A;
    CUSPARSE_spmvop_AT &at = state->spmv_AT;
    check_cusparse(cusparseCreate(&a.cusparseHandle),
                   "cusparseCreate reduced");
    // destroy_spmv owns the shared handle through spmv_AT.  Preserve that
    // ownership even when only one matrix direction needs cuSPARSE.
    at.cusparseHandle = a.cusparseHandle;
    check_cusparse(cusparseSetStream(a.cusparseHandle, workspace->stream),
                   "cusparseSetStream reduced");
    a.alpha = 1.0;
    a.beta = 0.0;
    at.alpha = 1.0;
    at.beta = 0.0;
    a.computeType = CUDA_R_64F;
    at.computeType = CUDA_R_64F;

    if (prepare_a) {
        check_cusparse(cusparseCreateDnVec(
            &a.x_bar_cusparseDescr, state->base_count, state->x_bar,
            CUDA_R_64F), "create reduced xbar");
        check_cusparse(cusparseCreateDnVec(
            &a.x_hat_cusparseDescr, state->base_count, state->x_hat,
            CUDA_R_64F), "create reduced xhat");
        check_cusparse(cusparseCreateDnVec(
            &a.x_temp_cusparseDescr, state->base_count, state->x,
            CUDA_R_64F), "create reduced xtemp");
        check_cusparse(cusparseCreateDnVec(
            &a.Ax_cusparseDescr, workspace->m, state->Ax,
            CUDA_R_64F), "create reduced Ax");
        check_cusparse(cusparseCreateCsr(
            &a.A_cusparseDescr, state->A.row, state->A.col,
            state->A.numElements, state->A.rowPtr, state->A.colIndex,
            state->A.value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create reduced A");
        check_cusparse(hprlp_prepare_spmvop(
            a.cusparseHandle, a.A_cusparseDescr, a.x_hat_cusparseDescr,
            a.Ax_cusparseDescr, a.Ax_cusparseDescr, a.computeType,
            &a.operation), "prepare reduced A SpMVOp");
    }
    if (prepare_at) {
        check_cusparse(cusparseCreateDnVec(
            &at.y_bar_cusparseDescr, workspace->m, workspace->y_bar,
            CUDA_R_64F), "create reduced ybar");
        check_cusparse(cusparseCreateDnVec(
            &at.y_cusparseDescr, workspace->m, workspace->y,
            CUDA_R_64F), "create reduced y");
        check_cusparse(cusparseCreateDnVec(
            &at.ATy_cusparseDescr, state->base_count, state->ATy,
            CUDA_R_64F), "create reduced ATy");
        check_cusparse(cusparseCreateCsr(
            &at.AT_cusparseDescr, state->AT.row, state->AT.col,
            state->AT.numElements, state->AT.rowPtr, state->AT.colIndex,
            state->AT.value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create reduced AT");
        check_cusparse(hprlp_prepare_spmvop(
            at.cusparseHandle, at.AT_cusparseDescr,
            at.y_cusparseDescr, at.ATy_cusparseDescr,
            at.ATy_cusparseDescr, at.computeType, &at.operation),
            "prepare reduced AT SpMVOp");
    }
}

bool reduced_compressed_autotune_enabled() {
    const char *value =
        std::getenv("HPRLP_USE_REDUCED_COMPRESSED_AUTOTUNE");
    if (value == nullptr) return true;
    const std::string setting(value);
    return setting != "0" && setting != "false" && setting != "FALSE" &&
        setting != "no" && setting != "NO";
}

bool reduced_defer_empty_rows_enabled() {
    const char *value =
        std::getenv("HPRLP_DEFER_REDUCED_EMPTY_ROWS_TO_CHECK");
    if (value == nullptr) return true;
    const std::string setting(value);
    return setting != "0" && setting != "false" && setting != "FALSE" &&
        setting != "no" && setting != "NO";
}

void autotune_reduced_compressed_backends(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters,
    int iteration);

bool reduced_nonempty_cusparse_solve_requested() {
    const char *value =
        std::getenv("HPRLP_USE_REDUCED_NONEMPTY_CUSPARSE");
    if (value == nullptr) return false;
    const std::string setting(value);
    return setting == "1" || setting == "true" || setting == "TRUE" ||
           setting == "yes" || setting == "YES";
}

__global__ void mark_nonempty_row_flags_kernel(
    const int *row_ptr, int *flags, int row_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < row_count) {
        flags[row] = row_ptr[row] != row_ptr[row + 1] ? 1 : 0;
    }
}

__global__ void finalize_nonempty_row_count_kernel(
    const int *flags, const int *prefix, int row_count, int *count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *count = row_count > 0
            ? prefix[row_count - 1] + flags[row_count - 1] : 0;
    }
}

__global__ void scatter_nonempty_row_metadata_kernel(
    const int *row_ptr, const int *flags, const int *prefix,
    int *original_to_nonempty, int *nonempty_to_original,
    int *nonempty_row_ptr, int row_count, int nonzero_count,
    int nonempty_count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < row_count) {
        if (flags[row] != 0) {
            const int compact = prefix[row];
            original_to_nonempty[row] = compact;
            nonempty_row_ptr[compact] = row_ptr[row];
            if (nonempty_to_original != nullptr) {
                nonempty_to_original[compact] = row;
            }
        } else {
            original_to_nonempty[row] = -1;
        }
    }
    if (row == 0) {
        nonempty_row_ptr[nonempty_count] = nonzero_count;
    }
}

int build_nonempty_row_metadata_gpu(
    const int *row_ptr, int row_count, int nonzero_count,
    bool build_inverse, int **original_to_nonempty,
    int **nonempty_to_original, int **nonempty_row_ptr,
    cudaStream_t stream) {
    int *flags = nullptr;
    int *prefix = nullptr;
    int *count_device = nullptr;
    allocate_device(&flags, row_count);
    allocate_device(&prefix, row_count);
    allocate_device(&count_device, 1);
    if (row_count > 0) {
        mark_nonempty_row_flags_kernel<<<
            numBlocks(row_count), numThreads, 0, stream>>>(
                row_ptr, flags, row_count);
    }
    std::size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, flags, prefix, row_count, stream));
    void *scan_temp = nullptr;
    if (scan_bytes > 0) CUDA_CHECK(cudaMalloc(&scan_temp, scan_bytes));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_temp, scan_bytes, flags, prefix, row_count, stream));
    finalize_nonempty_row_count_kernel<<<1, 1, 0, stream>>>(
        flags, prefix, row_count, count_device);
    int nonempty_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &nonempty_count, count_device, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    allocate_device(original_to_nonempty, row_count);
    allocate_device(
        nonempty_row_ptr, static_cast<std::size_t>(nonempty_count) + 1);
    if (build_inverse) {
        allocate_device(nonempty_to_original, nonempty_count);
    } else {
        *nonempty_to_original = nullptr;
    }
    if (row_count > 0) {
        scatter_nonempty_row_metadata_kernel<<<
            numBlocks(row_count), numThreads, 0, stream>>>(
                row_ptr, flags, prefix, *original_to_nonempty,
                *nonempty_to_original, *nonempty_row_ptr, row_count,
                nonzero_count, nonempty_count);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(scan_temp);
    cudaFree(count_device);
    cudaFree(prefix);
    cudaFree(flags);
    return nonempty_count;
}

struct HPRLP_device_vector_compare_result {
    unsigned long long mismatch_count;
    unsigned long long max_abs_bits;
    unsigned long long max_relative_bits;
};

__global__ void compare_vectors_kernel(
    const HPRLP_FLOAT *reference, const HPRLP_FLOAT *candidate,
    std::size_t count, HPRLP_device_vector_compare_result *result) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        if (reference[index] == candidate[index]) continue;
        atomicAdd(&result->mismatch_count, 1ULL);
        HPRLP_FLOAT difference = fabs(reference[index] - candidate[index]);
        HPRLP_FLOAT relative = difference /
            fmax(static_cast<HPRLP_FLOAT>(1.0), fabs(reference[index]));
        if (!isfinite(difference)) difference = INFINITY;
        if (!isfinite(relative)) relative = INFINITY;
        atomicMax(
            &result->max_abs_bits,
            static_cast<unsigned long long>(__double_as_longlong(difference)));
        atomicMax(
            &result->max_relative_bits,
            static_cast<unsigned long long>(__double_as_longlong(relative)));
    }
}

HPRLP_FLOAT hprlp_nonnegative_double_from_bits(
    unsigned long long bits) {
    HPRLP_FLOAT value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

HPRLP_device_vector_compare_result compare_vectors_gpu(
    const HPRLP_FLOAT *reference, const HPRLP_FLOAT *candidate,
    std::size_t count, cudaStream_t stream) {
    HPRLP_device_vector_compare_result *device_result = nullptr;
    allocate_device(&device_result, 1);
    CUDA_CHECK(cudaMemsetAsync(
        device_result, 0, sizeof(*device_result), stream));
    if (count > 0) {
        constexpr int threads = 256;
        const std::size_t required = (count + threads - 1) / threads;
        const int blocks = static_cast<int>(
            required < 4096 ? required : 4096);
        compare_vectors_kernel<<<blocks, threads, 0, stream>>>(
            reference, candidate, count, device_result);
        CUDA_CHECK(cudaGetLastError());
    }
    HPRLP_device_vector_compare_result host_result{};
    CUDA_CHECK(cudaMemcpyAsync(
        &host_result, device_result, sizeof(host_result),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(device_result);
    return host_result;
}

void prepare_reduced_nonempty_cusparse_y(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool allow_autotune_candidate = false) {
    if (workspace == nullptr || state == nullptr ||
        !reduced_nonempty_cusparse_solve_requested()) {
        return;
    }
    const bool generic_y_candidate =
        !state->use_packed_dictionary &&
        !state->use_signed_factorized &&
        !state->use_unit_factorized;
    if ((!reduced_uses_generic_y(state) &&
         !(allow_autotune_candidate && generic_y_candidate)) ||
        state->base_count <= 0 ||
        state->spmv_A.cusparseHandle == nullptr) {
        return;
    }
    if (!allow_autotune_candidate &&
        !state->defer_empty_rows_selected) {
        state->defer_empty_rows_to_observation = false;
    }

    const bool needs_nonempty_to_original =
        state->defer_empty_rows_to_observation;
    state->nonempty_A_count = build_nonempty_row_metadata_gpu(
        state->A.rowPtr, workspace->m, state->A.numElements,
        needs_nonempty_to_original, &state->original_to_nonempty_A,
        &state->nonempty_to_original_A, &state->nonempty_A_row_ptr,
        workspace->stream);
    allocate_device(&state->nonempty_Ax, state->nonempty_A_count);

    CUSPARSE_spmvop_A &full = state->spmv_A;
    check_cusparse(cusparseCreateCsr(
        &state->nonempty_A_descr, state->nonempty_A_count, state->A.col,
        state->A.numElements, state->nonempty_A_row_ptr,
        state->A.colIndex, state->A.value, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "create production nonempty reduced A");
    check_cusparse(cusparseCreateDnVec(
        &state->nonempty_x_hat_descr, state->base_count, state->x_hat,
        CUDA_R_64F), "create production nonempty reduced xhat");
    check_cusparse(cusparseCreateDnVec(
        &state->nonempty_Ax_descr, state->nonempty_A_count,
        state->nonempty_Ax, CUDA_R_64F),
        "create production nonempty reduced Ax");
    check_cusparse(hprlp_prepare_spmvop(
        full.cusparseHandle, state->nonempty_A_descr,
        state->nonempty_x_hat_descr, state->nonempty_Ax_descr,
        state->nonempty_Ax_descr, full.computeType,
        &state->nonempty_A_operation),
        "prepare production nonempty reduced A SpMVOp");
    state->use_nonempty_cusparse_y = true;
    if (!allow_autotune_candidate && state->profile_enabled) {
        std::cout << "  reduced nonempty cuSPARSE Y: rows="
                  << state->nonempty_A_count << "/" << workspace->m
                  << std::endl;
    }
}

void prepare_parallel_delta_cusparse_y(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (!state->use_nonempty_cusparse_y ||
        state->delta_count < kReducedParallelDeltaMinimumColumns ||
        state->delta_nnz < kReducedParallelDeltaMinimumNnz) {
        return;
    }

    allocate_device(&state->parallel_delta_input, state->delta_count);
    allocate_device(&state->parallel_delta_ax, workspace->m);
    CUSPARSE_spmvop_A &full = state->spmv_A;
    check_cusparse(cusparseCreateCsr(
        &state->parallel_delta_A_descr, workspace->m,
        state->delta_count, state->delta_nnz, state->delta_A.rowPtr,
        state->delta_A.colIndex, state->delta_A.value,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "create reduced parallel delta A");
    check_cusparse(cusparseCreateDnVec(
        &state->parallel_delta_input_descr, state->delta_count,
        state->parallel_delta_input, CUDA_R_64F),
        "create reduced parallel delta input");
    check_cusparse(cusparseCreateDnVec(
        &state->parallel_delta_ax_descr, workspace->m,
        state->parallel_delta_ax, CUDA_R_64F),
        "create reduced parallel delta Ax");
    check_cusparse(hprlp_prepare_spmvop(
        full.cusparseHandle, state->parallel_delta_A_descr,
        state->parallel_delta_input_descr,
        state->parallel_delta_ax_descr,
        state->parallel_delta_ax_descr, full.computeType,
        &state->parallel_delta_operation),
        "prepare reduced parallel delta A SpMVOp");
    state->use_parallel_delta_cusparse_y = true;
    std::cout << "  reduced parallel delta cuSPARSE Y: columns="
              << state->delta_count << ", nnz=" << state->delta_nnz
              << std::endl;
}

void compute_fixed_shift(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    HPRLP_FLOAT *fixed_values = nullptr;
    allocate_device(&fixed_values, workspace->n);
    fixed_values_from_mask_kernel<<<
        numBlocks(workspace->n), numThreads, 0, workspace->stream>>>(
        fixed_values, state->x_bar_mask, workspace->l, workspace->u,
        workspace->n);

    // SpMVOp only supports NON_TRANSPOSE, so compute the algebraically
    // equivalent A*x_fixed using the explicit CSR descriptor and plan for A.
    CUSPARSE_spmvop_A *spmv = workspace->spmv_A;
    cusparseDnVecDescr_t fixed_values_descr = nullptr;
    cusparseDnVecDescr_t fixed_shift_descr = nullptr;
    check_cusparse(cusparseCreateDnVec(
        &fixed_values_descr, workspace->n, fixed_values, CUDA_R_64F),
        "create fixed values descriptor");
    check_cusparse(cusparseCreateDnVec(
        &fixed_shift_descr, workspace->m, state->row_fixed_shift,
        CUDA_R_64F), "create fixed shift descriptor");
    HPRLP_FLOAT alpha = 1.0;
    HPRLP_FLOAT beta = 0.0;
    check_cusparse(hprlp_run_spmvop(
        spmv->cusparseHandle, spmv->operation, &alpha, &beta,
        fixed_values_descr, fixed_shift_descr, fixed_shift_descr),
        "compute fixed shift with A SpMVOp");
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    cusparseDestroyDnVec(fixed_values_descr);
    cusparseDestroyDnVec(fixed_shift_descr);
    cudaFree(fixed_values);
}

void build_row_buckets(
    const std::vector<int> &row_ptr,
    int **short_rows,
    int *short_count,
    int **warp_rows,
    int *warp_count,
    cudaStream_t stream) {
    std::vector<int> short_host;
    std::vector<int> warp_host;
    short_host.reserve(row_ptr.size());
    warp_host.reserve(row_ptr.size() / 2);
    for (int row = 0; row + 1 < static_cast<int>(row_ptr.size()); ++row) {
        if (row_ptr[row + 1] - row_ptr[row] <= 16) {
            short_host.push_back(row);
        } else {
            warp_host.push_back(row);
        }
    }
    *short_count = static_cast<int>(short_host.size());
    *warp_count = static_cast<int>(warp_host.size());
    allocate_device(short_rows, short_host.size());
    allocate_device(warp_rows, warp_host.size());
    if (!short_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *short_rows, short_host.data(), short_host.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream));
    if (!warp_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *warp_rows, warp_host.data(), warp_host.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream));
}

void build_signed_combined_buckets(
    const std::vector<int> &row_ptr,
    int **medium_rows,
    int *medium_count,
    int **long_rows,
    int *long_count,
    cudaStream_t stream) {
    std::vector<int> medium_host;
    std::vector<int> long_host;
    medium_host.reserve(row_ptr.size() / 2);
    long_host.reserve(row_ptr.size() / 64);
    for (int row = 0; row + 1 < static_cast<int>(row_ptr.size()); ++row) {
        const int row_nnz = row_ptr[row + 1] - row_ptr[row];
        const HPRLPRowBucket bucket = hprlp_row_bucket(row_nnz);
        if (bucket == HPRLP_ROW_WARP) {
            medium_host.push_back(row);
        } else if (bucket == HPRLP_ROW_BLOCK) {
            long_host.push_back(row);
        }
    }
    *medium_count = static_cast<int>(medium_host.size());
    *long_count = static_cast<int>(long_host.size());
    allocate_device(medium_rows, medium_host.size());
    allocate_device(long_rows, long_host.size());
    if (!medium_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *medium_rows, medium_host.data(), medium_host.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream));
    if (!long_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *long_rows, long_host.data(), long_host.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream));
}

void build_signed_empty_batch_rows(
    const std::vector<int> &row_ptr,
    int **short_nonempty_rows,
    int *short_nonempty_count,
    int **empty_rows,
    int *empty_count,
    cudaStream_t stream) {
    std::vector<int> short_nonempty_host;
    std::vector<int> empty_host;
    short_nonempty_host.reserve(row_ptr.size() / 2);
    empty_host.reserve(row_ptr.size() / 2);
    for (int row = 0; row + 1 < static_cast<int>(row_ptr.size()); ++row) {
        const int row_nnz = row_ptr[row + 1] - row_ptr[row];
        if (row_nnz == 0) {
            empty_host.push_back(row);
        } else if (row_nnz <= HPRLP_SCALAR_ROW_MAX_NNZ) {
            short_nonempty_host.push_back(row);
        }
    }
    *short_nonempty_count =
        static_cast<int>(short_nonempty_host.size());
    *empty_count = static_cast<int>(empty_host.size());
    allocate_device(short_nonempty_rows, short_nonempty_host.size());
    allocate_device(empty_rows, empty_host.size());
    if (!short_nonempty_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *short_nonempty_rows, short_nonempty_host.data(),
        short_nonempty_host.size() * sizeof(int), cudaMemcpyHostToDevice,
        stream));
    if (!empty_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *empty_rows, empty_host.data(), empty_host.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream));
}

void build_scan_empty_batch_rows(
    const std::vector<int> &row_ptr,
    int **short_nonempty_rows,
    int *short_nonempty_count,
    int *empty_count,
    int minimum_empty_percent,
    cudaStream_t stream) {
    std::vector<int> short_nonempty_host;
    short_nonempty_host.reserve(row_ptr.size() / 8);
    int empty_host_count = 0;
    for (int row = 0; row + 1 < static_cast<int>(row_ptr.size()); ++row) {
        const int row_nnz = row_ptr[row + 1] - row_ptr[row];
        if (row_nnz == 0) {
            ++empty_host_count;
        } else if (row_nnz <= HPRLP_SCALAR_ROW_MAX_NNZ) {
            short_nonempty_host.push_back(row);
        }
    }
    *empty_count = empty_host_count;
    const std::size_t row_count = row_ptr.empty() ? 0 : row_ptr.size() - 1;
    if (empty_host_count == 0 ||
        static_cast<long long>(empty_host_count) * 100 <
            static_cast<long long>(row_count) *
                minimum_empty_percent) {
        *short_nonempty_rows = nullptr;
        *short_nonempty_count = 0;
        return;
    }
    *short_nonempty_count =
        static_cast<int>(short_nonempty_host.size());
    allocate_device(short_nonempty_rows, short_nonempty_host.size());
    if (!short_nonempty_host.empty()) CUDA_CHECK(cudaMemcpyAsync(
        *short_nonempty_rows, short_nonempty_host.data(),
        short_nonempty_host.size() * sizeof(int), cudaMemcpyHostToDevice,
        stream));
}

void build_device_medium_long_buckets(
    const int *row_ptr,
    int row_count,
    int medium_count,
    int long_count,
    int **row_buckets,
    int **medium_rows,
    int **long_rows,
    int *scratch_counts,
    cudaStream_t stream) {
    *row_buckets = nullptr;
    *medium_rows = nullptr;
    *long_rows = nullptr;
    if (row_count <= 0 || (medium_count <= 0 && long_count <= 0)) return;
    allocate_device(row_buckets, row_count);
    CUDA_CHECK(cudaMemsetAsync(
        scratch_counts, 0, 2 * sizeof(int), stream));
    collect_compact_medium_long_rows_kernel<<<
        numBlocks(row_count), numThreads, 0, stream>>>(
        row_ptr, *row_buckets, scratch_counts, row_count);
    if (medium_count > 0) *medium_rows = *row_buckets;
    if (long_count > 0) {
        *long_rows = *row_buckets + row_count - long_count;
    }
}

void build_device_empty_short_buckets(
    const int *row_ptr,
    int row_count,
    int short_nonempty_count,
    int empty_count,
    int **row_buckets,
    int **short_nonempty_rows,
    int **empty_rows,
    int *scratch_counts,
    cudaStream_t stream) {
    *row_buckets = nullptr;
    *short_nonempty_rows = nullptr;
    *empty_rows = nullptr;
    if (row_count <= 0 ||
        (short_nonempty_count <= 0 && empty_count <= 0)) return;
    allocate_device(row_buckets, row_count);
    CUDA_CHECK(cudaMemsetAsync(
        scratch_counts, 0, 2 * sizeof(int), stream));
    collect_compact_empty_short_rows_kernel<<<
        numBlocks(row_count), numThreads, 0, stream>>>(
        row_ptr, *row_buckets, scratch_counts, row_count);
    if (short_nonempty_count > 0) {
        *short_nonempty_rows = *row_buckets;
    }
    if (empty_count > 0) {
        *empty_rows = *row_buckets + row_count - empty_count;
    }
}

bool build_reduced_workspace(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters,
    int iteration) {
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    const auto start = std::chrono::steady_clock::now();

    // Julia's findall(x_bar_mask .== INTERIOR) is a stable device-side
    // selection.  A warp-count prefix scan followed by an ordered scatter
    // preserves that exact ascending original-column order without copying
    // the full mask to host.
    if (workspace->n > 0) {
        count_interior_columns_by_warp_kernel<<<
            numBlocks(workspace->n), numThreads, 0, workspace->stream>>>(
            state->x_bar_mask, state->free_count_by_warp_device,
            workspace->n);
        std::size_t select_scan_temp_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr, select_scan_temp_bytes,
            state->free_count_by_warp_device,
            state->free_count_by_warp_device,
            state->free_count_warp_count, workspace->stream));
        ensure_device_byte_capacity(
            &state->construction_temp,
            &state->construction_temp_capacity,
            select_scan_temp_bytes);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            state->construction_temp, select_scan_temp_bytes,
            state->free_count_by_warp_device,
            state->free_count_by_warp_device,
            state->free_count_warp_count, workspace->stream));
        scatter_interior_column_indices_kernel<<<
            numBlocks(workspace->n), numThreads, 0, workspace->stream>>>(
            state->x_bar_mask, state->free_count_by_warp_device,
            state->delta_indices_device, workspace->n);
    }
    state->base_count = state->free_count;
    state->delta_work_accum = 0;

    allocate_device(&state->free_to_original, state->free_count);
    if (state->free_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            state->free_to_original, state->delta_indices_device,
            static_cast<std::size_t>(state->free_count) * sizeof(int),
            cudaMemcpyDeviceToDevice, workspace->stream));
    }

    state->AT.row = state->free_count;
    state->AT.col = workspace->m;
    allocate_device(
        &state->AT.rowPtr, static_cast<std::size_t>(state->free_count) + 1);
    CUDA_CHECK(cudaMemsetAsync(
        state->AT.rowPtr, 0, sizeof(int), workspace->stream));
    if (state->free_count > 0) {
        selected_csr_row_lengths_kernel<<<
            numBlocks(state->free_count), numThreads, 0,
            workspace->stream>>>(
            state->AT.rowPtr, state->free_to_original,
            workspace->AT->rowPtr, state->free_count);
    }

    std::size_t scan_temp_bytes = 0;
    if (state->free_count > 0) {
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr, scan_temp_bytes, state->AT.rowPtr + 1,
            state->AT.rowPtr + 1, state->free_count, workspace->stream));
        ensure_device_byte_capacity(
            &state->construction_temp,
            &state->construction_temp_capacity,
            scan_temp_bytes);
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            state->construction_temp, scan_temp_bytes,
            state->AT.rowPtr + 1,
            state->AT.rowPtr + 1, state->free_count, workspace->stream));
    }
    int reduced_nnz = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &reduced_nnz, state->AT.rowPtr + state->free_count, sizeof(int),
        cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->AT.numElements = reduced_nnz;
    state->base_reduced_nnz = reduced_nnz;
    allocate_device(&state->AT.colIndex, reduced_nnz);
    allocate_device(&state->AT.value, reduced_nnz);
    if (state->free_count > 0 && reduced_nnz > 0) {
        copy_selected_csr_rows_kernel<<<
            state->free_count, 256, 0, workspace->stream>>>(
            state->AT.colIndex, state->AT.value, state->AT.rowPtr,
            state->free_to_original, workspace->AT->rowPtr,
            workspace->AT->colIndex, workspace->AT->value,
            state->free_count);
    }
    transpose_csr(
        state->AT, &state->A, workspace->stream,
        state->transpose_handle, &state->construction_temp,
        &state->construction_temp_capacity);
    allocate_device(&state->AT_row_buckets, state->free_count);
    allocate_device(&state->A_row_buckets, workspace->m);
    CUDA_CHECK(cudaMemsetAsync(
        state->base_row_analysis_device, 0,
        2 * kReducedRowAnalysisFields * sizeof(int), workspace->stream));
    if (state->free_count > 0) {
        analyze_compact_rows_kernel<<<
            numBlocks(state->free_count), numThreads, 0,
            workspace->stream>>>(
            state->AT.rowPtr, state->AT_row_buckets,
            state->base_row_analysis_device, state->free_count);
    }
    if (workspace->m > 0) {
        analyze_compact_rows_kernel<<<
            numBlocks(workspace->m), numThreads, 0,
            workspace->stream>>>(
            state->A.rowPtr, state->A_row_buckets,
            state->base_row_analysis_device + kReducedRowAnalysisFields,
            workspace->m);
    }
    int row_analysis[2 * kReducedRowAnalysisFields] = {};
    CUDA_CHECK(cudaMemcpyAsync(
        row_analysis, state->base_row_analysis_device,
        sizeof(row_analysis), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    const int *const at_analysis = row_analysis;
    const int *const a_analysis =
        row_analysis + kReducedRowAnalysisFields;
    state->AT_short_count = at_analysis[kReducedRowShortCount];
    state->AT_warp_count = at_analysis[kReducedRowWarpCount];
    state->AT_rows_short = state->AT_short_count > 0
        ? state->AT_row_buckets : nullptr;
    state->AT_rows_warp = state->AT_warp_count > 0
        ? state->AT_row_buckets + state->free_count - state->AT_warp_count
        : nullptr;
    state->A_short_count = a_analysis[kReducedRowShortCount];
    state->A_warp_count = a_analysis[kReducedRowWarpCount];
    state->A_rows_short = state->A_short_count > 0
        ? state->A_row_buckets : nullptr;
    state->A_rows_warp = state->A_warp_count > 0
        ? state->A_row_buckets + workspace->m - state->A_warp_count
        : nullptr;
    const bool seed_fused_x = workspace->reduced_backend_autotune_done
        ? workspace->reduced_use_fused_x
        : workspace->x_backend == HPRLPXBackend::GenericFused;
    const bool seed_fused_y = workspace->reduced_backend_autotune_done
        ? workspace->reduced_use_fused_y
        : workspace->y_backend == HPRLPYBackend::GenericFused;
    state->use_fused_x = parameters != nullptr &&
        !parameters->CUSPARSE_spmv && seed_fused_x;
    state->use_fused_y = parameters != nullptr &&
        !parameters->CUSPARSE_spmv && seed_fused_y;
    const bool selected_unit_y_backend =
        workspace->x_backend == HPRLPXBackend::UnitFactorized &&
        (workspace->y_backend == HPRLPYBackend::UnitColTile ||
         workspace->y_backend == HPRLPYBackend::UnitColTileZeroBitset ||
         workspace->y_backend == HPRLPYBackend::UnitActiveScatter);
    state->use_unit_factorized = parameters != nullptr &&
        !parameters->CUSPARSE_spmv && selected_unit_y_backend &&
        workspace->unit_operator_x_ready && workspace->unit_coltile_ready &&
        workspace->uniform_unit_sign != 0 &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_y != nullptr &&
        workspace->unit_AT_col_index_u16 != nullptr &&
        workspace->m <= 65536;
    state->use_unit_active_scatter = state->use_unit_factorized &&
        workspace->y_backend == HPRLPYBackend::UnitActiveScatter;
    const int reduced_max_AT_row_nnz =
        at_analysis[kReducedRowMaxNnz];
    const int reduced_max_A_row_nnz =
        a_analysis[kReducedRowMaxNnz];
    const bool selected_signed_split_x =
        workspace->x_backend == HPRLPXBackend::SignedUnitSplitU16;
    const bool selected_signed_packed_x =
        workspace->x_backend == HPRLPXBackend::SignedUnitPacked;
    const bool selected_signed_combined_y =
        workspace->y_backend == HPRLPYBackend::SignedUnitPackedCombined;
    const bool selected_signed_packed_y =
        workspace->y_backend == HPRLPYBackend::SignedUnitPacked;
    const bool selected_signed_y =
        selected_signed_combined_y || selected_signed_packed_y;
    const bool reduced_signed_x_is_scalar = hprlp_all_rows_fit_scalar(
        state->free_count, reduced_max_AT_row_nnz,
        HPRLP_UNIT_SCALAR_ROW_MAX_NNZ);
    const bool signed_x_shape_supported =
        (selected_signed_split_x && workspace->m <= 65536 &&
         reduced_signed_x_is_scalar) ||
        selected_signed_packed_x;
    state->use_signed_factorized = parameters != nullptr &&
        !parameters->CUSPARSE_spmv && selected_signed_y &&
        signed_x_shape_supported &&
        workspace->signed_unit_operator_ready &&
        workspace->signed_unit_operator != nullptr &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_y != nullptr;
    state->signed_x_uses_split_u16 = state->use_signed_factorized &&
        selected_signed_split_x;
    state->signed_x_uses_scalar = state->use_signed_factorized &&
        reduced_signed_x_is_scalar;
    const bool reduced_signed_y_is_scalar = hprlp_all_rows_fit_scalar(
        workspace->m, reduced_max_A_row_nnz,
        HPRLP_UNIT_SCALAR_ROW_MAX_NNZ);
    const bool compact_prefers_combined_y =
        !reduced_signed_y_is_scalar &&
        static_cast<long long>(state->A_short_count) * 100 >=
            static_cast<long long>(workspace->m) * 99;
    state->signed_y_uses_combined = state->use_signed_factorized &&
        (selected_signed_combined_y || compact_prefers_combined_y);
    state->signed_y_uses_scalar = state->use_signed_factorized &&
        selected_signed_packed_y && !state->signed_y_uses_combined &&
        reduced_signed_y_is_scalar;
    state->signed_AT_uses_u16 = state->use_signed_factorized &&
        !state->signed_x_uses_split_u16 && workspace->m <= 32768;
    state->signed_A_uses_u16 = state->use_signed_factorized &&
        state->free_count > 0 && state->free_count <= 32768;
    if (state->use_signed_factorized &&
        !state->signed_x_uses_scalar) {
        state->signed_AT_medium_count =
            at_analysis[kReducedRowMediumCount];
        state->signed_AT_long_count =
            at_analysis[kReducedRowLongCount];
        build_device_medium_long_buckets(
            state->AT.rowPtr, state->free_count,
            state->signed_AT_medium_count, state->signed_AT_long_count,
            &state->signed_AT_row_buckets,
            &state->signed_AT_rows_medium, &state->signed_AT_rows_long,
            state->base_row_analysis_device, workspace->stream);
    }
    if (state->signed_y_uses_combined ||
        (state->use_signed_factorized &&
         !state->signed_y_uses_scalar)) {
        state->signed_A_medium_count =
            a_analysis[kReducedRowMediumCount];
        state->signed_A_long_count =
            a_analysis[kReducedRowLongCount];
        build_device_medium_long_buckets(
            state->A.rowPtr, workspace->m,
            state->signed_A_medium_count, state->signed_A_long_count,
            &state->signed_A_row_buckets,
            &state->signed_A_rows_medium, &state->signed_A_rows_long,
            state->base_row_analysis_device, workspace->stream);
    }
    state->use_signed_empty_row_batch =
        state->use_signed_factorized && state->signed_y_uses_combined;
    if (state->use_signed_empty_row_batch) {
        state->signed_A_short_nonempty_count =
            a_analysis[kReducedRowShortNonemptyCount];
        state->signed_A_empty_count =
            a_analysis[kReducedRowEmptyCount];
        state->use_signed_empty_row_batch =
            state->signed_A_empty_count >=
                kReducedSignedEmptyBatchMinimumRows;
        if (state->use_signed_empty_row_batch) {
            build_device_empty_short_buckets(
                state->A.rowPtr, workspace->m,
                state->signed_A_short_nonempty_count,
                state->signed_A_empty_count,
                &state->signed_A_empty_row_buckets,
                &state->signed_A_rows_short_nonempty,
                &state->signed_A_rows_empty,
                state->base_row_analysis_device, workspace->stream);
        }
    }
    const bool selected_fixed_degree_dictionary_x =
        workspace->x_backend ==
            HPRLPXBackend::FixedDegreePackedDictionary;
    const bool selected_packed_dictionary_x =
        workspace->x_backend == HPRLPXBackend::PackedDictionary ||
        selected_fixed_degree_dictionary_x;
    const bool selected_packed_dictionary_pair =
        selected_packed_dictionary_x &&
        workspace->y_backend == HPRLPYBackend::PackedDictionary;
    const auto packed_indices_fit = [](int index_count,
                                       unsigned code_bits) {
        if (index_count <= 0 || code_bits >= 32) return false;
        const unsigned index_bits = 32 - code_bits;
        const std::uint64_t capacity = UINT64_C(1) << index_bits;
        return static_cast<std::uint64_t>(index_count) <= capacity;
    };
    state->use_packed_dictionary_x = parameters != nullptr &&
        !parameters->CUSPARSE_spmv && selected_packed_dictionary_x &&
        workspace->dictionary_operator_x_ready &&
        workspace->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32 &&
        workspace->coefficient_dictionary != nullptr &&
        workspace->coefficient_dictionary_size > 0 &&
        workspace->AT_dictionary_packed_u32 != nullptr &&
        workspace->inverse_row_norm != nullptr &&
        workspace->inverse_col_norm != nullptr &&
        workspace->unit_scaled_y != nullptr &&
        packed_indices_fit(
            workspace->m, workspace->packed_dictionary_code_bits);
    state->use_packed_dictionary =
        state->use_packed_dictionary_x &&
        selected_packed_dictionary_pair &&
        workspace->dictionary_operator_y_ready &&
        workspace->A_packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32 &&
        workspace->A_coefficient_dictionary != nullptr &&
        workspace->A_coefficient_dictionary_size > 0 &&
        workspace->A_dictionary_packed_u32 != nullptr &&
        workspace->unit_scaled_x_hat != nullptr &&
        packed_indices_fit(
            state->free_count,
            workspace->A_packed_dictionary_code_bits);
    const bool autotune_generic_reduced_backends =
        reduced_compressed_autotune_enabled() &&
        parameters != nullptr && !parameters->CUSPARSE_spmv &&
        !workspace->reduced_backend_autotune_done &&
        state->base_count > 0 && workspace->m > 0 &&
        !state->use_unit_factorized &&
        !state->use_signed_factorized &&
        !state->use_packed_dictionary_x;
    int uniform_compact_AT_degree = 0;
    if (state->use_packed_dictionary_x && state->free_count > 0) {
        uniform_compact_AT_degree = at_analysis[kReducedRowNonuniform] == 0
            ? at_analysis[kReducedRowFirstDegree] : 0;
        if (uniform_compact_AT_degree < 1 ||
            uniform_compact_AT_degree > HPRLP_SCALAR_ROW_MAX_NNZ) {
            uniform_compact_AT_degree = 0;
        }
    }
    state->use_fixed_degree_dictionary_x =
        state->use_packed_dictionary_x &&
        selected_fixed_degree_dictionary_x &&
        uniform_compact_AT_degree != 0;
    state->fixed_degree_dictionary_x_degree =
        state->use_fixed_degree_dictionary_x
            ? uniform_compact_AT_degree : 0;
    if (state->use_packed_dictionary_x) {
        state->signed_AT_medium_count =
            at_analysis[kReducedRowMediumCount];
        state->signed_AT_long_count =
            at_analysis[kReducedRowLongCount];
        build_device_medium_long_buckets(
            state->AT.rowPtr, state->free_count,
            state->signed_AT_medium_count, state->signed_AT_long_count,
            &state->signed_AT_row_buckets,
            &state->signed_AT_rows_medium, &state->signed_AT_rows_long,
            state->base_row_analysis_device, workspace->stream);
    }
    if (state->use_packed_dictionary) {
        state->signed_A_medium_count =
            a_analysis[kReducedRowMediumCount];
        state->signed_A_long_count =
            a_analysis[kReducedRowLongCount];
        build_device_medium_long_buckets(
            state->A.rowPtr, workspace->m,
            state->signed_A_medium_count, state->signed_A_long_count,
            &state->signed_A_row_buckets,
            &state->signed_A_rows_medium, &state->signed_A_rows_long,
            state->base_row_analysis_device, workspace->stream);
        state->signed_A_empty_count =
            a_analysis[kReducedRowEmptyCount];
        state->use_dictionary_empty_row_batch =
            static_cast<long long>(state->signed_A_empty_count) * 100 >=
                static_cast<long long>(workspace->m) *
                    kReducedDictionaryEmptyBatchMinimumPercent;
        if (state->use_dictionary_empty_row_batch) {
            state->signed_A_short_nonempty_count =
                a_analysis[kReducedRowShortNonemptyCount];
            build_device_empty_short_buckets(
                state->A.rowPtr, workspace->m,
                state->signed_A_short_nonempty_count,
                state->signed_A_empty_count,
                &state->signed_A_empty_row_buckets,
                &state->signed_A_rows_short_nonempty,
                &state->signed_A_rows_empty,
                state->base_row_analysis_device, workspace->stream);
        }
    } else if (!state->use_signed_factorized &&
               !state->use_unit_active_scatter &&
               !state->use_unit_factorized &&
               (!state->use_fused_y ||
                autotune_generic_reduced_backends ||
                (state->defer_empty_rows_selected &&
                 !state->use_packed_dictionary_x &&
                 reduced_defer_empty_rows_enabled()))) {
        state->signed_A_empty_count =
            a_analysis[kReducedRowEmptyCount];
        state->use_cusparse_empty_row_batch =
            static_cast<long long>(state->signed_A_empty_count) * 100 >=
                static_cast<long long>(workspace->m) *
                    kReducedCusparseEmptyBatchMinimumPercent;
        if (state->use_cusparse_empty_row_batch) {
            state->signed_A_short_nonempty_count =
                a_analysis[kReducedRowShortNonemptyCount];
            build_device_empty_short_buckets(
                state->A.rowPtr, workspace->m,
                state->signed_A_short_nonempty_count,
                state->signed_A_empty_count,
                &state->signed_A_empty_row_buckets,
                &state->signed_A_rows_short_nonempty,
                &state->signed_A_rows_empty,
                state->base_row_analysis_device, workspace->stream);
        }
    }
    state->defer_empty_rows_to_observation =
        reduced_defer_empty_rows_enabled() &&
        state->signed_A_rows_empty != nullptr &&
        state->signed_A_empty_count > 0 &&
        (state->use_signed_empty_row_batch ||
         state->use_dictionary_empty_row_batch ||
         state->use_cusparse_empty_row_batch);
    state->deferred_empty_pending_iterations = 0;
    const bool trace_base_empty_rows =
        std::getenv("HPRLP_TRACE_REDUCED_BASE_EMPTY") != nullptr;
    if ((parameters != nullptr && parameters->autotune_verbose) ||
        trace_base_empty_rows) {
        const char *const reduced_x_backend = state->use_packed_dictionary_x
            ? (state->use_fixed_degree_dictionary_x
                ? "fixed-degree-packed-dictionary"
                : "packed-dictionary")
            : (state->use_signed_factorized
            ? (state->signed_x_uses_split_u16
                ? "signed-split-u16" : "signed-packed")
            : (state->use_unit_factorized
                ? "unit-factor"
                : (state->use_fused_x ? "fused" : "cusparse")));
        const char *const reduced_y_backend = state->use_unit_active_scatter
            ? "unit-active-scatter"
            : (state->use_packed_dictionary
                ? "packed-dictionary"
                : (state->use_signed_factorized
                ? (state->signed_y_uses_combined
                    ? "signed-packed-combined" : "signed-packed")
                : (state->use_unit_factorized
                    ? "unit-csr-block"
                    : (state->use_fused_y ? "fused" : "cusparse"))));
        std::cout << "  reduced backends: x="
                  << reduced_x_backend << ", y=" << reduced_y_backend
                  << std::endl;
        if (trace_base_empty_rows) {
            const int base_empty_count =
                a_analysis[kReducedRowEmptyCount];
            const double base_empty_ratio = workspace->m > 0
                ? static_cast<double>(base_empty_count) / workspace->m
                : 0.0;
            std::cout << "  reduced base-row stats: iteration=" << iteration
                      << " rows=" << workspace->m
                      << " empty=" << base_empty_count
                      << " empty_ratio=" << base_empty_ratio
                      << " x_backend=" << reduced_x_backend
                      << " y_backend=" << reduced_y_backend
                      << std::endl;
        }
        if (state->use_signed_empty_row_batch) {
            std::cout << "  reduced signed empty-row batch: empty="
                      << state->signed_A_empty_count
                      << ", short_nonempty="
                      << state->signed_A_short_nonempty_count
                      << ", medium=" << state->signed_A_medium_count
                      << ", long=" << state->signed_A_long_count
                      << std::endl;
        }
        if (state->use_dictionary_empty_row_batch) {
            std::cout << "  reduced dictionary empty-row batch: empty="
                      << state->signed_A_empty_count
                      << ", short_nonempty="
                      << state->signed_A_short_nonempty_count
                      << ", medium=" << state->signed_A_medium_count
                      << ", long=" << state->signed_A_long_count
                      << std::endl;
        }
        if (state->use_cusparse_empty_row_batch) {
            std::cout << "  reduced cusparse empty-row candidate: empty="
                      << state->signed_A_empty_count
                      << ", short_nonempty="
                      << state->signed_A_short_nonempty_count
                      << ", warp=" << state->A_warp_count
                      << std::endl;
        }
    }

    allocate_device(&state->row_fixed_shift, workspace->m);
    if (state->use_unit_factorized) {
        allocate_device(
            &state->unit_AT_constraint_index, state->AT.numElements);
        if (state->AT.numElements > 0) {
            compact_u16_indices_kernel<<<
                numBlocks(state->AT.numElements), numThreads, 0,
                workspace->stream>>>(
                state->unit_AT_constraint_index, state->AT.colIndex,
                state->AT.numElements);
        }
    }
    if (state->use_signed_factorized) {
        if (state->signed_x_uses_split_u16) {
            allocate_device(
                &state->signed_AT_constraint_index,
                state->AT.numElements);
            allocate_device(
                &state->signed_AT_negative, state->AT.numElements);
            if (state->AT.numElements > 0) {
                pack_signed_split_u16_kernel<<<
                    numBlocks(state->AT.numElements), numThreads, 0,
                    workspace->stream>>>(
                    state->signed_AT_constraint_index,
                    state->signed_AT_negative, state->AT.colIndex,
                    state->AT.value, state->AT.numElements);
            }
        } else if (state->signed_AT_uses_u16) {
            allocate_device(
                &state->signed_AT_entries_u16, state->AT.numElements);
            if (state->AT.numElements > 0) {
                pack_signed_entries_u16_kernel<<<
                    numBlocks(state->AT.numElements), numThreads, 0,
                    workspace->stream>>>(
                    state->signed_AT_entries_u16, state->AT.colIndex,
                    state->AT.value, state->AT.numElements);
            }
        } else {
            allocate_device(
                &state->signed_AT_entries_u32, state->AT.numElements);
            if (state->AT.numElements > 0) {
                pack_signed_entries_u32_kernel<<<
                    numBlocks(state->AT.numElements), numThreads, 0,
                    workspace->stream>>>(
                    state->signed_AT_entries_u32, state->AT.colIndex,
                    state->AT.value, state->AT.numElements);
            }
        }
        if (state->signed_A_uses_u16) {
            allocate_device(
                &state->signed_A_entries_u16, state->A.numElements);
            if (state->A.numElements > 0) {
                pack_signed_entries_u16_kernel<<<
                    numBlocks(state->A.numElements), numThreads, 0,
                    workspace->stream>>>(
                    state->signed_A_entries_u16, state->A.colIndex,
                    state->A.value, state->A.numElements);
            }
        } else {
            allocate_device(
                &state->signed_A_entries_u32, state->A.numElements);
            if (state->A.numElements > 0) {
                pack_signed_entries_u32_kernel<<<
                    numBlocks(state->A.numElements), numThreads, 0,
                    workspace->stream>>>(
                    state->signed_A_entries_u32, state->A.colIndex,
                    state->A.value, state->A.numElements);
            }
        }
    }
    if (state->use_packed_dictionary_x) {
        allocate_device(
            &state->dictionary_AT_entries_u32, state->AT.numElements);
        if (state->free_count > 0 && state->AT.numElements > 0) {
            copy_selected_packed_dictionary_rows_kernel<<<
                state->free_count, 256, 0, workspace->stream>>>(
                state->dictionary_AT_entries_u32, state->AT.rowPtr,
                state->free_to_original, workspace->AT->rowPtr,
                workspace->AT_dictionary_packed_u32,
                state->free_count);
        }
        if (state->use_fixed_degree_dictionary_x) {
            allocate_device(
                &state->dictionary_AT_entries_soa,
                state->AT.numElements);
            if (state->AT.numElements > 0) {
                repack_uniform_degree_dictionary_rows_soa_kernel<<<
                    numBlocks(state->AT.numElements), numThreads, 0,
                    workspace->stream>>>(
                    state->dictionary_AT_entries_u32,
                    state->dictionary_AT_entries_soa,
                    state->free_count,
                    state->fixed_degree_dictionary_x_degree,
                    state->AT.numElements);
            }
        }
        if (state->use_packed_dictionary) {
            allocate_device(
                &state->dictionary_A_entries_u32, state->A.numElements);
            allocate_device(
                &state->dictionary_AT_to_A_code,
                workspace->coefficient_dictionary_size);
            build_dictionary_code_translation_kernel<<<
                numBlocks(workspace->coefficient_dictionary_size),
                numThreads, 0, workspace->stream>>>(
                state->dictionary_AT_to_A_code,
                workspace->coefficient_dictionary,
                workspace->coefficient_dictionary_size,
                workspace->A_coefficient_dictionary,
                workspace->A_coefficient_dictionary_size);
            if (workspace->m > 0 && state->A.numElements > 0) {
                pack_compact_dictionary_A_from_AT_kernel<<<
                    workspace->m, 256, 0, workspace->stream>>>(
                    state->dictionary_A_entries_u32, state->A.rowPtr,
                    state->A.colIndex, state->AT.rowPtr,
                    state->AT.colIndex, state->dictionary_AT_entries_u32,
                    state->dictionary_AT_to_A_code,
                    workspace->packed_dictionary_code_bits,
                    workspace->A_packed_dictionary_code_bits,
                    workspace->m);
            }
        }
    }
    if (state->use_unit_factorized || state->use_signed_factorized ||
        state->use_packed_dictionary_x) {
        allocate_device(
            &state->compact_inverse_col_norm, state->free_count);
        if (state->free_count > 0) {
            gather_selected_values_kernel<<<
                numBlocks(state->free_count), numThreads, 0,
                workspace->stream>>>(
                state->compact_inverse_col_norm,
                workspace->inverse_col_norm,
                state->free_to_original, state->free_count);
        }
        if (state->use_unit_factorized || state->use_signed_factorized ||
            state->use_packed_dictionary) {
            allocate_device(
                &state->compact_scaled_x_hat, state->free_count);
        }
    }
    allocate_device(&state->x, state->free_count);
    allocate_device(&state->x_bar, state->free_count);
    allocate_device(&state->x_hat, state->free_count);
    allocate_device(&state->last_x, state->free_count);
    allocate_device(&state->lower, state->free_count);
    allocate_device(&state->upper, state->free_count);
    allocate_device(&state->objective, state->free_count);
    allocate_device(&state->bound_type, state->free_count);
    if (reduced_uses_cusparse_x(state) ||
        autotune_generic_reduced_backends) {
        allocate_device(&state->ATy, state->free_count);
    }
    if (reduced_uses_generic_y(state) || state->use_unit_active_scatter ||
        autotune_generic_reduced_backends) {
        allocate_device(&state->Ax, workspace->m);
    }
    if (state->free_count > 0) {
        gather_reduced_state_kernel<<<
            numBlocks(state->free_count), numThreads, 0,
            workspace->stream>>>(
            state->x, state->x_bar, state->x_hat, state->last_x,
            state->lower, state->upper, state->objective,
            state->bound_type, workspace->x, workspace->x_bar,
            workspace->x_hat,
            workspace->last_x, workspace->l, workspace->u, workspace->c,
            workspace->x_bound_type, state->free_to_original,
            state->free_count);
    }
    compute_fixed_shift(workspace, state);
    scatter_fixed_bounds_kernel<<<
        numBlocks(workspace->n), numThreads, 0, workspace->stream>>>(
        workspace->x, workspace->x_bar, workspace->x_hat,
        workspace->last_x, workspace->l, workspace->u,
        state->x_bar_mask, workspace->n);
    const auto compressed_autotune_started =
        std::chrono::steady_clock::now();
    prepare_reduced_spmv(
        workspace, state, autotune_generic_reduced_backends);
    if (autotune_generic_reduced_backends) {
        prepare_reduced_nonempty_cusparse_y(workspace, state, true);
        autotune_reduced_compressed_backends(
            workspace, state, parameters, iteration);
        // Candidate calls must not share cuSPARSE descriptor/preprocess state
        // with the production graph. Recreate only the selected directions.
        destroy_reduced_nonempty_cusparse_y(state);
        destroy_spmv(&state->spmv_A, &state->spmv_AT);
        prepare_reduced_spmv(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        const double isolated_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            compressed_autotune_started).count();
        state->profile_backend_autotune.calls += 1;
        state->profile_backend_autotune.seconds += isolated_seconds;
        std::cout << "  reduced compressed autotune isolated_total_ms="
                  << std::fixed << std::setprecision(3)
                  << isolated_seconds * 1000.0
                  << " (candidate and production descriptors included)"
                  << std::defaultfloat << std::setprecision(2)
                  << std::endl;
    } else if (!workspace->reduced_backend_autotune_done) {
        workspace->reduced_use_fused_x = state->use_fused_x;
        workspace->reduced_use_fused_y = state->use_fused_y;
        workspace->reduced_backend_autotune_done = true;
        state->defer_empty_rows_selected = false;
        const char *const reason = parameters != nullptr &&
                parameters->CUSPARSE_spmv
            ? "forced-cusparse"
            : (reduced_compressed_autotune_enabled()
                ? "inherited-specialized" : "disabled");
        std::cout << "  reduced compressed autotune skipped: reason="
                  << reason << ", x="
                  << (state->use_fused_x ? "fused" : "inherited")
                  << ", y="
                  << (state->use_fused_y ? "fused" : "inherited")
                  << std::endl;
    }
    prepare_reduced_nonempty_cusparse_y(workspace, state);
    if (state->defer_empty_rows_to_observation) {
        state->deferred_empty_halpern_capacity = std::max(
            parameters != nullptr ? parameters->check_iter : 0,
            HPRLP_NORMAL_GRAPH_BATCH_SIZE);
        allocate_device(
            &state->deferred_empty_halpern_factors,
            2 * static_cast<std::size_t>(
                state->deferred_empty_halpern_capacity));
        if (state->profile_enabled) {
            std::cout
                << "  reduced empty rows deferred to observation: rows="
                << state->signed_A_empty_count
                << " factor_capacity="
                << state->deferred_empty_halpern_capacity << std::endl;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->rebuilds += 1;
    state->built = true;
    state->last_rebuild_iteration = iteration;
    state->last_free_columns = state->free_count;
    if (parameters != nullptr && parameters->autotune_verbose) {
        std::cout << "  reduced build: iteration=" << iteration
                  << ", free=" << state->base_count
                  << ", nnz=" << state->base_reduced_nnz << std::endl;
    }
    state->build_time += std::chrono::duration<HPRLP_FLOAT>(
        std::chrono::steady_clock::now() - start).count();
    return true;
}

bool extend_reduced_delta_workspace(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    const int new_count = state->recorded_delta_count;
    if (!state->built || new_count <= 0 ||
        state->recorded_changed_count != new_count) {
        return false;
    }
    const int previous_count = state->delta_count;
    const int delta_count = previous_count + new_count;
    const int expected_total = state->base_count + delta_count;
    if (state->free_count != expected_total) return false;

    // Stream-ordered allocation removes the synchronous allocation barrier
    // for the specialized/fused delta path. Keep the generic cuSPARSE Y path
    // on the legacy allocator because paired measurements showed a regression
    // there and delta allocation is not its dominant cost.
    const bool use_stream_ordered_allocations =
        state->use_unit_factorized || state->use_signed_factorized ||
        state->use_packed_dictionary || state->use_fused_y;
    auto allocate_delta = [&](auto **pointer, std::size_t count) {
        if (use_stream_ordered_allocations) {
            allocate_device_async(pointer, count, workspace->stream);
        } else {
            allocate_device(pointer, count);
        }
    };

    int *accumulated_indices = nullptr;
    std::uint8_t *accumulated_old_mask = nullptr;
    allocate_delta(&accumulated_indices, delta_count);
    allocate_delta(&accumulated_old_mask, delta_count);
    if (previous_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            accumulated_indices, state->delta_free_to_original,
            static_cast<std::size_t>(previous_count) * sizeof(int),
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            accumulated_old_mask, state->delta_old_mask,
            static_cast<std::size_t>(previous_count) * sizeof(std::uint8_t),
            cudaMemcpyDeviceToDevice, workspace->stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        accumulated_indices + previous_count, state->delta_indices_device,
        static_cast<std::size_t>(new_count) * sizeof(int),
        cudaMemcpyDeviceToDevice, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        accumulated_old_mask + previous_count,
        state->delta_old_mask_device,
        static_cast<std::size_t>(new_count) * sizeof(std::uint8_t),
        cudaMemcpyDeviceToDevice, workspace->stream));

    destroy_reduced_graph(state);
    // Delta extensions occur after all preceding work was queued on the same
    // stream. Keep allocation and release stream ordered so growing a small
    // delta does not introduce a device-wide cudaFree/cudaMalloc barrier.
    if (use_stream_ordered_allocations) {
        destroy_delta_workspace_async(state, workspace->stream);
    } else {
        destroy_delta_workspace(state);
    }
    state->delta_free_to_original = accumulated_indices;
    state->delta_old_mask = accumulated_old_mask;
    state->delta_count = delta_count;

    state->delta_AT.row = delta_count;
    state->delta_AT.col = workspace->m;
    allocate_delta(
        &state->delta_AT.rowPtr, static_cast<std::size_t>(delta_count) + 1);
    CUDA_CHECK(cudaMemsetAsync(
        state->delta_AT.rowPtr, 0, sizeof(int), workspace->stream));
    selected_csr_row_lengths_kernel<<<
        numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
        state->delta_AT.rowPtr, state->delta_free_to_original,
        workspace->AT->rowPtr, delta_count);

    std::size_t scan_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_temp_bytes, state->delta_AT.rowPtr + 1,
        state->delta_AT.rowPtr + 1, delta_count, workspace->stream));
    ensure_device_byte_capacity(
        &state->construction_temp,
        &state->construction_temp_capacity,
        scan_temp_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        state->construction_temp, scan_temp_bytes,
        state->delta_AT.rowPtr + 1,
        state->delta_AT.rowPtr + 1, delta_count, workspace->stream));

    int delta_nnz = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &delta_nnz, state->delta_AT.rowPtr + delta_count, sizeof(int),
        cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->delta_nnz = delta_nnz;
    state->delta_AT.numElements = delta_nnz;
    allocate_delta(&state->delta_AT.colIndex, delta_nnz);
    allocate_delta(&state->delta_AT.value, delta_nnz);
    if (delta_nnz > 0) copy_selected_csr_rows_kernel<<<
        delta_count, 256, 0, workspace->stream>>>(
        state->delta_AT.colIndex, state->delta_AT.value,
        state->delta_AT.rowPtr, state->delta_free_to_original,
        workspace->AT->rowPtr, workspace->AT->colIndex,
        workspace->AT->value, delta_count);
    transpose_csr(
        state->delta_AT, &state->delta_A, workspace->stream,
        state->transpose_handle, &state->construction_temp,
        &state->construction_temp_capacity, use_stream_ordered_allocations);

    const bool use_signed_delta = state->use_signed_factorized;
    const bool use_empty_row_batch =
        state->use_signed_empty_row_batch ||
        state->use_dictionary_empty_row_batch ||
        state->use_cusparse_empty_row_batch ||
        state->use_nonempty_cusparse_y;
    if (use_empty_row_batch) {
        const std::size_t nonempty_word_count =
            (static_cast<std::size_t>(workspace->m) + 31) / 32;
        allocate_delta(
            &state->delta_signed_A_nonempty_words,
            nonempty_word_count);
        CUDA_CHECK(cudaMemsetAsync(
            state->delta_signed_A_nonempty_words, 0,
            nonempty_word_count * sizeof(std::uint32_t),
            workspace->stream));
        if (workspace->m > 0) {
            mark_nonempty_csr_rows_kernel<<<
                numBlocks(workspace->m), numThreads, 0,
                workspace->stream>>>(
                state->delta_A.rowPtr,
                state->delta_signed_A_nonempty_words,
                workspace->m);
        }
    }
    if (use_signed_delta) {
        allocate_delta(
            &state->delta_signed_AT_entries_u32, state->delta_nnz);
        allocate_delta(
            &state->delta_signed_A_entries_u32, state->delta_nnz);
        if (state->delta_nnz > 0) {
            pack_signed_entries_u32_kernel<<<
                numBlocks(state->delta_nnz), numThreads, 0,
                workspace->stream>>>(
                state->delta_signed_AT_entries_u32,
                state->delta_AT.colIndex, state->delta_AT.value,
                state->delta_nnz);
            pack_signed_entries_u32_kernel<<<
                numBlocks(state->delta_nnz), numThreads, 0,
                workspace->stream>>>(
                state->delta_signed_A_entries_u32,
                state->delta_A.colIndex, state->delta_A.value,
                state->delta_nnz);
        }
    }

    allocate_delta(&state->delta_AT_row_buckets, delta_count);
    const std::size_t delta_A_bucket_capacity =
        use_empty_row_batch
        ? static_cast<std::size_t>(delta_nnz)
        : static_cast<std::size_t>(workspace->m);
    allocate_delta(
        &state->delta_A_row_buckets, delta_A_bucket_capacity);
    CUDA_CHECK(cudaMemsetAsync(
        state->delta_row_bucket_counts_device, 0, 4 * sizeof(int),
        workspace->stream));
    build_compact_row_buckets_kernel<<<
        numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
        state->delta_AT.rowPtr, state->delta_AT_row_buckets,
        state->delta_row_bucket_counts_device, delta_count);
    if (use_empty_row_batch && workspace->m > 0 &&
        delta_nnz > 0) {
        build_delta_only_row_list_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            state->A.rowPtr, state->delta_A.rowPtr,
            state->delta_A_row_buckets,
            state->delta_row_bucket_counts_device + 2, workspace->m);
    } else if (workspace->m > 0 && !use_empty_row_batch) {
        build_compact_row_buckets_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            state->delta_A.rowPtr, state->delta_A_row_buckets,
            state->delta_row_bucket_counts_device + 2, workspace->m);
    }
    int bucket_counts[4] = {};
    CUDA_CHECK(cudaMemcpyAsync(
        bucket_counts, state->delta_row_bucket_counts_device,
        sizeof(bucket_counts), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->delta_AT_short_count = bucket_counts[0];
    state->delta_AT_warp_count = bucket_counts[1];
    state->delta_A_short_count = use_empty_row_batch
        ? 0 : bucket_counts[2];
    state->delta_A_warp_count = use_empty_row_batch
        ? 0 : bucket_counts[3];
    state->delta_AT_rows_short = state->delta_AT_row_buckets;
    state->delta_AT_rows_warp = state->delta_AT_row_buckets +
        (delta_count - state->delta_AT_warp_count);
    state->delta_A_rows_short = use_empty_row_batch
        ? nullptr : state->delta_A_row_buckets;
    state->delta_A_rows_warp =
        !use_empty_row_batch && workspace->m > 0
        ? state->delta_A_row_buckets +
            (workspace->m - state->delta_A_warp_count)
        : nullptr;
    state->signed_delta_only_rows = use_empty_row_batch
        ? state->delta_A_row_buckets : nullptr;
    state->signed_delta_only_count = use_empty_row_batch
        ? bucket_counts[2] : 0;
    allocate_delta(&state->delta_x, delta_count);
    allocate_delta(&state->delta_x_bar, delta_count);
    allocate_delta(&state->delta_x_hat, delta_count);
    allocate_delta(&state->delta_last_x, delta_count);
    allocate_delta(&state->delta_lower, delta_count);
    allocate_delta(&state->delta_upper, delta_count);
    allocate_delta(&state->delta_objective, delta_count);
    allocate_delta(&state->delta_bound_type, delta_count);
    if (use_signed_delta) {
        allocate_delta(&state->delta_inverse_col_norm, delta_count);
        allocate_delta(&state->delta_scaled_x_hat, delta_count);
        allocate_delta(&state->delta_scaled_fixed, delta_count);
    }
    gather_reduced_state_kernel<<<
        numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
        state->delta_x, state->delta_x_bar, state->delta_x_hat,
        state->delta_last_x, state->delta_lower, state->delta_upper,
        state->delta_objective, state->delta_bound_type,
        workspace->x, workspace->x_bar, workspace->x_hat,
        workspace->last_x, workspace->l, workspace->u, workspace->c,
        workspace->x_bound_type, state->delta_free_to_original,
        delta_count);
    if (state->use_nonempty_cusparse_y) {
        allocate_delta(&state->delta_fixed, delta_count);
        const bool parallel_delta =
            delta_count >= kReducedParallelDeltaMinimumColumns &&
            delta_nnz >= kReducedParallelDeltaMinimumNnz;
        if (!parallel_delta &&
            delta_nnz >= kReducedPrecomputedDeltaInputMinimumNnz) {
            allocate_delta(&state->delta_input, delta_count);
        }
        fixed_values_from_mask_kernel<<<
            numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
            state->delta_fixed, state->delta_old_mask,
            state->delta_lower, state->delta_upper, delta_count);
    }
    if (use_signed_delta) {
        gather_selected_values_kernel<<<
            numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
            state->delta_inverse_col_norm, workspace->inverse_col_norm,
            state->delta_free_to_original, delta_count);
        build_scaled_fixed_delta_kernel<<<
            numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
            state->delta_scaled_fixed, state->delta_inverse_col_norm,
            state->delta_old_mask, state->delta_lower, state->delta_upper,
            delta_count);
    }
    prepare_parallel_delta_cusparse_y(workspace, state);
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    state->last_free_columns = state->free_count;
    return true;
}

void enqueue_reduced_x_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool use_delta) {
    // Keep every signed base X/Y variant intact after fixed columns are
    // released.  The signed Y kernel assigned to each base row also traverses
    // that row's sparse delta entries, avoiding both a generic base SpMV and a
    // separate full-row correction pass.
    const bool use_specialized_delta_pair =
        use_delta &&
        state->use_signed_factorized;
    // Dictionary X/Y can likewise retain their packed base representation;
    // each Y row folds its sparse delta correction into the existing launch.
    const bool use_dictionary_delta_pair =
        use_delta &&
        state->use_packed_dictionary_x && state->use_packed_dictionary;
    const bool use_unit_delta_pair =
        use_delta && state->use_unit_factorized;
    if (state->base_count > 0 || use_delta) {
        if (state->use_packed_dictionary_x &&
            (!use_delta || use_dictionary_delta_pair ||
             !state->use_packed_dictionary ||
             state->use_fixed_degree_dictionary_x)) {
            if (state->use_fixed_degree_dictionary_x) {
                packed_dictionary_update_x_fixed_degree_run_kernel<<<
                    (state->base_count + kReducedSignedXThreads - 1) /
                        kReducedSignedXThreads,
                    kReducedSignedXThreads, 0, workspace->stream>>>(
                    state->x, state->x_hat, state->lower, state->upper,
                    state->bound_type, state->objective, state->last_x,
                    workspace->unit_scaled_y,
                    state->compact_inverse_col_norm,
                    state->use_packed_dictionary
                        ? state->compact_scaled_x_hat : nullptr,
                    HPRLPPackedStatePlan{},
                    workspace->coefficient_dictionary,
                    state->dictionary_AT_entries_soa,
                    workspace->packed_dictionary_code_bits,
                    workspace->Halpern_params,
                    workspace->halpern_factors, 0, state->base_count,
                    state->fixed_degree_dictionary_x_degree);
            } else {
                const HPRLP_packed_dictionary_x_view_gpu view{
                    state->x, state->x_hat, state->lower, state->upper,
                    state->bound_type, state->objective, state->last_x,
                    workspace->unit_scaled_y,
                    state->compact_inverse_col_norm,
                    state->use_packed_dictionary
                        ? state->compact_scaled_x_hat : nullptr,
                    HPRLPPackedStatePlan{},
                    workspace->coefficient_dictionary,
                    state->dictionary_AT_entries_u32, state->AT.rowPtr,
                    workspace->packed_dictionary_code_bits,
                    state->AT_rows_short, state->AT_short_count,
                    state->signed_AT_rows_medium,
                    state->signed_AT_medium_count,
                    state->signed_AT_rows_long,
                    state->signed_AT_long_count, true};
                hprlp_enqueue_packed_dictionary_x(
                    view, workspace->Halpern_params,
                    workspace->halpern_factors, kReducedSignedYThreads,
                    workspace->stream);
            }
        } else if (state->use_signed_factorized &&
                   (!use_delta || use_specialized_delta_pair)) {
            if (state->signed_x_uses_split_u16) {
                const HPRLP_signed_unit_x_split_u16_view_gpu view{
                    state->base_count, state->x, state->x_hat,
                    state->lower, state->upper, state->bound_type,
                    state->objective, state->last_x,
                    workspace->unit_scaled_y,
                    state->compact_scaled_x_hat, nullptr,
                    state->compact_inverse_col_norm, state->AT.rowPtr,
                    state->signed_AT_constraint_index,
                    state->signed_AT_negative};
                hprlp_enqueue_signed_unit_x_split_u16_scalar(
                    view, workspace->Halpern_params,
                    workspace->halpern_factors, kReducedSignedXThreads,
                    workspace->stream);
            } else if (state->signed_x_uses_scalar) {
                const HPRLP_signed_unit_x_packed_view_gpu view{
                    state->base_count, state->x, state->x_hat,
                    state->lower, state->upper, state->bound_type,
                    state->objective, state->last_x,
                    workspace->unit_scaled_y,
                    state->compact_scaled_x_hat, nullptr,
                    state->compact_inverse_col_norm, state->AT.rowPtr,
                    state->signed_AT_entries_u16,
                    state->signed_AT_entries_u32};
                hprlp_enqueue_signed_unit_x_packed_scalar(
                    view, workspace->Halpern_params,
                    workspace->halpern_factors, kReducedSignedXThreads,
                    workspace->stream);
            } else {
                const HPRLP_signed_unit_x_packed_bucket_view_gpu view{
                    {state->base_count, state->x, state->x_hat,
                     state->lower, state->upper, state->bound_type,
                     state->objective, state->last_x,
                     workspace->unit_scaled_y,
                     state->compact_scaled_x_hat, nullptr,
                     state->compact_inverse_col_norm, state->AT.rowPtr,
                     state->signed_AT_entries_u16,
                     state->signed_AT_entries_u32},
                    state->AT_rows_short, state->AT_short_count,
                    state->signed_AT_rows_medium,
                    state->signed_AT_medium_count,
                    state->signed_AT_rows_long,
                    state->signed_AT_long_count};
                hprlp_enqueue_signed_unit_x_packed_bucketed(
                    view, workspace->Halpern_params,
                    workspace->halpern_factors,
                    kReducedSignedXThreads, kReducedSignedYThreads,
                    workspace->stream);
            }
        } else if (state->use_unit_factorized &&
                   (!use_delta || use_unit_delta_pair)) {
            const HPRLP_unit_factorized_x_view_gpu view{
                workspace->m, state->base_count,
                state->x, state->x_hat, state->lower, state->upper,
                state->bound_type,
                state->objective, state->last_x, workspace->unit_scaled_y,
                state->compact_inverse_col_norm,
                state->use_unit_active_scatter
                    ? nullptr : state->compact_scaled_x_hat,
                state->use_unit_active_scatter ? state->Ax : nullptr,
                state->AT.rowPtr, state->unit_AT_constraint_index};
            hprlp_enqueue_unit_factorized_x_scalar(
                view, workspace->Halpern_params,
                workspace->halpern_factors, workspace->uniform_unit_sign,
                kReducedVectorThreads, workspace->stream);
        } else if (state->use_fused_x) {
            if (state->AT_short_count > 0) fused_reduced_x_short_kernel<<<
                numBlocks(state->AT_short_count), numThreads, 0,
                workspace->stream>>>(
                state->x, state->x_hat, state->lower, state->upper,
                state->bound_type, state->objective, state->last_x,
                workspace->y,
                state->AT.rowPtr, state->AT.colIndex, state->AT.value,
                workspace->Halpern_params, workspace->halpern_factors,
                state->AT_rows_short, state->AT_short_count);
            if (state->AT_warp_count > 0) fused_reduced_x_warp_kernel<<<
                (state->AT_warp_count + 15) / 16, 512, 0,
                workspace->stream>>>(
                state->x, state->x_hat, state->lower, state->upper,
                state->bound_type, state->objective, state->last_x,
                workspace->y,
                state->AT.rowPtr, state->AT.colIndex, state->AT.value,
                workspace->Halpern_params, workspace->halpern_factors,
                state->AT_rows_warp, state->AT_warp_count);
        } else {
            CUSPARSE_spmvop_AT &at = state->spmv_AT;
            check_cusparse(hprlp_run_spmvop(
                at.cusparseHandle, at.operation, &at.alpha, &at.beta,
                at.y_cusparseDescr, at.ATy_cusparseDescr,
                at.ATy_cusparseDescr), "capture reduced AT SpMVOp");
            update_reduced_x_from_aty_kernel<<<
                (state->base_count + kReducedVectorThreads - 1) /
                    kReducedVectorThreads,
                kReducedVectorThreads, 0,
                workspace->stream>>>(
                state->x, state->x_hat, state->lower, state->upper,
                state->bound_type, state->ATy, state->objective,
                state->last_x, workspace->Halpern_params,
                workspace->halpern_factors, state->base_count);
        }
        if (use_specialized_delta_pair) {
            const HPRLP_signed_unit_x_packed_bucket_view_gpu delta_view{
                {state->delta_count, state->delta_x, state->delta_x_hat,
                 state->delta_lower, state->delta_upper,
                 state->delta_bound_type, state->delta_objective,
                 state->delta_last_x, workspace->unit_scaled_y,
                 state->delta_scaled_x_hat, nullptr,
                 state->delta_inverse_col_norm, state->delta_AT.rowPtr,
                 nullptr, state->delta_signed_AT_entries_u32},
                state->delta_AT_rows_short,
                state->delta_AT_short_count,
                state->delta_AT_rows_warp,
                state->delta_AT_warp_count,
                nullptr, 0};
            hprlp_enqueue_signed_unit_x_packed_bucketed(
                delta_view, workspace->Halpern_params,
                workspace->halpern_factors,
                kReducedSignedXThreads, kReducedSignedYThreads,
                workspace->stream);
        } else {
            // Julia updates released delta columns independently of whether
            // the base reduced AT path is fused or cuSPARSE.
            if (state->delta_AT_short_count > 0) {
                if (state->delta_input != nullptr) {
                    fused_reduced_delta_x_short_with_input_kernel<<<
                        numBlocks(state->delta_AT_short_count), numThreads,
                        0, workspace->stream>>>(
                        state->delta_x, state->delta_x_hat,
                        state->delta_input, state->delta_fixed,
                        state->delta_lower, state->delta_upper,
                        state->delta_bound_type, state->delta_objective,
                        state->delta_last_x, workspace->y,
                        state->delta_AT.rowPtr,
                        state->delta_AT.colIndex, state->delta_AT.value,
                        workspace->Halpern_params,
                        workspace->halpern_factors,
                        state->delta_AT_rows_short,
                        state->delta_AT_short_count);
                } else {
                    fused_reduced_x_short_kernel<<<
                    numBlocks(state->delta_AT_short_count), numThreads, 0,
                    workspace->stream>>>(
                    state->delta_x, state->delta_x_hat,
                    state->delta_lower, state->delta_upper,
                    state->delta_bound_type, state->delta_objective,
                    state->delta_last_x, workspace->y,
                    state->delta_AT.rowPtr, state->delta_AT.colIndex,
                    state->delta_AT.value, workspace->Halpern_params,
                    workspace->halpern_factors,
                    state->delta_AT_rows_short,
                    state->delta_AT_short_count);
                }
            }
            if (state->delta_AT_warp_count > 0) {
                if (state->delta_input != nullptr) {
                    fused_reduced_delta_x_warp_with_input_kernel<<<
                        (state->delta_AT_warp_count + 15) / 16, 512, 0,
                        workspace->stream>>>(
                        state->delta_x, state->delta_x_hat,
                        state->delta_input, state->delta_fixed,
                        state->delta_lower, state->delta_upper,
                        state->delta_bound_type, state->delta_objective,
                        state->delta_last_x, workspace->y,
                        state->delta_AT.rowPtr,
                        state->delta_AT.colIndex, state->delta_AT.value,
                        workspace->Halpern_params,
                        workspace->halpern_factors,
                        state->delta_AT_rows_warp,
                        state->delta_AT_warp_count);
                } else {
                    fused_reduced_x_warp_kernel<<<
                    (state->delta_AT_warp_count + 15) / 16, 512, 0,
                    workspace->stream>>>(
                    state->delta_x, state->delta_x_hat,
                    state->delta_lower, state->delta_upper,
                    state->delta_bound_type, state->delta_objective,
                    state->delta_last_x, workspace->y,
                    state->delta_AT.rowPtr, state->delta_AT.colIndex,
                    state->delta_AT.value, workspace->Halpern_params,
                    workspace->halpern_factors,
                    state->delta_AT_rows_warp,
                    state->delta_AT_warp_count);
                }
            }
        }
    }
}

void enqueue_reduced_y_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool use_delta) {
    const bool use_specialized_delta_pair =
        use_delta && state->use_signed_factorized;
    const bool use_dictionary_delta_pair =
        use_delta &&
        state->use_packed_dictionary_x && state->use_packed_dictionary;
    const bool use_unit_delta_pair =
        use_delta && state->use_unit_factorized;
    if (state->use_packed_dictionary &&
        (!use_delta || use_dictionary_delta_pair)) {
        const HPRLP_packed_dictionary_y_view_gpu view{
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            state->compact_scaled_x_hat, workspace->inverse_row_norm,
            workspace->unit_scaled_y, state->row_fixed_shift,
            workspace->packed_state_plan,
            workspace->A_coefficient_dictionary,
            state->dictionary_A_entries_u32, state->A.rowPtr,
            workspace->A_packed_dictionary_code_bits,
            state->A_rows_short, state->A_short_count,
            state->signed_A_rows_medium, state->signed_A_medium_count,
            state->signed_A_rows_long, state->signed_A_long_count,
            use_dictionary_delta_pair ? state->delta_x_hat : nullptr,
            use_dictionary_delta_pair ? state->delta_lower : nullptr,
            use_dictionary_delta_pair ? state->delta_upper : nullptr,
            use_dictionary_delta_pair ? state->delta_old_mask : nullptr,
            use_dictionary_delta_pair ? state->delta_A.rowPtr : nullptr,
            use_dictionary_delta_pair ? state->delta_A.colIndex : nullptr,
            use_dictionary_delta_pair ? state->delta_A.value : nullptr,
            true};
        hprlp_enqueue_packed_dictionary_y(
            view, workspace->Halpern_params,
            workspace->halpern_factors, kReducedSignedYThreads,
            workspace->stream);
    } else if (state->use_signed_factorized &&
               (!use_delta || use_specialized_delta_pair)) {
        if (state->signed_y_uses_combined) {
            const HPRLP_signed_unit_y_combined_view_gpu view{
                workspace->m, workspace->y, workspace->AL, workspace->AU,
                workspace->y_bound_type, workspace->last_y,
                state->compact_scaled_x_hat, nullptr,
                workspace->unit_scaled_y, state->row_fixed_shift,
                workspace->inverse_row_norm, state->A.rowPtr,
                state->signed_A_entries_u16,
                state->signed_A_entries_u32,
                state->signed_A_rows_medium,
                state->signed_A_medium_count,
                state->signed_A_rows_long,
                state->signed_A_long_count,
                use_specialized_delta_pair
                    ? state->delta_scaled_x_hat : nullptr,
                use_specialized_delta_pair
                    ? state->delta_scaled_fixed : nullptr,
                use_specialized_delta_pair ? state->delta_A.rowPtr : nullptr,
                use_specialized_delta_pair
                    ? state->delta_signed_A_entries_u32 : nullptr,
                use_specialized_delta_pair
                    ? state->delta_signed_A_nonempty_words : nullptr};
            hprlp_enqueue_signed_unit_y_combined(
                view, workspace->Halpern_params,
                workspace->halpern_factors, kReducedSignedYThreads,
                workspace->stream);
        } else if (state->signed_y_uses_scalar) {
            const HPRLP_signed_unit_y_scalar_view_gpu base_view{
                workspace->m, workspace->y, workspace->AL, workspace->AU,
                workspace->y_bound_type, workspace->last_y,
                state->compact_scaled_x_hat, nullptr,
                workspace->unit_scaled_y, state->row_fixed_shift,
                workspace->inverse_row_norm, state->A.rowPtr,
                state->signed_A_entries_u16,
                state->signed_A_entries_u32};
            if (use_specialized_delta_pair) {
                const HPRLP_signed_unit_y_scalar_delta_view_gpu view{
                    base_view, state->delta_scaled_x_hat,
                    state->delta_scaled_fixed, state->delta_A.rowPtr,
                    state->delta_signed_A_entries_u32};
                hprlp_enqueue_signed_unit_y_scalar_delta(
                    view, workspace->Halpern_params,
                    workspace->halpern_factors, kReducedSignedXThreads,
                    workspace->stream);
            } else {
                hprlp_enqueue_signed_unit_y_scalar(
                    base_view, workspace->Halpern_params,
                    workspace->halpern_factors, kReducedSignedXThreads,
                    workspace->stream);
            }
        } else {
            const HPRLP_signed_unit_y_bucket_view_gpu view{
                {workspace->m, workspace->y, workspace->AL,
                 workspace->AU, workspace->y_bound_type,
                 workspace->last_y, state->compact_scaled_x_hat, nullptr,
                 workspace->unit_scaled_y, state->row_fixed_shift,
                 workspace->inverse_row_norm, state->A.rowPtr,
                 state->signed_A_entries_u16,
                 state->signed_A_entries_u32},
                state->A_rows_short, state->A_short_count,
                state->signed_A_rows_medium,
                state->signed_A_medium_count,
                state->signed_A_rows_long,
                state->signed_A_long_count,
                use_specialized_delta_pair
                    ? state->delta_scaled_x_hat : nullptr,
                use_specialized_delta_pair
                    ? state->delta_scaled_fixed : nullptr,
                use_specialized_delta_pair ? state->delta_A.rowPtr : nullptr,
                use_specialized_delta_pair
                    ? state->delta_signed_A_entries_u32 : nullptr};
            hprlp_enqueue_signed_unit_y_bucketed(
                view, workspace->Halpern_params,
                workspace->halpern_factors, kReducedSignedXThreads,
                kReducedSignedYThreads, workspace->stream);
        }
    } else if (state->use_unit_active_scatter &&
               (!use_delta || use_unit_delta_pair)) {
        reduced_unit_active_scatter_y_kernel<<<
            (workspace->m + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, state->Ax,
            workspace->inverse_row_norm, workspace->unit_scaled_y,
            state->row_fixed_shift, state->delta_x_hat,
            state->delta_lower, state->delta_upper, state->delta_old_mask,
            state->delta_A.rowPtr, state->delta_A.colIndex,
            state->delta_A.value, use_unit_delta_pair,
            workspace->Halpern_params,
            workspace->halpern_factors, workspace->uniform_unit_sign,
            workspace->m);
    } else if (state->use_unit_factorized &&
               (!use_delta || use_unit_delta_pair)) {
        reduced_unit_factorized_y_kernel<<<
            workspace->m, kReducedUnitYThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            state->compact_scaled_x_hat, workspace->inverse_row_norm,
            workspace->unit_scaled_y, state->row_fixed_shift,
            state->A.rowPtr, state->A.colIndex, state->delta_x_hat,
            state->delta_lower, state->delta_upper, state->delta_old_mask,
            state->delta_A.rowPtr, state->delta_A.colIndex,
            state->delta_A.value, use_unit_delta_pair,
            workspace->Halpern_params,
            workspace->halpern_factors, workspace->uniform_unit_sign,
            workspace->m);
    } else if (state->use_fused_y) {
        if (state->A_short_count > 0) fused_reduced_y_short_kernel<<<
            numBlocks(state->A_short_count), numThreads, 0,
            workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, state->x_hat, state->A.rowPtr, state->A.colIndex, state->A.value,
            state->row_fixed_shift, state->delta_x_hat,
            state->delta_lower, state->delta_upper, state->delta_old_mask,
            state->delta_A.rowPtr, state->delta_A.colIndex,
            state->delta_A.value, use_delta, workspace->Halpern_params,
            workspace->halpern_factors, state->A_rows_short,
            state->A_short_count);
        if (state->A_warp_count > 0) fused_reduced_y_warp_kernel<<<
            (state->A_warp_count + 15) / 16, 512, 0,
            workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, state->x_hat, state->A.rowPtr, state->A.colIndex, state->A.value,
            state->row_fixed_shift, state->delta_x_hat,
            state->delta_lower, state->delta_upper, state->delta_old_mask,
            state->delta_A.rowPtr, state->delta_A.colIndex,
            state->delta_A.value, use_delta, workspace->Halpern_params,
            workspace->halpern_factors, state->A_rows_warp,
            state->A_warp_count);
    } else {
        if (state->base_count > 0) {
            CUSPARSE_spmvop_A &a = state->spmv_A;
            check_cusparse(hprlp_run_spmvop(
                a.cusparseHandle, a.operation, &a.alpha, &a.beta,
                a.x_hat_cusparseDescr, a.Ax_cusparseDescr,
                a.Ax_cusparseDescr), "capture reduced A SpMVOp");
        } else {
            CUDA_CHECK(cudaMemsetAsync(
                state->Ax, 0,
                static_cast<std::size_t>(workspace->m) * sizeof(HPRLP_FLOAT),
                workspace->stream));
        }
        if (use_delta) {
            update_reduced_y_delta_kernel<<<
                (workspace->m + kReducedVectorThreads - 1) /
                    kReducedVectorThreads,
                kReducedVectorThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU,
                workspace->y_bound_type, workspace->last_y, state->Ax,
                state->row_fixed_shift, state->delta_x_hat,
                state->delta_lower, state->delta_upper,
                state->delta_old_mask, state->delta_A.rowPtr,
                state->delta_A.colIndex, state->delta_A.value,
                state->use_packed_dictionary_x
                    ? workspace->unit_scaled_y : nullptr,
                workspace->inverse_row_norm,
                workspace->Halpern_params, workspace->halpern_factors,
                workspace->m);
        } else {
            add_fixed_shift_kernel<<<
                (workspace->m + kReducedVectorThreads - 1) /
                    kReducedVectorThreads,
                kReducedVectorThreads, 0,
                workspace->stream>>>(
                state->Ax, state->row_fixed_shift, workspace->m);
            update_y_normal_kernel<<<
                (workspace->m + kReducedYThreads - 1) / kReducedYThreads,
                kReducedYThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU, state->Ax,
                workspace->last_y, workspace->Halpern_params,
                workspace->halpern_factors, workspace->m);
        }
    }
    // Packed-dictionary X consumes y pre-scaled by the row norm. Paired
    // dictionary Y maintains that cache without a delta, and the generic
    // delta-Y kernel above can write it while updating y. Other mixed paths
    // need one explicit refresh for the next iteration.
    const bool y_kernel_wrote_scaled_y =
        (state->use_packed_dictionary && !use_delta) ||
        (use_delta && !state->use_fused_y);
    if (state->use_packed_dictionary_x && !y_kernel_wrote_scaled_y) {
        vector_dot_product_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            workspace->y, workspace->inverse_row_norm,
            workspace->unit_scaled_y, workspace->m, false);
    }
}

void enqueue_reduced_signed_nonempty_y_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    const bool use_delta = state->delta_count > 0;
    const HPRLP_signed_unit_y_bucket_view_gpu view{
        {workspace->m, workspace->y, workspace->AL,
         workspace->AU, workspace->y_bound_type,
         workspace->last_y, state->compact_scaled_x_hat, nullptr,
         workspace->unit_scaled_y, state->row_fixed_shift,
         workspace->inverse_row_norm, state->A.rowPtr,
         state->signed_A_entries_u16,
         state->signed_A_entries_u32},
        state->signed_A_rows_short_nonempty,
        state->signed_A_short_nonempty_count,
        state->signed_A_rows_medium,
        state->signed_A_medium_count,
        state->signed_A_rows_long,
        state->signed_A_long_count,
        use_delta ? state->delta_scaled_x_hat : nullptr,
        use_delta ? state->delta_scaled_fixed : nullptr,
        use_delta ? state->delta_A.rowPtr : nullptr,
        use_delta ? state->delta_signed_A_entries_u32 : nullptr};
    hprlp_enqueue_signed_unit_y_bucketed(
        view, workspace->Halpern_params, workspace->halpern_factors,
        kReducedSignedXThreads, kReducedSignedYThreads, workspace->stream);

    if (use_delta && state->signed_delta_only_count > 0) {
        update_reduced_signed_delta_only_y_kernel<<<
            (state->signed_delta_only_count + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            workspace->unit_scaled_y, workspace->inverse_row_norm,
            state->row_fixed_shift, state->delta_scaled_x_hat,
            state->delta_scaled_fixed, state->delta_A.rowPtr,
            state->delta_signed_A_entries_u32,
            state->signed_delta_only_rows,
            state->signed_delta_only_count, workspace->Halpern_params,
            workspace->halpern_factors);
    }
}

void enqueue_reduced_generic_nonempty_y_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool use_delta) {
    if (state->signed_A_short_nonempty_count > 0)
        fused_reduced_y_short_kernel<<<
            numBlocks(state->signed_A_short_nonempty_count), numThreads, 0,
            workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, state->x_hat,
            state->A.rowPtr, state->A.colIndex, state->A.value,
            state->row_fixed_shift, state->delta_x_hat,
            state->delta_lower, state->delta_upper, state->delta_old_mask,
            state->delta_A.rowPtr, state->delta_A.colIndex,
            state->delta_A.value, use_delta, workspace->Halpern_params,
            workspace->halpern_factors,
            state->signed_A_rows_short_nonempty,
            state->signed_A_short_nonempty_count);
    if (state->A_warp_count > 0)
        fused_reduced_y_warp_kernel<<<
            (state->A_warp_count + 15) / 16, 512, 0,
            workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, state->x_hat,
            state->A.rowPtr, state->A.colIndex, state->A.value,
            state->row_fixed_shift, state->delta_x_hat,
            state->delta_lower, state->delta_upper, state->delta_old_mask,
            state->delta_A.rowPtr, state->delta_A.colIndex,
            state->delta_A.value, use_delta, workspace->Halpern_params,
            workspace->halpern_factors, state->A_rows_warp,
            state->A_warp_count);
    if (use_delta && state->signed_delta_only_count > 0)
        update_reduced_dictionary_delta_only_y_kernel<<<
            (state->signed_delta_only_count + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, nullptr,
            workspace->inverse_row_norm, state->row_fixed_shift,
            state->delta_x_hat, state->delta_lower, state->delta_upper,
            state->delta_old_mask, state->delta_A.rowPtr,
            state->delta_A.colIndex, state->delta_A.value,
            state->signed_delta_only_rows,
            state->signed_delta_only_count, workspace->Halpern_params,
            workspace->halpern_factors);
}

void enqueue_reduced_generic_empty_y_batch(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (state->signed_A_empty_count <= 0 || workspace->m <= 0) return;
    update_reduced_dictionary_empty_y_batch_kernel<
        HPRLP_NORMAL_GRAPH_BATCH_SIZE><<<
            (workspace->m + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y, nullptr,
            workspace->inverse_row_norm, state->row_fixed_shift,
            state->A.rowPtr,
            state->delta_count > 0
                ? state->delta_signed_A_nonempty_words : nullptr,
            workspace->m, workspace->Halpern_params,
            workspace->halpern_factor_batch);
}

void enqueue_reduced_signed_empty_y_batch(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (state->signed_A_empty_count <= 0) return;
    update_reduced_signed_empty_y_batch_kernel<
        HPRLP_NORMAL_GRAPH_BATCH_SIZE><<<
            (state->signed_A_empty_count + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            workspace->unit_scaled_y, workspace->inverse_row_norm,
            state->row_fixed_shift,
            state->delta_count > 0
                ? state->delta_signed_A_nonempty_words : nullptr,
            state->signed_A_rows_empty, state->signed_A_empty_count,
            workspace->Halpern_params, workspace->halpern_factor_batch);
}

void enqueue_reduced_dictionary_nonempty_y_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    const bool use_delta = state->delta_count > 0;
    const HPRLP_packed_dictionary_y_view_gpu view{
        workspace->y, workspace->AL, workspace->AU,
        workspace->y_bound_type, workspace->last_y,
        state->compact_scaled_x_hat, workspace->inverse_row_norm,
        workspace->unit_scaled_y, state->row_fixed_shift,
        workspace->packed_state_plan,
        workspace->A_coefficient_dictionary,
        state->dictionary_A_entries_u32, state->A.rowPtr,
        workspace->A_packed_dictionary_code_bits,
        state->signed_A_rows_short_nonempty,
        state->signed_A_short_nonempty_count,
        state->signed_A_rows_medium, state->signed_A_medium_count,
        state->signed_A_rows_long, state->signed_A_long_count,
        use_delta ? state->delta_x_hat : nullptr,
        use_delta ? state->delta_lower : nullptr,
        use_delta ? state->delta_upper : nullptr,
        use_delta ? state->delta_old_mask : nullptr,
        use_delta ? state->delta_A.rowPtr : nullptr,
        use_delta ? state->delta_A.colIndex : nullptr,
        use_delta ? state->delta_A.value : nullptr,
        true};
    hprlp_enqueue_packed_dictionary_y(
        view, workspace->Halpern_params, workspace->halpern_factors,
        kReducedSignedYThreads, workspace->stream);

    if (use_delta && state->signed_delta_only_count > 0) {
        update_reduced_dictionary_delta_only_y_kernel<<<
            (state->signed_delta_only_count + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            workspace->unit_scaled_y, workspace->inverse_row_norm,
            state->row_fixed_shift, state->delta_x_hat,
            state->delta_lower, state->delta_upper,
            state->delta_old_mask, state->delta_A.rowPtr,
            state->delta_A.colIndex, state->delta_A.value,
            state->signed_delta_only_rows,
            state->signed_delta_only_count, workspace->Halpern_params,
            workspace->halpern_factors);
    }
}

void enqueue_reduced_dictionary_empty_y_batch(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (state->signed_A_empty_count <= 0 || workspace->m <= 0) return;
    update_reduced_dictionary_empty_y_batch_kernel<
        HPRLP_NORMAL_GRAPH_BATCH_SIZE><<<
            (workspace->m + kReducedVectorThreads - 1) /
                kReducedVectorThreads,
            kReducedVectorThreads, 0, workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            workspace->unit_scaled_y, workspace->inverse_row_norm,
            state->row_fixed_shift, state->A.rowPtr,
            state->delta_count > 0
                ? state->delta_signed_A_nonempty_words : nullptr,
            workspace->m, workspace->Halpern_params,
            workspace->halpern_factor_batch);
}

void enqueue_reduced_nonempty_cusparse_y_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool use_delta);

void enqueue_reduced_nonempty_y_updates_for_deferred_rows(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (state->use_signed_empty_row_batch) {
        enqueue_reduced_signed_nonempty_y_updates(workspace, state);
    } else if (state->use_dictionary_empty_row_batch) {
        enqueue_reduced_dictionary_nonempty_y_updates(workspace, state);
    } else if (state->use_nonempty_cusparse_y) {
        enqueue_reduced_nonempty_cusparse_y_updates(
            workspace, state, state->delta_count > 0);
    } else {
        enqueue_reduced_generic_nonempty_y_updates(
            workspace, state, state->delta_count > 0);
    }
}

void enqueue_reduced_deferred_empty_rows(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    int deferred_iterations) {
    if (!state->defer_empty_rows_to_observation ||
        deferred_iterations <= 0 || state->signed_A_empty_count <= 0 ||
        state->signed_A_rows_empty == nullptr ||
        state->deferred_empty_halpern_factors == nullptr) {
        return;
    }
    if (deferred_iterations > state->deferred_empty_halpern_capacity) {
        throw std::runtime_error(
            "deferred reduced empty-row interval exceeds factor capacity");
    }
    prepare_reduced_deferred_halpern_factors_kernel<<<
        1, 1, 0, workspace->stream>>>(
        workspace->halpern_inner,
        state->deferred_empty_halpern_factors,
        deferred_iterations);
    HPRLP_FLOAT *const scaled_y =
        (state->use_signed_factorized || state->use_packed_dictionary_x)
        ? workspace->unit_scaled_y : nullptr;
    update_reduced_deferred_empty_y_kernel<<<
        (state->signed_A_empty_count + kReducedVectorThreads - 1) /
            kReducedVectorThreads,
        kReducedVectorThreads, 0, workspace->stream>>>(
        workspace->y, workspace->AL, workspace->AU,
        workspace->y_bound_type, workspace->last_y, scaled_y,
        workspace->inverse_row_norm, state->row_fixed_shift,
        state->delta_count > 0
            ? state->delta_signed_A_nonempty_words : nullptr,
        state->signed_A_rows_empty, state->signed_A_empty_count,
        workspace->Halpern_params,
        state->deferred_empty_halpern_factors,
        deferred_iterations);
}

void enqueue_reduced_nonempty_cusparse_y_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    bool use_delta) {
    CUSPARSE_spmvop_A &full = state->spmv_A;
    check_cusparse(hprlp_run_spmvop(
        full.cusparseHandle, state->nonempty_A_operation,
        &full.alpha, &full.beta, state->nonempty_x_hat_descr,
        state->nonempty_Ax_descr, state->nonempty_Ax_descr),
        "capture production nonempty reduced A SpMVOp");
    const bool parallel_delta =
        use_delta && state->use_parallel_delta_cusparse_y;
    if (parallel_delta) {
        build_unscaled_delta_input_kernel<<<
            numBlocks(state->delta_count), kReducedVectorThreads, 0,
            workspace->stream>>>(
            state->parallel_delta_input, state->delta_x_hat,
            state->delta_old_mask, state->delta_lower,
            state->delta_upper, state->delta_count);
        check_cusparse(hprlp_run_spmvop(
            full.cusparseHandle, state->parallel_delta_operation,
            &full.alpha, &full.beta, state->parallel_delta_input_descr,
            state->parallel_delta_ax_descr,
            state->parallel_delta_ax_descr),
            "capture production parallel delta A SpMVOp");
    }
    HPRLP_FLOAT *const scaled_y = state->use_packed_dictionary_x
        ? workspace->unit_scaled_y : nullptr;
    if (state->defer_empty_rows_to_observation) {
        if (state->nonempty_A_count > 0) {
            update_reduced_nonempty_y_from_compact_ax_kernel<<<
                numBlocks(state->nonempty_A_count),
                kReducedVectorThreads, 0, workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU,
                workspace->y_bound_type, workspace->last_y,
                state->nonempty_Ax, state->nonempty_to_original_A,
                state->row_fixed_shift, scaled_y,
                workspace->inverse_row_norm,
                use_delta && !parallel_delta ? state->delta_input : nullptr,
                use_delta && !parallel_delta ? state->delta_x_hat : nullptr,
                use_delta && !parallel_delta ? state->delta_fixed : nullptr,
                use_delta && !parallel_delta
                    ? state->delta_A.rowPtr : nullptr,
                use_delta && !parallel_delta
                    ? state->delta_A.colIndex : nullptr,
                use_delta && !parallel_delta ? state->delta_A.value : nullptr,
                parallel_delta ? state->parallel_delta_ax : nullptr,
                use_delta ? state->delta_signed_A_nonempty_words : nullptr,
                use_delta, state->nonempty_A_count,
                workspace->Halpern_params, workspace->halpern_factors);
        }
        if (use_delta && state->signed_delta_only_count > 0) {
            update_reduced_delta_only_y_for_nonempty_cusparse_kernel<<<
                numBlocks(state->signed_delta_only_count),
                kReducedVectorThreads, 0, workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU,
                workspace->y_bound_type, workspace->last_y,
                state->row_fixed_shift, scaled_y,
                workspace->inverse_row_norm,
                !parallel_delta ? state->delta_input : nullptr,
                !parallel_delta ? state->delta_x_hat : nullptr,
                !parallel_delta ? state->delta_fixed : nullptr,
                !parallel_delta ? state->delta_A.rowPtr : nullptr,
                !parallel_delta ? state->delta_A.colIndex : nullptr,
                !parallel_delta ? state->delta_A.value : nullptr,
                parallel_delta ? state->parallel_delta_ax : nullptr,
                state->signed_delta_only_rows,
                state->signed_delta_only_count,
                workspace->Halpern_params, workspace->halpern_factors);
        }
    } else if (workspace->m > 0) {
        update_reduced_full_order_y_from_compact_ax_kernel<<<
            numBlocks(workspace->m), kReducedVectorThreads, 0,
            workspace->stream>>>(
            workspace->y, workspace->AL, workspace->AU,
            workspace->y_bound_type, workspace->last_y,
            state->nonempty_Ax, state->original_to_nonempty_A,
            state->row_fixed_shift, scaled_y,
            workspace->inverse_row_norm,
            use_delta && !parallel_delta ? state->delta_input : nullptr,
            use_delta && !parallel_delta ? state->delta_x_hat : nullptr,
            use_delta && !parallel_delta ? state->delta_fixed : nullptr,
            use_delta && !parallel_delta ? state->delta_A.rowPtr : nullptr,
            use_delta && !parallel_delta ? state->delta_A.colIndex : nullptr,
            use_delta && !parallel_delta ? state->delta_A.value : nullptr,
            parallel_delta ? state->parallel_delta_ax : nullptr,
            use_delta ? state->delta_signed_A_nonempty_words : nullptr,
            use_delta, workspace->m,
            workspace->Halpern_params, workspace->halpern_factors);
    }
}

void enqueue_reduced_iteration_updates(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    const bool use_delta = state->delta_count > 0;
    enqueue_reduced_x_updates(workspace, state, use_delta);
    enqueue_reduced_y_updates(workspace, state, use_delta);
}

void enqueue_reduced_production_iteration_body(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    const bool use_delta = state->delta_count > 0;
    if (state->defer_empty_rows_to_observation) {
        enqueue_reduced_x_updates(workspace, state, use_delta);
        enqueue_reduced_nonempty_y_updates_for_deferred_rows(
            workspace, state);
    } else if (state->use_nonempty_cusparse_y) {
        enqueue_reduced_x_updates(workspace, state, use_delta);
        enqueue_reduced_nonempty_cusparse_y_updates(
            workspace, state, use_delta);
    } else {
        enqueue_reduced_iteration_updates(workspace, state);
    }
}

void autotune_reduced_compressed_backends(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    const HPRLP_parameters *parameters,
    int iteration) {
    constexpr int short_probe_iterations = 3;
    constexpr int finalist_probe_iterations = 8;
    constexpr double minimum_improvement = 0.05;
    constexpr double early_abort_ratio = 1.50;
    const auto total_started = std::chrono::steady_clock::now();

    const std::size_t x_bytes =
        static_cast<std::size_t>(state->base_count) * sizeof(HPRLP_FLOAT);
    const std::size_t y_bytes =
        static_cast<std::size_t>(workspace->m) * sizeof(HPRLP_FLOAT);
    HPRLP_FLOAT *saved_x = nullptr;
    HPRLP_FLOAT *saved_x_hat = nullptr;
    HPRLP_FLOAT *saved_y = nullptr;
    allocate_device(&saved_x, state->base_count);
    allocate_device(&saved_x_hat, state->base_count);
    allocate_device(&saved_y, workspace->m);
    CUDA_CHECK(cudaMemcpyAsync(
        saved_x, state->x, x_bytes, cudaMemcpyDeviceToDevice,
        workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        saved_x_hat, state->x_hat, x_bytes, cudaMemcpyDeviceToDevice,
        workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        saved_y, workspace->y, y_bytes, cudaMemcpyDeviceToDevice,
        workspace->stream));

    auto restore_vectors = [&]() {
        CUDA_CHECK(cudaMemcpyAsync(
            state->x, saved_x, x_bytes, cudaMemcpyDeviceToDevice,
            workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            state->x_hat, saved_x_hat, x_bytes,
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            workspace->y, saved_y, y_bytes, cudaMemcpyDeviceToDevice,
            workspace->stream));
    };
    const bool nonempty_cusparse_available =
        state->use_nonempty_cusparse_y;
    auto configure = [&](bool fused_x, bool fused_y) {
        state->use_fused_x = fused_x;
        state->use_fused_y = fused_y;
        state->use_nonempty_cusparse_y =
            nonempty_cusparse_available && !fused_y;
    };

    struct Candidate {
        bool fused_x = false;
        bool fused_y = false;
        double short_ms = std::numeric_limits<double>::infinity();
        double final_ms = std::numeric_limits<double>::infinity();
        bool finalist = false;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(4);
    auto add_candidate = [&](bool fused_x, bool fused_y) {
        for (const Candidate &candidate : candidates) {
            if (candidate.fused_x == fused_x &&
                candidate.fused_y == fused_y) {
                return;
            }
        }
        candidates.push_back(Candidate{fused_x, fused_y});
    };
    const bool incumbent_fused_x = state->use_fused_x;
    const bool incumbent_fused_y = state->use_fused_y;
    add_candidate(incumbent_fused_x, incumbent_fused_y);
    add_candidate(false, false);
    add_candidate(true, false);
    add_candidate(false, true);
    add_candidate(true, true);

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));
    auto time_candidate = [&](Candidate &candidate, int probe_iterations) {
        restore_vectors();
        configure(candidate.fused_x, candidate.fused_y);
        enqueue_reduced_production_iteration_body(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        restore_vectors();
        CUDA_CHECK(cudaEventRecord(event_start, workspace->stream));
        for (int probe = 0; probe < probe_iterations; ++probe) {
            enqueue_reduced_production_iteration_body(workspace, state);
        }
        CUDA_CHECK(cudaEventRecord(event_stop, workspace->stream));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, event_start, event_stop));
        return static_cast<double>(elapsed_ms) / probe_iterations;
    };

    for (Candidate &candidate : candidates) {
        candidate.short_ms =
            time_candidate(candidate, short_probe_iterations);
    }
    const double incumbent_short_ms = candidates.front().short_ms;
    int best_alternate = -1;
    for (int index = 1; index < static_cast<int>(candidates.size());
         ++index) {
        if (candidates[index].short_ms <
            (best_alternate >= 0
                ? candidates[best_alternate].short_ms
                : std::numeric_limits<double>::infinity())) {
            best_alternate = index;
        }
    }

    candidates.front().finalist = true;
    candidates.front().final_ms = time_candidate(
        candidates.front(), finalist_probe_iterations);
    if (best_alternate >= 0 &&
        candidates[best_alternate].short_ms <=
            incumbent_short_ms * early_abort_ratio) {
        candidates[best_alternate].finalist = true;
        candidates[best_alternate].final_ms = time_candidate(
            candidates[best_alternate], finalist_probe_iterations);
    }
    auto combined_time = [&](const Candidate &candidate) {
        if (!candidate.finalist || !std::isfinite(candidate.final_ms)) {
            return candidate.short_ms;
        }
        return (candidate.short_ms * short_probe_iterations +
                candidate.final_ms * finalist_probe_iterations) /
            (short_probe_iterations + finalist_probe_iterations);
    };

    const double incumbent_ms = combined_time(candidates.front());
    int selected = 0;
    double selected_ms = incumbent_ms;
    if (best_alternate >= 0 && candidates[best_alternate].finalist) {
        const double alternate_ms =
            combined_time(candidates[best_alternate]);
        if (std::isfinite(alternate_ms) && alternate_ms < selected_ms) {
            selected = best_alternate;
            selected_ms = alternate_ms;
        }
    }
    const double probe_total_ms = 1000.0 * std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_started).count();
    const double saved_ms = incumbent_ms - selected_ms;
    const double improvement = incumbent_ms > 0.0
        ? saved_ms / incumbent_ms : 0.0;
    const double payback_iterations = saved_ms > 0.0
        ? probe_total_ms / saved_ms
        : std::numeric_limits<double>::infinity();
    const int remaining_iterations = parameters != nullptr
        ? std::max(0, parameters->max_iter - iteration) : 0;
    if (selected != 0 &&
        (improvement < minimum_improvement ||
         payback_iterations > remaining_iterations)) {
        selected = 0;
        selected_ms = incumbent_ms;
    }

    const Candidate &choice = candidates[selected];
    configure(choice.fused_x, choice.fused_y);
    workspace->reduced_use_fused_x = choice.fused_x;
    workspace->reduced_use_fused_y = choice.fused_y;
    workspace->reduced_backend_autotune_done = true;
    state->defer_empty_rows_selected =
        state->defer_empty_rows_to_observation;
    if (choice.fused_y &&
        !state->defer_empty_rows_to_observation) {
        state->use_cusparse_empty_row_batch = false;
    }
    restore_vectors();
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

    auto backend_name = [&](const Candidate &candidate) {
        const std::string x_name = candidate.fused_x
            ? "fused" : "cusparse";
        std::string y_name;
        if (state->defer_empty_rows_to_observation) {
            y_name = !candidate.fused_y && nonempty_cusparse_available
                ? "nonempty-cusparse+defer" : "fused+defer";
        } else if (!candidate.fused_y && nonempty_cusparse_available) {
            y_name = "nonempty-cusparse";
        } else {
            y_name = candidate.fused_y ? "fused" : "cusparse";
        }
        return x_name + "/" + y_name;
    };
    std::cout << "  reduced compressed autotune: iteration=" << iteration
              << ", shape=" << workspace->m << "x"
              << state->base_count << ", nnz=" << state->A.numElements
              << ", short_probes=" << short_probe_iterations
              << ", finalist_probes=" << finalist_probe_iterations
              << ", probe_total_ms=" << std::fixed
              << std::setprecision(3) << probe_total_ms << std::endl;
    for (const Candidate &candidate : candidates) {
        std::cout << "    candidate=" << backend_name(candidate)
                  << " short_ms=" << std::fixed << std::setprecision(6)
                  << candidate.short_ms;
        if (candidate.finalist) {
            std::cout << " combined_ms=" << combined_time(candidate);
        } else {
            std::cout << " early_rejected=1";
        }
        std::cout << std::endl;
    }
    std::cout << "    selected=" << backend_name(choice)
              << " incumbent=" << backend_name(candidates.front())
              << " measured_improvement=" << std::fixed
              << std::setprecision(4) << improvement
              << " payback_iterations=" << std::setprecision(1)
              << payback_iterations
              << std::defaultfloat << std::setprecision(2)
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));
    CUDA_CHECK(cudaFree(saved_x));
    CUDA_CHECK(cudaFree(saved_x_hat));
    CUDA_CHECK(cudaFree(saved_y));
}

void capture_reduced_graphs(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    destroy_reduced_graph(state);
    refresh_reduced_factor_scaled_y(workspace, state);

    CUDA_CHECK(cudaStreamBeginCapture(
        workspace->stream, cudaStreamCaptureModeGlobal));
    enqueue_reduced_production_iteration_body(workspace, state);
    advance_halpern_factors(workspace);
    CUDA_CHECK(cudaStreamEndCapture(workspace->stream, &state->graph));
    CUDA_CHECK(cudaGraphInstantiate(
        &state->graph_exec, state->graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaStreamBeginCapture(
        workspace->stream, cudaStreamCaptureModeGlobal));
    HPRLP_FLOAT *const canonical_halpern_factors =
        workspace->halpern_factors;
    prepare_halpern_factor_batch_kernel<<<1, 1, 0, workspace->stream>>>(
        workspace->halpern_inner, canonical_halpern_factors,
        workspace->halpern_factor_batch, HPRLP_NORMAL_GRAPH_BATCH_SIZE);
    if (state->defer_empty_rows_to_observation) {
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_x_updates(
                workspace, state, state->delta_count > 0);
            enqueue_reduced_nonempty_y_updates_for_deferred_rows(
                workspace, state);
        }
    } else if (state->use_nonempty_cusparse_y) {
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_x_updates(
                workspace, state, state->delta_count > 0);
            enqueue_reduced_nonempty_cusparse_y_updates(
                workspace, state, state->delta_count > 0);
        }
    } else if (state->use_signed_empty_row_batch) {
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_x_updates(
                workspace, state, state->delta_count > 0);
            enqueue_reduced_signed_nonempty_y_updates(workspace, state);
        }
        enqueue_reduced_signed_empty_y_batch(workspace, state);
    } else if (state->use_dictionary_empty_row_batch) {
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_x_updates(
                workspace, state, state->delta_count > 0);
            enqueue_reduced_dictionary_nonempty_y_updates(workspace, state);
        }
        enqueue_reduced_dictionary_empty_y_batch(workspace, state);
    } else if (state->use_cusparse_empty_row_batch) {
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_x_updates(
                workspace, state, state->delta_count > 0);
            enqueue_reduced_generic_nonempty_y_updates(
                workspace, state, state->delta_count > 0);
        }
        enqueue_reduced_generic_empty_y_batch(workspace, state);
    } else {
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_iteration_updates(workspace, state);
        }
    }
    workspace->halpern_factors = canonical_halpern_factors;
    CUDA_CHECK(cudaStreamEndCapture(
        workspace->stream, &state->graph_batch));
    CUDA_CHECK(cudaGraphInstantiate(
        &state->graph_exec_batch, state->graph_batch, nullptr, nullptr, 0));
}

bool reduced_profile_enabled() {
    const char *value = std::getenv("HPRLP_PROFILE_REDUCED");
    if (value == nullptr) return false;
    const std::string setting(value);
    return setting == "1" || setting == "true" || setting == "TRUE" ||
        setting == "yes" || setting == "YES";
}

bool reduced_delta_profile_requested() {
    const char *value = std::getenv("HPRLP_PROFILE_REDUCED_DELTA");
    if (value == nullptr) return false;
    const std::string setting(value);
    return setting == "1" || setting == "true" || setting == "TRUE" ||
           setting == "yes" || setting == "YES";
}

bool reduced_nonempty_cusparse_profile_requested() {
    const char *value =
        std::getenv("HPRLP_PROFILE_REDUCED_NONEMPTY_CUSPARSE");
    if (value == nullptr) return false;
    const std::string setting(value);
    return setting == "1" || setting == "true" || setting == "TRUE" ||
           setting == "yes" || setting == "YES";
}

const std::vector<int> &reduced_profile_iteration_schedule() {
    static const std::vector<int> schedule = []() {
        std::vector<int> result;
        const char *value =
            std::getenv("HPRLP_PROFILE_REDUCED_ITERATIONS");
        if (value == nullptr) return result;
        const std::string setting(value);
        std::size_t begin = 0;
        while (begin < setting.size()) {
            const std::size_t end = setting.find(',', begin);
            const std::string token = setting.substr(
                begin, end == std::string::npos
                    ? std::string::npos : end - begin);
            const int iteration = std::atoi(token.c_str());
            if (iteration >= 0) result.push_back(iteration);
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return result;
    }();
    return schedule;
}

void profile_reduced_graph(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state,
    int iteration) {
    if (state->profile_done || !state->profile_enabled) return;
    const std::vector<int> &schedule =
        reduced_profile_iteration_schedule();
    const bool uses_schedule = !schedule.empty();
    if (uses_schedule &&
        state->profile_count >= static_cast<int>(schedule.size())) {
        state->profile_done = true;
        return;
    }
    const char *minimum_iteration_setting =
        std::getenv("HPRLP_PROFILE_REDUCED_MIN_ITERATION");
    const int minimum_iteration = uses_schedule
        ? schedule[state->profile_count]
        : (minimum_iteration_setting != nullptr
               ? std::atoi(minimum_iteration_setting) : 0);
    if (iteration < minimum_iteration) return;
    if (reduced_delta_profile_requested() && state->delta_count == 0) return;
    state->profile_count += 1;
    state->profile_done = !uses_schedule ||
        state->profile_count >= static_cast<int>(schedule.size());

    constexpr int repeats = 100;
    const std::size_t base_bytes =
        static_cast<std::size_t>(state->base_count) * sizeof(HPRLP_FLOAT);
    const std::size_t delta_bytes =
        static_cast<std::size_t>(state->delta_count) * sizeof(HPRLP_FLOAT);
    const std::size_t y_bytes =
        static_cast<std::size_t>(workspace->m) * sizeof(HPRLP_FLOAT);

    HPRLP_FLOAT *x_backup = nullptr;
    HPRLP_FLOAT *x_hat_backup = nullptr;
    HPRLP_FLOAT *delta_x_backup = nullptr;
    HPRLP_FLOAT *delta_x_hat_backup = nullptr;
    HPRLP_FLOAT *y_backup = nullptr;
    int *halpern_inner_backup = nullptr;
    HPRLP_FLOAT *halpern_factors_backup = nullptr;
    if (state->base_count > 0) {
        allocate_device(&x_backup, state->base_count);
        allocate_device(&x_hat_backup, state->base_count);
    }
    if (state->delta_count > 0) {
        allocate_device(&delta_x_backup, state->delta_count);
        allocate_device(&delta_x_hat_backup, state->delta_count);
    }
    allocate_device(&y_backup, workspace->m);
    allocate_device(&halpern_inner_backup, 1);
    allocate_device(&halpern_factors_backup, 2);

    auto save_mutable_state = [&]() {
        if (base_bytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                x_backup, state->x, base_bytes, cudaMemcpyDeviceToDevice,
                workspace->stream));
            CUDA_CHECK(cudaMemcpyAsync(
                x_hat_backup, state->x_hat, base_bytes,
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        if (delta_bytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                delta_x_backup, state->delta_x, delta_bytes,
                cudaMemcpyDeviceToDevice, workspace->stream));
            CUDA_CHECK(cudaMemcpyAsync(
                delta_x_hat_backup, state->delta_x_hat, delta_bytes,
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            y_backup, workspace->y, y_bytes, cudaMemcpyDeviceToDevice,
            workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            halpern_inner_backup, workspace->halpern_inner, sizeof(int),
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            halpern_factors_backup, workspace->halpern_factors,
            2 * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice,
            workspace->stream));
    };
    auto restore_mutable_state = [&]() {
        if (base_bytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                state->x, x_backup, base_bytes, cudaMemcpyDeviceToDevice,
                workspace->stream));
            CUDA_CHECK(cudaMemcpyAsync(
                state->x_hat, x_hat_backup, base_bytes,
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        if (delta_bytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                state->delta_x, delta_x_backup, delta_bytes,
                cudaMemcpyDeviceToDevice, workspace->stream));
            CUDA_CHECK(cudaMemcpyAsync(
                state->delta_x_hat, delta_x_hat_backup, delta_bytes,
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            workspace->y, y_backup, y_bytes, cudaMemcpyDeviceToDevice,
            workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            workspace->halpern_inner, halpern_inner_backup, sizeof(int),
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            workspace->halpern_factors, halpern_factors_backup,
            2 * sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToDevice,
            workspace->stream));
    };

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto time_repeated = [&](const auto &enqueue) {
        CUDA_CHECK(cudaEventRecord(start, workspace->stream));
        for (int repeat = 0; repeat < repeats; ++repeat) enqueue();
        CUDA_CHECK(cudaEventRecord(stop, workspace->stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed / repeats;
    };

    save_mutable_state();
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    const float graph_ms = time_repeated([&]() {
        CUDA_CHECK(cudaGraphLaunch(state->graph_exec, workspace->stream));
    });
    restore_mutable_state();
    refresh_reduced_factor_scaled_y(workspace, state);
    refresh_reduced_factor_scaled_x_hat(workspace, state);
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    const float graph_batch_ms = time_repeated([&]() {
        CUDA_CHECK(cudaGraphLaunch(
            state->graph_exec_batch, workspace->stream));
    }) / HPRLP_NORMAL_GRAPH_BATCH_SIZE;
    restore_mutable_state();
    refresh_reduced_factor_scaled_y(workspace, state);
    refresh_reduced_factor_scaled_x_hat(workspace, state);
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

    float x_stage_ms = -1.0f;
    float y_stage_ms = -1.0f;
    float y_nonempty_only_ms = -1.0f;
    float y_empty_batch_ms_per_iteration = -1.0f;
    float y_base_only_ms = -1.0f;
    float y_alternate_ms = -1.0f;
    float y_cusparse_spmv_ms = -1.0f;
    float y_dense_add_shift_ms = -1.0f;
    float y_dense_update_ms = -1.0f;
    float y_nonempty_cusparse_graph_batch_ms = -1.0f;
    float y_nonempty_cusparse_parallel_delta_graph_batch_ms = -1.0f;
    float y_nonempty_cusparse_spmv_ms = -1.0f;
    float y_nonempty_cusparse_update_ms = -1.0f;
    double y_nonempty_cusparse_build_ms = -1.0;
    int y_nonempty_cusparse_rows = -1;
    int y_nonempty_cusparse_delta_only_rows = -1;
    long long y_nonempty_cusparse_x_mismatches = -1;
    long long y_nonempty_cusparse_x_hat_mismatches = -1;
    long long y_nonempty_cusparse_y_mismatches = -1;
    long long y_nonempty_cusparse_scaled_y_mismatches = -1;
    int y_nonempty_cusparse_halpern_mismatches = -1;
    HPRLP_FLOAT y_nonempty_cusparse_max_abs_diff = -1.0;
    HPRLP_FLOAT y_nonempty_cusparse_max_relative_diff = -1.0;
    long long y_nonempty_cusparse_parallel_delta_mismatches = -1;
    HPRLP_FLOAT y_nonempty_cusparse_parallel_delta_max_abs_diff = -1.0;
    HPRLP_FLOAT y_nonempty_cusparse_parallel_delta_max_relative_diff = -1.0;
    float halpern_ms = -1.0f;
    const bool use_delta = state->delta_count > 0;
    if (state->base_count > 0 || use_delta) {
        x_stage_ms = time_repeated([&]() {
            enqueue_reduced_x_updates(workspace, state, use_delta);
        });
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        if (!use_delta && state->use_cusparse_empty_row_batch) {
            y_alternate_ms = time_repeated([&]() {
                enqueue_reduced_generic_nonempty_y_updates(
                    workspace, state, false);
            });
            restore_mutable_state();
            refresh_reduced_factor_scaled_y(workspace, state);
            refresh_reduced_factor_scaled_x_hat(workspace, state);
            CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        }
    }

    y_stage_ms = time_repeated([&]() {
        enqueue_reduced_y_updates(workspace, state, use_delta);
    });
    restore_mutable_state();
    refresh_reduced_factor_scaled_y(workspace, state);
    refresh_reduced_factor_scaled_x_hat(workspace, state);
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

    const bool uses_generic_cusparse_y =
        state->base_count > 0 && !state->use_packed_dictionary &&
        !state->use_signed_factorized && !state->use_unit_active_scatter &&
        !state->use_unit_factorized && !state->use_fused_y;
    if (uses_generic_cusparse_y) {
        CUSPARSE_spmvop_A &a = state->spmv_A;
        y_cusparse_spmv_ms = time_repeated([&]() {
            check_cusparse(hprlp_run_spmvop(
                a.cusparseHandle, a.operation, &a.alpha, &a.beta,
                a.x_hat_cusparseDescr, a.Ax_cusparseDescr,
                a.Ax_cusparseDescr), "profile reduced A SpMVOp");
        });
        y_dense_add_shift_ms = time_repeated([&]() {
            add_fixed_shift_kernel<<<
                (workspace->m + kReducedVectorThreads - 1) /
                    kReducedVectorThreads,
                kReducedVectorThreads, 0, workspace->stream>>>(
                state->Ax, state->row_fixed_shift, workspace->m);
        });
        y_dense_update_ms = time_repeated([&]() {
            update_y_normal_kernel<<<
                (workspace->m + kReducedYThreads - 1) / kReducedYThreads,
                kReducedYThreads, 0, workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU, state->Ax,
                workspace->last_y, workspace->Halpern_params,
                workspace->halpern_factors, workspace->m);
        });
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    }

    const bool profile_nonempty_cusparse =
        reduced_nonempty_cusparse_profile_requested();
    if (uses_generic_cusparse_y && profile_nonempty_cusparse &&
        state->defer_empty_rows_to_observation) {
        std::cout << "  reduced nonempty cuSPARSE candidate profile skipped: "
                  << "production defers empty rows" << std::endl;
    }
    if (uses_generic_cusparse_y && profile_nonempty_cusparse &&
        !state->defer_empty_rows_to_observation) {
        const auto candidate_build_started = std::chrono::steady_clock::now();
        int *original_to_nonempty = nullptr;
        int *unused_nonempty_to_original = nullptr;
        int *nonempty_row_ptr = nullptr;
        HPRLP_FLOAT *nonempty_ax = nullptr;
        const int nonempty_count = build_nonempty_row_metadata_gpu(
            state->A.rowPtr, workspace->m, state->A.numElements, false,
            &original_to_nonempty, &unused_nonempty_to_original,
            &nonempty_row_ptr, workspace->stream);
        y_nonempty_cusparse_rows = nonempty_count;
        allocate_device(&nonempty_ax, nonempty_count);

        cusparseHandle_t compact_handle = nullptr;
        cusparseSpMatDescr_t compact_matrix = nullptr;
        cusparseDnVecDescr_t compact_x_hat = nullptr;
        cusparseDnVecDescr_t compact_ax = nullptr;
        HPRLP_spmvop compact_operation{};
        HPRLP_FLOAT compact_alpha = 1.0;
        HPRLP_FLOAT compact_beta = 0.0;
        check_cusparse(cusparseCreate(&compact_handle),
                       "create nonempty profile handle");
        check_cusparse(cusparseSetStream(
            compact_handle, workspace->stream),
            "set nonempty profile stream");
        check_cusparse(cusparseCreateCsr(
            &compact_matrix, nonempty_count, state->A.col,
            state->A.numElements, nonempty_row_ptr, state->A.colIndex,
            state->A.value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "create nonempty profile A");
        check_cusparse(cusparseCreateDnVec(
            &compact_x_hat, state->base_count, state->x_hat, CUDA_R_64F),
            "create nonempty profile xhat");
        check_cusparse(cusparseCreateDnVec(
            &compact_ax, nonempty_count, nonempty_ax, CUDA_R_64F),
            "create nonempty profile Ax");
        check_cusparse(hprlp_prepare_spmvop(
            compact_handle, compact_matrix, compact_x_hat, compact_ax,
            compact_ax, CUDA_R_64F, &compact_operation),
            "prepare nonempty profile A SpMVOp");

        std::uint32_t *candidate_delta_nonempty_words = nullptr;
        int *candidate_delta_only_rows = nullptr;
        int *candidate_delta_only_count_device = nullptr;
        int candidate_delta_only_count = 0;
        if (use_delta) {
            const std::size_t word_count =
                (static_cast<std::size_t>(workspace->m) + 31) / 32;
            allocate_device(
                &candidate_delta_nonempty_words, word_count);
            CUDA_CHECK(cudaMemsetAsync(
                candidate_delta_nonempty_words, 0,
                word_count * sizeof(std::uint32_t), workspace->stream));
            mark_nonempty_csr_rows_kernel<<<
                numBlocks(workspace->m), kReducedVectorThreads, 0,
                workspace->stream>>>(
                state->delta_A.rowPtr,
                candidate_delta_nonempty_words, workspace->m);
            const std::size_t delta_only_capacity =
                std::max<std::size_t>(
                    1, static_cast<std::size_t>(state->delta_nnz));
            allocate_device(
                &candidate_delta_only_rows, delta_only_capacity);
            allocate_device(&candidate_delta_only_count_device, 1);
            CUDA_CHECK(cudaMemsetAsync(
                candidate_delta_only_count_device, 0, sizeof(int),
                workspace->stream));
            build_delta_only_row_list_kernel<<<
                numBlocks(workspace->m), kReducedVectorThreads, 0,
                workspace->stream>>>(
                state->A.rowPtr, state->delta_A.rowPtr,
                candidate_delta_only_rows,
                candidate_delta_only_count_device, workspace->m);
            CUDA_CHECK(cudaMemcpyAsync(
                &candidate_delta_only_count,
                candidate_delta_only_count_device, sizeof(int),
                cudaMemcpyDeviceToHost, workspace->stream));
            CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        }
        y_nonempty_cusparse_delta_only_rows =
            candidate_delta_only_count;

        HPRLP_FLOAT *candidate_delta_fixed = nullptr;
        HPRLP_FLOAT *candidate_delta_input = nullptr;
        if (use_delta) {
            allocate_device(&candidate_delta_fixed, state->delta_count);
            if (state->delta_nnz >=
                kReducedPrecomputedDeltaInputMinimumNnz) {
                allocate_device(
                    &candidate_delta_input, state->delta_count);
            }
            fixed_values_from_mask_kernel<<<
                numBlocks(state->delta_count), kReducedVectorThreads, 0,
                workspace->stream>>>(
                candidate_delta_fixed, state->delta_old_mask,
                state->delta_lower, state->delta_upper,
                state->delta_count);
            if (candidate_delta_input != nullptr) {
                build_unscaled_delta_input_kernel<<<
                    numBlocks(state->delta_count), kReducedVectorThreads, 0,
                    workspace->stream>>>(
                    candidate_delta_input, state->delta_x_hat,
                    state->delta_old_mask, state->delta_lower,
                    state->delta_upper, state->delta_count);
            }
        }

        HPRLP_FLOAT *parallel_delta_input = nullptr;
        HPRLP_FLOAT *parallel_delta_ax = nullptr;
        cusparseSpMatDescr_t parallel_delta_matrix = nullptr;
        cusparseDnVecDescr_t parallel_delta_input_descr = nullptr;
        cusparseDnVecDescr_t parallel_delta_ax_descr = nullptr;
        HPRLP_spmvop parallel_delta_operation{};
        if (use_delta) {
            allocate_device(&parallel_delta_input, state->delta_count);
            allocate_device(&parallel_delta_ax, workspace->m);
            check_cusparse(cusparseCreateCsr(
                &parallel_delta_matrix, workspace->m, state->delta_count,
                state->delta_nnz, state->delta_A.rowPtr,
                state->delta_A.colIndex, state->delta_A.value,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "create parallel delta profile A");
            check_cusparse(cusparseCreateDnVec(
                &parallel_delta_input_descr, state->delta_count,
                parallel_delta_input, CUDA_R_64F),
                "create parallel delta profile input");
            check_cusparse(cusparseCreateDnVec(
                &parallel_delta_ax_descr, workspace->m,
                parallel_delta_ax, CUDA_R_64F),
                "create parallel delta profile Ax");
            check_cusparse(hprlp_prepare_spmvop(
                compact_handle, parallel_delta_matrix,
                parallel_delta_input_descr, parallel_delta_ax_descr,
                parallel_delta_ax_descr, CUDA_R_64F,
                &parallel_delta_operation),
                "prepare parallel delta profile A SpMVOp");
        }

        HPRLP_FLOAT *const candidate_scaled_y =
            state->use_packed_dictionary_x ? workspace->unit_scaled_y
                                           : nullptr;
        auto enqueue_candidate_nonempty_y = [&]() {
            check_cusparse(hprlp_run_spmvop(
                compact_handle, compact_operation, &compact_alpha,
                &compact_beta, compact_x_hat, compact_ax, compact_ax),
                "run nonempty profile A SpMVOp");
            if (workspace->m > 0) {
                update_reduced_full_order_y_from_compact_ax_kernel<<<
                    numBlocks(workspace->m), kReducedVectorThreads, 0,
                    workspace->stream>>>(
                    workspace->y, workspace->AL, workspace->AU,
                    workspace->y_bound_type, workspace->last_y,
                    nonempty_ax, original_to_nonempty,
                    state->row_fixed_shift, candidate_scaled_y,
                    workspace->inverse_row_norm,
                    use_delta ? candidate_delta_input : nullptr,
                    use_delta ? state->delta_x_hat : nullptr,
                    candidate_delta_fixed,
                    use_delta ? state->delta_A.rowPtr : nullptr,
                    use_delta ? state->delta_A.colIndex : nullptr,
                    use_delta ? state->delta_A.value : nullptr,
                    nullptr,
                    candidate_delta_nonempty_words,
                    use_delta, workspace->m,
                    workspace->Halpern_params,
                    workspace->halpern_factors);
            }
        };
        auto enqueue_candidate_parallel_delta_y = [&]() {
            check_cusparse(hprlp_run_spmvop(
                compact_handle, compact_operation, &compact_alpha,
                &compact_beta, compact_x_hat, compact_ax, compact_ax),
                "run parallel-delta profile base A SpMVOp");
            build_unscaled_delta_input_kernel<<<
                numBlocks(state->delta_count), kReducedVectorThreads, 0,
                workspace->stream>>>(
                parallel_delta_input, state->delta_x_hat,
                state->delta_old_mask, state->delta_lower,
                state->delta_upper, state->delta_count);
            check_cusparse(hprlp_run_spmvop(
                compact_handle, parallel_delta_operation, &compact_alpha,
                &compact_beta, parallel_delta_input_descr,
                parallel_delta_ax_descr, parallel_delta_ax_descr),
                "run parallel-delta profile delta A SpMVOp");
            if (workspace->m > 0) {
                update_reduced_full_order_y_from_compact_ax_kernel<<<
                    numBlocks(workspace->m), kReducedVectorThreads, 0,
                    workspace->stream>>>(
                    workspace->y, workspace->AL, workspace->AU,
                    workspace->y_bound_type, workspace->last_y,
                    nonempty_ax, original_to_nonempty,
                    state->row_fixed_shift, candidate_scaled_y,
                    workspace->inverse_row_norm, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr,
                    parallel_delta_ax, candidate_delta_nonempty_words,
                    true, workspace->m,
                    workspace->Halpern_params,
                    workspace->halpern_factors);
            }
        };
        y_nonempty_cusparse_spmv_ms = time_repeated([&]() {
            check_cusparse(hprlp_run_spmvop(
                compact_handle, compact_operation, &compact_alpha,
                &compact_beta, compact_x_hat, compact_ax, compact_ax),
                "time nonempty profile A SpMVOp");
        });
        y_nonempty_cusparse_update_ms = time_repeated([&]() {
            if (workspace->m <= 0) return;
            update_reduced_full_order_y_from_compact_ax_kernel<<<
                numBlocks(workspace->m), kReducedVectorThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->AL, workspace->AU,
                workspace->y_bound_type, workspace->last_y,
                nonempty_ax, original_to_nonempty,
                state->row_fixed_shift, candidate_scaled_y,
                workspace->inverse_row_norm,
                use_delta ? candidate_delta_input : nullptr,
                use_delta ? state->delta_x_hat : nullptr,
                candidate_delta_fixed,
                use_delta ? state->delta_A.rowPtr : nullptr,
                use_delta ? state->delta_A.colIndex : nullptr,
                use_delta ? state->delta_A.value : nullptr,
                nullptr, candidate_delta_nonempty_words,
                use_delta, workspace->m,
                workspace->Halpern_params,
                workspace->halpern_factors);
        });
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        cudaGraph_t candidate_graph = nullptr;
        cudaGraphExec_t candidate_graph_exec = nullptr;
        HPRLP_FLOAT *const saved_delta_fixed = state->delta_fixed;
        HPRLP_FLOAT *const saved_delta_input = state->delta_input;
        if (use_delta) {
            state->delta_fixed = candidate_delta_fixed;
            state->delta_input = candidate_delta_input;
        }
        CUDA_CHECK(cudaStreamBeginCapture(
            workspace->stream, cudaStreamCaptureModeGlobal));
        HPRLP_FLOAT *const canonical_halpern_factors =
            workspace->halpern_factors;
        prepare_halpern_factor_batch_kernel<<<1, 1, 0, workspace->stream>>>(
            workspace->halpern_inner, canonical_halpern_factors,
            workspace->halpern_factor_batch,
            HPRLP_NORMAL_GRAPH_BATCH_SIZE);
        for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
            workspace->halpern_factors =
                workspace->halpern_factor_batch + 2 * i;
            enqueue_reduced_x_updates(workspace, state, use_delta);
            enqueue_candidate_nonempty_y();
        }
        workspace->halpern_factors = canonical_halpern_factors;
        CUDA_CHECK(cudaStreamEndCapture(
            workspace->stream, &candidate_graph));
        CUDA_CHECK(cudaGraphInstantiate(
            &candidate_graph_exec, candidate_graph, nullptr, nullptr, 0));

        cudaGraph_t parallel_delta_graph = nullptr;
        cudaGraphExec_t parallel_delta_graph_exec = nullptr;
        if (use_delta) {
            state->delta_input = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(
                workspace->stream, cudaStreamCaptureModeGlobal));
            prepare_halpern_factor_batch_kernel<<<
                1, 1, 0, workspace->stream>>>(
                workspace->halpern_inner, canonical_halpern_factors,
                workspace->halpern_factor_batch,
                HPRLP_NORMAL_GRAPH_BATCH_SIZE);
            for (int i = 0; i < HPRLP_NORMAL_GRAPH_BATCH_SIZE; ++i) {
                workspace->halpern_factors =
                    workspace->halpern_factor_batch + 2 * i;
                enqueue_reduced_x_updates(workspace, state, true);
                enqueue_candidate_parallel_delta_y();
            }
            workspace->halpern_factors = canonical_halpern_factors;
            CUDA_CHECK(cudaStreamEndCapture(
                workspace->stream, &parallel_delta_graph));
            CUDA_CHECK(cudaGraphInstantiate(
                &parallel_delta_graph_exec, parallel_delta_graph,
                nullptr, nullptr, 0));
        }
        state->delta_fixed = saved_delta_fixed;
        state->delta_input = saved_delta_input;
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        y_nonempty_cusparse_build_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - candidate_build_started)
                .count();

        HPRLP_FLOAT *production_x = nullptr;
        HPRLP_FLOAT *production_x_hat = nullptr;
        HPRLP_FLOAT *production_y = nullptr;
        HPRLP_FLOAT *production_scaled_y = nullptr;
        allocate_device(&production_x, state->base_count);
        allocate_device(&production_x_hat, state->base_count);
        allocate_device(&production_y, workspace->m);
        if (candidate_scaled_y != nullptr) {
            allocate_device(&production_scaled_y, workspace->m);
        }
        int production_halpern_inner = 0;
        HPRLP_FLOAT production_halpern_factors[2] = {};

        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        CUDA_CHECK(cudaGraphLaunch(
            state->graph_exec_batch, workspace->stream));
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            production_x, state->x,
            static_cast<std::size_t>(state->base_count) *
                sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            production_x_hat, state->x_hat,
            static_cast<std::size_t>(state->base_count) *
                sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice, workspace->stream));
        CUDA_CHECK(cudaMemcpyAsync(
            production_y, workspace->y,
            static_cast<std::size_t>(workspace->m) * sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice, workspace->stream));
        if (candidate_scaled_y != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(
                production_scaled_y, workspace->unit_scaled_y,
                static_cast<std::size_t>(workspace->m) *
                    sizeof(HPRLP_FLOAT),
                cudaMemcpyDeviceToDevice, workspace->stream));
        }
        CUDA_CHECK(cudaMemcpy(
            &production_halpern_inner, workspace->halpern_inner,
            sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            production_halpern_factors, workspace->halpern_factors,
            sizeof(production_halpern_factors), cudaMemcpyDeviceToHost));

        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        CUDA_CHECK(cudaGraphLaunch(candidate_graph_exec, workspace->stream));
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        HPRLP_FLOAT max_abs_diff = 0.0;
        HPRLP_FLOAT max_relative_diff = 0.0;
        auto compare_float_vector = [&](const HPRLP_FLOAT *reference,
                                        const HPRLP_FLOAT *candidate,
                                        std::size_t count,
                                        long long *mismatches) {
            const HPRLP_device_vector_compare_result result =
                compare_vectors_gpu(
                    reference, candidate, count, workspace->stream);
            *mismatches = static_cast<long long>(result.mismatch_count);
            max_abs_diff = std::max(
                max_abs_diff,
                hprlp_nonnegative_double_from_bits(result.max_abs_bits));
            max_relative_diff = std::max(
                max_relative_diff,
                hprlp_nonnegative_double_from_bits(
                    result.max_relative_bits));
        };
        compare_float_vector(
            production_x, state->x,
            static_cast<std::size_t>(state->base_count),
            &y_nonempty_cusparse_x_mismatches);
        compare_float_vector(
            production_x_hat, state->x_hat,
            static_cast<std::size_t>(state->base_count),
            &y_nonempty_cusparse_x_hat_mismatches);
        compare_float_vector(
            production_y, workspace->y,
            static_cast<std::size_t>(workspace->m),
            &y_nonempty_cusparse_y_mismatches);
        if (candidate_scaled_y != nullptr) {
            compare_float_vector(
                production_scaled_y, workspace->unit_scaled_y,
                static_cast<std::size_t>(workspace->m),
                &y_nonempty_cusparse_scaled_y_mismatches);
        } else {
            y_nonempty_cusparse_scaled_y_mismatches = 0;
        }
        int candidate_halpern_inner = 0;
        HPRLP_FLOAT candidate_halpern_factors[2] = {};
        CUDA_CHECK(cudaMemcpy(
            &candidate_halpern_inner, workspace->halpern_inner,
            sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            candidate_halpern_factors, workspace->halpern_factors,
            sizeof(candidate_halpern_factors), cudaMemcpyDeviceToHost));
        y_nonempty_cusparse_halpern_mismatches =
            (production_halpern_inner != candidate_halpern_inner ? 1 : 0) +
            (production_halpern_factors[0] !=
                     candidate_halpern_factors[0]
                 ? 1
                 : 0) +
            (production_halpern_factors[1] !=
                     candidate_halpern_factors[1]
                 ? 1
                 : 0);
        y_nonempty_cusparse_max_abs_diff = max_abs_diff;
        y_nonempty_cusparse_max_relative_diff = max_relative_diff;

        if (use_delta) {
            restore_mutable_state();
            refresh_reduced_factor_scaled_y(workspace, state);
            refresh_reduced_factor_scaled_x_hat(workspace, state);
            CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
            CUDA_CHECK(cudaGraphLaunch(
                parallel_delta_graph_exec, workspace->stream));
            CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

            max_abs_diff = 0.0;
            max_relative_diff = 0.0;
            long long mismatch_count = 0;
            long long total_mismatches = 0;
            compare_float_vector(
                production_x, state->x,
                static_cast<std::size_t>(state->base_count),
                &mismatch_count);
            total_mismatches += mismatch_count;
            compare_float_vector(
                production_x_hat, state->x_hat,
                static_cast<std::size_t>(state->base_count),
                &mismatch_count);
            total_mismatches += mismatch_count;
            compare_float_vector(
                production_y, workspace->y,
                static_cast<std::size_t>(workspace->m),
                &mismatch_count);
            total_mismatches += mismatch_count;
            if (candidate_scaled_y != nullptr) {
                compare_float_vector(
                    production_scaled_y, workspace->unit_scaled_y,
                    static_cast<std::size_t>(workspace->m),
                    &mismatch_count);
                total_mismatches += mismatch_count;
            }
            CUDA_CHECK(cudaMemcpy(
                &candidate_halpern_inner, workspace->halpern_inner,
                sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                candidate_halpern_factors, workspace->halpern_factors,
                sizeof(candidate_halpern_factors),
                cudaMemcpyDeviceToHost));
            total_mismatches +=
                production_halpern_inner != candidate_halpern_inner ? 1 : 0;
            total_mismatches +=
                production_halpern_factors[0] !=
                    candidate_halpern_factors[0] ? 1 : 0;
            total_mismatches +=
                production_halpern_factors[1] !=
                    candidate_halpern_factors[1] ? 1 : 0;
            y_nonempty_cusparse_parallel_delta_mismatches =
                total_mismatches;
            y_nonempty_cusparse_parallel_delta_max_abs_diff =
                max_abs_diff;
            y_nonempty_cusparse_parallel_delta_max_relative_diff =
                max_relative_diff;
        }

        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        y_nonempty_cusparse_graph_batch_ms = time_repeated([&]() {
            CUDA_CHECK(cudaGraphLaunch(
                candidate_graph_exec, workspace->stream));
        }) / HPRLP_NORMAL_GRAPH_BATCH_SIZE;
        if (use_delta) {
            restore_mutable_state();
            refresh_reduced_factor_scaled_y(workspace, state);
            refresh_reduced_factor_scaled_x_hat(workspace, state);
            CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
            y_nonempty_cusparse_parallel_delta_graph_batch_ms =
                time_repeated([&]() {
                    CUDA_CHECK(cudaGraphLaunch(
                        parallel_delta_graph_exec, workspace->stream));
                }) / HPRLP_NORMAL_GRAPH_BATCH_SIZE;
        }
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        if (parallel_delta_graph_exec != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(parallel_delta_graph_exec));
            CUDA_CHECK(cudaGraphDestroy(parallel_delta_graph));
        }
        hprlp_destroy_spmvop(&parallel_delta_operation);
        if (parallel_delta_ax_descr != nullptr) {
            check_cusparse(cusparseDestroyDnVec(parallel_delta_ax_descr),
                           "destroy parallel delta profile Ax");
            check_cusparse(cusparseDestroyDnVec(
                parallel_delta_input_descr),
                "destroy parallel delta profile input");
            check_cusparse(cusparseDestroySpMat(parallel_delta_matrix),
                           "destroy parallel delta profile A");
        }
        CUDA_CHECK(cudaFree(parallel_delta_ax));
        CUDA_CHECK(cudaFree(parallel_delta_input));
        CUDA_CHECK(cudaGraphExecDestroy(candidate_graph_exec));
        CUDA_CHECK(cudaGraphDestroy(candidate_graph));
        hprlp_destroy_spmvop(&compact_operation);
        check_cusparse(cusparseDestroyDnVec(compact_ax),
                       "destroy nonempty profile Ax");
        check_cusparse(cusparseDestroyDnVec(compact_x_hat),
                       "destroy nonempty profile xhat");
        check_cusparse(cusparseDestroySpMat(compact_matrix),
                       "destroy nonempty profile A");
        check_cusparse(cusparseDestroy(compact_handle),
                       "destroy nonempty profile handle");
        CUDA_CHECK(cudaFree(nonempty_ax));
        CUDA_CHECK(cudaFree(nonempty_row_ptr));
        CUDA_CHECK(cudaFree(original_to_nonempty));
        CUDA_CHECK(cudaFree(production_x));
        CUDA_CHECK(cudaFree(production_x_hat));
        CUDA_CHECK(cudaFree(production_y));
        CUDA_CHECK(cudaFree(production_scaled_y));
        CUDA_CHECK(cudaFree(candidate_delta_only_count_device));
        CUDA_CHECK(cudaFree(candidate_delta_only_rows));
        CUDA_CHECK(cudaFree(candidate_delta_nonempty_words));
        CUDA_CHECK(cudaFree(candidate_delta_fixed));
        CUDA_CHECK(cudaFree(candidate_delta_input));
    }

    if (!use_delta && state->use_dictionary_empty_row_batch) {
        int *const all_short_rows = state->A_rows_short;
        const int all_short_count = state->A_short_count;
        state->A_rows_short = state->signed_A_rows_short_nonempty;
        state->A_short_count = state->signed_A_short_nonempty_count;
        y_nonempty_only_ms = time_repeated([&]() {
            enqueue_reduced_y_updates(workspace, state, false);
        });
        state->A_rows_short = all_short_rows;
        state->A_short_count = all_short_count;
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        prepare_halpern_factor_batch_kernel<<<1, 1, 0, workspace->stream>>>(
            workspace->halpern_inner, workspace->halpern_factors,
            workspace->halpern_factor_batch, HPRLP_NORMAL_GRAPH_BATCH_SIZE);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
        y_empty_batch_ms_per_iteration = time_repeated([&]() {
            enqueue_reduced_dictionary_empty_y_batch(workspace, state);
        }) / HPRLP_NORMAL_GRAPH_BATCH_SIZE;
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    }

    if (use_delta && state->use_signed_factorized) {
        // Diagnostic-only comparisons use the same production enqueue
        // function. Mutable solver state is restored before execution
        // resumes, so neither comparison changes the numerical trajectory.
        y_base_only_ms = time_repeated([&]() {
            enqueue_reduced_y_updates(workspace, state, false);
        });
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        const bool production_uses_combined =
            state->signed_y_uses_combined;
        const bool production_uses_scalar =
            state->signed_y_uses_scalar;
        state->signed_y_uses_combined = !production_uses_combined;
        state->signed_y_uses_scalar = false;
        y_alternate_ms = time_repeated([&]() {
            enqueue_reduced_y_updates(workspace, state, true);
        });
        state->signed_y_uses_combined = production_uses_combined;
        state->signed_y_uses_scalar = production_uses_scalar;
        restore_mutable_state();
        refresh_reduced_factor_scaled_y(workspace, state);
        refresh_reduced_factor_scaled_x_hat(workspace, state);
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    }

    halpern_ms = time_repeated([&]() {
        advance_halpern_factors(workspace);
    });
    restore_mutable_state();
    refresh_reduced_factor_scaled_y(workspace, state);
    refresh_reduced_factor_scaled_x_hat(workspace, state);
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x_backup);
    cudaFree(x_hat_backup);
    cudaFree(delta_x_backup);
    cudaFree(delta_x_hat_backup);
    cudaFree(y_backup);
    cudaFree(halpern_inner_backup);
    cudaFree(halpern_factors_backup);

    std::cout << std::fixed << std::setprecision(6)
              << "  reduced profile: iteration=" << iteration
              << " repeats=" << repeats
              << " rows=" << workspace->m
              << " base_columns=" << state->base_count
              << " base_nnz=" << state->base_reduced_nnz
              << " delta_columns=" << state->delta_count
              << " delta_nnz=" << state->delta_nnz
              << " signed_y_scalar="
              << (state->signed_y_uses_scalar ? "yes" : "no")
              << " graph_ms=" << graph_ms
              << " graph_batch_ms_per_iteration=" << graph_batch_ms
              << " x_stage_ms=" << x_stage_ms
              << " y_stage_ms=" << y_stage_ms
              << " y_nonempty_only_ms=" << y_nonempty_only_ms
              << " y_empty_batch_ms_per_iteration="
              << y_empty_batch_ms_per_iteration
              << " y_base_only_ms=" << y_base_only_ms
              << " y_alternate_ms=" << y_alternate_ms
              << " y_cusparse_spmv_ms=" << y_cusparse_spmv_ms
              << " y_dense_add_shift_ms=" << y_dense_add_shift_ms
              << " y_dense_update_ms=" << y_dense_update_ms
              << " y_nonempty_cusparse_graph_batch_ms_per_iteration="
              << y_nonempty_cusparse_graph_batch_ms
              << " y_nonempty_cusparse_parallel_delta_graph_batch_ms_per_iteration="
              << y_nonempty_cusparse_parallel_delta_graph_batch_ms
              << " y_nonempty_cusparse_spmv_ms="
              << y_nonempty_cusparse_spmv_ms
              << " y_nonempty_cusparse_update_ms="
              << y_nonempty_cusparse_update_ms
              << " y_nonempty_cusparse_build_ms="
              << y_nonempty_cusparse_build_ms
              << " y_nonempty_cusparse_rows="
              << y_nonempty_cusparse_rows
              << " y_nonempty_cusparse_delta_only_rows="
              << y_nonempty_cusparse_delta_only_rows
              << " y_nonempty_cusparse_x_mismatches="
              << y_nonempty_cusparse_x_mismatches
              << " y_nonempty_cusparse_x_hat_mismatches="
              << y_nonempty_cusparse_x_hat_mismatches
              << " y_nonempty_cusparse_y_mismatches="
              << y_nonempty_cusparse_y_mismatches
              << " y_nonempty_cusparse_scaled_y_mismatches="
              << y_nonempty_cusparse_scaled_y_mismatches
              << " y_nonempty_cusparse_halpern_mismatches="
              << y_nonempty_cusparse_halpern_mismatches
              << " y_nonempty_cusparse_max_abs_diff="
              << y_nonempty_cusparse_max_abs_diff
              << " y_nonempty_cusparse_parallel_delta_mismatches="
              << y_nonempty_cusparse_parallel_delta_mismatches
              << " y_nonempty_cusparse_parallel_delta_max_abs_diff="
              << y_nonempty_cusparse_parallel_delta_max_abs_diff
              << " y_nonempty_cusparse_parallel_delta_max_relative_diff="
              << y_nonempty_cusparse_parallel_delta_max_relative_diff
              << " y_production_variant="
              << (state->signed_y_uses_combined ? "combined" : "bucketed")
              << " A_short_rows=" << state->A_short_count
              << " A_short_nonempty_rows="
              << state->signed_A_short_nonempty_count
              << " A_empty_rows=" << state->signed_A_empty_count
              << " A_delta_only_rows=" << state->signed_delta_only_count
              << " A_medium_rows=" << state->signed_A_medium_count
              << " A_long_rows=" << state->signed_A_long_count
              << " halpern_ms=" << halpern_ms
              << std::defaultfloat << std::endl;
    if (y_nonempty_cusparse_max_abs_diff >= 0.0) {
        std::cout << std::scientific << std::setprecision(17)
                  << "  reduced nonempty cusparse comparison: max_abs_diff="
                  << y_nonempty_cusparse_max_abs_diff
                  << " max_relative_diff="
                  << y_nonempty_cusparse_max_relative_diff
                  << std::defaultfloat << std::endl;
    }
    if (y_nonempty_cusparse_parallel_delta_max_abs_diff >= 0.0) {
        std::cout << std::scientific << std::setprecision(17)
                  << "  reduced parallel delta comparison: max_abs_diff="
                  << y_nonempty_cusparse_parallel_delta_max_abs_diff
                  << " max_relative_diff="
                  << y_nonempty_cusparse_parallel_delta_max_relative_diff
                  << std::defaultfloat << std::endl;
    }
}

}  // namespace

void hprlp_initialize_reduced_matrix_state(HPRLP_workspace_gpu *workspace) {
    if (workspace == nullptr || workspace->reduced_matrix != nullptr) return;
    auto *state = new HPRLP_reduced_matrix_state;
    state->profile_enabled = reduced_profile_enabled();
    state->free_count = workspace->n;
    state->last_free_columns = workspace->n;
    state->free_count_warp_count = (workspace->n + 31) / 32;
    allocate_device(&state->x_bar_mask, workspace->n);
    allocate_device(&state->changed_device, 1);
    allocate_device(&state->changed_count_device, 1);
    allocate_device(&state->free_count_device, 1);
    allocate_device(
        &state->free_count_by_warp_device, state->free_count_warp_count);
    allocate_device(&state->delta_count_device, 1);
    allocate_device(&state->delta_indices_device, workspace->n);
    allocate_device(&state->delta_old_mask_device, workspace->n);
    allocate_device(&state->delta_row_bucket_counts_device, 4);
    allocate_device(
        &state->base_row_analysis_device,
        2 * kReducedRowAnalysisFields);
    check_cusparse(
        cusparseCreate(&state->transpose_handle),
        "cusparseCreate reduced transpose");
    check_cusparse(
        cusparseSetStream(state->transpose_handle, workspace->stream),
        "cusparseSetStream reduced transpose");
    if (workspace->n > 0) {
        CUDA_CHECK(cub::DeviceReduce::Sum(
            nullptr, state->free_count_reduce_temp_bytes,
            state->free_count_by_warp_device, state->free_count_device,
            state->free_count_warp_count, workspace->stream));
        if (state->free_count_reduce_temp_bytes > 0) {
            CUDA_CHECK(cudaMalloc(
                &state->free_count_reduce_temp,
                state->free_count_reduce_temp_bytes));
        }
        CUDA_CHECK(cudaMemsetAsync(
            state->x_bar_mask, HPRLP_XBAR_INTERIOR,
            static_cast<std::size_t>(workspace->n) * sizeof(std::uint8_t),
            workspace->stream));
    }
    if (reduced_rows_enabled()) {
        state->row_active_count = workspace->m;
        state->row_active_count_warp_count = (workspace->m + 31) / 32;
        allocate_device(&state->y_bar_mask, workspace->m);
        allocate_device(&state->row_base_mask, workspace->m);
        allocate_device(&state->row_changed_device, 1);
        allocate_device(&state->row_active_count_device, 1);
        allocate_device(
            &state->row_active_count_by_warp_device,
            state->row_active_count_warp_count);
        if (workspace->m > 0) {
            CUDA_CHECK(cudaMemsetAsync(
                state->y_bar_mask, 1,
                static_cast<std::size_t>(workspace->m) *
                    sizeof(std::uint8_t), workspace->stream));
            CUDA_CHECK(cudaMemsetAsync(
                state->row_base_mask, 0,
                static_cast<std::size_t>(workspace->m) *
                    sizeof(std::uint8_t), workspace->stream));
            CUDA_CHECK(cub::DeviceReduce::Sum(
                nullptr, state->row_count_reduce_temp_bytes,
                state->row_active_count_by_warp_device,
                state->row_active_count_device,
                state->row_active_count_warp_count, workspace->stream));
            if (state->row_count_reduce_temp_bytes > 0) {
                CUDA_CHECK(cudaMalloc(
                    &state->row_count_reduce_temp,
                    state->row_count_reduce_temp_bytes));
            }
        }
    }
    workspace->reduced_matrix = state;
}

void hprlp_free_reduced_matrix_state(HPRLP_workspace_gpu *workspace) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr) return;
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    destroy_reduced_row_workspace(state);
    destroy_reduced_workspace(state);
    cudaFree(state->x_bar_mask);
    cudaFree(state->changed_device);
    cudaFree(state->changed_count_device);
    cudaFree(state->free_count_device);
    cudaFree(state->free_count_by_warp_device);
    cudaFree(state->free_count_reduce_temp);
    cudaFree(state->delta_count_device);
    cudaFree(state->delta_indices_device);
    cudaFree(state->delta_old_mask_device);
    cudaFree(state->delta_row_bucket_counts_device);
    cudaFree(state->base_row_analysis_device);
    cudaFree(state->y_bar_mask);
    cudaFree(state->row_base_mask);
    cudaFree(state->row_changed_device);
    cudaFree(state->row_active_count_device);
    cudaFree(state->row_active_count_by_warp_device);
    cudaFree(state->row_count_reduce_temp);
    cudaFree(state->construction_temp);
    if (state->transpose_handle) {
        cusparseDestroy(state->transpose_handle);
    }
    delete state;
    workspace->reduced_matrix = nullptr;
}

bool hprlp_refresh_reduced_matrix_mask(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_parameters *parameters,
    const HPRLP_residuals *residuals,
    bool force_refresh) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr ||
        parameters == nullptr || residuals == nullptr) {
        return false;
    }
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    const bool maintenance_active =
        parameters->use_reduced_matrix &&
        residuals->KKTx_and_gap_org_bar < reduced_enter_residual();
    if (!maintenance_active) {
        const bool changed = state->mask_started || state->row_mask_started;
        if (changed && workspace->n > 0) {
            CUDA_CHECK(cudaMemsetAsync(
                state->x_bar_mask, HPRLP_XBAR_INTERIOR,
                static_cast<std::size_t>(workspace->n) *
                    sizeof(std::uint8_t),
                workspace->stream));
        }
        state->mask_started = false;
        state->mask_changed = changed;
        state->mask_count_stale = false;
        state->free_count = workspace->n;
        state->recorded_changed_count = 0;
        state->recorded_delta_count = 0;
        if (state->row_mask_started && workspace->m > 0) {
            CUDA_CHECK(cudaMemsetAsync(
                state->y_bar_mask, 1,
                static_cast<std::size_t>(workspace->m) *
                    sizeof(std::uint8_t), workspace->stream));
        }
        state->row_mask_started = false;
        state->row_mask_changed = changed;
        state->row_active_count = workspace->m;
        state->row_stable_count = 0;
        return changed;
    }

    // force_refresh is used only at termination.  Avoid a duplicate recount
    // if the regular refresh on the same checkpoint already produced one.
    if (force_refresh && !state->mask_count_stale) return false;

    const bool row_mask_changed =
        !force_refresh && refresh_reduced_row_mask(workspace, state);

    // During one maintenance epoch update_x_bar_mask_kernel only releases
    // fixed columns; it never fixes an interior column.  Therefore the free
    // ratio cannot fall back below the entry threshold.  Once it blocks
    // reduced mode, keep releasing the remaining fixed mask entries but omit
    // the reduction, host copies, and stream synchronization.  A forced full
    // recount at termination keeps the monotone mask statistics exact.
    const bool entry_ratio_cannot_recover =
        !recompute_reduced_prebuild_mask() &&
        !force_refresh && state->mask_started && !state->built &&
        workspace->n > 0 &&
        static_cast<HPRLP_FLOAT>(state->free_count) / workspace->n >=
            HPRLP_REDUCED_ENTER_RATIO;
    if (entry_ratio_cannot_recover) {
        release_x_bar_mask_kernel<<<
            numBlocks(workspace->n), numThreads, 0, workspace->stream>>>(
            state->x_bar_mask, workspace->x_bar,
            workspace->l, workspace->u, workspace->n);
        state->mask_changed = false;
        state->mask_count_stale = true;
        state->recorded_changed_count = 0;
        state->recorded_delta_count = 0;
        return row_mask_changed;
    }

    state->mask_count_stale = false;

    CUDA_CHECK(cudaMemsetAsync(
        state->changed_device, 0, sizeof(int), workspace->stream));
    CUDA_CHECK(cudaMemsetAsync(
        state->changed_count_device, 0, sizeof(int), workspace->stream));
    CUDA_CHECK(cudaMemsetAsync(
        state->delta_count_device, 0, sizeof(int), workspace->stream));
    state->recorded_changed_count = 0;
    state->recorded_delta_count = 0;
    if (workspace->n > 0) {
        if (state->mask_started &&
            !(recompute_reduced_prebuild_mask() && !state->built)) {
            update_x_bar_mask_kernel<<<
                numBlocks(workspace->n), numThreads, 0,
                workspace->stream>>>(
                state->x_bar_mask, workspace->x_bar,
                workspace->l, workspace->u, state->changed_device,
                state->changed_count_device,
                state->free_count_by_warp_device,
                state->delta_count_device, state->delta_indices_device,
                state->delta_old_mask_device, state->built, workspace->n);
        } else {
            initialize_x_bar_mask_kernel<<<
                numBlocks(workspace->n), numThreads, 0,
                workspace->stream>>>(
                state->x_bar_mask, workspace->x_bar,
                workspace->l, workspace->u, state->changed_device,
                state->free_count_by_warp_device, workspace->n);
            state->mask_started = true;
        }
        CUDA_CHECK(cub::DeviceReduce::Sum(
            state->free_count_reduce_temp,
            state->free_count_reduce_temp_bytes,
            state->free_count_by_warp_device, state->free_count_device,
            state->free_count_warp_count, workspace->stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(
            state->free_count_device, 0, sizeof(int), workspace->stream));
    }
    int changed = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &changed, state->changed_device, sizeof(int),
        cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &state->free_count, state->free_count_device, sizeof(int),
        cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &state->recorded_changed_count, state->changed_count_device,
        sizeof(int), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &state->recorded_delta_count, state->delta_count_device,
        sizeof(int), cudaMemcpyDeviceToHost, workspace->stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    if (state->recorded_delta_count > 1) {
        // update_x_bar_mask_kernel discovers released columns in parallel.
        // The atomic append order is scheduler-dependent and becomes the CSR
        // order of the delta operator, changing floating-point accumulation.
        // Canonicalize each released batch by original column index while
        // carrying its previous-bound tag with it.
        thrust::sort_by_key(
            thrust::cuda::par.on(workspace->stream),
            thrust::device_pointer_cast(state->delta_indices_device),
            thrust::device_pointer_cast(
                state->delta_indices_device + state->recorded_delta_count),
            thrust::device_pointer_cast(state->delta_old_mask_device));
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    }
    state->mask_changed = changed != 0;
    return state->mask_changed || row_mask_changed;
}

void hprlp_maybe_reset_reduced_matrix_mask_on_restart(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_parameters *parameters,
    const HPRLP_residuals *residuals,
    int iteration,
    int restart_flag) {
    if (!reduced_restart_mask_reset_enabled() || restart_flag <= 0 ||
        workspace == nullptr || workspace->reduced_matrix == nullptr ||
        parameters == nullptr || residuals == nullptr ||
        !parameters->use_reduced_matrix ||
        residuals->KKTx_and_gap_org_bar >= reduced_enter_residual()) {
        return;
    }
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    const bool check_columns = workspace->n > 0 &&
        state->built && state->mask_started;
    const bool check_rows = reduced_rows_enabled() && workspace->m > 0 &&
        state->row_built && state->row_mask_started;
    if (!check_columns && !check_rows) return;

    if (check_columns) {
        count_current_interior_by_warp_kernel<<<
            numBlocks(workspace->n), numThreads, 0, workspace->stream>>>(
            workspace->x_bar, workspace->l, workspace->u,
            state->free_count_by_warp_device, workspace->n);
        CUDA_CHECK(cub::DeviceReduce::Sum(
            state->free_count_reduce_temp,
            state->free_count_reduce_temp_bytes,
            state->free_count_by_warp_device, state->free_count_device,
            state->free_count_warp_count, workspace->stream));
        int fresh_free_count = workspace->n;
        CUDA_CHECK(cudaMemcpyAsync(
            &fresh_free_count, state->free_count_device, sizeof(int),
            cudaMemcpyDeviceToHost, workspace->stream));
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        state->restart_mask_checks += 1;
        const int current_free_count = state->free_count;
        const HPRLP_FLOAT recovery = current_free_count > 0
            ? 1.0 - static_cast<HPRLP_FLOAT>(fresh_free_count) /
                        current_free_count
            : 0.0;
        const long long saved_columns =
            static_cast<long long>(current_free_count) - fresh_free_count;
        const bool reset = fresh_free_count < current_free_count &&
            current_free_count >=
                reduced_restart_mask_min_current_columns() &&
            recovery >= reduced_restart_mask_min_recovery() &&
            saved_columns >= reduced_restart_mask_min_saved_columns();
        std::cout << "Reduced restart mask check: iteration=" << iteration
                  << " restart_flag=" << restart_flag
                  << " current_free=" << current_free_count
                  << " fresh_free=" << fresh_free_count
                  << " saved_columns=" << saved_columns
                  << " recovery=" << std::fixed << std::setprecision(5)
                  << recovery
                  << " min_recovery="
                  << reduced_restart_mask_min_recovery()
                  << " min_saved_columns="
                  << reduced_restart_mask_min_saved_columns()
                  << " min_current_columns="
                  << reduced_restart_mask_min_current_columns()
                  << " reset=" << (reset ? "yes" : "no")
                  << std::defaultfloat << std::setprecision(2)
                  << std::endl;
        if (reset) {
            hprlp_flush_reduced_matrix_state(workspace);
            destroy_reduced_workspace(state);
            CUDA_CHECK(cudaMemsetAsync(
                state->x_bar_mask, HPRLP_XBAR_INTERIOR,
                static_cast<std::size_t>(workspace->n) *
                    sizeof(std::uint8_t),
                workspace->stream));
            state->mask_started = false;
            state->mask_changed = true;
            state->mask_count_stale = false;
            state->free_count = workspace->n;
            state->stable_count = 0;
            state->recorded_changed_count = 0;
            state->recorded_delta_count = 0;
            state->last_rebuild_iteration = -1;
            state->restart_mask_resets += 1;
        }
    }

    if (check_rows) {
        count_current_active_rows_by_warp_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            workspace->y_bar, workspace->AL, workspace->AU,
            workspace->y_bound_type,
            state->row_active_count_by_warp_device, workspace->m);
        CUDA_CHECK(cub::DeviceReduce::Sum(
            state->row_count_reduce_temp,
            state->row_count_reduce_temp_bytes,
            state->row_active_count_by_warp_device,
            state->row_active_count_device,
            state->row_active_count_warp_count, workspace->stream));
        int fresh_active_count = workspace->m;
        CUDA_CHECK(cudaMemcpyAsync(
            &fresh_active_count, state->row_active_count_device,
            sizeof(int), cudaMemcpyDeviceToHost, workspace->stream));
        CUDA_CHECK(cudaStreamSynchronize(workspace->stream));

        state->row_restart_mask_checks += 1;
        const int current_active_count = state->row_active_count;
        const HPRLP_FLOAT recovery = current_active_count > 0
            ? 1.0 - static_cast<HPRLP_FLOAT>(fresh_active_count) /
                        current_active_count
            : 0.0;
        const long long saved_rows =
            static_cast<long long>(current_active_count) -
            fresh_active_count;
        const bool reset = fresh_active_count < current_active_count &&
            current_active_count >=
                reduced_restart_mask_min_current_columns() &&
            recovery >= reduced_restart_mask_min_recovery() &&
            saved_rows >= reduced_restart_mask_min_saved_columns();
        std::cout << "Reduced row restart mask check: iteration="
                  << iteration << " restart_flag=" << restart_flag
                  << " current_active=" << current_active_count
                  << " fresh_active=" << fresh_active_count
                  << " saved_rows=" << saved_rows
                  << " recovery=" << std::fixed << std::setprecision(5)
                  << recovery
                  << " min_recovery="
                  << reduced_restart_mask_min_recovery()
                  << " min_saved_rows="
                  << reduced_restart_mask_min_saved_columns()
                  << " min_current_rows="
                  << reduced_restart_mask_min_current_columns()
                  << " reset=" << (reset ? "yes" : "no")
                  << std::defaultfloat << std::setprecision(2)
                  << std::endl;
        if (reset) {
            hprlp_flush_reduced_matrix_state(workspace);
            destroy_reduced_row_workspace(state);
            CUDA_CHECK(cudaMemsetAsync(
                state->y_bar_mask, 1,
                static_cast<std::size_t>(workspace->m) *
                    sizeof(std::uint8_t),
                workspace->stream));
            CUDA_CHECK(cudaMemsetAsync(
                state->row_base_mask, 0,
                static_cast<std::size_t>(workspace->m) *
                    sizeof(std::uint8_t),
                workspace->stream));
            state->row_mask_started = false;
            state->row_mask_changed = true;
            state->row_active_count = workspace->m;
            state->row_stable_count = 0;
            state->row_last_rebuild_iteration = -1;
            state->row_restart_mask_resets += 1;
        }
    }
}

void materialize_reduced_deferred_empty_rows(
    HPRLP_workspace_gpu *workspace,
    HPRLP_reduced_matrix_state *state) {
    if (workspace == nullptr || state == nullptr ||
        state->deferred_empty_pending_iterations <= 0) {
        return;
    }
    const int deferred_iterations =
        state->deferred_empty_pending_iterations;
    const HPRLP_reduced_profile_token profile =
        begin_reduced_stage_profile(workspace, state);
    enqueue_reduced_deferred_empty_rows(
        workspace, state, deferred_iterations);
    finish_reduced_stage_profile(
        workspace, profile, &state->profile_deferred_empty_flush);
    state->deferred_empty_pending_iterations = 0;
    state->deferred_empty_total_iterations += deferred_iterations;
    state->deferred_empty_flushes += 1;
}

void hprlp_flush_reduced_matrix_state(HPRLP_workspace_gpu *workspace) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr) return;
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    if (state->mode == HPRLP_reduced_mode::Rows) {
        if (!state->use_compact_row_y || !state->full_dirty) return;
    } else {
        if (!state->full_dirty &&
            state->deferred_empty_pending_iterations <= 0) return;
        materialize_reduced_deferred_empty_rows(workspace, state);
    }
    const HPRLP_reduced_profile_token profile =
        begin_reduced_stage_profile(workspace, state);
    if (state->mode == HPRLP_reduced_mode::Rows) {
        if (state->row_base_count > 0) {
            scatter_reduced_row_state_kernel<<<
                numBlocks(state->row_base_count), numThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->last_y,
                state->row_y, state->row_last_y,
                state->row_active_to_original, state->row_base_count);
        }
        if (state->row_delta_count > 0) {
            scatter_reduced_row_state_kernel<<<
                numBlocks(state->row_delta_count), numThreads, 0,
                workspace->stream>>>(
                workspace->y, workspace->last_y,
                state->row_delta_y, state->row_delta_last_y,
                state->row_delta_to_original, state->row_delta_count);
        }
        zero_inactive_y_state_kernel<<<
            numBlocks(workspace->m), numThreads, 0, workspace->stream>>>(
            workspace->y, workspace->last_y,
            state->y_bar_mask, workspace->m);
        state->full_dirty = false;
        finish_reduced_stage_profile(
            workspace, profile, &state->profile_flush_to_full);
        return;
    }
    if (state->base_count > 0) scatter_reduced_state_kernel<<<
        numBlocks(state->base_count), numThreads, 0, workspace->stream>>>(
        workspace->x, workspace->x_bar, workspace->x_hat,
        state->x, state->x_bar, state->x_hat,
        state->free_to_original, state->base_count);
    const int delta_count = state->delta_count;
    if (delta_count > 0) scatter_reduced_state_kernel<<<
        numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
        workspace->x, workspace->x_bar, workspace->x_hat,
        state->delta_x, state->delta_x_bar, state->delta_x_hat,
        state->delta_free_to_original, delta_count);
    state->full_dirty = false;
    finish_reduced_stage_profile(
        workspace, profile, &state->profile_flush_to_full);
}

void hprlp_sync_reduced_matrix_state_from_full(
    HPRLP_workspace_gpu *workspace) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr) return;
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    if (!state->active) return;
    if (state->mode == HPRLP_reduced_mode::Rows &&
        !state->use_compact_row_y) return;
    const HPRLP_reduced_profile_token profile =
        begin_reduced_stage_profile(workspace, state);
    if (state->mode == HPRLP_reduced_mode::Rows) {
        if (state->row_base_count > 0) {
            gather_reduced_row_state_kernel<<<
                numBlocks(state->row_base_count), numThreads, 0,
                workspace->stream>>>(
                state->row_y, state->row_last_y,
                state->row_lower, state->row_upper,
                workspace->y, workspace->last_y,
                workspace->AL, workspace->AU,
                state->row_active_to_original, state->row_base_count);
            if (state->row_use_signed_x ||
                state->row_use_packed_dictionary_x) {
                gather_reduced_row_backend_state_kernel<<<
                    numBlocks(state->row_base_count), numThreads, 0,
                    workspace->stream>>>(
                    nullptr, state->row_scaled_y, nullptr,
                    workspace->inverse_row_norm, workspace->y,
                    workspace->y_bound_type,
                    state->row_active_to_original,
                    state->row_base_count);
            }
        }
        if (state->row_delta_count > 0) {
            gather_reduced_row_state_kernel<<<
                numBlocks(state->row_delta_count), numThreads, 0,
                workspace->stream>>>(
                state->row_delta_y, state->row_delta_last_y,
                state->row_delta_lower, state->row_delta_upper,
                workspace->y, workspace->last_y,
                workspace->AL, workspace->AU,
                state->row_delta_to_original, state->row_delta_count);
        }
        state->full_dirty = false;
        finish_reduced_stage_profile(
            workspace, profile, &state->profile_gather_from_full);
        return;
    }
    if (state->base_count > 0) gather_reduced_state_kernel<<<
        numBlocks(state->base_count), numThreads, 0, workspace->stream>>>(
        state->x, state->x_bar, state->x_hat, state->last_x,
        state->lower, state->upper, state->objective, state->bound_type,
        workspace->x, workspace->x_bar, workspace->x_hat,
        workspace->last_x, workspace->l, workspace->u, workspace->c,
        workspace->x_bound_type, state->free_to_original,
        state->base_count);
    const int delta_count = state->delta_count;
    if (delta_count > 0) gather_reduced_state_kernel<<<
        numBlocks(delta_count), numThreads, 0, workspace->stream>>>(
        state->delta_x, state->delta_x_bar, state->delta_x_hat,
        state->delta_last_x, state->delta_lower, state->delta_upper,
        state->delta_objective, state->delta_bound_type,
        workspace->x, workspace->x_bar, workspace->x_hat,
        workspace->last_x, workspace->l, workspace->u, workspace->c,
        workspace->x_bound_type, state->delta_free_to_original,
        delta_count);
    refresh_reduced_factor_scaled_y(workspace, state);
    finish_reduced_stage_profile(
        workspace, profile, &state->profile_gather_from_full);
}

void hprlp_update_reduced_matrix_mode(
    HPRLP_workspace_gpu *workspace,
    LP_info_gpu *lp,
    Scaling_info *scaling,
    const HPRLP_parameters *parameters,
    const HPRLP_residuals *residuals,
    int iteration,
    bool residuals_refreshed,
    bool next_iteration_checks) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr ||
        parameters == nullptr || residuals == nullptr ||
        !residuals_refreshed) {
        return;
    }
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    state->activation_checks += 1;
    if (state->mask_changed) {
        state->stable_count = 0;
    } else {
        state->stable_count += 1;
    }
    if (state->row_mask_changed) {
        state->row_stable_count = 0;
    } else if (state->row_mask_started) {
        state->row_stable_count += 1;
    }
    const bool mask_enabled =
        parameters->use_reduced_matrix && state->mask_started &&
        residuals->KKTx_and_gap_org_bar < reduced_enter_residual();
    const HPRLP_FLOAT free_ratio = workspace->n > 0
        ? static_cast<HPRLP_FLOAT>(state->free_count) / workspace->n
        : 0.0;
    const HPRLP_FLOAT enter_ratio = state->built
        ? HPRLP_REDUCED_ENTER_RATIO : reduced_prebuild_enter_ratio();
    const char *full_alignment_env =
        std::getenv("HPRLP_REDUCED_FULL_ALIGNMENT");
    const bool force_full_alignment = full_alignment_env != nullptr &&
        std::string(full_alignment_env) != "0";
    const bool column_ready =
        !force_full_alignment &&
        mask_enabled && free_ratio < enter_ratio &&
        (state->stable_count >= HPRLP_REDUCED_ENTER_STABLE ||
         free_ratio < HPRLP_REDUCED_ENTER_FAST_RATIO);

    const HPRLP_FLOAT row_ratio = workspace->m > 0
        ? static_cast<HPRLP_FLOAT>(state->row_active_count) / workspace->m
        : 1.0;
    const bool row_ready =
        reduced_rows_enabled() && !force_full_alignment && mask_enabled &&
        state->row_mask_started && state->row_active_count > 0 &&
        row_ratio < reduced_row_enter_ratio() &&
        (state->row_stable_count >= HPRLP_REDUCED_ENTER_STABLE ||
         row_ratio < HPRLP_REDUCED_ENTER_FAST_RATIO);
    const bool select_rows = row_ready &&
        (!column_ready || row_ratio < free_ratio);

    if (select_rows) {
        if (!workspace->reduced_backend_autotune_done) {
            const HPRLP_reduced_profile_token autotune_profile =
                begin_reduced_stage_profile(workspace, state);
            autotune_reduced_update_backends(
                workspace, lp, scaling, parameters);
            finish_reduced_stage_profile(
                workspace, autotune_profile,
                &state->profile_backend_autotune);
        }
        if (state->built) {
            hprlp_flush_reduced_matrix_state(workspace);
            const HPRLP_reduced_profile_token destroy_profile =
                begin_reduced_stage_profile(workspace, state);
            destroy_reduced_workspace(state);
            finish_reduced_stage_profile(
                workspace, destroy_profile,
                &state->profile_workspace_destroy);
        }
        const long long row_rebuild_work =
            2LL * state->row_A.numElements + workspace->n +
            state->row_base_count;
        const bool adaptive_row_rebase_due =
            reduced_adaptive_row_rebase_enabled() && state->row_built &&
            state->row_delta_count > 0 &&
            state->row_delta_work_accum >= row_rebuild_work;
        const bool specialized_row_rebase_due =
            state->row_built && state->row_mask_changed &&
            (state->row_use_signed_x ||
             state->row_use_packed_dictionary_x);
        if (specialized_row_rebase_due) {
            // The full-path packed X launchers consume one compact AT.  Until
            // they accept a second row-delta operator, rebuild the compact
            // base at the checkpoint so no newly active row can be omitted.
            hprlp_flush_reduced_matrix_state(workspace);
            const HPRLP_reduced_profile_token destroy_profile =
                begin_reduced_stage_profile(workspace, state);
            destroy_reduced_row_workspace(state);
            finish_reduced_stage_profile(
                workspace, destroy_profile,
                &state->profile_workspace_destroy);
            state->row_last_rebuild_iteration = -1;
        } else if (adaptive_row_rebase_due) {
            hprlp_flush_reduced_matrix_state(workspace);
            const HPRLP_reduced_profile_token destroy_profile =
                begin_reduced_stage_profile(workspace, state);
            destroy_reduced_row_workspace(state);
            finish_reduced_stage_profile(
                workspace, destroy_profile,
                &state->profile_workspace_destroy);
        } else if (state->row_built && state->row_mask_changed) {
            hprlp_flush_reduced_matrix_state(workspace);
            const HPRLP_reduced_profile_token delta_profile =
                begin_reduced_stage_profile(workspace, state);
            const bool extended = rebuild_reduced_row_delta_workspace(
                workspace, state);
            finish_reduced_stage_profile(
                workspace, delta_profile, &state->profile_delta_extend);
            if (!extended) {
                hprlp_flush_reduced_matrix_state(workspace);
                const HPRLP_reduced_profile_token destroy_profile =
                    begin_reduced_stage_profile(workspace, state);
                destroy_reduced_row_workspace(state);
                finish_reduced_stage_profile(
                    workspace, destroy_profile,
                    &state->profile_workspace_destroy);
            }
        }
        if (!state->row_built && !next_iteration_checks) {
            const int interval =
                row_ratio < HPRLP_REDUCED_REBUILD_FAST_RATIO
                ? HPRLP_REDUCED_REBUILD_FAST_INTERVAL
                : HPRLP_REDUCED_REBUILD_INTERVAL;
            const bool interval_ready =
                state->row_last_rebuild_iteration < 0 ||
                iteration - state->row_last_rebuild_iteration >= interval;
            if (!interval_ready && !specialized_row_rebase_due) {
                state->active = false;
                state->mode = HPRLP_reduced_mode::None;
                return;
            }
            const HPRLP_reduced_profile_token build_profile =
                begin_reduced_stage_profile(workspace, state);
            build_reduced_row_workspace(
                workspace, state, parameters, iteration);
            finish_reduced_stage_profile(
                workspace, build_profile, &state->profile_full_build);
        }
        state->active = state->row_built;
        state->mode = state->row_built
            ? HPRLP_reduced_mode::Rows : HPRLP_reduced_mode::None;
        if (state->row_built) {
            if (state->row_first_iteration < 0) {
                state->row_first_iteration = iteration;
            }
            state->row_last_active_ratio = row_ratio;
        }
        return;
    }

    if (state->row_built) {
        hprlp_flush_reduced_matrix_state(workspace);
        const HPRLP_reduced_profile_token destroy_profile =
            begin_reduced_stage_profile(workspace, state);
        destroy_reduced_row_workspace(state);
        finish_reduced_stage_profile(
            workspace, destroy_profile, &state->profile_workspace_destroy);
    }

    if (!column_ready) {
        // Before reduced mode is built there is no device workspace to flush
        // or release.  Avoid issuing dozens of cudaFree(nullptr) calls at
        // every residual checkpoint for problems that never enter it.
        if (state->built) {
            hprlp_flush_reduced_matrix_state(workspace);
            const HPRLP_reduced_profile_token destroy_profile =
                begin_reduced_stage_profile(workspace, state);
            destroy_reduced_workspace(state);
            finish_reduced_stage_profile(
                workspace, destroy_profile,
                &state->profile_workspace_destroy);
        }
        state->mode = HPRLP_reduced_mode::None;
        state->active = false;
        return;
    }
    if (state->first_iteration < 0) state->first_iteration = iteration;
    state->last_trigger_iteration = iteration;
    state->last_free_ratio = free_ratio;
    state->last_trigger_residual = residuals->KKTx_and_gap_org_bar;
    state->last_trigger_sigma = workspace->sigma;

    // Match Julia's adaptive-rebase lifecycle: accumulated delta work is
    // assessed only at a fresh residual checkpoint, before applying the
    // latest mask delta. Rebuilding inside an arbitrary reduced iteration
    // changes the sparse accumulation path between checkpoints and made MCF
    // trajectories diverge sharply after reduced-mode entry.
    const long long delta_count =
        static_cast<long long>(state->delta_count);
    const long long rebuild_work =
        2 * state->base_reduced_nnz + workspace->m + state->base_count;
    const bool adaptive_rebase_due =
        state->built && delta_count > 0 &&
        state->delta_work_accum >= rebuild_work;
    if (adaptive_rebase_due) {
        hprlp_flush_reduced_matrix_state(workspace);
        const HPRLP_reduced_profile_token destroy_profile =
            begin_reduced_stage_profile(workspace, state);
        destroy_reduced_workspace(state);
        finish_reduced_stage_profile(
            workspace, destroy_profile,
            &state->profile_workspace_destroy);
    }

    const bool changed_existing_workspace =
        state->mask_changed && state->built;
    bool extension_failed = false;
    if (changed_existing_workspace) {
        hprlp_flush_reduced_matrix_state(workspace);
        const HPRLP_reduced_profile_token delta_profile =
            begin_reduced_stage_profile(workspace, state);
        const bool extended = extend_reduced_delta_workspace(workspace, state);
        finish_reduced_stage_profile(
            workspace, delta_profile, &state->profile_delta_extend);
        if (!extended) {
            const HPRLP_reduced_profile_token destroy_profile =
                begin_reduced_stage_profile(workspace, state);
            destroy_reduced_workspace(state);
            finish_reduced_stage_profile(
                workspace, destroy_profile,
                &state->profile_workspace_destroy);
            extension_failed = true;
        }
    }
    if (!state->built && !next_iteration_checks) {
        const int interval =
            free_ratio < HPRLP_REDUCED_REBUILD_FAST_RATIO
            ? HPRLP_REDUCED_REBUILD_FAST_INTERVAL
            : HPRLP_REDUCED_REBUILD_INTERVAL;
        const bool interval_ready =
            state->last_rebuild_iteration < 0 ||
            iteration - state->last_rebuild_iteration >= interval;
        if (interval_ready || extension_failed || adaptive_rebase_due) {
            const HPRLP_reduced_profile_token build_profile =
                begin_reduced_stage_profile(workspace, state);
            build_reduced_workspace(workspace, state, parameters, iteration);
            finish_reduced_stage_profile(
                workspace, build_profile, &state->profile_full_build);
        }
    }
    state->active = state->built;
    state->mode = state->built
        ? HPRLP_reduced_mode::Columns : HPRLP_reduced_mode::None;
}

int hprlp_launch_reduced_matrix_iterations(
    HPRLP_workspace_gpu *workspace,
    int iteration,
    int restart_flag,
    int requested_iterations) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr) return 1;
    HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    if (!state->active) return 1;
    if (state->mode == HPRLP_reduced_mode::Rows) {
        if (state->row_graph_exec == nullptr || restart_flag > 0) {
            const HPRLP_reduced_profile_token graph_profile =
                begin_reduced_stage_profile(workspace, state);
            capture_reduced_row_graphs(workspace, state);
            finish_reduced_stage_profile(
                workspace, graph_profile, &state->profile_graph_capture);
        }
        const bool launch_batch =
            requested_iterations == HPRLP_NORMAL_GRAPH_BATCH_SIZE &&
            state->row_graph_exec_batch != nullptr;
        const int launched_iterations = launch_batch
            ? HPRLP_NORMAL_GRAPH_BATCH_SIZE : 1;
        CUDA_CHECK(cudaGraphLaunch(
            launch_batch ? state->row_graph_exec_batch
                         : state->row_graph_exec,
            workspace->stream));
        state->full_dirty = state->use_compact_row_y;
        state->row_active_iterations += launched_iterations;
        state->row_active_iteration_sum +=
            static_cast<long double>(state->row_active_count) *
            launched_iterations;
        state->row_nnz_iteration_sum +=
            static_cast<long double>(
                state->row_A.numElements +
                state->row_delta_A.numElements) *
            launched_iterations;
        state->row_minimum_active = std::min(
            state->row_minimum_active,
            static_cast<long long>(state->row_active_count));
        state->row_minimum_nnz = std::min(
            state->row_minimum_nnz,
            static_cast<long long>(state->row_A.numElements) +
                state->row_delta_A.numElements);
        if (state->row_delta_count > 0) {
            const long long per_iteration =
                2LL * state->row_delta_A.numElements +
                state->row_delta_count;
            const long long limit = std::numeric_limits<long long>::max();
            const long long batch_work = per_iteration >
                    limit / launched_iterations
                ? limit : per_iteration * launched_iterations;
            state->row_delta_work_accum =
                state->row_delta_work_accum > limit - batch_work
                ? limit : state->row_delta_work_accum + batch_work;
        }
        return launched_iterations;
    }
    if (state->graph_exec == nullptr || restart_flag > 0) {
        const HPRLP_reduced_profile_token graph_profile =
            begin_reduced_stage_profile(workspace, state);
        capture_reduced_graphs(workspace, state);
        finish_reduced_stage_profile(
            workspace, graph_profile, &state->profile_graph_capture);
    }
    profile_reduced_graph(workspace, state, iteration);
    const bool launch_batch =
        requested_iterations == HPRLP_NORMAL_GRAPH_BATCH_SIZE &&
        state->graph_exec_batch != nullptr;
    const int launched_iterations = launch_batch
        ? HPRLP_NORMAL_GRAPH_BATCH_SIZE : 1;
    CUDA_CHECK(cudaGraphLaunch(
        launch_batch ? state->graph_exec_batch : state->graph_exec,
        workspace->stream));
    if (state->defer_empty_rows_to_observation) {
        state->deferred_empty_pending_iterations += launched_iterations;
    }
    state->active_iterations += launched_iterations;
    const long long active_columns =
        static_cast<long long>(state->base_count) + state->delta_count;
    const long long active_nnz =
        state->base_reduced_nnz + state->delta_nnz;
    state->active_column_iteration_sum +=
        static_cast<long double>(active_columns) * launched_iterations;
    state->active_nnz_iteration_sum +=
        static_cast<long double>(active_nnz) * launched_iterations;
    state->minimum_active_columns =
        std::min(state->minimum_active_columns, active_columns);
    state->minimum_active_nnz =
        std::min(state->minimum_active_nnz, active_nnz);
    state->full_dirty = true;
    const long long delta_count =
        static_cast<long long>(state->delta_count);
    if (delta_count > 0) {
        const long long per_iteration =
            2 * state->delta_nnz + delta_count;
        const long long limit = std::numeric_limits<long long>::max();
        const long long batch_work = per_iteration >
                limit / launched_iterations
            ? limit : per_iteration * launched_iterations;
        state->delta_work_accum = state->delta_work_accum >
                limit - batch_work
            ? limit : state->delta_work_accum + batch_work;
    }
    return launched_iterations;
}


bool hprlp_reduced_matrix_active(const HPRLP_workspace_gpu *workspace) {
    return workspace != nullptr && workspace->reduced_matrix != nullptr &&
        workspace->reduced_matrix->active;
}

void hprlp_print_reduced_matrix_profile(
    const HPRLP_workspace_gpu *workspace) {
    if (workspace == nullptr || workspace->reduced_matrix == nullptr) return;
    const HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    if (!state->profile_enabled) return;
    const auto print_stage = [](
            const char *name, const HPRLP_reduced_stage_profile &profile) {
        const double total_ms = 1000.0 * profile.seconds;
        const double mean_ms = profile.calls > 0
            ? total_ms / static_cast<double>(profile.calls) : 0.0;
        std::cout << "    " << name << ": calls=" << profile.calls
                  << " total_ms=" << std::fixed << std::setprecision(3)
                  << total_ms << " mean_ms=" << mean_ms << "\n";
    };
    std::cout << "  reduced-maintenance-detail:\n";
    print_stage("backend_autotune", state->profile_backend_autotune);
    print_stage("full_build", state->profile_full_build);
    print_stage("delta_extend", state->profile_delta_extend);
    print_stage("workspace_destroy", state->profile_workspace_destroy);
    print_stage("graph_capture", state->profile_graph_capture);
    print_stage(
        "deferred_empty_flush", state->profile_deferred_empty_flush);
    print_stage("flush_to_full", state->profile_flush_to_full);
    print_stage("gather_from_full", state->profile_gather_from_full);
}

void hprlp_collect_reduced_matrix_results(
    const HPRLP_workspace_gpu *workspace,
    HPRLP_results *results) {
    if (workspace == nullptr || results == nullptr ||
        workspace->reduced_matrix == nullptr) {
        return;
    }
    const HPRLP_reduced_matrix_state *state = workspace->reduced_matrix;
    if (state->row_active_iterations > 0) {
        const long double iterations = state->row_active_iterations;
        const long double full_rows = workspace->m;
        const long double full_nnz = workspace->A != nullptr
            ? workspace->A->numElements : 0;
        const double average_row_ratio = full_rows > 0
            ? static_cast<double>(state->row_active_iteration_sum /
                (iterations * full_rows)) : 1.0;
        const double average_nnz_ratio = full_nnz > 0
            ? static_cast<double>(state->row_nnz_iteration_sum /
                (iterations * full_nnz)) : 1.0;
        const double minimum_row_ratio = full_rows > 0
            ? static_cast<double>(state->row_minimum_active / full_rows)
            : 1.0;
        const double minimum_nnz_ratio = full_nnz > 0
            ? static_cast<double>(state->row_minimum_nnz / full_nnz)
            : 1.0;
        std::cout << "Row reduction summary: first_iteration="
                  << state->row_first_iteration
                  << " active_iterations=" << state->row_active_iterations
                  << " rebuilds=" << state->row_rebuilds
                  << " build_time=" << state->row_build_time
                  << " last_active_ratio=" << state->row_last_active_ratio
                  << " average_row_ratio=" << average_row_ratio
                  << " average_nnz_ratio=" << average_nnz_ratio
                  << " minimum_row_ratio=" << minimum_row_ratio
                  << " minimum_nnz_ratio=" << minimum_nnz_ratio
                  << std::endl;
    }
    if (state->restart_mask_checks > 0) {
        std::cout << "Reduced restart mask summary: checks="
                  << state->restart_mask_checks
                  << " resets=" << state->restart_mask_resets << std::endl;
    }
    if (state->deferred_empty_flushes > 0) {
        std::cout << "Reduced deferred empty-row summary: flushes="
                  << state->deferred_empty_flushes
                  << " iterations="
                  << state->deferred_empty_total_iterations << std::endl;
    }
    if (state->row_restart_mask_checks > 0) {
        std::cout << "Reduced row restart mask summary: checks="
                  << state->row_restart_mask_checks
                  << " resets=" << state->row_restart_mask_resets
                  << std::endl;
    }
    results->interior_percentage = workspace->n > 0
        ? 100.0 * static_cast<HPRLP_FLOAT>(state->free_count) / workspace->n
        : 0.0;
    results->reduced_activation_checks = state->activation_checks;
    results->reduced_active_iterations = state->active_iterations;
    results->reduced_first_iteration = state->first_iteration;
    results->reduced_rebuilds = state->rebuilds;
    results->reduced_build_time = state->build_time;
    results->reduced_last_trigger_iteration = state->last_trigger_iteration;
    results->reduced_last_free_ratio = state->last_free_ratio;
    results->reduced_last_trigger_residual = state->last_trigger_residual;
    results->reduced_last_trigger_sigma = state->last_trigger_sigma;
    results->reduced_last_free_columns = state->last_free_columns;
    if (state->active_iterations > 0) {
        const long double active_iterations = state->active_iterations;
        const long double full_columns = workspace->n;
        const long double full_nnz = workspace->AT != nullptr
            ? workspace->AT->numElements : 0;
        results->reduced_active_iteration_ratio = results->iter > 0
            ? static_cast<HPRLP_FLOAT>(active_iterations / results->iter)
            : 0.0;
        results->reduced_average_column_ratio = full_columns > 0
            ? static_cast<HPRLP_FLOAT>(
                state->active_column_iteration_sum /
                (active_iterations * full_columns))
            : 0.0;
        results->reduced_average_nnz_ratio = full_nnz > 0
            ? static_cast<HPRLP_FLOAT>(
                state->active_nnz_iteration_sum /
                (active_iterations * full_nnz))
            : 0.0;
        results->reduced_minimum_column_ratio = full_columns > 0
            ? static_cast<HPRLP_FLOAT>(
                state->minimum_active_columns / full_columns)
            : 0.0;
        results->reduced_minimum_nnz_ratio = full_nnz > 0
            ? static_cast<HPRLP_FLOAT>(
                state->minimum_active_nnz / full_nnz)
            : 0.0;
    }
}
