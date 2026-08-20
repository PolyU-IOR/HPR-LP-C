#include "solver/internal/solver_output.h"
#include "api/version.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <streambuf>
#include <string>

namespace {

bool starts_with(const std::string &text, const char *prefix) {
    return text.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
}

bool is_iteration_row(const std::string &line) {
    const std::size_t first = line.find_first_not_of(' ');
    if (first == std::string::npos || line[first] < '0' || line[first] > '9') {
        return false;
    }
    // Progress rows have eight whitespace-separated numeric fields.  Requiring
    // scientific notation prevents unrelated integer diagnostics from leaking.
    return line.find("e+") != std::string::npos ||
           line.find("e-") != std::string::npos;
}

class HPRLP_quiet_streambuf final : public std::streambuf {
public:
    explicit HPRLP_quiet_streambuf(std::streambuf *destination)
        : destination_(destination) {}

    ~HPRLP_quiet_streambuf() override { flush_pending(); }

protected:
    int overflow(int character) override {
        if (character == traits_type::eof()) return traits_type::not_eof(character);
        append(static_cast<char>(character));
        return character;
    }

    std::streamsize xsputn(const char *data, std::streamsize size) override {
        for (std::streamsize i = 0; i < size; ++i) append(data[i]);
        return size;
    }

    int sync() override {
        destination_->pubsync();
        return 0;
    }

private:
    void append(char character) {
        pending_.push_back(character);
        if (character == '\n') flush_pending();
    }

    bool allowed(const std::string &line) {
        if (line == "\n" || line == "\r\n") {
            in_parameter_block_ = false;
            if (!banner_started_) return false;
            return true;
        }

        if (line == "==================================================================\n") {
            banner_started_ = true;
            return banner_separator_count_++ < 2;
        }
        if (
            starts_with(line, "                          HPR-LP Solver") ||
            starts_with(line, "     Halpern Peaceman-Rachford") ||
            starts_with(line, "  Version: ")) {
            return true;
        }

        if (!banner_started_ && (starts_with(line, "Reading file ") ||
            starts_with(line, "Reading time: ") ||
            starts_with(line, "Original Model:") ||
            starts_with(line, "Presolved ("))) {
            return true;
        }

        if (starts_with(line, "Solver Parameters:")) {
            in_parameter_block_ = true;
            return true;
        }
        if (in_parameter_block_) {
            return true;
        }

        if (starts_with(line, " iter     errRp")) return true;
        if (is_iteration_row(line)) return true;

        if (starts_with(line, "Total Time: ")) {
            summary_complete_ = true;
            return true;
        }

        if (!summary_complete_ &&
            (starts_with(line, "=== Solution Summary ===") ||
            starts_with(line, "Status: ") ||
            starts_with(line, "Iterations: ") ||
            starts_with(line, "Primal Objective: ") ||
            starts_with(line, "Dual Objective: ") ||
            starts_with(line, "Objective Gap: ") ||
            starts_with(line, "Primal Residual: ") ||
            starts_with(line, "Dual Residual: "))) {
            return true;
        }

        // Never hide actual failures. Informational [warn] messages remain on
        // stderr and therefore also bypass this stdout-only filter.
        return starts_with(line, "Error:") || starts_with(line, "[error]");
    }

    void flush_pending() {
        if (pending_.empty()) return;
        const bool blank = pending_ == "\n" || pending_ == "\r\n";
        if (allowed(pending_) && !(blank && last_emitted_blank_)) {
            destination_->sputn(pending_.data(),
                                static_cast<std::streamsize>(pending_.size()));
            last_emitted_blank_ = blank;
        }
        pending_.clear();
    }

    std::streambuf *destination_;
    std::string pending_;
    bool in_parameter_block_ = false;
    bool banner_started_ = false;
    int banner_separator_count_ = 0;
    bool last_emitted_blank_ = false;
    bool summary_complete_ = false;
};

bool show_parameter(const HPRLP_parameters *param, std::uint64_t bit) {
    return param->print_debug_info || (param->specified_parameter_mask & bit) != 0;
}

} // namespace

HPRLP_scoped_output_filter::HPRLP_scoped_output_filter(bool print_debug_info) {
    if (print_debug_info) return;
    original_ = std::cout.rdbuf();
    filter_.reset(new HPRLP_quiet_streambuf(original_));
    std::cout.rdbuf(filter_.get());
}

HPRLP_scoped_output_filter::~HPRLP_scoped_output_filter() {
    if (!original_) return;
    std::cout.flush();
    std::cout.rdbuf(original_);
    filter_.reset();
}

/**
 * Print HPRLP startup banner with version information
 */
void print_solver_banner() {
    std::cout << "\n";
    std::cout << "==================================================================\n";
    std::cout << "                          HPR-LP Solver                           \n";
    std::cout << "     Halpern Peaceman-Rachford Linear Programming Solver          \n";
    std::cout << "                                                                  \n";
    std::cout << "  Version: " << HPRLP_VERSION_STRING << "                          \n";
    std::cout << "                                                                  \n";
    std::cout << "==================================================================\n";
    std::cout << "\n";
}

/**
 * Print solver parameters
 */
void print_solver_parameters(const HPRLP_parameters *param) {
    if (!param->print_debug_info && param->specified_parameter_mask == 0) return;
    std::cout << "Solver Parameters:\n";
    if (show_parameter(param, HPRLP_PARAM_DEVICE)) std::cout << "  Device:              GPU " << param->device_number << "\n";
    if (show_parameter(param, HPRLP_PARAM_MAX_ITER)) std::cout << "  Max Iterations:      " << param->max_iter << "\n";
    if (show_parameter(param, HPRLP_PARAM_STOP_TOL)) std::cout << "  Stopping Tolerance:  " << std::scientific << std::setprecision(1) << param->stop_tol << "\n" << std::defaultfloat;
    if (show_parameter(param, HPRLP_PARAM_TIME_LIMIT)) std::cout << "  Time Limit:          " << std::fixed << std::setprecision(1) << param->time_limit << " seconds\n" << std::defaultfloat;
    if (show_parameter(param, HPRLP_PARAM_CHECK_ITER)) std::cout << "  Check Interval:      " << param->check_iter << " iterations\n";
    if (show_parameter(param, HPRLP_PARAM_CUSPARSE_SPMV)) std::cout << "  cuSPARSE SpMV:       " << (param->CUSPARSE_spmv ? "Forced" : "Auto") << "\n";
    if (show_parameter(param, HPRLP_PARAM_AUTOTUNE_VERBOSE)) std::cout << "  Autotune Verbose:    " << (param->autotune_verbose ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_REDUCED_MATRIX)) std::cout << "  Reduced Matrix:      " << (param->use_reduced_matrix ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_AUTO_MEMORY_POLICY)) std::cout << "  Auto Memory Policy:  " << (param->auto_reduced_compression_policy ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_PROGRESS_MONITOR)) std::cout << "  Progress Monitor:    " << (param->enable_progress_monitor ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_PROGRESS_CONTROL)) std::cout << "  Progress Control:    " << (param->enable_progress_control ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_SIGMA_REBALANCE)) std::cout << "  Sigma Rebalance:     " << (param->enable_sigma_rebalance_restart ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_RESTART_GUARD)) std::cout << "  Long Restart Guard:  " << (param->use_progress_restart_guard ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_RESTART_COOLDOWN)) std::cout << "  Restart Cooldown:    " << param->restart_cooldown_checks << " checks\n";
    if (show_parameter(param, HPRLP_PARAM_DEBUG_RESTART)) std::cout << "  Restart Debug:       " << (param->debug_restart ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_DEBUG_SIGMA)) std::cout << "  Sigma Debug:         " << (param->debug_sigma ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_FIXED_SIGMA)) {
        if (std::isfinite(param->fixed_sigma) && param->fixed_sigma > 0.0) std::cout << "  Fixed Sigma:         " << param->fixed_sigma << "\n";
        else std::cout << "  Fixed Sigma:         Adaptive\n";
    }
    const char *presolver_name = "Disabled";
    if (param->use_presolve && param->presolver != HPRLP_PRESOLVER_NONE) {
        presolver_name = param->presolver == HPRLP_PRESOLVER_GPU
            ? "GPU-Presolver-C"
            : "PSLP";
    }
    if (show_parameter(param, HPRLP_PARAM_PRESOLVER)) std::cout << "  Presolver:           " << presolver_name << "\n";
    if (show_parameter(param, HPRLP_PARAM_GPU_FOLDING) && param->use_presolve && param->presolver == HPRLP_PRESOLVER_GPU) {
        std::cout << "  GPU Folding:         "
                  << (param->enable_gpu_folding ? "Enabled" : "Disabled") << "\n";
    }
    const bool show_scaling = param->print_debug_info ||
        (param->specified_parameter_mask & (HPRLP_PARAM_CR_SCALING |
         HPRLP_PARAM_RUIZ_SCALING | HPRLP_PARAM_POCK_SCALING |
         HPRLP_PARAM_BC_SCALING)) != 0;
    if (show_scaling) std::cout << "  Scaling:\n";
    if (show_parameter(param, HPRLP_PARAM_CR_SCALING)) std::cout << "    - Curtis-Reid:     " << (param->use_CR_scaling ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_RUIZ_SCALING)) std::cout << "    - Ruiz:            " << (param->use_Ruiz_scaling ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_POCK_SCALING)) std::cout << "    - Pock-Chambolle:  " << (param->use_Pock_Chambolle_scaling ? "Enabled" : "Disabled") << "\n";
    if (show_parameter(param, HPRLP_PARAM_BC_SCALING)) std::cout << "    - Bounds/Cost:     " << (param->use_bc_scaling ? "Enabled" : "Disabled") << "\n";
    if (param->print_debug_info) {
        std::cout << "  Debug Information:  Enabled\n";
    }
    std::cout << "\n";
}

/** Print the final solution and timing summary. */
void print_solution_summary(const HPRLP_results& result,
                            HPRLP_FLOAT primal_objective,
                            HPRLP_FLOAT dual_objective,
                            HPRLP_FLOAT objective_gap,
                            HPRLP_FLOAT primal_residual,
                            HPRLP_FLOAT dual_residual) {
    const std::streamsize saved_precision = std::cout.precision();
    const std::ios::fmtflags saved_flags = std::cout.flags();

    std::cout << "\n=== Solution Summary ===\n"
              << "Status: " << result.status << "\n"
              << "Iterations: " << result.iter << "\n"
              << std::fixed << std::setprecision(8)
              << "Primal Objective: " << primal_objective << "\n"
              << "Dual Objective: " << dual_objective << "\n"
              << std::scientific << std::setprecision(2)
              << "Objective Gap: " << objective_gap << "\n"
              << "Primal Residual: " << primal_residual << "\n"
              << "Dual Residual: " << dual_residual << "\n"
              << std::fixed << std::setprecision(2)
              << "Total Time: " << result.timing.total_time << " seconds\n";
    // Detailed timing and reduced-state statistics are retained verbatim in
    // debug mode; the quiet stream filter removes these lines otherwise.
    std::cout << std::fixed << std::setprecision(5)
              << "Presolve Time: " << result.timing.presolve_time << " seconds\n"
              << "Setup Time: " << result.timing.setup_time << " seconds\n"
              << "Scaling Time: " << result.timing.scaling_time << " seconds\n"
              << "Analyze Time: " << result.timing.analyze_time << " seconds\n"
              << "Power Time: " << result.timing.power_iteration_time << " seconds\n"
              << "Solve Time: " << result.timing.solve_time << " seconds\n"
              << "Folding Time: " << result.folding_time << " seconds\n"
              << "Interior Percentage: " << result.interior_percentage << "%\n"
              << "Reduced Activation Checks: " << result.reduced_activation_checks << "\n"
              << "Reduced Active Iterations: " << result.reduced_active_iterations << "\n"
              << "Reduced First Iteration: " << result.reduced_first_iteration << "\n"
              << "Reduced Rebuilds: " << result.reduced_rebuilds << "\n"
              << "Reduced Build Time: " << result.reduced_build_time << " seconds\n"
              << "Reduced Last Trigger Iteration: " << result.reduced_last_trigger_iteration << "\n"
              << "Reduced Last Free Ratio: " << result.reduced_last_free_ratio << "\n"
              << "Reduced Last Trigger Residual: " << result.reduced_last_trigger_residual << "\n"
              << "Reduced Last Trigger Sigma: " << result.reduced_last_trigger_sigma << "\n"
              << "Reduced Last Free Columns: " << result.reduced_last_free_columns << "\n"
              << "Reduced Active Iteration Ratio: " << result.reduced_active_iteration_ratio << "\n"
              << "Reduced Average Column Ratio: " << result.reduced_average_column_ratio << "\n"
              << "Reduced Average NNZ Ratio: " << result.reduced_average_nnz_ratio << "\n"
              << "Reduced Minimum Column Ratio: " << result.reduced_minimum_column_ratio << "\n"
              << "Reduced Minimum NNZ Ratio: " << result.reduced_minimum_nnz_ratio << "\n\n"
              << std::flush;

    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
}


static HPRLP_finite_abs_range finite_abs_range(const HPRLP_FLOAT* values,
                                       std::size_t count) {
    HPRLP_finite_abs_range range;
    if (!values) {
        return range;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isinf(values[i])) {
            if (std::signbit(values[i])) {
                range.has_negative_infinity = true;
            } else {
                range.has_positive_infinity = true;
            }
            continue;
        }
        if (!std::isfinite(values[i])) {
            continue;
        }
        const HPRLP_FLOAT magnitude = std::abs(values[i]);
        range.minimum = std::min(range.minimum, magnitude);
        range.maximum = std::max(range.maximum, magnitude);
        range.has_value = true;
    }
    return range;
}

static void print_finite_abs_range(const char* name,
                                   const HPRLP_finite_abs_range& range) {
    std::cout << "  " << std::left << std::setw(3) << name << std::right << " ";
    if (!range.has_value) {
        if (range.has_negative_infinity && range.has_positive_infinity) {
            std::cout << "[-Inf, Inf]\n";
        } else if (range.has_negative_infinity) {
            std::cout << "[-Inf, -Inf]\n";
        } else if (range.has_positive_infinity) {
            std::cout << "[Inf, Inf]\n";
        } else {
            std::cout << "[n/a, n/a]\n";
        }
        return;
    }
    std::cout << "[" << range.minimum << ", " << range.maximum << "]\n";
}

void print_numerical_ranges(const HPRLP_numerical_ranges &ranges,
                            const char *stage) {
    const std::streamsize saved_precision = std::cout.precision();
    const std::ios::fmtflags saved_flags = std::cout.flags();

    std::cout << "\nNumerical ranges " << stage
              << " (finite absolute values):\n"
              << std::scientific << std::setprecision(2);
    print_finite_abs_range("A", ranges.A);
    print_finite_abs_range("AL", ranges.AL);
    print_finite_abs_range("AU", ranges.AU);
    print_finite_abs_range("l", ranges.l);
    print_finite_abs_range("u", ranges.u);
    print_finite_abs_range("c", ranges.c);
    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
    std::cout << std::flush;
}

void print_numerical_ranges(const LP_info_cpu *model, const char *stage) {
    if (!model || !model->A) {
        const std::streamsize saved_precision = std::cout.precision();
        const std::ios::fmtflags saved_flags = std::cout.flags();
        std::cout << "\nNumerical ranges " << stage
                  << " (finite absolute values):\n"
                  << "  unavailable\n";
        std::cout.flags(saved_flags);
        std::cout.precision(saved_precision);
        std::cout << std::flush;
        return;
    }

    HPRLP_numerical_ranges ranges;
    ranges.A = finite_abs_range(
        model->A->value,
        static_cast<std::size_t>(model->A->numElements));
    ranges.AL = finite_abs_range(
        model->AL, static_cast<std::size_t>(model->m));
    ranges.AU = finite_abs_range(
        model->AU, static_cast<std::size_t>(model->m));
    ranges.l = finite_abs_range(
        model->l, static_cast<std::size_t>(model->n));
    ranges.u = finite_abs_range(
        model->u, static_cast<std::size_t>(model->n));
    ranges.c = finite_abs_range(
        model->c, static_cast<std::size_t>(model->n));
    print_numerical_ranges(ranges, stage);
}
