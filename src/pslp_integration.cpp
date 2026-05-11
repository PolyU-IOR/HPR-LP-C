#include "pslp_integration.h"

#include "mps_reader.h"
#include "preprocess.h"

#include "PSLP_API.h"
#include "PSLP_sol.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iostream>
#include <new>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct EmbeddedPresolverHandle {
    pid_t child_pid = -1;
    int to_child_fd = -1;
    int from_child_fd = -1;
    int reduced_m = 0;
    int reduced_n = 0;
};

enum class WorkerMessageKind : uint32_t {
    PresolveSuccess = 1,
    PresolveFailure = 2,
    PostsolveRequest = 3,
    PostsolveSuccess = 4,
    PostsolveFailure = 5,
};

struct PresolveSuccessPayload {
    int32_t reduced_m = 0;
    int32_t reduced_n = 0;
    int32_t reduced_nnz = 0;
    HPRLP_FLOAT obj_constant = 0.0;
};

struct PostsolveRequestPayload {
    int32_t reduced_m = 0;
    int32_t reduced_n = 0;
};

struct PostsolveSuccessPayload {
    int32_t original_m = 0;
    int32_t original_n = 0;
};

struct OriginalKktMetrics {
    HPRLP_FLOAT primal_obj = 0.0;
    HPRLP_FLOAT dual_obj = 0.0;
    HPRLP_FLOAT primal_feas = 0.0;
    HPRLP_FLOAT dual_feas = 0.0;
    HPRLP_FLOAT gap = 0.0;
};

void release_result_vectors(HPRLP_results *result);

void initialize_empty_model(LP_info_cpu *model) {
    if (!model) {
        return;
    }
    model->m = 0;
    model->n = 0;
    model->A = nullptr;
    model->AL = nullptr;
    model->AU = nullptr;
    model->c = nullptr;
    model->l = nullptr;
    model->u = nullptr;
    model->obj_constant = 0.0;
}

bool write_exact(int fd, const void *buffer, size_t byte_count) {
    const char *cursor = static_cast<const char*>(buffer);
    size_t total_written = 0;
    while (total_written < byte_count) {
        const ssize_t written = ::write(fd, cursor + total_written, byte_count - total_written);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        total_written += static_cast<size_t>(written);
    }
    return true;
}

bool read_exact(int fd, void *buffer, size_t byte_count) {
    char *cursor = static_cast<char*>(buffer);
    size_t total_read = 0;
    while (total_read < byte_count) {
        const ssize_t bytes_read = ::read(fd, cursor + total_read, byte_count - total_read);
        if (bytes_read == 0) {
            return false;
        }
        if (bytes_read < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        total_read += static_cast<size_t>(bytes_read);
    }
    return true;
}

template <typename T>
bool write_pod(int fd, const T &value) {
    return write_exact(fd, &value, sizeof(T));
}

template <typename T>
bool read_pod(int fd, T *value) {
    return read_exact(fd, value, sizeof(T));
}

template <typename T>
bool write_array(int fd, const T *values, size_t count) {
    if (count == 0) {
        return true;
    }
    return write_exact(fd, values, count * sizeof(T));
}

template <typename T>
bool read_array(int fd, T *values, size_t count) {
    if (count == 0) {
        return true;
    }
    return read_exact(fd, values, count * sizeof(T));
}

bool write_message_kind(int fd, WorkerMessageKind kind) {
    const uint32_t raw_kind = static_cast<uint32_t>(kind);
    return write_pod(fd, raw_kind);
}

bool read_message_kind(int fd, WorkerMessageKind *kind) {
    uint32_t raw_kind = 0;
    if (!read_pod(fd, &raw_kind)) {
        return false;
    }
    *kind = static_cast<WorkerMessageKind>(raw_kind);
    return true;
}

void close_fd_if_open(int *fd) {
    if (!fd || *fd < 0) {
        return;
    }
    ::close(*fd);
    *fd = -1;
}

void wait_for_child_process(pid_t child_pid) {
    if (child_pid <= 0) {
        return;
    }

    int status = 0;
    while (::waitpid(child_pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return;
        }
    }
}

void close_worker_channels(EmbeddedPresolverHandle *handle) {
    if (!handle) {
        return;
    }
    close_fd_if_open(&handle->to_child_fd);
    close_fd_if_open(&handle->from_child_fd);
}

void destroy_worker_handle(EmbeddedPresolverHandle *handle) {
    if (!handle) {
        return;
    }
    close_worker_channels(handle);
    wait_for_child_process(handle->child_pid);
    delete handle;
}

void cleanup_worker_resources(pid_t child_pid, int *to_child_fd, int *from_child_fd) {
    close_fd_if_open(to_child_fd);
    close_fd_if_open(from_child_fd);
    wait_for_child_process(child_pid);
}

[[noreturn]] void child_worker_exit(Presolver *presolver,
                                    Settings *settings,
                                    int request_fd,
                                    int response_fd,
                                    int exit_code) {
    close_fd_if_open(&request_fd);
    close_fd_if_open(&response_fd);
    if (presolver) {
        free_presolver(presolver);
    }
    if (settings) {
        free_settings(settings);
    }
    _exit(exit_code);
}

[[noreturn]] void pslp_worker_main(const LP_info_cpu *model,
                                   const HPRLP_parameters *param,
                                   int request_fd,
                                   int response_fd) {
    Settings *settings = default_settings();
    Presolver *presolver = nullptr;

    if (!settings) {
        write_message_kind(response_fd, WorkerMessageKind::PresolveFailure);
        child_worker_exit(nullptr, nullptr, request_fd, response_fd, EXIT_FAILURE);
    }

    settings->verbose = false;
    if (param && std::isfinite(param->time_limit) && param->time_limit > 0.0) {
        settings->max_time = std::min(settings->max_time, static_cast<double>(param->time_limit));
    }

    presolver = new_presolver(model->A->value,
                              model->A->colIndex,
                              model->A->rowPtr,
                              static_cast<size_t>(model->m),
                              static_cast<size_t>(model->n),
                              static_cast<size_t>(model->A->numElements),
                              model->AL,
                              model->AU,
                              model->l,
                              model->u,
                              model->c,
                              settings);
    if (!presolver) {
        write_message_kind(response_fd, WorkerMessageKind::PresolveFailure);
        child_worker_exit(nullptr, settings, request_fd, response_fd, EXIT_FAILURE);
    }

    run_presolver(presolver);
    if (!presolver->reduced_prob) {
        write_message_kind(response_fd, WorkerMessageKind::PresolveFailure);
        child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
    }

    PresolvedProblem *problem = presolver->reduced_prob;
    PresolveSuccessPayload success_payload;
    success_payload.reduced_m = static_cast<int32_t>(problem->m);
    success_payload.reduced_n = static_cast<int32_t>(problem->n);
    success_payload.reduced_nnz = static_cast<int32_t>(problem->nnz);
    success_payload.obj_constant = model->obj_constant + problem->obj_offset;

    const bool wrote_presolve_result =
        write_message_kind(response_fd, WorkerMessageKind::PresolveSuccess) &&
        write_pod(response_fd, success_payload) &&
        write_array(response_fd, problem->Ap, static_cast<size_t>(problem->m) + 1) &&
        write_array(response_fd, problem->Ai, static_cast<size_t>(problem->nnz)) &&
        write_array(response_fd, problem->Ax, static_cast<size_t>(problem->nnz)) &&
        write_array(response_fd, problem->lhs, static_cast<size_t>(problem->m)) &&
        write_array(response_fd, problem->rhs, static_cast<size_t>(problem->m)) &&
        write_array(response_fd, problem->lbs, static_cast<size_t>(problem->n)) &&
        write_array(response_fd, problem->ubs, static_cast<size_t>(problem->n)) &&
        write_array(response_fd, problem->c, static_cast<size_t>(problem->n));
    if (!wrote_presolve_result) {
        child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
    }

    while (true) {
        WorkerMessageKind request_kind;
        if (!read_message_kind(request_fd, &request_kind)) {
            child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_SUCCESS);
        }

        if (request_kind != WorkerMessageKind::PostsolveRequest) {
            write_message_kind(response_fd, WorkerMessageKind::PostsolveFailure);
            child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
        }

        PostsolveRequestPayload request_payload;
        if (!read_pod(request_fd, &request_payload) ||
            request_payload.reduced_m != success_payload.reduced_m ||
            request_payload.reduced_n != success_payload.reduced_n) {
            write_message_kind(response_fd, WorkerMessageKind::PostsolveFailure);
            child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
        }

        HPRLP_FLOAT *x = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(request_payload.reduced_n) * sizeof(HPRLP_FLOAT)));
        HPRLP_FLOAT *y = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(request_payload.reduced_m) * sizeof(HPRLP_FLOAT)));
        HPRLP_FLOAT *z = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(request_payload.reduced_n) * sizeof(HPRLP_FLOAT)));
        if ((!x && request_payload.reduced_n > 0) ||
            (!y && request_payload.reduced_m > 0) ||
            (!z && request_payload.reduced_n > 0) ||
            !read_array(request_fd, x, static_cast<size_t>(request_payload.reduced_n)) ||
            !read_array(request_fd, y, static_cast<size_t>(request_payload.reduced_m)) ||
            !read_array(request_fd, z, static_cast<size_t>(request_payload.reduced_n))) {
            std::free(x);
            std::free(y);
            std::free(z);
            write_message_kind(response_fd, WorkerMessageKind::PostsolveFailure);
            child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
        }

        postsolve(presolver, x, y, z);
        std::free(x);
        std::free(y);
        std::free(z);

        if (!presolver->sol || !presolver->sol->x || !presolver->sol->y || !presolver->sol->z) {
            write_message_kind(response_fd, WorkerMessageKind::PostsolveFailure);
            child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
        }

        PostsolveSuccessPayload postsolve_payload;
        postsolve_payload.original_m = static_cast<int32_t>(presolver->sol->dim_y);
        postsolve_payload.original_n = static_cast<int32_t>(presolver->sol->dim_x);
        const bool wrote_postsolve_result =
            write_message_kind(response_fd, WorkerMessageKind::PostsolveSuccess) &&
            write_pod(response_fd, postsolve_payload) &&
            write_array(response_fd, presolver->sol->x, static_cast<size_t>(presolver->sol->dim_x)) &&
            write_array(response_fd, presolver->sol->y, static_cast<size_t>(presolver->sol->dim_y)) &&
            write_array(response_fd, presolver->sol->z, static_cast<size_t>(presolver->sol->dim_x));
        if (!wrote_postsolve_result) {
            child_worker_exit(presolver, settings, request_fd, response_fd, EXIT_FAILURE);
        }
    }
}

bool receive_reduced_model(int fd,
                           const PresolveSuccessPayload &payload,
                           LP_info_cpu *reduced_model) {
    try {
        std::vector<int> row_ptr(static_cast<size_t>(payload.reduced_m) + 1);
        std::vector<int> col_idx(static_cast<size_t>(payload.reduced_nnz));
        std::vector<HPRLP_FLOAT> values(static_cast<size_t>(payload.reduced_nnz));
        std::vector<HPRLP_FLOAT> lhs(static_cast<size_t>(payload.reduced_m));
        std::vector<HPRLP_FLOAT> rhs(static_cast<size_t>(payload.reduced_m));
        std::vector<HPRLP_FLOAT> lbs(static_cast<size_t>(payload.reduced_n));
        std::vector<HPRLP_FLOAT> ubs(static_cast<size_t>(payload.reduced_n));
        std::vector<HPRLP_FLOAT> c(static_cast<size_t>(payload.reduced_n));

        const bool read_ok =
            read_array(fd, row_ptr.data(), row_ptr.size()) &&
            read_array(fd, col_idx.data(), col_idx.size()) &&
            read_array(fd, values.data(), values.size()) &&
            read_array(fd, lhs.data(), lhs.size()) &&
            read_array(fd, rhs.data(), rhs.size()) &&
            read_array(fd, lbs.data(), lbs.size()) &&
            read_array(fd, ubs.data(), ubs.size()) &&
            read_array(fd, c.data(), c.size());
        if (!read_ok) {
            return false;
        }

        const CSRMatrix reduced_csr{
            payload.reduced_m,
            payload.reduced_n,
            payload.reduced_nnz,
            row_ptr.data(),
            col_idx.data(),
            values.data(),
        };

        initialize_empty_model(reduced_model);
        build_model_from_arrays(&reduced_csr,
                                lhs.data(),
                                rhs.data(),
                                lbs.data(),
                                ubs.data(),
                                c.data(),
                                payload.obj_constant,
                                reduced_model);
        return reduced_model->A && reduced_model->m > 0 && reduced_model->n > 0;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool read_postsolve_result(int fd,
                           int original_m,
                           int original_n,
                           HPRLP_results *result) {
    PostsolveSuccessPayload payload;
    if (!read_pod(fd, &payload) || payload.original_m != original_m || payload.original_n != original_n) {
        return false;
    }

    HPRLP_FLOAT *x_full = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(original_n) * sizeof(HPRLP_FLOAT)));
    HPRLP_FLOAT *y_full = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(original_m) * sizeof(HPRLP_FLOAT)));
    HPRLP_FLOAT *z_full = static_cast<HPRLP_FLOAT*>(std::malloc(static_cast<size_t>(original_n) * sizeof(HPRLP_FLOAT)));
    if ((!x_full && original_n > 0) || (!y_full && original_m > 0) || (!z_full && original_n > 0) ||
        !read_array(fd, x_full, static_cast<size_t>(original_n)) ||
        !read_array(fd, y_full, static_cast<size_t>(original_m)) ||
        !read_array(fd, z_full, static_cast<size_t>(original_n))) {
        std::free(x_full);
        std::free(y_full);
        std::free(z_full);
        return false;
    }

    release_result_vectors(result);
    result->x = x_full;
    result->y = y_full;
    result->z = z_full;
    return true;
}

void release_result_vectors(HPRLP_results *result) {
    if (!result) {
        return;
    }
    if (result->x) {
        std::free(result->x);
        result->x = nullptr;
    }
    if (result->y) {
        std::free(result->y);
        result->y = nullptr;
    }
    if (result->z) {
        std::free(result->z);
        result->z = nullptr;
    }
}

void csr_matvec(const sparseMatrix *A, const HPRLP_FLOAT *x, HPRLP_FLOAT *out) {
    for (int row = 0; row < A->row; ++row) {
        HPRLP_FLOAT sum = 0.0;
        for (int idx = A->rowPtr[row]; idx < A->rowPtr[row + 1]; ++idx) {
            sum += A->value[idx] * x[A->colIndex[idx]];
        }
        out[row] = sum;
    }
}

void csr_transpose_matvec(const sparseMatrix *A, const HPRLP_FLOAT *y, HPRLP_FLOAT *out) {
    std::fill(out, out + A->col, 0.0);
    for (int row = 0; row < A->row; ++row) {
        const HPRLP_FLOAT y_row = y[row];
        for (int idx = A->rowPtr[row]; idx < A->rowPtr[row + 1]; ++idx) {
            out[A->colIndex[idx]] += A->value[idx] * y_row;
        }
    }
}

void project_row_duals(const LP_info_cpu *model, std::vector<HPRLP_FLOAT> *y_proj) {
    for (int i = 0; i < model->m; ++i) {
        const bool lower_inf = std::isinf(model->AL[i]) && model->AL[i] < 0.0;
        const bool upper_inf = std::isinf(model->AU[i]) && model->AU[i] > 0.0;
        if (lower_inf && upper_inf) {
            (*y_proj)[i] = 0.0;
        } else if (upper_inf) {
            (*y_proj)[i] = std::max((*y_proj)[i], static_cast<HPRLP_FLOAT>(0.0));
        } else if (lower_inf) {
            (*y_proj)[i] = std::min((*y_proj)[i], static_cast<HPRLP_FLOAT>(0.0));
        }
    }
}

void project_bound_duals(const LP_info_cpu *model, std::vector<HPRLP_FLOAT> *z_proj) {
    for (int j = 0; j < model->n; ++j) {
        const bool lower_inf = std::isinf(model->l[j]) && model->l[j] < 0.0;
        const bool upper_inf = std::isinf(model->u[j]) && model->u[j] > 0.0;
        if (lower_inf && upper_inf) {
            (*z_proj)[j] = 0.0;
        } else if (upper_inf) {
            (*z_proj)[j] = std::max((*z_proj)[j], static_cast<HPRLP_FLOAT>(0.0));
        } else if (lower_inf) {
            (*z_proj)[j] = std::min((*z_proj)[j], static_cast<HPRLP_FLOAT>(0.0));
        }
    }
}

HPRLP_FLOAT squared_norm_of_conceptual_rhs(const HPRLP_FLOAT *lower,
                                           const HPRLP_FLOAT *upper,
                                           int len) {
    HPRLP_FLOAT sum_sq = 0.0;
    for (int i = 0; i < len; ++i) {
        const HPRLP_FLOAT lower_val = std::isfinite(lower[i]) ? std::abs(lower[i]) : 0.0;
        const HPRLP_FLOAT upper_val = std::isfinite(upper[i]) ? std::abs(upper[i]) : 0.0;
        const HPRLP_FLOAT rhs_val = std::max(lower_val, upper_val);
        sum_sq += rhs_val * rhs_val;
    }
    return sum_sq;
}

OriginalKktMetrics compute_original_kkt_metrics(const LP_info_cpu *model,
                                                const HPRLP_FLOAT *x,
                                                const HPRLP_FLOAT *y,
                                                const HPRLP_FLOAT *z) {
    OriginalKktMetrics metrics;

    std::vector<HPRLP_FLOAT> y_proj(y, y + model->m);
    std::vector<HPRLP_FLOAT> z_proj(z, z + model->n);
    project_row_duals(model, &y_proj);
    project_bound_duals(model, &z_proj);

    std::vector<HPRLP_FLOAT> Ax(model->m, 0.0);
    std::vector<HPRLP_FLOAT> ATy(model->n, 0.0);
    csr_matvec(model->A, x, Ax.data());
    csr_transpose_matvec(model->A, y_proj.data(), ATy.data());

    HPRLP_FLOAT norm_c_sq = 0.0;
    for (int j = 0; j < model->n; ++j) {
        norm_c_sq += model->c[j] * model->c[j];
    }

    const HPRLP_FLOAT norm_b = 1.0 + std::sqrt(
        squared_norm_of_conceptual_rhs(model->AL, model->AU, model->m));
    const HPRLP_FLOAT norm_c = 1.0 + std::sqrt(norm_c_sq);

    HPRLP_FLOAT err_Ax_sq = 0.0;
    for (int i = 0; i < model->m; ++i) {
        HPRLP_FLOAT violation = 0.0;
        if (std::isfinite(model->AL[i]) && Ax[i] < model->AL[i]) {
            violation = std::max(violation, model->AL[i] - Ax[i]);
        }
        if (std::isfinite(model->AU[i]) && Ax[i] > model->AU[i]) {
            violation = std::max(violation, Ax[i] - model->AU[i]);
        }
        err_Ax_sq += violation * violation;
    }

    HPRLP_FLOAT err_x_sq = 0.0;
    for (int j = 0; j < model->n; ++j) {
        HPRLP_FLOAT violation = 0.0;
        if (std::isfinite(model->l[j]) && x[j] < model->l[j]) {
            violation = std::max(violation, model->l[j] - x[j]);
        }
        if (std::isfinite(model->u[j]) && x[j] > model->u[j]) {
            violation = std::max(violation, x[j] - model->u[j]);
        }
        err_x_sq += violation * violation;
    }
    metrics.primal_feas = std::max(std::sqrt(err_Ax_sq), std::sqrt(err_x_sq)) / norm_b;

    HPRLP_FLOAT dual_residual_sq = 0.0;
    for (int j = 0; j < model->n; ++j) {
        const HPRLP_FLOAT dual_residual = model->c[j] - ATy[j] - z_proj[j];
        dual_residual_sq += dual_residual * dual_residual;
    }
    metrics.dual_feas = std::sqrt(dual_residual_sq) / norm_c;

    HPRLP_FLOAT p_lin = 0.0;
    for (int j = 0; j < model->n; ++j) {
        p_lin += model->c[j] * x[j];
    }

    HPRLP_FLOAT d_lin = 0.0;
    for (int i = 0; i < model->m; ++i) {
        const HPRLP_FLOAT support = y_proj[i] >= 0.0 ?
            (std::isfinite(model->AL[i]) ? model->AL[i] : 0.0) :
            (std::isfinite(model->AU[i]) ? model->AU[i] : 0.0);
        d_lin += y_proj[i] * support;
    }
    for (int j = 0; j < model->n; ++j) {
        const HPRLP_FLOAT support = z_proj[j] >= 0.0 ?
            (std::isfinite(model->l[j]) ? model->l[j] : 0.0) :
            (std::isfinite(model->u[j]) ? model->u[j] : 0.0);
        d_lin += z_proj[j] * support;
    }

    metrics.gap = std::abs(d_lin - p_lin) /
                  (1.0 + std::abs(d_lin) + std::abs(p_lin));
    metrics.primal_obj = p_lin + model->obj_constant;
    metrics.dual_obj = d_lin + model->obj_constant;
    return metrics;
}

std::vector<std::string> check_org_recovery_failures(HPRLP_FLOAT primal_feas,
                                                     HPRLP_FLOAT dual_feas,
                                                     HPRLP_FLOAT gap,
                                                     HPRLP_FLOAT stop_tol) {
    std::vector<std::string> failures;
    if (primal_feas > stop_tol) {
        failures.emplace_back("primal recover failed");
    }
    if (dual_feas > stop_tol || gap > stop_tol) {
        failures.emplace_back("dual recover failed");
    }
    return failures;
}

void print_postsolve_kkt_validation(const OriginalKktMetrics &metrics,
                                    const HPRLP_parameters *param) {
    const HPRLP_FLOAT stop_tol = param ? param->stop_tol : 1e-4;
    const HPRLP_FLOAT original_kkt_error = std::max(metrics.primal_feas,
                                                    std::max(metrics.dual_feas, metrics.gap));
    if (original_kkt_error <= stop_tol) {
        std::cout << "Postsolve original KKT check passed" << std::endl;
        return;
    }

    const std::vector<std::string> failures = check_org_recovery_failures(
        metrics.primal_feas, metrics.dual_feas, metrics.gap, stop_tol);

    std::cout << "Warning: postsolve original KKT check failed"
              << " (but the primal solution and objective are reliable): ";
    for (size_t i = 0; i < failures.size(); ++i) {
        if (i > 0) {
            std::cout << "; ";
        }
        std::cout << failures[i];
    }
    std::cout << std::endl;
    std::cout << "Stop Tolerance: " << stop_tol << std::endl;
    std::cout << "Primal Objective: " << metrics.primal_obj << std::endl;
    std::cout << "Dual Objective: " << metrics.dual_obj << std::endl;
    std::cout << "Primal Residual: " << metrics.primal_feas << std::endl;
    std::cout << "Dual Residual: " << metrics.dual_feas << std::endl;
    std::cout << "Relative Gap: " << metrics.gap << std::endl;
}

} // namespace

bool run_embedded_pslp_presolve(const LP_info_cpu *model,
                                const HPRLP_parameters *param,
                                LP_info_cpu *reduced_model,
                                void **presolver_handle_out) {
    if (!model || !model->A || !reduced_model || !presolver_handle_out) {
        return false;
    }

    initialize_empty_model(reduced_model);
    *presolver_handle_out = nullptr;

    int parent_to_child[2] = {-1, -1};
    int child_to_parent[2] = {-1, -1};
    if (::pipe(parent_to_child) != 0 || ::pipe(child_to_parent) != 0) {
        close_fd_if_open(&parent_to_child[0]);
        close_fd_if_open(&parent_to_child[1]);
        close_fd_if_open(&child_to_parent[0]);
        close_fd_if_open(&child_to_parent[1]);
        std::cerr << "[warn] Failed to start isolated PSLP worker; solving original model" << std::endl;
        return false;
    }

    const pid_t child_pid = ::fork();
    if (child_pid < 0) {
        close_fd_if_open(&parent_to_child[0]);
        close_fd_if_open(&parent_to_child[1]);
        close_fd_if_open(&child_to_parent[0]);
        close_fd_if_open(&child_to_parent[1]);
        std::cerr << "[warn] Failed to fork isolated PSLP worker; solving original model" << std::endl;
        return false;
    }

    if (child_pid == 0) {
        close_fd_if_open(&parent_to_child[1]);
        close_fd_if_open(&child_to_parent[0]);
        pslp_worker_main(model, param, parent_to_child[0], child_to_parent[1]);
    }

    close_fd_if_open(&parent_to_child[0]);
    close_fd_if_open(&child_to_parent[1]);

    std::cout << "Doing presolve (PSLP)..." << std::endl;
    const auto presolve_start = std::chrono::steady_clock::now();
    WorkerMessageKind message_kind;
    const bool received_message = read_message_kind(child_to_parent[0], &message_kind);
    const double presolve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - presolve_start).count();
    std::cout << "PSLP presolve time: " << presolve_seconds << " seconds" << std::endl;

    if (!received_message || message_kind != WorkerMessageKind::PresolveSuccess) {
        cleanup_worker_resources(child_pid, &parent_to_child[1], &child_to_parent[0]);
        std::cerr << "[warn] PSLP worker terminated before returning a reduced problem; solving original model"
                  << std::endl;
        return false;
    }

    PresolveSuccessPayload payload;
    if (!read_pod(child_to_parent[0], &payload) || !receive_reduced_model(child_to_parent[0], payload, reduced_model)) {
        cleanup_worker_resources(child_pid, &parent_to_child[1], &child_to_parent[0]);
        std::cerr << "[warn] Failed to build reduced HPRLP model from PSLP output; solving original model" << std::endl;
        free_lp_info_cpu(reduced_model);
        initialize_empty_model(reduced_model);
        return false;
    }

    EmbeddedPresolverHandle *handle = new (std::nothrow) EmbeddedPresolverHandle;
    if (!handle) {
        cleanup_worker_resources(child_pid, &parent_to_child[1], &child_to_parent[0]);
        free_lp_info_cpu(reduced_model);
        initialize_empty_model(reduced_model);
        std::cerr << "[warn] Failed to allocate isolated PSLP worker handle; solving original model"
                  << std::endl;
        return false;
    }

    handle->child_pid = child_pid;
    handle->to_child_fd = parent_to_child[1];
    handle->from_child_fd = child_to_parent[0];
    handle->reduced_m = reduced_model->m;
    handle->reduced_n = reduced_model->n;
    *presolver_handle_out = handle;

    std::cout << "PSLP presolve reduced problem: (" << model->m << ", " << model->n << ") -> ("
              << reduced_model->m << ", " << reduced_model->n << ")" << std::endl;
    return true;
}

bool apply_embedded_pslp_postsolve(HPRLP_results *result,
                                   void *presolver_handle,
                                   int original_m,
                                   int original_n) {
    if (!result || !presolver_handle || !result->x || !result->y || !result->z) {
        return false;
    }

    EmbeddedPresolverHandle *handle = static_cast<EmbeddedPresolverHandle*>(presolver_handle);
    if (handle->to_child_fd < 0 || handle->from_child_fd < 0 || handle->reduced_m <= 0 || handle->reduced_n <= 0) {
        return false;
    }

    PostsolveRequestPayload request_payload;
    request_payload.reduced_m = handle->reduced_m;
    request_payload.reduced_n = handle->reduced_n;
    const bool sent_request =
        write_message_kind(handle->to_child_fd, WorkerMessageKind::PostsolveRequest) &&
        write_pod(handle->to_child_fd, request_payload) &&
        write_array(handle->to_child_fd, result->x, static_cast<size_t>(handle->reduced_n)) &&
        write_array(handle->to_child_fd, result->y, static_cast<size_t>(handle->reduced_m)) &&
        write_array(handle->to_child_fd, result->z, static_cast<size_t>(handle->reduced_n));
    if (!sent_request) {
        std::cerr << "[warn] Failed to communicate reduced solution to isolated PSLP worker" << std::endl;
        return false;
    }

    WorkerMessageKind response_kind;
    if (!read_message_kind(handle->from_child_fd, &response_kind)) {
        std::cerr << "[warn] Isolated PSLP worker terminated during postsolve" << std::endl;
        return false;
    }

    if (response_kind != WorkerMessageKind::PostsolveSuccess) {
        std::cerr << "[warn] PSLP postsolve did not return an original-space solution" << std::endl;
        return false;
    }

    if (!read_postsolve_result(handle->from_child_fd, original_m, original_n, result)) {
        std::cerr << "[warn] PSLP postsolve returned inconsistent solution dimensions" << std::endl;
        return false;
    }

    return true;
}

bool postsolve_and_validate_original_kkt(HPRLP_results *result,
                                         const LP_info_cpu *original_model,
                                         void *presolver_handle,
                                         const HPRLP_parameters *param) {
    if (!result || !original_model) {
        return false;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "PSLP POSTSOLVE" << std::endl;
    std::cout << "================================================================================" << std::endl;

    if (!apply_embedded_pslp_postsolve(result, presolver_handle, original_model->m, original_model->n)) {
        return false;
    }

    if (std::strcmp(result->status, "OPTIMAL") != 0) {
        std::cout << "Skipping postsolve original KKT check since the reduced solution is not optimal"
                  << std::endl;
        return true;
    }

    const OriginalKktMetrics metrics = compute_original_kkt_metrics(
        original_model, result->x, result->y, result->z);
    print_postsolve_kkt_validation(metrics, param);
    return true;
}

void free_embedded_pslp_presolver(void *presolver_handle) {
    if (!presolver_handle) {
        return;
    }

    EmbeddedPresolverHandle *handle = static_cast<EmbeddedPresolverHandle*>(presolver_handle);
    destroy_worker_handle(handle);
}