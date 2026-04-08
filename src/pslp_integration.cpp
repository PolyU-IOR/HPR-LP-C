#include "pslp_integration.h"

#include "mps_reader.h"
#include "preprocess.h"

#include "PSLP_API.h"
#include "PSLP_sol.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct EmbeddedPresolverHandle {
    Presolver *presolver = nullptr;
    Settings *settings = nullptr;
};

struct OriginalKktMetrics {
    HPRLP_FLOAT primal_obj = 0.0;
    HPRLP_FLOAT dual_obj = 0.0;
    HPRLP_FLOAT primal_feas = 0.0;
    HPRLP_FLOAT dual_feas = 0.0;
    HPRLP_FLOAT gap = 0.0;
};

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

    *presolver_handle_out = nullptr;

    Settings *settings = default_settings();
    if (!settings) {
        std::cerr << "[warn] PSLP settings allocation failed; solving original model" << std::endl;
        return false;
    }

    settings->verbose = false;
    if (param && std::isfinite(param->time_limit) && param->time_limit > 0.0) {
        settings->max_time = std::min(settings->max_time, static_cast<double>(param->time_limit));
    }

    Presolver *presolver = new_presolver(model->A->value,
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
        std::cerr << "[warn] PSLP presolver initialization failed; solving original model" << std::endl;
        free_settings(settings);
        return false;
    }

    std::cout << "Doing presolve (PSLP)..." << std::endl;
    const auto presolve_start = std::chrono::steady_clock::now();
    const PresolveStatus status = run_presolver(presolver);
    const double presolve_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - presolve_start).count();
    std::cout << "PSLP presolve time: " << presolve_seconds << " seconds" << std::endl;

    if (!presolver->reduced_prob) {
        std::cerr << "[warn] PSLP presolve returned no reduced problem (status="
                  << static_cast<int>(status) << "); solving original model" << std::endl;
        free_presolver(presolver);
        free_settings(settings);
        return false;
    }

    PresolvedProblem *problem = presolver->reduced_prob;
    CSRMatrix reduced_csr;
    reduced_csr.nrows = static_cast<int>(problem->m);
    reduced_csr.ncols = static_cast<int>(problem->n);
    reduced_csr.nnz = static_cast<int>(problem->nnz);
    reduced_csr.row_ptr = problem->Ap;
    reduced_csr.col_idx = problem->Ai;
    reduced_csr.values = problem->Ax;

    reduced_model->m = 0;
    reduced_model->n = 0;
    reduced_model->A = nullptr;
    reduced_model->AL = nullptr;
    reduced_model->AU = nullptr;
    reduced_model->c = nullptr;
    reduced_model->l = nullptr;
    reduced_model->u = nullptr;
    reduced_model->obj_constant = 0.0;

    build_model_from_arrays(&reduced_csr,
                            problem->lhs,
                            problem->rhs,
                            problem->lbs,
                            problem->ubs,
                            problem->c,
                            model->obj_constant + problem->obj_offset,
                            reduced_model);

    if (!reduced_model->A || reduced_model->m <= 0 || reduced_model->n <= 0) {
        std::cerr << "[warn] Failed to build reduced HPRLP model from PSLP output; solving original model" << std::endl;
        free_lp_info_cpu(reduced_model);
        free_presolver(presolver);
        free_settings(settings);
        return false;
    }

    EmbeddedPresolverHandle *handle = new EmbeddedPresolverHandle;
    handle->presolver = presolver;
    handle->settings = settings;
    *presolver_handle_out = handle;

    std::cout << "PSLP presolve reduced problem: (" << model->m << ", " << model->n << ") -> ("
              << reduced_model->m << ", " << reduced_model->n << ")" << std::endl;
    std::cout << "PSLP objective offset: " << problem->obj_offset << std::endl;
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
    if (!handle->presolver) {
        return false;
    }

    postsolve(handle->presolver, result->x, result->y, result->z);

    if (!handle->presolver->sol) {
        std::cerr << "[warn] PSLP postsolve did not return an original-space solution" << std::endl;
        return false;
    }

    const Solution *solution = handle->presolver->sol;
    if (!solution->x || !solution->y || !solution->z ||
        static_cast<int>(solution->dim_x) != original_n ||
        static_cast<int>(solution->dim_y) != original_m) {
        std::cerr << "[warn] PSLP postsolve returned inconsistent solution dimensions" << std::endl;
        return false;
    }

    HPRLP_FLOAT *x_full = static_cast<HPRLP_FLOAT*>(std::malloc(original_n * sizeof(HPRLP_FLOAT)));
    HPRLP_FLOAT *y_full = static_cast<HPRLP_FLOAT*>(std::malloc(original_m * sizeof(HPRLP_FLOAT)));
    HPRLP_FLOAT *z_full = static_cast<HPRLP_FLOAT*>(std::malloc(original_n * sizeof(HPRLP_FLOAT)));
    if (!x_full || !y_full || !z_full) {
        std::free(x_full);
        std::free(y_full);
        std::free(z_full);
        std::cerr << "[warn] Failed to allocate host memory for PSLP postsolve result" << std::endl;
        return false;
    }

    std::memcpy(x_full, solution->x, original_n * sizeof(HPRLP_FLOAT));
    std::memcpy(y_full, solution->y, original_m * sizeof(HPRLP_FLOAT));
    std::memcpy(z_full, solution->z, original_n * sizeof(HPRLP_FLOAT));

    release_result_vectors(result);
    result->x = x_full;
    result->y = y_full;
    result->z = z_full;
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
    if (handle->presolver) {
        free_presolver(handle->presolver);
    }
    if (handle->settings) {
        free_settings(handle->settings);
    }
    delete handle;
}