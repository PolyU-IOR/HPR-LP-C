#pragma once

#include "gpu_presolver/folding/folding_structs.hpp"

#include <vector>

namespace gpu_presolver::folding {

std::vector<double> fold_initial_vector(const std::vector<double>* value,
                                        const std::vector<std::int32_t>& color_index,
                                        std::int32_t reduced_length);

FoldingRunSummary run_folding(const presolve::LPInfoGpu& original_model,
                              double tolerance,
                              bool verbose = false);

void free_folded_lp(FoldingRunSummary& summary);

FoldingMapHost copy_map_to_host(const FoldingMapDevice& map);

void unfold_solution(const FoldingMapHost& map,
                     const std::vector<double>& x_red,
                     const std::vector<double>& y_red,
                     const std::vector<double>& z_red,
                     std::vector<double>* x,
                     std::vector<double>* y,
                     std::vector<double>* z);

UnfoldedSolutionHost unfold_solution_to_host(const FoldingMapDevice& map,
                                             const std::vector<double>& x_red,
                                             const std::vector<double>& y_red,
                                             const std::vector<double>& z_red);

FoldingPipelineSummary run_gpu_presolve_with_folding(const presolve::LPInfoGpu& lp,
                                                     const presolve::PresolveParams& params,
                                                     bool keep_reduced_lp,
                                                     bool keep_folded_lp);

UnfoldedSolutionHost postsolve_and_unfold_to_host(double* x_red,
                                                  double* y_red,
                                                  double* z_red,
                                                  const FoldingPipelineSummary& summary);

}  // namespace gpu_presolver::folding
