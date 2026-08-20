#pragma once

#include "gpu_presolver/folding/folding_structs.hpp"

#include <cstdint>

namespace gpu_presolver::folding {

struct ColorPair {
  std::int32_t color_id = 0;
  float color_sig = 0.0f;
};

struct FoldingWorkspace {
  std::int32_t num_row = 0;
  std::int32_t num_col = 0;
  std::int32_t num_row_color = 1;
  std::int32_t num_col_color = 1;
  ColorPair* row_color = nullptr;
  ColorPair* col_color = nullptr;
  std::int32_t* row_color_id = nullptr;
  std::int32_t* col_color_id = nullptr;
  double* row_color_sig = nullptr;
  double* col_color_sig = nullptr;
  std::int32_t* row_color_start = nullptr;
  std::int32_t* col_color_start = nullptr;
  std::int32_t* row_perm = nullptr;
  std::int32_t* col_perm = nullptr;
};

void init_workspace(FoldingWorkspace& workspace, std::int32_t num_row, std::int32_t num_col);
void free_workspace(FoldingWorkspace& workspace);
std::int32_t folding_update_row_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol);
std::int32_t folding_update_col_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol);
void folding_write_row_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side);
void folding_write_col_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side);

bool refine_color(const presolve::LPInfoGpu& device_model,
                  FoldingWorkspace& workspace,
                  double tolerance,
                  bool verbose,
                  FoldingRefinementProfile* profile);

presolve::LPInfoGpu reduce_size(const FoldingWorkspace& workspace,
                                const presolve::LPInfoGpu& model,
                                double tolerance,
                                FoldingReduceProfile* profile);

FoldingMapDevice build_device_map(const FoldingWorkspace& workspace,
                                  FoldingMapProfile* profile);

void free_folding_map_device(FoldingMapDevice& map);

UnfoldedSolutionHost unfold_solution_device_to_host(const FoldingMapDevice& map,
                                                    const double* x_red_device,
                                                    const double* y_red_device,
                                                    const double* z_red_device);

void free_lp_device(presolve::LPInfoGpu& lp);

}  // namespace gpu_presolver::folding
