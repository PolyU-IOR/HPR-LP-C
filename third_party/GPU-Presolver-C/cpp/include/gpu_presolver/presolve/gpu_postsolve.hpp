#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <vector>

namespace gpu_presolver::presolve {

void postsolve_restore_structural_primal_gpu(
    double* x_org,
    const std::vector<StructuralL1PrimalRecoveryStep>& recoveries,
    const double* original_l = nullptr);

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const double* original_l = nullptr);

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const LPInfoGpu* original_model,
                                 const double* original_l = nullptr);

}  // namespace gpu_presolver::presolve
