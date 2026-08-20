#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_empty_rows(PresolvePlanGpu& plan,
                           const LPInfoGpu& lp,
                           const PresolveStatsGpu& stats,
                           const PresolveParams& pparams);

}  // namespace gpu_presolver::presolve
