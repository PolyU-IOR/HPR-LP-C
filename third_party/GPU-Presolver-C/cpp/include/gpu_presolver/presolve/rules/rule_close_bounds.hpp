#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// Direct port of Julia `apply_rule_close_bounds!`.
void apply_rule_close_bounds(PresolvePlanGpu& plan,
                             const LPInfoGpu& lp,
                             const PresolveParams& pparams);

}  // namespace gpu_presolver::presolve
