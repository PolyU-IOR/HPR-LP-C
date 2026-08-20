#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_structural_l1_substitution(PresolvePlanGpu& plan,
                                           const LPInfoGpu& lp,
                                           const PresolveStatsGpu& stats,
                                           const PresolveParams& pparams);

bool structural_l1_prefix_screen_passes(const LPInfoGpu& lp);

}  // namespace gpu_presolver::presolve
