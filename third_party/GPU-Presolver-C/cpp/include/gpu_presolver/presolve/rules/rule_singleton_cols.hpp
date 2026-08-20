#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_singleton_cols_eq(PresolvePlanGpu& plan,
                                  const LPInfoGpu& lp,
                                  const PresolveStatsGpu& stats,
                                  const PresolveParams& pparams);

void apply_rule_singleton_cols_dual_infer(PresolvePlanGpu& plan,
                                          const LPInfoGpu& lp,
                                          const PresolveStatsGpu& stats,
                                          const PresolveParams& pparams);

}  // namespace gpu_presolver::presolve
