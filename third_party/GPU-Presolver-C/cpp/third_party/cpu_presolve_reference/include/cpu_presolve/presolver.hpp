#pragma once

#include "cpu_presolve/lp_model.hpp"
#include "cpu_presolve/presolve_options.hpp"
#include "cpu_presolve/presolve_result.hpp"

namespace cpu_presolve {

class Presolver {
public:
  static PresolveResult run(const LpModel& model,
                            const PresolveOptions& options = PresolveOptions{});
};

}  // namespace cpu_presolve
