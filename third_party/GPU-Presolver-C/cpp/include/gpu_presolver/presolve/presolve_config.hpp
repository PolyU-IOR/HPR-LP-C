#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <string>

namespace gpu_presolver::presolve {

// Loads the supported TOML settings into params. Existing values are retained
// for keys that are not present. If parsing fails, params is left unchanged.
void load_presolve_params_from_toml(const std::string& path, PresolveParams& params);

}  // namespace gpu_presolver::presolve
