#pragma once

#include "cpu_presolve/lp_model.hpp"

#include <iosfwd>
#include <string>

namespace cpu_presolve {

struct MpsModel {
  std::string name;
  LpModel lp;
};

MpsModel read_mps(std::istream& input);

}  // namespace cpu_presolve
