#pragma once

#include "cpu_presolve/lp_model.hpp"

#include <utility>
#include <vector>

namespace cpu_presolve {

enum class PresolveStatus {
  kOk,
  kInfeasible,
  kUnbounded,
};

struct FixedColumnRecord {
  int original_col;
  double value;
  double objective_coefficient;
};

class PresolveResult {
public:
  PresolveResult(LpModel reduced_model,
                 PresolveStatus status,
                 bool changed,
                 std::vector<FixedColumnRecord> fixed_columns,
                 double objective_shift);

  const LpModel& reduced_model() const& noexcept { return reduced_model_; }
  LpModel&& reduced_model() && noexcept { return std::move(reduced_model_); }
  PresolveStatus status() const noexcept { return status_; }
  bool changed() const noexcept { return changed_; }
  const std::vector<FixedColumnRecord>& fixed_columns() const noexcept { return fixed_columns_; }
  double objective_shift() const noexcept { return objective_shift_; }

private:
  LpModel reduced_model_;
  PresolveStatus status_;
  bool changed_;
  std::vector<FixedColumnRecord> fixed_columns_;
  double objective_shift_;
};

}  // namespace cpu_presolve
