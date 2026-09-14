#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gpu_presolver::model {

struct CscMatrix {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<std::int32_t> col_ptr;
  std::vector<std::int32_t> row_idx;
  std::vector<double> values;

  [[nodiscard]] std::size_t nnz() const;
  void validate() const;
};

class LpProblem {
 public:
  LpProblem(CscMatrix matrix,
            std::vector<double> objective,
            std::vector<double> row_lower,
            std::vector<double> row_upper,
            std::vector<double> col_lower,
            std::vector<double> col_upper,
            double objective_constant);

  [[nodiscard]] std::int32_t rows() const;
  [[nodiscard]] std::int32_t cols() const;
  [[nodiscard]] std::size_t nnz() const;
  [[nodiscard]] double objective_constant() const;

 private:
  CscMatrix matrix_;
  std::vector<double> objective_;
  std::vector<double> row_lower_;
  std::vector<double> row_upper_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  double objective_constant_ = 0.0;
};

}  // namespace gpu_presolver::model
