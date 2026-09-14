#include "gpu_presolver/model/lp_problem.hpp"

#include <algorithm>

namespace gpu_presolver::model {

std::size_t CscMatrix::nnz() const { return values.size(); }

void CscMatrix::validate() const {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("CscMatrix dimensions must be nonnegative");
  }
  if (col_ptr.size() != static_cast<std::size_t>(cols) + 1U) {
    throw std::invalid_argument("CscMatrix col_ptr length must equal cols + 1");
  }
  if (row_idx.size() != values.size()) {
    throw std::invalid_argument("CscMatrix row_idx length must equal values length");
  }
  if (col_ptr.empty()) {
    return;
  }
  if (col_ptr.front() != 0) {
    throw std::invalid_argument("CscMatrix col_ptr must start at 0");
  }
  if (col_ptr.back() != static_cast<std::int32_t>(values.size())) {
    throw std::invalid_argument("CscMatrix col_ptr must end at nnz");
  }
  if (!std::is_sorted(col_ptr.begin(), col_ptr.end())) {
    throw std::invalid_argument("CscMatrix col_ptr must be sorted");
  }
  for (const auto row : row_idx) {
    if (row < 0 || row >= rows) {
      throw std::invalid_argument("CscMatrix row index out of range");
    }
  }
}

LpProblem::LpProblem(CscMatrix matrix,
                     std::vector<double> objective,
                     std::vector<double> row_lower,
                     std::vector<double> row_upper,
                     std::vector<double> col_lower,
                     std::vector<double> col_upper,
                     double objective_constant)
    : matrix_(std::move(matrix)),
      objective_(std::move(objective)),
      row_lower_(std::move(row_lower)),
      row_upper_(std::move(row_upper)),
      col_lower_(std::move(col_lower)),
      col_upper_(std::move(col_upper)),
      objective_constant_(objective_constant) {
  matrix_.validate();
  const auto row_count = static_cast<std::size_t>(matrix_.rows);
  const auto col_count = static_cast<std::size_t>(matrix_.cols);
  if (objective_.size() != col_count || col_lower_.size() != col_count ||
      col_upper_.size() != col_count) {
    throw std::invalid_argument("LP column vector lengths must equal cols");
  }
  if (row_lower_.size() != row_count || row_upper_.size() != row_count) {
    throw std::invalid_argument("LP row bound lengths must equal rows");
  }
}

std::int32_t LpProblem::rows() const { return matrix_.rows; }

std::int32_t LpProblem::cols() const { return matrix_.cols; }

std::size_t LpProblem::nnz() const { return matrix_.nnz(); }

double LpProblem::objective_constant() const { return objective_constant_; }

}  // namespace gpu_presolver::model
