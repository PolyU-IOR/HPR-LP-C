#include "cpu_presolve/lp_model.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace cpu_presolve {
namespace {

void require_size(const char* name, std::size_t actual, std::size_t expected) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(name) + " has size " +
                                std::to_string(actual) + ", expected " +
                                std::to_string(expected));
  }
}

void require_bounds(const char* lower_name,
                    const char* upper_name,
                    const std::vector<double>& lower,
                    const std::vector<double>& upper) {
  for (std::size_t k = 0; k < lower.size(); ++k) {
    if (lower[k] > upper[k]) {
      throw std::invalid_argument(std::string(lower_name) + "[" + std::to_string(k) +
                                  "] exceeds " + upper_name + "[" + std::to_string(k) + "]");
    }
  }
}

}  // namespace

LpModel::LpModel(CscMatrix matrix,
                 std::vector<double> objective,
                 std::vector<double> row_lower,
                 std::vector<double> row_upper,
                 std::vector<double> col_lower,
                 std::vector<double> col_upper,
                 double obj_constant)
    : csc_(std::make_shared<CscMatrix>(std::move(matrix))),
      csr_(std::make_shared<CsrMatrix>(csc_->to_csr())),
      objective_(std::make_shared<std::vector<double>>(std::move(objective))),
      row_lower_(std::make_shared<std::vector<double>>(std::move(row_lower))),
      row_upper_(std::make_shared<std::vector<double>>(std::move(row_upper))),
      col_lower_(std::make_shared<std::vector<double>>(std::move(col_lower))),
      col_upper_(std::make_shared<std::vector<double>>(std::move(col_upper))),
      obj_constant_(obj_constant) {
  require_size("objective", objective_->size(), static_cast<std::size_t>(num_cols()));
  require_size("row_lower", row_lower_->size(), static_cast<std::size_t>(num_rows()));
  require_size("row_upper", row_upper_->size(), static_cast<std::size_t>(num_rows()));
  require_size("col_lower", col_lower_->size(), static_cast<std::size_t>(num_cols()));
  require_size("col_upper", col_upper_->size(), static_cast<std::size_t>(num_cols()));
  require_bounds("row_lower", "row_upper", *row_lower_, *row_upper_);
  require_bounds("col_lower", "col_upper", *col_lower_, *col_upper_);
}

LpModel::LpModel(CsrMatrix row_matrix,
                 std::vector<double> objective,
                 std::vector<double> row_lower,
                 std::vector<double> row_upper,
                 std::vector<double> col_lower,
                 std::vector<double> col_upper,
                 double obj_constant)
    : csc_(nullptr),
      csr_(std::make_shared<CsrMatrix>(std::move(row_matrix))),
      objective_(std::make_shared<std::vector<double>>(std::move(objective))),
      row_lower_(std::make_shared<std::vector<double>>(std::move(row_lower))),
      row_upper_(std::make_shared<std::vector<double>>(std::move(row_upper))),
      col_lower_(std::make_shared<std::vector<double>>(std::move(col_lower))),
      col_upper_(std::make_shared<std::vector<double>>(std::move(col_upper))),
      obj_constant_(obj_constant) {
  require_size("objective", objective_->size(), static_cast<std::size_t>(num_cols()));
  require_size("row_lower", row_lower_->size(), static_cast<std::size_t>(num_rows()));
  require_size("row_upper", row_upper_->size(), static_cast<std::size_t>(num_rows()));
  require_size("col_lower", col_lower_->size(), static_cast<std::size_t>(num_cols()));
  require_size("col_upper", col_upper_->size(), static_cast<std::size_t>(num_cols()));
  require_bounds("row_lower", "row_upper", *row_lower_, *row_upper_);
  require_bounds("col_lower", "col_upper", *col_lower_, *col_upper_);
}

LpModel::LpModel(CscMatrix matrix,
                 CsrMatrix row_matrix,
                 std::vector<double> objective,
                 std::vector<double> row_lower,
                 std::vector<double> row_upper,
                 std::vector<double> col_lower,
                 std::vector<double> col_upper,
                 double obj_constant)
    : csc_(std::make_shared<CscMatrix>(std::move(matrix))),
      csr_(std::make_shared<CsrMatrix>(std::move(row_matrix))),
      objective_(std::make_shared<std::vector<double>>(std::move(objective))),
      row_lower_(std::make_shared<std::vector<double>>(std::move(row_lower))),
      row_upper_(std::make_shared<std::vector<double>>(std::move(row_upper))),
      col_lower_(std::make_shared<std::vector<double>>(std::move(col_lower))),
      col_upper_(std::make_shared<std::vector<double>>(std::move(col_upper))),
      obj_constant_(obj_constant) {
  if (csc_->rows() != csr_->rows() || csc_->cols() != csr_->cols() || csc_->nnz() != csr_->nnz()) {
    throw std::invalid_argument("CSC and CSR storage must describe the same shape");
  }
  require_size("objective", objective_->size(), static_cast<std::size_t>(num_cols()));
  require_size("row_lower", row_lower_->size(), static_cast<std::size_t>(num_rows()));
  require_size("row_upper", row_upper_->size(), static_cast<std::size_t>(num_rows()));
  require_size("col_lower", col_lower_->size(), static_cast<std::size_t>(num_cols()));
  require_size("col_upper", col_upper_->size(), static_cast<std::size_t>(num_cols()));
  require_bounds("row_lower", "row_upper", *row_lower_, *row_upper_);
  require_bounds("col_lower", "col_upper", *col_lower_, *col_upper_);
}

LpModel::LpModel(std::shared_ptr<const CscMatrix> csc,
                 std::shared_ptr<const CsrMatrix> csr,
                 std::shared_ptr<const std::vector<double>> objective,
                 std::shared_ptr<const std::vector<double>> row_lower,
                 std::shared_ptr<const std::vector<double>> row_upper,
                 std::shared_ptr<const std::vector<double>> col_lower,
                 std::shared_ptr<const std::vector<double>> col_upper,
                 double obj_constant)
    : csc_(std::move(csc)),
      csr_(std::move(csr)),
      objective_(std::move(objective)),
      row_lower_(std::move(row_lower)),
      row_upper_(std::move(row_upper)),
      col_lower_(std::move(col_lower)),
      col_upper_(std::move(col_upper)),
      obj_constant_(obj_constant) {
  if (!csc_ || !csr_ || !objective_ || !row_lower_ || !row_upper_ || !col_lower_ || !col_upper_) {
    throw std::invalid_argument("model storage pointers must not be null");
  }
  if (csc_->rows() != csr_->rows() || csc_->cols() != csr_->cols() || csc_->nnz() != csr_->nnz()) {
    throw std::invalid_argument("CSC and CSR storage must describe the same shape");
  }
  require_size("objective", objective_->size(), static_cast<std::size_t>(num_cols()));
  require_size("row_lower", row_lower_->size(), static_cast<std::size_t>(num_rows()));
  require_size("row_upper", row_upper_->size(), static_cast<std::size_t>(num_rows()));
  require_size("col_lower", col_lower_->size(), static_cast<std::size_t>(num_cols()));
  require_size("col_upper", col_upper_->size(), static_cast<std::size_t>(num_cols()));
  require_bounds("row_lower", "row_upper", *row_lower_, *row_upper_);
  require_bounds("col_lower", "col_upper", *col_lower_, *col_upper_);
}

int LpModel::num_rows() const noexcept {
  return csc_ ? csc_->rows() : csr_->rows();
}

int LpModel::num_cols() const noexcept {
  return csc_ ? csc_->cols() : csr_->cols();
}

const CscMatrix& LpModel::csc() const {
  if (!csc_) {
    csc_ = std::make_shared<CscMatrix>(csr_->to_csc());
  }
  return *csc_;
}

const CsrMatrix& LpModel::csr() const {
  if (!csr_) {
    csr_ = std::make_shared<CsrMatrix>(csc_->to_csr());
  }
  return *csr_;
}

LpModel LpModel::with_bounds(std::vector<double> row_lower,
                             std::vector<double> row_upper,
                             std::vector<double> col_lower,
                             std::vector<double> col_upper,
                             double obj_constant) const {
  return LpModel(
      csc_,
      csr_,
      objective_,
      std::make_shared<std::vector<double>>(std::move(row_lower)),
      std::make_shared<std::vector<double>>(std::move(row_upper)),
      std::make_shared<std::vector<double>>(std::move(col_lower)),
      std::make_shared<std::vector<double>>(std::move(col_upper)),
      obj_constant);
}

LpModel LpModel::with_row_bounds(std::vector<double> row_lower,
                                 std::vector<double> row_upper,
                                 double obj_constant) const {
  return LpModel(
      csc_,
      csr_,
      objective_,
      std::make_shared<std::vector<double>>(std::move(row_lower)),
      std::make_shared<std::vector<double>>(std::move(row_upper)),
      col_lower_,
      col_upper_,
      obj_constant);
}

LpModel LpModel::with_col_bounds(std::vector<double> col_lower,
                                 std::vector<double> col_upper,
                                 double obj_constant) const {
  return LpModel(
      csc_,
      csr_,
      objective_,
      row_lower_,
      row_upper_,
      std::make_shared<std::vector<double>>(std::move(col_lower)),
      std::make_shared<std::vector<double>>(std::move(col_upper)),
      obj_constant);
}

}  // namespace cpu_presolve
