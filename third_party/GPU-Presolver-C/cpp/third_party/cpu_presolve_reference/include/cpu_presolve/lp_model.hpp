#pragma once

#include "cpu_presolve/sparse_matrix.hpp"

#include <memory>
#include <vector>

namespace cpu_presolve {

class LpModel {
public:
  LpModel(CscMatrix matrix,
          std::vector<double> objective,
          std::vector<double> row_lower,
          std::vector<double> row_upper,
          std::vector<double> col_lower,
          std::vector<double> col_upper,
          double obj_constant);
  LpModel(CsrMatrix row_matrix,
          std::vector<double> objective,
          std::vector<double> row_lower,
          std::vector<double> row_upper,
          std::vector<double> col_lower,
          std::vector<double> col_upper,
          double obj_constant);
  LpModel(CscMatrix matrix,
          CsrMatrix row_matrix,
          std::vector<double> objective,
          std::vector<double> row_lower,
          std::vector<double> row_upper,
          std::vector<double> col_lower,
          std::vector<double> col_upper,
          double obj_constant);

  LpModel with_bounds(std::vector<double> row_lower,
                      std::vector<double> row_upper,
                      std::vector<double> col_lower,
                      std::vector<double> col_upper,
                      double obj_constant) const;
  LpModel with_row_bounds(std::vector<double> row_lower,
                          std::vector<double> row_upper,
                          double obj_constant) const;
  LpModel with_col_bounds(std::vector<double> col_lower,
                          std::vector<double> col_upper,
                          double obj_constant) const;

  int num_rows() const noexcept;
  int num_cols() const noexcept;

  const CscMatrix& csc() const;
  const CsrMatrix& csr() const;

  const std::vector<double>& objective() const noexcept { return *objective_; }
  const std::vector<double>& row_lower() const noexcept { return *row_lower_; }
  const std::vector<double>& row_upper() const noexcept { return *row_upper_; }
  const std::vector<double>& col_lower() const noexcept { return *col_lower_; }
  const std::vector<double>& col_upper() const noexcept { return *col_upper_; }
  double obj_constant() const noexcept { return obj_constant_; }

private:
  LpModel(std::shared_ptr<const CscMatrix> csc,
          std::shared_ptr<const CsrMatrix> csr,
          std::shared_ptr<const std::vector<double>> objective,
          std::shared_ptr<const std::vector<double>> row_lower,
          std::shared_ptr<const std::vector<double>> row_upper,
          std::shared_ptr<const std::vector<double>> col_lower,
          std::shared_ptr<const std::vector<double>> col_upper,
          double obj_constant);

  mutable std::shared_ptr<const CscMatrix> csc_;
  mutable std::shared_ptr<const CsrMatrix> csr_;
  std::shared_ptr<const std::vector<double>> objective_;
  std::shared_ptr<const std::vector<double>> row_lower_;
  std::shared_ptr<const std::vector<double>> row_upper_;
  std::shared_ptr<const std::vector<double>> col_lower_;
  std::shared_ptr<const std::vector<double>> col_upper_;
  double obj_constant_;
};

}  // namespace cpu_presolve
