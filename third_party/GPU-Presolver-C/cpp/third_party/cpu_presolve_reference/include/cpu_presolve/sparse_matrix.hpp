#pragma once

#include <vector>

namespace cpu_presolve {

class CscMatrix;

class CsrMatrix {
public:
  CsrMatrix(int rows,
            int cols,
            std::vector<int> row_ptr,
            std::vector<int> col_idx,
            std::vector<double> values);

  int rows() const noexcept { return rows_; }
  int cols() const noexcept { return cols_; }
  int nnz() const noexcept { return static_cast<int>(values_.size()); }

  const std::vector<int>& row_ptr() const noexcept { return row_ptr_; }
  const std::vector<int>& col_idx() const noexcept { return col_idx_; }
  const std::vector<double>& values() const noexcept { return values_; }

  CscMatrix to_csc() const;

private:
  int rows_;
  int cols_;
  std::vector<int> row_ptr_;
  std::vector<int> col_idx_;
  std::vector<double> values_;
};

class CscMatrix {
public:
  CscMatrix(int rows,
            int cols,
            std::vector<int> col_ptr,
            std::vector<int> row_idx,
            std::vector<double> values);

  int rows() const noexcept { return rows_; }
  int cols() const noexcept { return cols_; }
  int nnz() const noexcept { return static_cast<int>(values_.size()); }

  const std::vector<int>& col_ptr() const noexcept { return col_ptr_; }
  const std::vector<int>& row_idx() const noexcept { return row_idx_; }
  const std::vector<double>& values() const noexcept { return values_; }

  CsrMatrix to_csr() const;

private:
  int rows_;
  int cols_;
  std::vector<int> col_ptr_;
  std::vector<int> row_idx_;
  std::vector<double> values_;
};

}  // namespace cpu_presolve
