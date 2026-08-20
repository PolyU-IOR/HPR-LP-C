#include "cpu_presolve/sparse_matrix.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace cpu_presolve {
namespace {

void validate_shape(int rows, int cols) {
  if (rows < 0) {
    throw std::invalid_argument("matrix row count must be non-negative");
  }
  if (cols < 0) {
    throw std::invalid_argument("matrix column count must be non-negative");
  }
}

void validate_compressed_matrix(const char* pointer_name,
                                const char* index_name,
                                int major_size,
                                int minor_size,
                                const std::vector<int>& pointer,
                                const std::vector<int>& index,
                                const std::vector<double>& values) {
  if (pointer.size() != static_cast<std::size_t>(major_size + 1)) {
    throw std::invalid_argument(std::string(pointer_name) + " size must equal major dimension + 1");
  }
  if (index.size() != values.size()) {
    throw std::invalid_argument(std::string(index_name) + " and values must have the same size");
  }
  if (pointer.empty() || pointer.front() != 0) {
    throw std::invalid_argument(std::string(pointer_name) + " must start at zero");
  }
  for (std::size_t k = 1; k < pointer.size(); ++k) {
    if (pointer[k] < pointer[k - 1]) {
      throw std::invalid_argument(std::string(pointer_name) + " must be monotone non-decreasing");
    }
  }
  if (pointer.back() != static_cast<int>(values.size())) {
    throw std::invalid_argument(std::string(pointer_name) + " final entry must equal nnz");
  }
  for (int idx : index) {
    if (idx < 0 || idx >= minor_size) {
      throw std::invalid_argument(std::string(index_name) + " contains an out-of-range index");
    }
  }
}

}  // namespace

CsrMatrix::CsrMatrix(int rows,
                     int cols,
                     std::vector<int> row_ptr,
                     std::vector<int> col_idx,
                     std::vector<double> values)
    : rows_(rows),
      cols_(cols),
      row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)),
      values_(std::move(values)) {
  validate_shape(rows_, cols_);
  validate_compressed_matrix("row_ptr", "col_idx", rows_, cols_, row_ptr_, col_idx_, values_);
}

CscMatrix CsrMatrix::to_csc() const {
  std::vector<int> col_counts(cols_, 0);
  for (int col : col_idx_) {
    ++col_counts[col];
  }

  std::vector<int> col_ptr(cols_ + 1, 0);
  for (int col = 0; col < cols_; ++col) {
    col_ptr[col + 1] = col_ptr[col] + col_counts[col];
  }

  std::vector<int> next = col_ptr;
  std::vector<int> row_idx(values_.size(), 0);
  std::vector<double> csc_values(values_.size(), 0.0);

  for (int row = 0; row < rows_; ++row) {
    for (int pos = row_ptr_[row]; pos < row_ptr_[row + 1]; ++pos) {
      const int col = col_idx_[pos];
      const int dest = next[col]++;
      row_idx[dest] = row;
      csc_values[dest] = values_[pos];
    }
  }

  return CscMatrix(rows_, cols_, std::move(col_ptr), std::move(row_idx), std::move(csc_values));
}

CscMatrix::CscMatrix(int rows,
                     int cols,
                     std::vector<int> col_ptr,
                     std::vector<int> row_idx,
                     std::vector<double> values)
    : rows_(rows),
      cols_(cols),
      col_ptr_(std::move(col_ptr)),
      row_idx_(std::move(row_idx)),
      values_(std::move(values)) {
  validate_shape(rows_, cols_);
  validate_compressed_matrix("col_ptr", "row_idx", cols_, rows_, col_ptr_, row_idx_, values_);
}

CsrMatrix CscMatrix::to_csr() const {
  std::vector<int> row_counts(rows_, 0);
  for (int row : row_idx_) {
    ++row_counts[row];
  }

  std::vector<int> row_ptr(rows_ + 1, 0);
  for (int row = 0; row < rows_; ++row) {
    row_ptr[row + 1] = row_ptr[row] + row_counts[row];
  }

  std::vector<int> next = row_ptr;
  std::vector<int> col_idx(values_.size(), 0);
  std::vector<double> csr_values(values_.size(), 0.0);

  for (int col = 0; col < cols_; ++col) {
    for (int pos = col_ptr_[col]; pos < col_ptr_[col + 1]; ++pos) {
      const int row = row_idx_[pos];
      const int dest = next[row]++;
      col_idx[dest] = col;
      csr_values[dest] = values_[pos];
    }
  }

  return CsrMatrix(rows_, cols_, std::move(row_ptr), std::move(col_idx), std::move(csr_values));
}

}  // namespace cpu_presolve

