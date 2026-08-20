#ifndef HPRLP_STRUCTURED_OPERATOR_ENCODING_H
#define HPRLP_STRUCTURED_OPERATOR_ENCODING_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct HPRLPHostCsrView {
    int rows;
    int cols;
    int nonzeros;
    const int *row_ptr;
    const int *col_index;
    const double *values;
};

struct HPRLPStructuredOperatorHost {
    int coefficient_bias = 0;
    bool coefficient_has_escape = false;
    int coefficient_escape_value = 0;

    std::vector<std::uint16_t> dense_rows;
    std::vector<std::uint16_t> dense_cols;
    std::vector<std::uint16_t> dense_local_cols;
    std::vector<std::uint16_t> dense_A_values_u16;
    std::vector<std::uint16_t> dense_AT_values_u16;

    std::vector<std::uint16_t> sparse_rows;
    std::vector<std::uint16_t> sparse_col0;
    std::vector<std::uint16_t> sparse_col1;
    std::vector<std::int8_t> sparse_second_sign;

    std::vector<std::uint16_t> short_AT_output_cols;
    std::vector<int> short_AT_row_ptr;
    std::vector<std::uint16_t> short_AT_rows;
    std::vector<std::uint16_t> short_AT_values_u16;
};

namespace hprlp_structured_detail {

constexpr int kMinimumDenseRows = 64;
constexpr int kMinimumDenseColumns = 8;

inline bool valid_csr(const HPRLPHostCsrView &matrix) {
    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.nonzeros <= 0 ||
        matrix.row_ptr == nullptr || matrix.col_index == nullptr ||
        matrix.values == nullptr || matrix.row_ptr[0] != 0 ||
        matrix.row_ptr[matrix.rows] != matrix.nonzeros) {
        return false;
    }
    for (int row = 0; row < matrix.rows; ++row) {
        if (matrix.row_ptr[row] > matrix.row_ptr[row + 1]) {
            return false;
        }
    }
    for (int index = 0; index < matrix.nonzeros; ++index) {
        if (matrix.col_index[index] < 0 ||
            matrix.col_index[index] >= matrix.cols ||
            !std::isfinite(matrix.values[index])) {
            return false;
        }
    }
    return true;
}

inline bool integer_code(double value, int bias, bool has_escape,
                         int escape_value, std::uint16_t *code) {
    if (std::trunc(value) != value) {
        return false;
    }
    if (has_escape && value == static_cast<double>(escape_value)) {
        *code = 65535;
        return true;
    }
    const double encoded = value + static_cast<double>(bias);
    if (encoded < 0.0 || encoded > 65534.0) {
        return false;
    }
    *code = static_cast<std::uint16_t>(encoded);
    return true;
}

inline bool find_integer_encoding(const HPRLPHostCsrView &matrix, int *bias,
                                  bool *has_escape, int *escape_value) {
    double minimum = matrix.values[0];
    double maximum = matrix.values[0];
    for (int index = 0; index < matrix.nonzeros; ++index) {
        const double value = matrix.values[index];
        if (std::trunc(value) != value) {
            return false;
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    if (minimum < -2147483647.0 || minimum > 2147483647.0 ||
        maximum < -2147483647.0 || maximum > 2147483647.0) {
        return false;
    }
    *bias = -static_cast<int>(minimum);
    *has_escape = false;
    *escape_value = 0;
    for (int index = 0; index < matrix.nonzeros; ++index) {
        const double value = matrix.values[index];
        const double encoded = value + static_cast<double>(*bias);
        if (encoded <= 65534.0) {
            continue;
        }
        const int integer_value = static_cast<int>(value);
        if (!*has_escape) {
            *has_escape = true;
            *escape_value = integer_value;
        } else if (*escape_value != integer_value) {
            return false;
        }
    }
    return true;
}

}  // namespace hprlp_structured_detail

inline bool hprlp_build_dense_two_sparse_operator(
    const HPRLPHostCsrView &A,
    const HPRLPHostCsrView &AT,
    HPRLPStructuredOperatorHost *output) {
    using namespace hprlp_structured_detail;
    if (output == nullptr || !valid_csr(A) || !valid_csr(AT) ||
        A.rows != AT.cols || A.cols != AT.rows ||
        A.nonzeros != AT.nonzeros || A.rows > 65536 || A.cols > 65536) {
        return false;
    }

    HPRLPStructuredOperatorHost candidate;
    if (!find_integer_encoding(A, &candidate.coefficient_bias,
                               &candidate.coefficient_has_escape,
                               &candidate.coefficient_escape_value)) {
        return false;
    }

    int maximum_row_degree = 0;
    for (int row = 0; row < A.rows; ++row) {
        maximum_row_degree = std::max(
            maximum_row_degree, A.row_ptr[row + 1] - A.row_ptr[row]);
    }
    if (maximum_row_degree < kMinimumDenseColumns + 1) {
        return false;
    }

    for (int row = 0; row < A.rows; ++row) {
        const int degree = A.row_ptr[row + 1] - A.row_ptr[row];
        if (degree == maximum_row_degree) {
            candidate.dense_rows.push_back(static_cast<std::uint16_t>(row));
        } else if (degree == 2) {
            candidate.sparse_rows.push_back(static_cast<std::uint16_t>(row));
        } else {
            return false;
        }
    }
    if (candidate.dense_rows.size() <
            static_cast<std::size_t>(kMinimumDenseRows) ||
        candidate.sparse_rows.empty()) {
        return false;
    }

    for (int col = 0; col < AT.rows; ++col) {
        const int degree = AT.row_ptr[col + 1] - AT.row_ptr[col];
        if (degree == static_cast<int>(candidate.dense_rows.size())) {
            candidate.dense_cols.push_back(static_cast<std::uint16_t>(col));
        } else if (degree == 2 || degree == 3) {
            candidate.short_AT_output_cols.push_back(
                static_cast<std::uint16_t>(col));
        } else {
            return false;
        }
    }
    if (candidate.dense_cols.size() <
            static_cast<std::size_t>(kMinimumDenseColumns) ||
        candidate.dense_cols.size() + 1 !=
            static_cast<std::size_t>(maximum_row_degree) ||
        candidate.short_AT_output_cols.empty()) {
        return false;
    }

    std::vector<unsigned char> local_col_seen(A.cols, 0);
    candidate.dense_local_cols.reserve(candidate.dense_rows.size());
    candidate.dense_A_values_u16.reserve(
        candidate.dense_rows.size() *
        static_cast<std::size_t>(maximum_row_degree));
    for (std::size_t dense_index = 0;
         dense_index < candidate.dense_rows.size(); ++dense_index) {
        const int row = candidate.dense_rows[dense_index];
        const int start = A.row_ptr[row];
        for (std::size_t col_index = 0;
             col_index < candidate.dense_cols.size(); ++col_index) {
            if (A.col_index[start + static_cast<int>(col_index)] !=
                candidate.dense_cols[col_index]) {
                return false;
            }
            std::uint16_t code = 0;
            if (!integer_code(A.values[start +
                                       static_cast<int>(col_index)],
                              candidate.coefficient_bias,
                              candidate.coefficient_has_escape,
                              candidate.coefficient_escape_value, &code)) {
                return false;
            }
            candidate.dense_A_values_u16.push_back(code);
        }
        const int local_col = A.col_index[start + maximum_row_degree - 1];
        if (std::binary_search(candidate.dense_cols.begin(),
                               candidate.dense_cols.end(),
                               static_cast<std::uint16_t>(local_col)) ||
            local_col_seen[local_col] != 0 ||
            AT.row_ptr[local_col + 1] - AT.row_ptr[local_col] != 3) {
            return false;
        }
        local_col_seen[local_col] = 1;
        candidate.dense_local_cols.push_back(
            static_cast<std::uint16_t>(local_col));
        std::uint16_t local_code = 0;
        if (!integer_code(A.values[start + maximum_row_degree - 1],
                          candidate.coefficient_bias,
                          candidate.coefficient_has_escape,
                          candidate.coefficient_escape_value, &local_code)) {
            return false;
        }
        candidate.dense_A_values_u16.push_back(local_code);
    }

    candidate.sparse_col0.reserve(candidate.sparse_rows.size());
    candidate.sparse_col1.reserve(candidate.sparse_rows.size());
    candidate.sparse_second_sign.reserve(candidate.sparse_rows.size());
    for (std::size_t sparse_index = 0;
         sparse_index < candidate.sparse_rows.size(); ++sparse_index) {
        const int row = candidate.sparse_rows[sparse_index];
        const int start = A.row_ptr[row];
        if (A.values[start] != 1.0 ||
            (A.values[start + 1] != 1.0 && A.values[start + 1] != -1.0)) {
            return false;
        }
        candidate.sparse_col0.push_back(
            static_cast<std::uint16_t>(A.col_index[start]));
        candidate.sparse_col1.push_back(
            static_cast<std::uint16_t>(A.col_index[start + 1]));
        candidate.sparse_second_sign.push_back(
            A.values[start + 1] == 1.0 ? 1 : -1);
    }

    candidate.dense_AT_values_u16.reserve(
        candidate.dense_cols.size() * candidate.dense_rows.size());
    for (std::size_t dense_col_index = 0;
         dense_col_index < candidate.dense_cols.size(); ++dense_col_index) {
        const int col = candidate.dense_cols[dense_col_index];
        const int start = AT.row_ptr[col];
        for (std::size_t row_index = 0;
             row_index < candidate.dense_rows.size(); ++row_index) {
            if (AT.col_index[start + static_cast<int>(row_index)] !=
                candidate.dense_rows[row_index]) {
                return false;
            }
            std::uint16_t code = 0;
            if (!integer_code(AT.values[start +
                                        static_cast<int>(row_index)],
                              candidate.coefficient_bias,
                              candidate.coefficient_has_escape,
                              candidate.coefficient_escape_value, &code)) {
                return false;
            }
            candidate.dense_AT_values_u16.push_back(code);
        }
    }

    candidate.short_AT_row_ptr.reserve(
        candidate.short_AT_output_cols.size() + 1);
    candidate.short_AT_row_ptr.push_back(0);
    for (std::size_t output_index = 0;
         output_index < candidate.short_AT_output_cols.size();
         ++output_index) {
        const int col = candidate.short_AT_output_cols[output_index];
        for (int index = AT.row_ptr[col]; index < AT.row_ptr[col + 1];
             ++index) {
            std::uint16_t code = 0;
            if (!integer_code(AT.values[index], candidate.coefficient_bias,
                              candidate.coefficient_has_escape,
                              candidate.coefficient_escape_value, &code)) {
                return false;
            }
            candidate.short_AT_rows.push_back(
                static_cast<std::uint16_t>(AT.col_index[index]));
            candidate.short_AT_values_u16.push_back(code);
        }
        candidate.short_AT_row_ptr.push_back(
            static_cast<int>(candidate.short_AT_rows.size()));
    }

    *output = std::move(candidate);
    return true;
}

#endif
