#ifndef HPRLP_GRID_SLACK_LAPLACIAN_OPERATOR_H
#define HPRLP_GRID_SLACK_LAPLACIAN_OPERATOR_H

#include "gpu/preprocessing/operators/common/structured_operator_encoding.h"

#include <cmath>
#include <vector>

struct HPRLPGridSlackLaplacianShape {
    int grid_side = 0;
    int interior_side = 0;
    int interior_count = 0;
    int boundary_half_count = 0;
    int grid_count = 0;
    int slack_count = 0;
    int vertical_negative_row0 = 0;
    int vertical_positive_row0 = 0;
    int horizontal_negative_row0 = 0;
    int horizontal_positive_row0 = 0;
    int laplacian_row0 = 0;
    int boundary_slack0 = 0;
    double interior_negative = 0.0;
    double interior_positive = 0.0;
    double boundary_negative = 0.0;
    double boundary_positive = 0.0;
    double slack_coefficient = 0.0;
    double laplacian_off_diagonal = 0.0;
    double laplacian_diagonal = 0.0;
};

#ifdef __CUDACC__
#define HPRLP_GRID_SLACK_HD __host__ __device__
#else
#define HPRLP_GRID_SLACK_HD
#endif

HPRLP_GRID_SLACK_HD inline int hprlp_grid_slack_grid_col(
    const HPRLPGridSlackLaplacianShape &shape, int gy, int gx) {
    if (gy == 0) return gx - 1;
    if (gy == shape.grid_side - 1) {
        return gy * shape.grid_side + gx - 3;
    }
    return gy * shape.grid_side + gx - 2;
}

HPRLP_GRID_SLACK_HD inline int hprlp_grid_slack_interior_col(
    const HPRLPGridSlackLaplacianShape &shape, int index) {
    return hprlp_grid_slack_grid_col(
        shape, 1 + index / shape.interior_side,
        1 + index % shape.interior_side);
}

HPRLP_GRID_SLACK_HD inline int hprlp_grid_slack_vertical_col(
    const HPRLPGridSlackLaplacianShape &shape, int index) {
    const int gy = 1 + index / 2;
    const int gx =
        index % 2 == 0 ? 0 : shape.grid_side - 1;
    return hprlp_grid_slack_grid_col(shape, gy, gx);
}

HPRLP_GRID_SLACK_HD inline int hprlp_grid_slack_horizontal_col(
    const HPRLPGridSlackLaplacianShape &shape, int index) {
    return index % 2 == 0
        ? index / 2
        : shape.grid_count - shape.interior_side + index / 2;
}

inline bool hprlp_use_grid_slack_laplacian_operator(
    const HPRLPGridSlackLaplacianShape &shape) {
    return shape.grid_side >= 5 && shape.interior_count >= 4096;
}

namespace hprlp_grid_slack_detail {

inline bool row_matches(const HPRLPHostCsrView &A, int row, int col0,
                        double value0, int col1, double value1) {
    const int start = A.row_ptr[row];
    return A.row_ptr[row + 1] == start + 2 &&
           A.col_index[start] == col0 && A.values[start] == value0 &&
           A.col_index[start + 1] == col1 &&
           A.values[start + 1] == value1;
}

inline bool exact_transpose(const HPRLPHostCsrView &A,
                            const HPRLPHostCsrView &AT) {
    std::vector<int> expected_row_ptr(A.cols + 1, 0);
    for (int entry = 0; entry < A.nonzeros; ++entry) {
        const int col = A.col_index[entry];
        if (col < 0 || col >= A.cols) return false;
        ++expected_row_ptr[col + 1];
    }
    for (int row = 0; row < A.cols; ++row) {
        expected_row_ptr[row + 1] += expected_row_ptr[row];
    }
    for (int row = 0; row <= A.cols; ++row) {
        if (AT.row_ptr[row] != expected_row_ptr[row]) return false;
    }
    std::vector<int> next = expected_row_ptr;
    for (int row = 0; row < A.rows; ++row) {
        for (int entry = A.row_ptr[row]; entry < A.row_ptr[row + 1];
             ++entry) {
            const int position = next[A.col_index[entry]]++;
            if (AT.col_index[position] != row ||
                AT.values[position] != A.values[entry]) return false;
        }
    }
    return true;
}

}  // namespace hprlp_grid_slack_detail

inline bool hprlp_detect_grid_slack_laplacian_operator(
    const HPRLPHostCsrView &A, const HPRLPHostCsrView &AT,
    HPRLPGridSlackLaplacianShape *output) {
    if (output == nullptr || !hprlp_structured_detail::valid_csr(A) ||
        !hprlp_structured_detail::valid_csr(AT) ||
        A.rows != AT.cols || A.cols != AT.rows ||
        A.nonzeros != AT.nonzeros || A.rows < 16 || A.cols < 16) {
        return false;
    }

    int laplacian_row0 = 0;
    while (laplacian_row0 < A.rows &&
           A.row_ptr[laplacian_row0 + 1] -
                   A.row_ptr[laplacian_row0] ==
               2) {
        ++laplacian_row0;
    }
    const int interior_count = A.rows - laplacian_row0;
    const int interior_side = static_cast<int>(
        std::sqrt(static_cast<double>(interior_count)));
    if (interior_side < 3 ||
        interior_side * interior_side != interior_count) return false;

    HPRLPGridSlackLaplacianShape shape;
    shape.grid_side = interior_side + 2;
    shape.interior_side = interior_side;
    shape.interior_count = interior_count;
    shape.boundary_half_count = 2 * interior_side;
    shape.grid_count = shape.grid_side * shape.grid_side - 4;
    shape.slack_count =
        shape.interior_count + shape.boundary_half_count;
    shape.vertical_negative_row0 = 2 * shape.interior_count;
    shape.vertical_positive_row0 =
        shape.vertical_negative_row0 + shape.boundary_half_count;
    shape.horizontal_negative_row0 =
        shape.vertical_positive_row0 + shape.boundary_half_count;
    shape.horizontal_positive_row0 =
        shape.horizontal_negative_row0 + shape.boundary_half_count;
    shape.laplacian_row0 =
        shape.horizontal_positive_row0 + shape.boundary_half_count;
    shape.boundary_slack0 = shape.grid_count + shape.interior_count;

    if (laplacian_row0 != shape.laplacian_row0 ||
        A.cols != shape.grid_count + shape.slack_count ||
        A.nonzeros != 2 * shape.laplacian_row0 +
                            5 * shape.interior_count) {
        return false;
    }
    for (int row = shape.laplacian_row0; row < A.rows; ++row) {
        if (A.row_ptr[row + 1] - A.row_ptr[row] != 5) return false;
    }

    shape.interior_negative = A.values[A.row_ptr[0]];
    shape.interior_positive =
        A.values[A.row_ptr[shape.interior_count]];
    shape.boundary_negative =
        A.values[A.row_ptr[shape.vertical_negative_row0]];
    shape.boundary_positive =
        A.values[A.row_ptr[shape.vertical_positive_row0]];
    shape.slack_coefficient = A.values[A.row_ptr[0] + 1];
    shape.laplacian_off_diagonal =
        A.values[A.row_ptr[shape.laplacian_row0]];
    shape.laplacian_diagonal =
        A.values[A.row_ptr[shape.laplacian_row0] + 2];

    using hprlp_grid_slack_detail::row_matches;
    for (int q = 0; q < shape.interior_count; ++q) {
        const int col = hprlp_grid_slack_interior_col(shape, q);
        const int slack = shape.grid_count + q;
        if (!row_matches(A, q, col, shape.interior_negative,
                         slack, shape.slack_coefficient) ||
            !row_matches(A, shape.interior_count + q, col,
                         shape.interior_positive, slack,
                         shape.slack_coefficient)) return false;
    }
    for (int q = 0; q < shape.boundary_half_count; ++q) {
        const int vertical = hprlp_grid_slack_vertical_col(shape, q);
        const int horizontal = hprlp_grid_slack_horizontal_col(shape, q);
        const int slack = shape.boundary_slack0 + q;
        if (!row_matches(A, shape.vertical_negative_row0 + q,
                         vertical, shape.boundary_negative, slack,
                         shape.slack_coefficient) ||
            !row_matches(A, shape.vertical_positive_row0 + q,
                         vertical, shape.boundary_positive, slack,
                         shape.slack_coefficient) ||
            !row_matches(A, shape.horizontal_negative_row0 + q,
                         horizontal, shape.boundary_negative, slack,
                         shape.slack_coefficient) ||
            !row_matches(A, shape.horizontal_positive_row0 + q,
                         horizontal, shape.boundary_positive, slack,
                         shape.slack_coefficient)) return false;
    }
    for (int q = 0; q < shape.interior_count; ++q) {
        const int row = shape.laplacian_row0 + q;
        const int gy = 1 + q / shape.interior_side;
        const int gx = 1 + q % shape.interior_side;
        const int start = A.row_ptr[row];
        const int expected_cols[5] = {
            hprlp_grid_slack_grid_col(shape, gy - 1, gx),
            hprlp_grid_slack_grid_col(shape, gy, gx - 1),
            hprlp_grid_slack_grid_col(shape, gy, gx),
            hprlp_grid_slack_grid_col(shape, gy, gx + 1),
            hprlp_grid_slack_grid_col(shape, gy + 1, gx)};
        for (int local = 0; local < 5; ++local) {
            const double expected_value =
                local == 2 ? shape.laplacian_diagonal
                           : shape.laplacian_off_diagonal;
            if (A.col_index[start + local] != expected_cols[local] ||
                A.values[start + local] != expected_value) return false;
        }
    }
    if (!hprlp_grid_slack_detail::exact_transpose(A, AT)) return false;
    *output = shape;
    return true;
}

#undef HPRLP_GRID_SLACK_HD

#endif
