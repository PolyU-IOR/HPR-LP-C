#ifndef HPRLP_WINDOWED_STENCIL_OPERATOR_H
#define HPRLP_WINDOWED_STENCIL_OPERATOR_H

#include "gpu/preprocessing/operators/common/structured_operator_encoding.h"

#include <algorithm>

struct HPRLPWindowedStencilShape {
    int dense_col = 0;
    int grid_col0 = 1;
    int grid_height = 0;
    int grid_width = 0;
    int equation_width = 0;
    int observation_height = 0;
    int observation_width = 0;
    int observation_count = 0;
    int observation_y0 = 0;
    int observation_x0 = 0;
    int stencil_row0 = 0;
    double dense_negative = 0.0;
    double dense_positive = 0.0;
    double observation_coefficient = 0.0;
    double off_diagonal = 0.0;
    double diagonal_boundary_left = 0.0;
    double diagonal_boundary = 0.0;
    double diagonal_interior_left = 0.0;
    double diagonal_interior = 0.0;
};

inline double hprlp_windowed_stencil_diagonal(
    const HPRLPWindowedStencilShape &shape, int gy, int gx) {
    const bool boundary_y =
        gy == 0 || gy == shape.grid_height - 1;
    if (boundary_y) {
        return gx == 0 ? shape.diagonal_boundary_left
                       : shape.diagonal_boundary;
    }
    return gx == 0 ? shape.diagonal_interior_left
                   : shape.diagonal_interior;
}

inline bool hprlp_use_windowed_stencil_operator(
    const HPRLPWindowedStencilShape &shape) {
    return shape.grid_height > 0 && shape.grid_width > 0 &&
           shape.observation_height > 0 &&
           shape.observation_width > 0 &&
           shape.grid_width <= 512 &&
           2 * shape.observation_width <= 512 &&
           shape.stencil_row0 +
                   shape.grid_height * shape.equation_width >=
               4096;
}

inline bool hprlp_detect_windowed_stencil_operator(
    const HPRLPHostCsrView &A, const HPRLPHostCsrView &AT,
    HPRLPWindowedStencilShape *output) {
    if (output == nullptr || !hprlp_structured_detail::valid_csr(A) ||
        !hprlp_structured_detail::valid_csr(AT) ||
        A.rows != AT.cols || A.cols != AT.rows ||
        A.nonzeros != AT.nonzeros || A.rows < 8 || A.cols < 8) {
        return false;
    }

    int degree2_prefix = 0;
    while (degree2_prefix < A.rows &&
           A.row_ptr[degree2_prefix + 1] - A.row_ptr[degree2_prefix] == 2) {
        ++degree2_prefix;
    }
    if (degree2_prefix < 2 || (degree2_prefix & 1) != 0) return false;

    HPRLPWindowedStencilShape shape;
    shape.observation_count = degree2_prefix / 2;
    shape.stencil_row0 = degree2_prefix;
    shape.dense_col = A.col_index[A.row_ptr[0]];
    if (shape.dense_col != 0) return false;
    shape.grid_col0 = 1;
    shape.dense_negative = A.values[A.row_ptr[0]];
    shape.observation_coefficient = A.values[A.row_ptr[0] + 1];
    shape.dense_positive =
        A.values[A.row_ptr[shape.observation_count]];

    const int grid_variables = A.cols - 1;
    const int stencil_rows = A.rows - shape.stencil_row0;
    shape.grid_height = grid_variables - stencil_rows;
    if (shape.grid_height < 3 ||
        grid_variables % shape.grid_height != 0) return false;
    shape.grid_width = grid_variables / shape.grid_height;
    shape.equation_width = shape.grid_width - 1;
    if (shape.grid_width < 3 ||
        stencil_rows != shape.grid_height * shape.equation_width) {
        return false;
    }

    const int first_observation_col = A.col_index[A.row_ptr[0] + 1];
    if (first_observation_col < shape.grid_col0 ||
        first_observation_col >= A.cols) return false;
    const int first_grid_index = first_observation_col - shape.grid_col0;
    shape.observation_y0 = first_grid_index / shape.grid_width;
    shape.observation_x0 = first_grid_index % shape.grid_width;
    shape.observation_width = 1;
    while (shape.observation_width < shape.observation_count) {
        const int previous = A.col_index[
            A.row_ptr[shape.observation_width - 1] + 1];
        const int current =
            A.col_index[A.row_ptr[shape.observation_width] + 1];
        if (current != previous + 1) break;
        ++shape.observation_width;
    }
    if (shape.observation_count % shape.observation_width != 0) {
        return false;
    }
    shape.observation_height =
        shape.observation_count / shape.observation_width;
    if (shape.observation_y0 + shape.observation_height >
            shape.grid_height ||
        shape.observation_x0 + shape.observation_width >
            shape.grid_width) {
        return false;
    }

    for (int observation = 0;
         observation < shape.observation_count; ++observation) {
        const int oy = observation / shape.observation_width;
        const int ox = observation % shape.observation_width;
        const int expected_col = shape.grid_col0 +
            shape.grid_width * (shape.observation_y0 + oy) +
            shape.observation_x0 + ox;
        const int negative_begin = A.row_ptr[observation];
        const int positive_row = shape.observation_count + observation;
        const int positive_begin = A.row_ptr[positive_row];
        if (A.col_index[negative_begin] != shape.dense_col ||
            A.col_index[negative_begin + 1] != expected_col ||
            A.values[negative_begin] != shape.dense_negative ||
            A.values[negative_begin + 1] !=
                shape.observation_coefficient ||
            A.col_index[positive_begin] != shape.dense_col ||
            A.col_index[positive_begin + 1] != expected_col ||
            A.values[positive_begin] != shape.dense_positive ||
            A.values[positive_begin + 1] !=
                shape.observation_coefficient) {
            return false;
        }
    }

    const auto diagonal_at = [&](int gy, int gx, double *value) {
        const int row = shape.stencil_row0 +
            shape.equation_width * gy + gx;
        const int center = shape.grid_col0 + shape.grid_width * gy + gx;
        for (int entry = A.row_ptr[row]; entry < A.row_ptr[row + 1]; ++entry) {
            if (A.col_index[entry] == center) {
                *value = A.values[entry];
                return true;
            }
        }
        return false;
    };
    const int representative_row =
        shape.stencil_row0 + shape.equation_width + 1;
    shape.off_diagonal = A.values[A.row_ptr[representative_row]];
    if (!diagonal_at(0, 0, &shape.diagonal_boundary_left) ||
        !diagonal_at(0, 1, &shape.diagonal_boundary) ||
        !diagonal_at(1, 0, &shape.diagonal_interior_left) ||
        !diagonal_at(1, 1, &shape.diagonal_interior)) {
        return false;
    }

    for (int gy = 0; gy < shape.grid_height; ++gy) {
        for (int gx = 0; gx < shape.equation_width; ++gx) {
            const int row = shape.stencil_row0 +
                shape.equation_width * gy + gx;
            const int center =
                shape.grid_col0 + shape.grid_width * gy + gx;
            int entry = A.row_ptr[row];
            if (gy > 0) {
                if (A.col_index[entry] != center - shape.grid_width ||
                    A.values[entry++] != shape.off_diagonal) return false;
            }
            if (gx > 0) {
                if (A.col_index[entry] != center - 1 ||
                    A.values[entry++] != shape.off_diagonal) return false;
            }
            if (A.col_index[entry] != center ||
                A.values[entry++] !=
                    hprlp_windowed_stencil_diagonal(shape, gy, gx)) {
                return false;
            }
            if (A.col_index[entry] != center + 1 ||
                A.values[entry++] != shape.off_diagonal) return false;
            if (gy < shape.grid_height - 1) {
                if (A.col_index[entry] != center + shape.grid_width ||
                    A.values[entry++] != shape.off_diagonal) return false;
            }
            if (entry != A.row_ptr[row + 1]) return false;
        }
    }

    int dense_entry = AT.row_ptr[shape.dense_col];
    if (AT.row_ptr[shape.dense_col + 1] - dense_entry !=
        2 * shape.observation_count) return false;
    for (int observation = 0;
         observation < shape.observation_count; ++observation) {
        if (AT.col_index[dense_entry] != observation ||
            AT.values[dense_entry++] != shape.dense_negative) return false;
    }
    for (int observation = 0;
         observation < shape.observation_count; ++observation) {
        if (AT.col_index[dense_entry] !=
                shape.observation_count + observation ||
            AT.values[dense_entry++] != shape.dense_positive) return false;
    }

    for (int gy = 0; gy < shape.grid_height; ++gy) {
        for (int gx = 0; gx < shape.grid_width; ++gx) {
            const int col = shape.grid_col0 + shape.grid_width * gy + gx;
            int entry = AT.row_ptr[col];
            const int end = AT.row_ptr[col + 1];
            const auto expect = [&](int row, double value) {
                if (entry >= end || AT.col_index[entry] != row ||
                    AT.values[entry] != value) return false;
                ++entry;
                return true;
            };
            const bool observed =
                gy >= shape.observation_y0 &&
                gy < shape.observation_y0 + shape.observation_height &&
                gx >= shape.observation_x0 &&
                gx < shape.observation_x0 + shape.observation_width;
            if (observed) {
                const int observation = shape.observation_width *
                    (gy - shape.observation_y0) +
                    gx - shape.observation_x0;
                if (!expect(observation, shape.observation_coefficient) ||
                    !expect(shape.observation_count + observation,
                            shape.observation_coefficient)) return false;
            }
            if (gy > 0 && gx < shape.equation_width &&
                !expect(shape.stencil_row0 +
                            shape.equation_width * (gy - 1) + gx,
                        shape.off_diagonal)) return false;
            if (gx > 0 &&
                !expect(shape.stencil_row0 +
                            shape.equation_width * gy + gx - 1,
                        shape.off_diagonal)) return false;
            if (gx < shape.equation_width &&
                !expect(shape.stencil_row0 +
                            shape.equation_width * gy + gx,
                        hprlp_windowed_stencil_diagonal(shape, gy, gx))) {
                return false;
            }
            if (gx < shape.equation_width - 1 &&
                !expect(shape.stencil_row0 +
                            shape.equation_width * gy + gx + 1,
                        shape.off_diagonal)) return false;
            if (gy < shape.grid_height - 1 &&
                gx < shape.equation_width &&
                !expect(shape.stencil_row0 +
                            shape.equation_width * (gy + 1) + gx,
                        shape.off_diagonal)) return false;
            if (entry != end) return false;
        }
    }

    *output = shape;
    return true;
}

#endif
