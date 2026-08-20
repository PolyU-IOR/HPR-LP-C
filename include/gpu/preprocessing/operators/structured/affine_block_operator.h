#ifndef HPRLP_AFFINE_BLOCK_OPERATOR_H
#define HPRLP_AFFINE_BLOCK_OPERATOR_H

#include "gpu/preprocessing/policies/row_bucket_policy.h"
#include "gpu/preprocessing/operators/common/structured_operator_encoding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// A lossless compiler for a broad class of structured sparse matrices.
// Within each encoded run, every row has the same degree and coefficients,
// while each column position is an affine function of the local row number.
// The detector depends only on CSR contents; it has no model-name or
// dimension-specific rules.
constexpr int HPRLP_AFFINE_BLOCK_MAX_DEGREE = 16;
constexpr int HPRLP_AFFINE_BLOCK_MIN_RUN_ROWS = 64;
constexpr int HPRLP_AFFINE_BLOCK_MAX_RUN_ROWS = 512;
constexpr int HPRLP_AFFINE_BLOCK_MIN_ROWS = 4096;

struct HPRLPAffineBlockHost {
    int row_count = 0;
    int encoded_row_count = 0;
    int maximum_row_degree = 0;
    std::vector<int> block_row_begin;
    std::vector<int> block_row_count;
    std::vector<int> block_entry_ptr;
    std::vector<int> entry_base_columns;
    std::vector<int> entry_column_strides;
    std::vector<double> entry_values;
    std::vector<int> fallback_short_rows;
    std::vector<int> fallback_warp_rows;
    std::vector<int> fallback_block_rows;
    std::vector<int> fallback_row_ptr;
    std::vector<int> fallback_col_indices;
    std::vector<double> fallback_values;
};

namespace hprlp_affine_block_detail {

inline bool raw_equal(double left, double right) {
    std::uint64_t left_bits = 0;
    std::uint64_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

inline bool rows_define_affine_step(const HPRLPHostCsrView &matrix,
                                    int first, int second, int degree,
                                    std::vector<int> *strides) {
    if (second >= matrix.rows ||
        matrix.row_ptr[second + 1] - matrix.row_ptr[second] != degree) {
        return false;
    }
    strides->resize(static_cast<std::size_t>(degree));
    const int first_entry = matrix.row_ptr[first];
    const int second_entry = matrix.row_ptr[second];
    for (int position = 0; position < degree; ++position) {
        if (!raw_equal(matrix.values[first_entry + position],
                       matrix.values[second_entry + position])) {
            return false;
        }
        (*strides)[static_cast<std::size_t>(position)] =
            matrix.col_index[second_entry + position] -
            matrix.col_index[first_entry + position];
    }
    return true;
}

inline bool row_matches_affine(const HPRLPHostCsrView &matrix, int first,
                               int row, int degree,
                               const std::vector<int> &strides) {
    if (matrix.row_ptr[row + 1] - matrix.row_ptr[row] != degree) {
        return false;
    }
    const int first_entry = matrix.row_ptr[first];
    const int row_entry = matrix.row_ptr[row];
    const long long local = static_cast<long long>(row - first);
    for (int position = 0; position < degree; ++position) {
        if (!raw_equal(matrix.values[first_entry + position],
                       matrix.values[row_entry + position])) {
            return false;
        }
        const long long reconstructed =
            static_cast<long long>(matrix.col_index[first_entry + position]) +
            local * static_cast<long long>(
                        strides[static_cast<std::size_t>(position)]);
        if (reconstructed != matrix.col_index[row_entry + position]) {
            return false;
        }
    }
    return true;
}

}  // namespace hprlp_affine_block_detail

inline bool hprlp_build_affine_block_operator(
    const HPRLPHostCsrView &matrix, HPRLPAffineBlockHost *output) {
    using namespace hprlp_affine_block_detail;
    if (output == nullptr || !hprlp_structured_detail::valid_csr(matrix)) {
        return false;
    }

    HPRLPAffineBlockHost candidate;
    candidate.row_count = matrix.rows;
    std::vector<unsigned char> encoded(static_cast<std::size_t>(matrix.rows),
                                       0);
    candidate.block_entry_ptr.push_back(0);

    int first = 0;
    while (first + HPRLP_AFFINE_BLOCK_MIN_RUN_ROWS <= matrix.rows) {
        const int degree = matrix.row_ptr[first + 1] - matrix.row_ptr[first];
        candidate.maximum_row_degree =
            std::max(candidate.maximum_row_degree, degree);
        if (degree <= 0 || degree > HPRLP_AFFINE_BLOCK_MAX_DEGREE) {
            ++first;
            continue;
        }

        std::vector<int> strides;
        if (!rows_define_affine_step(matrix, first, first + 1, degree,
                                     &strides)) {
            ++first;
            continue;
        }
        int end = first + 2;
        while (end < matrix.rows &&
               row_matches_affine(matrix, first, end, degree, strides)) {
            ++end;
        }
        if (end - first < HPRLP_AFFINE_BLOCK_MIN_RUN_ROWS) {
            ++first;
            continue;
        }

        int block_begin = first;
        while (block_begin < end) {
            const int block_rows = std::min(
                HPRLP_AFFINE_BLOCK_MAX_RUN_ROWS, end - block_begin);
            // A small tail shares the same affine law, so retain it rather
            // than creating CSR fallback work after a certified long run.
            candidate.block_row_begin.push_back(block_begin);
            candidate.block_row_count.push_back(block_rows);
            const int first_entry = matrix.row_ptr[block_begin];
            for (int position = 0; position < degree; ++position) {
                candidate.entry_base_columns.push_back(
                    matrix.col_index[first_entry + position]);
                candidate.entry_column_strides.push_back(
                    strides[static_cast<std::size_t>(position)]);
                candidate.entry_values.push_back(
                    matrix.values[first_entry + position]);
            }
            candidate.block_entry_ptr.push_back(
                static_cast<int>(candidate.entry_values.size()));
            for (int row = block_begin; row < block_begin + block_rows;
                 ++row) {
                encoded[static_cast<std::size_t>(row)] = 1;
            }
            candidate.encoded_row_count += block_rows;
            block_begin += block_rows;
        }
        first = end;
    }
    for (; first < matrix.rows; ++first) {
        candidate.maximum_row_degree = std::max(
            candidate.maximum_row_degree,
            matrix.row_ptr[first + 1] - matrix.row_ptr[first]);
    }

    // Re-verify every reconstructed entry before enabling the representation.
    for (std::size_t block = 0; block < candidate.block_row_begin.size();
         ++block) {
        const int row_begin = candidate.block_row_begin[block];
        const int row_count = candidate.block_row_count[block];
        const int entry_begin = candidate.block_entry_ptr[block];
        const int entry_end = candidate.block_entry_ptr[block + 1];
        const int degree = entry_end - entry_begin;
        for (int local = 0; local < row_count; ++local) {
            const int row = row_begin + local;
            if (matrix.row_ptr[row + 1] - matrix.row_ptr[row] != degree) {
                return false;
            }
            for (int position = 0; position < degree; ++position) {
                const int entry = matrix.row_ptr[row] + position;
                const long long reconstructed =
                    static_cast<long long>(candidate.entry_base_columns[
                        static_cast<std::size_t>(entry_begin + position)]) +
                    static_cast<long long>(local) *
                        candidate.entry_column_strides[
                            static_cast<std::size_t>(entry_begin + position)];
                if (reconstructed != matrix.col_index[entry] ||
                    !raw_equal(candidate.entry_values[
                                   static_cast<std::size_t>(entry_begin +
                                                            position)],
                               matrix.values[entry])) {
                    return false;
                }
            }
        }
    }

    const int minimum_coverage_rows = std::max(
        HPRLP_AFFINE_BLOCK_MIN_ROWS, (2 * matrix.rows + 4) / 5);
    if (candidate.block_row_begin.empty() ||
        candidate.encoded_row_count < minimum_coverage_rows) {
        return false;
    }

    candidate.fallback_row_ptr.resize(
        static_cast<std::size_t>(matrix.rows) + 1);
    for (int row = 0; row < matrix.rows; ++row) {
        candidate.fallback_row_ptr[static_cast<std::size_t>(row)] =
            static_cast<int>(candidate.fallback_col_indices.size());
        if (encoded[static_cast<std::size_t>(row)] != 0) continue;
        const int degree = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
        switch (hprlp_row_bucket(degree)) {
            case HPRLP_ROW_SCALAR:
                candidate.fallback_short_rows.push_back(row);
                break;
            case HPRLP_ROW_WARP:
                candidate.fallback_warp_rows.push_back(row);
                break;
            case HPRLP_ROW_BLOCK:
                candidate.fallback_block_rows.push_back(row);
                break;
        }
        for (int entry = matrix.row_ptr[row];
             entry < matrix.row_ptr[row + 1]; ++entry) {
            candidate.fallback_col_indices.push_back(matrix.col_index[entry]);
            candidate.fallback_values.push_back(matrix.values[entry]);
        }
    }
    candidate.fallback_row_ptr[static_cast<std::size_t>(matrix.rows)] =
        static_cast<int>(candidate.fallback_col_indices.size());

    *output = std::move(candidate);
    return true;
}

#endif
