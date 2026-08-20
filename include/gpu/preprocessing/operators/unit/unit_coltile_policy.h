#ifndef HPRLP_UNIT_COLTILE_POLICY_H
#define HPRLP_UNIT_COLTILE_POLICY_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

constexpr int HPRLP_UNIT_COLTILE_COLUMNS = 65536;
constexpr std::size_t HPRLP_UNIT_COLTILE_MIN_NONZEROS = 1u << 20;

struct HPRLPUnitColTileHost {
    int rows = 0;
    int columns = 0;
    int tile_cols = HPRLP_UNIT_COLTILE_COLUMNS;
    int tile_count = 0;
    std::vector<int> row_tile_offsets;
    std::vector<std::uint16_t> local_cols;
};

inline bool hprlp_use_unit_coltile(int uniform_unit_sign,
                                  std::size_t row_count,
                                  std::size_t column_count,
                                  std::size_t nonzeros) {
    if (uniform_unit_sign == 0 || row_count == 0 ||
        column_count <= HPRLP_UNIT_COLTILE_COLUMNS ||
        nonzeros < HPRLP_UNIT_COLTILE_MIN_NONZEROS) {
        return false;
    }
    const std::size_t tile_count =
        (column_count + HPRLP_UNIT_COLTILE_COLUMNS - 1) /
        HPRLP_UNIT_COLTILE_COLUMNS;
    return row_count <=
        (static_cast<std::size_t>(std::numeric_limits<int>::max()) - 1) /
            tile_count;
}

inline bool hprlp_build_unit_coltile(
    int rows, int columns, int nonzeros, const int *row_ptr,
    const int *col_index, HPRLPUnitColTileHost *output) {
    if (output == nullptr || rows <= 0 ||
        columns <= HPRLP_UNIT_COLTILE_COLUMNS || nonzeros <= 0 ||
        row_ptr == nullptr || col_index == nullptr || row_ptr[0] != 0 ||
        row_ptr[rows] != nonzeros) {
        return false;
    }
    HPRLPUnitColTileHost candidate;
    candidate.rows = rows;
    candidate.columns = columns;
    candidate.tile_count =
        static_cast<int>((static_cast<long long>(columns) +
                          candidate.tile_cols - 1) /
                         candidate.tile_cols);
    const long long pair_count =
        static_cast<long long>(rows) * candidate.tile_count;
    if (pair_count >= std::numeric_limits<int>::max()) {
        return false;
    }
    candidate.row_tile_offsets.assign(
        static_cast<std::size_t>(pair_count) + 1, 0);
    for (int row = 0; row < rows; ++row) {
        if (row_ptr[row] > row_ptr[row + 1]) {
            return false;
        }
        const long long row_base =
            static_cast<long long>(row) * candidate.tile_count;
        for (int index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
            const int column = col_index[index];
            if (column < 0 || column >= columns) {
                return false;
            }
            const int tile = column / candidate.tile_cols;
            ++candidate.row_tile_offsets[
                static_cast<std::size_t>(row_base + tile + 1)];
        }
    }
    for (std::size_t index = 1;
         index < candidate.row_tile_offsets.size(); ++index) {
        candidate.row_tile_offsets[index] +=
            candidate.row_tile_offsets[index - 1];
    }

    candidate.local_cols.resize(static_cast<std::size_t>(nonzeros));
    std::vector<int> cursor = candidate.row_tile_offsets;
    for (int row = 0; row < rows; ++row) {
        const long long row_base =
            static_cast<long long>(row) * candidate.tile_count;
        for (int index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
            const int column = col_index[index];
            const int tile = column / candidate.tile_cols;
            const int destination = cursor[
                static_cast<std::size_t>(row_base + tile)]++;
            candidate.local_cols[static_cast<std::size_t>(destination)] =
                static_cast<std::uint16_t>(
                    column - tile * candidate.tile_cols);
        }
    }
    for (long long pair = 0; pair < pair_count; ++pair) {
        const int begin = candidate.row_tile_offsets[
            static_cast<std::size_t>(pair)];
        const int end = candidate.row_tile_offsets[
            static_cast<std::size_t>(pair + 1)];
        std::sort(candidate.local_cols.begin() + begin,
                  candidate.local_cols.begin() + end);
    }
    *output = std::move(candidate);
    return true;
}

#endif
