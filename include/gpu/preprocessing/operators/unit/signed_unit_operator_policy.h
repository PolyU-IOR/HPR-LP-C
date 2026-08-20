#ifndef HPRLP_SIGNED_UNIT_OPERATOR_POLICY_H
#define HPRLP_SIGNED_UNIT_OPERATOR_POLICY_H

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr std::uint16_t HPRLP_SIGNED_U16_SIGN_MASK = 0x8000u;
constexpr std::uint16_t HPRLP_SIGNED_U16_INDEX_MASK = 0x7fffu;
constexpr std::uint32_t HPRLP_SIGNED_U32_SIGN_MASK = 0x80000000u;
constexpr std::uint32_t HPRLP_SIGNED_U32_INDEX_MASK = 0x7fffffffu;

struct HPRLPSignedUnitPackedHost {
    bool uses_u16 = false;
    bool split_u16_ready = false;
    std::vector<std::uint16_t> entries_u16;
    std::vector<std::uint32_t> entries_u32;
    std::vector<std::uint16_t> split_indices_u16;
    std::vector<std::uint8_t> split_negative_u8;
};

struct HPRLPFixedDegreeRunHost {
    int degree = 0;
    int row_begin = 0;
    int row_count = 0;
    int entry_begin = 0;
};

inline HPRLPFixedDegreeRunHost hprlp_find_longest_fixed_degree_run(
    int row_count, const int *row_ptr, int degree) {
    HPRLPFixedDegreeRunHost best;
    if (row_count <= 0 || row_ptr == nullptr || degree <= 0) {
        return best;
    }

    int current_begin = 0;
    int current_count = 0;
    for (int row = 0; row < row_count; ++row) {
        const int row_degree = row_ptr[row + 1] - row_ptr[row];
        if (row_degree < 0) {
            return HPRLPFixedDegreeRunHost();
        }
        if (row_degree == degree) {
            if (current_count == 0) {
                current_begin = row;
            }
            ++current_count;
            if (current_count > best.row_count) {
                best.degree = degree;
                best.row_begin = current_begin;
                best.row_count = current_count;
                best.entry_begin = row_ptr[current_begin];
            }
        } else {
            current_count = 0;
        }
    }
    return best;
}

inline bool hprlp_use_fixed_degree_run(
    const HPRLPFixedDegreeRunHost &run, int total_rows) {
    // A large majority run amortizes the extra dispatch predicate while
    // avoiding two CSR row-pointer reads and a variable-trip loop per row.
    constexpr int kMinimumRunRows = 65536;
    return total_rows > 0 && run.degree > 0 &&
           run.row_begin >= 0 && run.row_count >= kMinimumRunRows &&
           static_cast<long long>(run.row_count) * 2 >= total_rows;
}

template <typename T>
inline bool hprlp_all_signed_unit(const T *values,
                                  std::size_t value_count) {
    if (values == nullptr || value_count == 0) {
        return false;
    }
    for (std::size_t index = 0; index < value_count; ++index) {
        if (values[index] != T(1) && values[index] != T(-1)) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline bool hprlp_mixed_signed_unit(const T *values,
                                    std::size_t value_count) {
    if (!hprlp_all_signed_unit(values, value_count)) {
        return false;
    }
    bool has_positive = false;
    bool has_negative = false;
    for (std::size_t index = 0; index < value_count; ++index) {
        has_positive = has_positive || values[index] == T(1);
        has_negative = has_negative || values[index] == T(-1);
    }
    return has_positive && has_negative;
}

inline bool hprlp_signed_index_uses_u16(int column_count) {
    return column_count > 0 && column_count <= 32768;
}

inline int hprlp_signed_u16_index(std::uint16_t entry) {
    return static_cast<int>(entry & HPRLP_SIGNED_U16_INDEX_MASK);
}

inline int hprlp_signed_u16_sign(std::uint16_t entry) {
    return (entry & HPRLP_SIGNED_U16_SIGN_MASK) == 0 ? 1 : -1;
}

inline int hprlp_signed_u32_index(std::uint32_t entry) {
    return static_cast<int>(entry & HPRLP_SIGNED_U32_INDEX_MASK);
}

inline int hprlp_signed_u32_sign(std::uint32_t entry) {
    return (entry & HPRLP_SIGNED_U32_SIGN_MASK) == 0 ? 1 : -1;
}

template <typename T>
inline bool hprlp_build_signed_unit_entries(
    int column_count, int nonzeros, const int *column_indices,
    const T *values, HPRLPSignedUnitPackedHost *packed) {
    if (packed == nullptr || column_count <= 0 || nonzeros <= 0 ||
        column_indices == nullptr ||
        !hprlp_all_signed_unit(values,
                               static_cast<std::size_t>(nonzeros))) {
        return false;
    }

    HPRLPSignedUnitPackedHost result;
    result.uses_u16 = hprlp_signed_index_uses_u16(column_count);
    result.split_u16_ready = column_count <= 65536;
    if (result.uses_u16) {
        result.entries_u16.resize(static_cast<std::size_t>(nonzeros));
    } else {
        result.entries_u32.resize(static_cast<std::size_t>(nonzeros));
    }
    if (result.split_u16_ready) {
        result.split_indices_u16.resize(static_cast<std::size_t>(nonzeros));
        result.split_negative_u8.resize(static_cast<std::size_t>(nonzeros));
    }

    for (int entry = 0; entry < nonzeros; ++entry) {
        const int column = column_indices[entry];
        if (column < 0 || column >= column_count) {
            return false;
        }
        const bool negative = values[entry] == T(-1);
        if (result.uses_u16) {
            result.entries_u16[entry] =
                static_cast<std::uint16_t>(column) |
                (negative ? HPRLP_SIGNED_U16_SIGN_MASK : 0u);
        } else {
            result.entries_u32[entry] =
                static_cast<std::uint32_t>(column) |
                (negative ? HPRLP_SIGNED_U32_SIGN_MASK : 0u);
        }
        if (result.split_u16_ready) {
            result.split_indices_u16[entry] =
                static_cast<std::uint16_t>(column);
            result.split_negative_u8[entry] = negative ? 1 : 0;
        }
    }

    *packed = result;
    return true;
}

#endif
