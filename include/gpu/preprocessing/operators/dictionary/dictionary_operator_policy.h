#ifndef HPRLP_DICTIONARY_OPERATOR_POLICY_H
#define HPRLP_DICTIONARY_OPERATOR_POLICY_H

#include <cstddef>

constexpr std::size_t HPRLP_DICTIONARY_COMPACT_MAX_VALUES = 256;
constexpr int HPRLP_DICTIONARY_BALANCED_MAX_ROW_NNZ = 4096;

inline bool hprlp_use_dictionary_operator_x(
    bool has_exact_original_dictionary,
    bool all_positive_unit_coefficients,
    bool packed_u32_feasible,
    std::size_t constraint_count,
    std::size_t nonzeros,
    std::size_t dictionary_size,
    int maximum_forward_row_nnz,
    int maximum_transpose_row_nnz) {
    // Packed-u32 feasibility already certifies that the exact column index
    // and exact coefficient code fit in one 32-bit entry.  A separate
    // coefficient-count cap is neither a correctness requirement nor a
    // useful structural criterion; runtime autotuning decides whether the
    // compressed kernel is profitable for the detected matrix.
    const bool basic_eligibility =
        has_exact_original_dictionary && !all_positive_unit_coefficients &&
        packed_u32_feasible && nonzeros > 0 && constraint_count > 0 &&
        dictionary_size > 0;
    const bool bounded_reduction_depth =
        maximum_forward_row_nnz >= 0 && maximum_transpose_row_nnz >= 0 &&
        maximum_forward_row_nnz <= HPRLP_DICTIONARY_BALANCED_MAX_ROW_NNZ &&
        maximum_transpose_row_nnz <= HPRLP_DICTIONARY_BALANCED_MAX_ROW_NNZ;
    // Keep the already validated low-cardinality region.  Broader exact
    // dictionaries are admitted only when neither orientation contains an
    // extreme reduction, which is a conservative numerical-stability guard.
    return basic_eligibility &&
           (dictionary_size <= HPRLP_DICTIONARY_COMPACT_MAX_VALUES ||
            bounded_reduction_depth);
}

#endif
