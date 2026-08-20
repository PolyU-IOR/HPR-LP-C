#ifndef HPRLP_ROW_TEMPLATE_OPERATOR_H
#define HPRLP_ROW_TEMPLATE_OPERATOR_H

#include "gpu/preprocessing/operators/common/structured_operator_encoding.h"
#include "gpu/preprocessing/policies/row_bucket_policy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

constexpr std::uint8_t HPRLP_ROW_TEMPLATE_FALLBACK = 255;
constexpr int HPRLP_ROW_TEMPLATE_MAX_DEGREE = 16;
constexpr int HPRLP_ROW_TEMPLATE_MAX_TEMPLATES = 254;
constexpr int HPRLP_ROW_TEMPLATE_MIN_REUSE = 128;
constexpr int HPRLP_ROW_TEMPLATE_MIN_ROWS = 4096;

struct HPRLPRowTemplateHost {
    int row_count = 0;
    int template_count = 0;
    int encoded_row_count = 0;
    int maximum_row_degree = 0;
    std::vector<std::uint8_t> row_template_ids;
    std::vector<int> row_bases;
    std::vector<int> template_ptr;
    std::vector<int> template_offsets;
    std::vector<double> template_values;
    std::vector<int> fallback_short_rows;
    std::vector<int> fallback_warp_rows;
    std::vector<int> fallback_block_rows;
    std::vector<int> fallback_row_ptr;
    std::vector<int> fallback_col_indices;
    std::vector<double> fallback_values;
};

namespace hprlp_row_template_detail {

using Signature = std::vector<std::uint64_t>;

inline std::uint64_t raw_bits(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline Signature make_signature(const HPRLPHostCsrView &matrix, int row) {
    const int begin = matrix.row_ptr[row];
    const int end = matrix.row_ptr[row + 1];
    const int degree = end - begin;
    Signature signature;
    signature.reserve(1 + 2 * static_cast<std::size_t>(degree));
    signature.push_back(static_cast<std::uint64_t>(degree));
    if (degree == 0) return signature;
    const int base = matrix.col_index[begin];
    for (int entry = begin; entry < end; ++entry) {
        const int offset = matrix.col_index[entry] - base;
        signature.push_back(static_cast<std::uint64_t>(
            static_cast<std::uint32_t>(offset)));
        signature.push_back(raw_bits(matrix.values[entry]));
    }
    return signature;
}

}  // namespace hprlp_row_template_detail

// Build a lossless, row-local encoding. Eligibility depends only on repeated
// row structure: degree, relative column offsets, and exact binary64 values.
// Rows without a sufficiently reused template retain the canonical CSR path.
inline bool hprlp_build_row_template_operator(
    const HPRLPHostCsrView &matrix,
    HPRLPRowTemplateHost *output) {
    using namespace hprlp_row_template_detail;
    if (output == nullptr || !hprlp_structured_detail::valid_csr(matrix)) {
        return false;
    }

    HPRLPRowTemplateHost candidate;
    candidate.row_count = matrix.rows;
    candidate.row_template_ids.assign(
        static_cast<std::size_t>(matrix.rows), HPRLP_ROW_TEMPLATE_FALLBACK);
    candidate.row_bases.assign(static_cast<std::size_t>(matrix.rows), 0);

    std::map<Signature, int> counts;
    std::vector<Signature> row_signatures(static_cast<std::size_t>(matrix.rows));
    for (int row = 0; row < matrix.rows; ++row) {
        const int degree = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
        candidate.maximum_row_degree =
            std::max(candidate.maximum_row_degree, degree);
        if (degree <= 0 || degree > HPRLP_ROW_TEMPLATE_MAX_DEGREE) continue;
        row_signatures[static_cast<std::size_t>(row)] =
            make_signature(matrix, row);
        ++counts[row_signatures[static_cast<std::size_t>(row)]];
    }
    std::vector<std::pair<int, Signature> > reusable;
    reusable.reserve(counts.size());
    for (std::map<Signature, int>::const_iterator it = counts.begin();
         it != counts.end(); ++it) {
        if (it->second >= HPRLP_ROW_TEMPLATE_MIN_REUSE) {
            reusable.push_back(std::make_pair(it->second, it->first));
        }
    }
    std::sort(reusable.begin(), reusable.end(),
              [](const std::pair<int, Signature> &left,
                 const std::pair<int, Signature> &right) {
                  if (left.first != right.first) return left.first > right.first;
                  return left.second < right.second;
              });
    if (reusable.size() > HPRLP_ROW_TEMPLATE_MAX_TEMPLATES) {
        reusable.resize(HPRLP_ROW_TEMPLATE_MAX_TEMPLATES);
    }

    std::map<Signature, std::uint8_t> template_ids;
    candidate.template_ptr.push_back(0);
    for (std::size_t template_index = 0;
         template_index < reusable.size(); ++template_index) {
        const Signature &signature = reusable[template_index].second;
        const std::uint8_t id = static_cast<std::uint8_t>(template_index);
        template_ids.emplace(signature, id);
        const int degree = static_cast<int>(signature[0]);
        for (int position = 0; position < degree; ++position) {
            candidate.template_offsets.push_back(static_cast<int>(
                static_cast<std::uint32_t>(signature[1 + 2 * position])));
            const std::uint64_t bits = signature[2 + 2 * position];
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            candidate.template_values.push_back(value);
        }
        candidate.template_ptr.push_back(
            static_cast<int>(candidate.template_offsets.size()));
    }

    for (int row = 0; row < matrix.rows; ++row) {
        const Signature &signature = row_signatures[static_cast<std::size_t>(row)];
        if (signature.empty()) continue;
        const std::map<Signature, std::uint8_t>::const_iterator found =
            template_ids.find(signature);
        if (found == template_ids.end()) continue;
        candidate.row_template_ids[static_cast<std::size_t>(row)] = found->second;
        candidate.row_bases[static_cast<std::size_t>(row)] =
            matrix.col_index[matrix.row_ptr[row]];
        ++candidate.encoded_row_count;
    }
    candidate.fallback_row_ptr.resize(
        static_cast<std::size_t>(matrix.rows) + 1);
    for (int row = 0; row < matrix.rows; ++row) {
        candidate.fallback_row_ptr[static_cast<std::size_t>(row)] =
            static_cast<int>(candidate.fallback_col_indices.size());
        if (candidate.row_template_ids[static_cast<std::size_t>(row)] !=
            HPRLP_ROW_TEMPLATE_FALLBACK) {
            continue;
        }
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
            candidate.fallback_col_indices.push_back(
                matrix.col_index[entry]);
            candidate.fallback_values.push_back(matrix.values[entry]);
        }
    }
    candidate.fallback_row_ptr[static_cast<std::size_t>(matrix.rows)] =
        static_cast<int>(candidate.fallback_col_indices.size());
    candidate.template_count = static_cast<int>(reusable.size());

    const int minimum_coverage_rows = std::max(
        HPRLP_ROW_TEMPLATE_MIN_ROWS, (2 * matrix.rows + 4) / 5);
    if (candidate.template_count == 0 ||
        candidate.encoded_row_count < minimum_coverage_rows) {
        return false;
    }
    *output = std::move(candidate);
    return true;
}

#endif
