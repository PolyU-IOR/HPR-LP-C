#ifndef HPRLP_PACKED_DICTIONARY_OPERATOR_H
#define HPRLP_PACKED_DICTIONARY_OPERATOR_H

#include "api/structs.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

constexpr std::size_t HPRLP_PACKED_DICTIONARY_MAX_VALUES = 65535;

struct HPRLPPackedDictionaryHost {
    HPRLPPackedDictionaryStorage storage =
        HPRLPPackedDictionaryStorage::None;
    bool cap_exceeded = false;
    unsigned index_bits = 0;
    unsigned code_bits = 0;
    std::vector<HPRLP_FLOAT> dictionary;
    std::vector<std::uint32_t> packed_u32;
    std::vector<std::uint16_t> indices_u16;
    std::vector<std::uint32_t> indices_u32;
    std::vector<std::uint8_t> codes_u8;
    std::vector<std::uint16_t> codes_u16;
};

inline std::uint64_t hprlp_fp64_bits(HPRLP_FLOAT value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "FP64 bit width mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline unsigned hprlp_bits_for_count(std::size_t count) {
    if (count <= 1) return 0;
    unsigned bits = 0;
    std::size_t maximum = count - 1;
    while (maximum != 0) {
        ++bits;
        maximum >>= 1;
    }
    return bits;
}

inline const char *hprlp_packed_dictionary_storage_name(
    HPRLPPackedDictionaryStorage storage) {
    switch (storage) {
        case HPRLPPackedDictionaryStorage::None: return "none";
        case HPRLPPackedDictionaryStorage::PackedU32: return "packed-u32";
        case HPRLPPackedDictionaryStorage::SeparateU16U8:
            return "u16-index+u8-code";
        case HPRLPPackedDictionaryStorage::SeparateU16U16:
            return "u16-index+u16-code";
        case HPRLPPackedDictionaryStorage::SeparateU32U8:
            return "u32-index+u8-code";
        case HPRLPPackedDictionaryStorage::SeparateU32U16:
            return "u32-index+u16-code";
    }
    return "unknown";
}

inline bool hprlp_build_packed_dictionary_operator(
    const HPRLP_FLOAT *values, const int *input_indices,
    std::size_t nonzeros, std::size_t input_count,
    HPRLPPackedDictionaryHost *encoding) {
    if (encoding == nullptr || values == nullptr || input_indices == nullptr ||
        nonzeros == 0 || input_count == 0) {
        return false;
    }
    *encoding = HPRLPPackedDictionaryHost{};
    encoding->index_bits = hprlp_bits_for_count(input_count);

    std::unordered_map<std::uint64_t, std::uint32_t> dictionary_codes;
    dictionary_codes.reserve(
        nonzeros < HPRLP_PACKED_DICTIONARY_MAX_VALUES
            ? nonzeros
            : HPRLP_PACKED_DICTIONARY_MAX_VALUES);
    encoding->dictionary.reserve(
        nonzeros < HPRLP_PACKED_DICTIONARY_MAX_VALUES
            ? nonzeros
            : HPRLP_PACKED_DICTIONARY_MAX_VALUES);
    for (std::size_t entry = 0; entry < nonzeros; ++entry) {
        const std::uint64_t key = hprlp_fp64_bits(values[entry]);
        if (dictionary_codes.find(key) != dictionary_codes.end()) continue;
        if (encoding->dictionary.size() ==
            HPRLP_PACKED_DICTIONARY_MAX_VALUES) {
            encoding->cap_exceeded = true;
            encoding->code_bits = 17;
            encoding->storage = HPRLPPackedDictionaryStorage::None;
            return false;
        }
        const std::uint32_t code = static_cast<std::uint32_t>(
            encoding->dictionary.size());
        dictionary_codes.emplace(key, code);
        encoding->dictionary.push_back(values[entry]);
    }

    encoding->code_bits = hprlp_bits_for_count(encoding->dictionary.size());
    const bool compact_index = input_count <= 65536;
    const bool compact_code = encoding->dictionary.size() <= 256;
    if (encoding->index_bits + encoding->code_bits <= 32) {
        encoding->storage = HPRLPPackedDictionaryStorage::PackedU32;
        encoding->packed_u32.resize(nonzeros);
    } else if (compact_index && compact_code) {
        encoding->storage = HPRLPPackedDictionaryStorage::SeparateU16U8;
        encoding->indices_u16.resize(nonzeros);
        encoding->codes_u8.resize(nonzeros);
    } else if (compact_index) {
        encoding->storage = HPRLPPackedDictionaryStorage::SeparateU16U16;
        encoding->indices_u16.resize(nonzeros);
        encoding->codes_u16.resize(nonzeros);
    } else if (compact_code) {
        encoding->storage = HPRLPPackedDictionaryStorage::SeparateU32U8;
        encoding->indices_u32.resize(nonzeros);
        encoding->codes_u8.resize(nonzeros);
    } else {
        encoding->storage = HPRLPPackedDictionaryStorage::SeparateU32U16;
        encoding->indices_u32.resize(nonzeros);
        encoding->codes_u16.resize(nonzeros);
    }

    const std::uint32_t code_mask = encoding->code_bits == 0
        ? 0u
        : ((UINT32_C(1) << encoding->code_bits) - 1u);
    for (std::size_t entry = 0; entry < nonzeros; ++entry) {
        const int signed_index = input_indices[entry];
        if (signed_index < 0 ||
            static_cast<std::size_t>(signed_index) >= input_count) {
            *encoding = HPRLPPackedDictionaryHost{};
            return false;
        }
        const std::uint32_t input_index =
            static_cast<std::uint32_t>(signed_index);
        const auto found = dictionary_codes.find(hprlp_fp64_bits(values[entry]));
        if (found == dictionary_codes.end()) {
            *encoding = HPRLPPackedDictionaryHost{};
            return false;
        }
        const std::uint32_t code = found->second;
        switch (encoding->storage) {
            case HPRLPPackedDictionaryStorage::PackedU32:
                encoding->packed_u32[entry] =
                    (input_index << encoding->code_bits) |
                    (code & code_mask);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU16U8:
                encoding->indices_u16[entry] =
                    static_cast<std::uint16_t>(input_index);
                encoding->codes_u8[entry] = static_cast<std::uint8_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU16U16:
                encoding->indices_u16[entry] =
                    static_cast<std::uint16_t>(input_index);
                encoding->codes_u16[entry] = static_cast<std::uint16_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU32U8:
                encoding->indices_u32[entry] = input_index;
                encoding->codes_u8[entry] = static_cast<std::uint8_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU32U16:
                encoding->indices_u32[entry] = input_index;
                encoding->codes_u16[entry] = static_cast<std::uint16_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::None:
                return false;
        }
    }
    return true;
}

#endif
