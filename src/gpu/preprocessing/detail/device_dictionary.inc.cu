namespace {

struct HPRLPValueToBits {
    __host__ __device__ std::uint64_t operator()(HPRLP_FLOAT value) const {
        union {
            HPRLP_FLOAT value;
            std::uint64_t bits;
        } converter;
        converter.value = value;
        return converter.bits;
    }
};

struct HPRLPBitsToValue {
    __host__ __device__ HPRLP_FLOAT operator()(std::uint64_t bits) const {
        union {
            HPRLP_FLOAT value;
            std::uint64_t bits;
        } converter;
        converter.bits = bits;
        return converter.value;
    }
};

__global__ void hprlp_encode_sorted_dictionary_kernel(
    const int *indices,
    const HPRLP_FLOAT *values,
    std::size_t nonzeros,
    std::size_t input_count,
    const std::uint64_t *lookup_bits,
    const std::uint32_t *lookup_codes,
    int dictionary_size,
    unsigned code_bits, HPRLPPackedDictionaryStorage storage,
    std::uint32_t *packed, std::uint16_t *indices_u16,
    std::uint32_t *indices_u32, std::uint8_t *codes_u8,
    std::uint16_t *codes_u16,
    int *invalid) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t entry =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         entry < nonzeros; entry += stride) {
        const int signed_index = indices[entry];
        if (signed_index < 0 ||
            static_cast<std::size_t>(signed_index) >= input_count) {
            atomicExch(invalid, 1);
            continue;
        }
        const std::uint64_t bits = HPRLPValueToBits{}(values[entry]);
        int left = 0;
        int right = dictionary_size;
        while (left < right) {
            const int middle = left + (right - left) / 2;
            if (lookup_bits[middle] < bits) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        if (left >= dictionary_size || lookup_bits[left] != bits) {
            atomicExch(invalid, 1);
            continue;
        }
        const std::uint32_t code = lookup_codes[left];
        const std::uint32_t code_mask = code_bits == 0
            ? 0u : ((UINT32_C(1) << code_bits) - 1u);
        const std::uint32_t input_index =
            static_cast<std::uint32_t>(signed_index);
        switch (storage) {
            case HPRLPPackedDictionaryStorage::PackedU32:
                packed[entry] = (input_index << code_bits) |
                    (code & code_mask);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU16U8:
                indices_u16[entry] = static_cast<std::uint16_t>(input_index);
                codes_u8[entry] = static_cast<std::uint8_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU16U16:
                indices_u16[entry] = static_cast<std::uint16_t>(input_index);
                codes_u16[entry] = static_cast<std::uint16_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU32U8:
                indices_u32[entry] = input_index;
                codes_u8[entry] = static_cast<std::uint8_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::SeparateU32U16:
                indices_u32[entry] = input_index;
                codes_u16[entry] = static_cast<std::uint16_t>(code);
                break;
            case HPRLPPackedDictionaryStorage::None:
                atomicExch(invalid, 1);
                break;
        }
    }
}

struct HPRLPDeviceDictionaryEncoding {
    int dictionary_size = 0;
    unsigned index_bits = 0;
    unsigned code_bits = 0;
    HPRLPPackedDictionaryStorage storage =
        HPRLPPackedDictionaryStorage::None;
    HPRLP_FLOAT *dictionary = nullptr;
    std::uint32_t *packed = nullptr;
    std::uint16_t *indices_u16 = nullptr;
    std::uint32_t *indices_u32 = nullptr;
    std::uint8_t *codes_u8 = nullptr;
    std::uint16_t *codes_u16 = nullptr;
};

void hprlp_release_device_dictionary_encoding(
    HPRLPDeviceDictionaryEncoding *encoding) {
    if (encoding == nullptr) return;
    hprlp_device_free(encoding->dictionary);
    hprlp_device_free(encoding->packed);
    hprlp_device_free(encoding->indices_u16);
    hprlp_device_free(encoding->indices_u32);
    hprlp_device_free(encoding->codes_u8);
    hprlp_device_free(encoding->codes_u16);
    *encoding = HPRLPDeviceDictionaryEncoding{};
}

bool hprlp_build_device_dictionary_encoding(
    const sparseMatrix *matrix,
    std::size_t input_count,
    HPRLPDeviceDictionaryEncoding *encoding) {
    if (matrix == nullptr || matrix->value == nullptr ||
        matrix->colIndex == nullptr || matrix->numElements <= 0 ||
        input_count == 0 || encoding == nullptr) {
        return false;
    }
    *encoding = HPRLPDeviceDictionaryEncoding{};
    const std::size_t nonzeros =
        static_cast<std::size_t>(matrix->numElements);

    // Stable sorting by coefficient bits keeps entry order within each value.
    // Reducing the original entry indices therefore recovers the first
    // occurrence of every distinct coefficient.  Sorting those first indices
    // recreates the baseline host dictionary order exactly.
    thrust::device_vector<std::uint64_t> sorted_bits(nonzeros);
    thrust::device_vector<std::uint32_t> sorted_entries(nonzeros);
    thrust::transform(
        thrust::device_pointer_cast(matrix->value),
        thrust::device_pointer_cast(matrix->value) + nonzeros,
        sorted_bits.begin(), HPRLPValueToBits{});
    thrust::sequence(
        sorted_entries.begin(), sorted_entries.end(), std::uint32_t{0});
    thrust::stable_sort_by_key(
        sorted_bits.begin(), sorted_bits.end(), sorted_entries.begin());

    thrust::device_vector<std::uint64_t> unique_bits(nonzeros);
    thrust::device_vector<std::uint32_t> first_entries(nonzeros);
    const auto unique_end = thrust::reduce_by_key(
        sorted_bits.begin(), sorted_bits.end(), sorted_entries.begin(),
        unique_bits.begin(), first_entries.begin(),
        thrust::equal_to<std::uint64_t>(),
        thrust::minimum<std::uint32_t>());
    const std::size_t dictionary_size = static_cast<std::size_t>(
        unique_end.first - unique_bits.begin());
    if (dictionary_size == 0 ||
        dictionary_size > HPRLP_PACKED_DICTIONARY_MAX_VALUES) {
        return false;
    }
    unique_bits.resize(dictionary_size);
    first_entries.resize(dictionary_size);
    thrust::sort_by_key(
        first_entries.begin(), first_entries.end(), unique_bits.begin());

    const unsigned code_bits = hprlp_bits_for_count(dictionary_size);
    const unsigned index_bits = hprlp_bits_for_count(input_count);
    const bool compact_index = input_count <= 65536;
    const bool compact_code = dictionary_size <= 256;
    HPRLPPackedDictionaryStorage storage =
        HPRLPPackedDictionaryStorage::None;
    if (index_bits + code_bits <= 32) {
        storage = HPRLPPackedDictionaryStorage::PackedU32;
    } else if (compact_index && compact_code) {
        storage = HPRLPPackedDictionaryStorage::SeparateU16U8;
    } else if (compact_index) {
        storage = HPRLPPackedDictionaryStorage::SeparateU16U16;
    } else if (compact_code) {
        storage = HPRLPPackedDictionaryStorage::SeparateU32U8;
    } else {
        storage = HPRLPPackedDictionaryStorage::SeparateU32U16;
    }

    thrust::device_vector<HPRLP_FLOAT> dictionary(dictionary_size);
    thrust::transform(
        unique_bits.begin(), unique_bits.end(), dictionary.begin(),
        HPRLPBitsToValue{});

    // The output dictionary remains in first-occurrence order, while a second
    // bit-sorted view supplies O(log d) code lookup to the parallel encoder.
    thrust::device_vector<std::uint64_t> lookup_bits = unique_bits;
    thrust::device_vector<std::uint32_t> lookup_codes(dictionary_size);
    thrust::sequence(
        lookup_codes.begin(), lookup_codes.end(), std::uint32_t{0});
    thrust::sort_by_key(
        lookup_bits.begin(), lookup_bits.end(), lookup_codes.begin());

    HPRLP_FLOAT *dictionary_out = nullptr;
    std::uint32_t *packed_out = nullptr;
    std::uint16_t *indices_u16_out = nullptr;
    std::uint32_t *indices_u32_out = nullptr;
    std::uint8_t *codes_u8_out = nullptr;
    std::uint16_t *codes_u16_out = nullptr;
    int *invalid_device = nullptr;
    try {
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &dictionary_out,
            dictionary_size * sizeof(HPRLP_FLOAT)));
        if (storage == HPRLPPackedDictionaryStorage::PackedU32) {
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &packed_out, nonzeros * sizeof(std::uint32_t)));
        } else if (storage ==
                       HPRLPPackedDictionaryStorage::SeparateU16U8 ||
                   storage ==
                       HPRLPPackedDictionaryStorage::SeparateU16U16) {
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &indices_u16_out, nonzeros * sizeof(std::uint16_t)));
        } else {
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &indices_u32_out, nonzeros * sizeof(std::uint32_t)));
        }
        if (storage == HPRLPPackedDictionaryStorage::SeparateU16U8 ||
            storage == HPRLPPackedDictionaryStorage::SeparateU32U8) {
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &codes_u8_out, nonzeros * sizeof(std::uint8_t)));
        } else if (storage != HPRLPPackedDictionaryStorage::PackedU32) {
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &codes_u16_out, nonzeros * sizeof(std::uint16_t)));
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&invalid_device), sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            dictionary_out, thrust::raw_pointer_cast(dictionary.data()),
            dictionary_size * sizeof(HPRLP_FLOAT),
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(invalid_device, 0, sizeof(int)));
        const int threads = 256;
        const int blocks = static_cast<int>(std::min<std::size_t>(
            65535, (nonzeros + threads - 1) / threads));
        hprlp_encode_sorted_dictionary_kernel<<<blocks, threads>>>(
            matrix->colIndex, matrix->value, nonzeros, input_count,
            thrust::raw_pointer_cast(lookup_bits.data()),
            thrust::raw_pointer_cast(lookup_codes.data()),
            static_cast<int>(dictionary_size), code_bits, storage, packed_out,
            indices_u16_out, indices_u32_out, codes_u8_out, codes_u16_out,
            invalid_device);
        CUDA_CHECK(cudaGetLastError());
        int invalid_host = 0;
        CUDA_CHECK(cudaMemcpy(
            &invalid_host, invalid_device, sizeof(int),
            cudaMemcpyDeviceToHost));
        cudaFree(invalid_device);
        invalid_device = nullptr;
        if (invalid_host != 0) {
            hprlp_device_free(dictionary_out);
            hprlp_device_free(packed_out);
            hprlp_device_free(indices_u16_out);
            hprlp_device_free(indices_u32_out);
            hprlp_device_free(codes_u8_out);
            hprlp_device_free(codes_u16_out);
            return false;
        }
    } catch (...) {
        cudaFree(invalid_device);
        hprlp_device_free(dictionary_out);
        hprlp_device_free(packed_out);
        hprlp_device_free(indices_u16_out);
        hprlp_device_free(indices_u32_out);
        hprlp_device_free(codes_u8_out);
        hprlp_device_free(codes_u16_out);
        throw;
    }

    encoding->dictionary_size = static_cast<int>(dictionary_size);
    encoding->index_bits = index_bits;
    encoding->code_bits = code_bits;
    encoding->storage = storage;
    encoding->dictionary = dictionary_out;
    encoding->packed = packed_out;
    encoding->indices_u16 = indices_u16_out;
    encoding->indices_u32 = indices_u32_out;
    encoding->codes_u8 = codes_u8_out;
    encoding->codes_u16 = codes_u16_out;
    return true;
}

}  // namespace

bool prepare_device_packed_dictionary_metadata(LP_info_gpu *lp) {
    if (lp == nullptr || lp->A == nullptr || lp->AT == nullptr) return false;
    HPRLPDeviceDictionaryEncoding x_encoding;
    HPRLPDeviceDictionaryEncoding y_encoding;
    if (!hprlp_build_device_dictionary_encoding(
            lp->AT, static_cast<std::size_t>(lp->m), &x_encoding)) {
        return false;
    }
    lp->has_original_coefficient_dictionary = true;
    lp->coefficient_dictionary_size = x_encoding.dictionary_size;
    lp->coefficient_dictionary = x_encoding.dictionary;
    lp->packed_dictionary_storage = x_encoding.storage;
    lp->packed_dictionary_index_bits = x_encoding.index_bits;
    lp->packed_dictionary_code_bits = x_encoding.code_bits;
    lp->AT_dictionary_packed_u32 = x_encoding.packed;
    lp->AT_dictionary_indices_u16 = x_encoding.indices_u16;
    lp->AT_dictionary_indices_u32 = x_encoding.indices_u32;
    lp->AT_dictionary_codes_u8 = x_encoding.codes_u8;
    lp->AT_dictionary_codes_u16 = x_encoding.codes_u16;

    const bool y_allowed =
        x_encoding.storage == HPRLPPackedDictionaryStorage::PackedU32 ||
        x_encoding.storage ==
            HPRLPPackedDictionaryStorage::SeparateU32U16;
    const bool y_built = y_allowed &&
        hprlp_build_device_dictionary_encoding(
            lp->A, static_cast<std::size_t>(lp->n), &y_encoding);
    const bool y_supported = y_built &&
        (y_encoding.storage == HPRLPPackedDictionaryStorage::PackedU32 ||
         y_encoding.storage ==
             HPRLPPackedDictionaryStorage::SeparateU32U16);
    if (y_supported) {
        lp->A_coefficient_dictionary_size = y_encoding.dictionary_size;
        lp->A_coefficient_dictionary = y_encoding.dictionary;
        lp->A_packed_dictionary_storage = y_encoding.storage;
        lp->A_packed_dictionary_code_bits = y_encoding.code_bits;
        lp->A_dictionary_packed_u32 = y_encoding.packed;
        lp->A_dictionary_indices_u32 = y_encoding.indices_u32;
        lp->A_dictionary_codes_u16 = y_encoding.codes_u16;
    } else if (y_built) {
        hprlp_release_device_dictionary_encoding(&y_encoding);
    }
    return true;
}
