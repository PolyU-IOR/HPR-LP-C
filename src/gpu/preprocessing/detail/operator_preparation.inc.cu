namespace {

std::uint64_t float_bits(HPRLP_FLOAT value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value),
                  "unexpected HPRLP_FLOAT size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

using XStaticKey =
    std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint8_t>;
using YStaticKey =
    std::tuple<std::uint64_t, std::uint64_t, std::uint8_t>;

XStaticKey x_static_key(const HPRLP_factorized_x_static_record &record) {
    return {float_bits(record.lower), float_bits(record.upper),
            float_bits(record.objective), record.bound_type};
}

YStaticKey y_static_key(const HPRLP_factorized_y_static_record &record) {
    return {float_bits(record.lower), float_bits(record.upper),
            record.bound_type};
}

template <typename Record, typename Key, typename KeyFn>
std::size_t encode_static_records_hybrid(
    const std::vector<Record> &expanded, KeyFn key_of,
    std::vector<std::uint8_t> *codes, std::vector<Record> *records) {
    struct UniqueRecord {
        Key key;
        Record record;
        std::size_t count;
        std::size_t first_index;
    };

    std::map<Key, std::size_t> unique_index;
    std::vector<UniqueRecord> unique_records;
    unique_records.reserve(256);
    for (std::size_t index = 0; index < expanded.size(); ++index) {
        const Key key = key_of(expanded[index]);
        const auto inserted =
            unique_index.emplace(key, unique_records.size());
        if (inserted.second) {
            unique_records.push_back(
                {key, expanded[index], 1, index});
        } else {
            ++unique_records[inserted.first->second].count;
        }
    }

    std::vector<std::size_t> frequency_order(unique_records.size());
    for (std::size_t index = 0; index < frequency_order.size(); ++index) {
        frequency_order[index] = index;
    }
    std::sort(frequency_order.begin(), frequency_order.end(),
              [&unique_records](std::size_t lhs, std::size_t rhs) {
                  if (unique_records[lhs].count !=
                      unique_records[rhs].count) {
                      return unique_records[lhs].count >
                             unique_records[rhs].count;
                  }
                  return unique_records[lhs].first_index <
                         unique_records[rhs].first_index;
              });

    // Code 255 is an exact escape to the original full-width arrays.  The
    // other 255 codes cover the most frequent records, so cardinality never
    // disables the lossless representation.
    constexpr std::size_t kStaticRecordCount = 255;
    constexpr std::uint8_t kEscapeCode = 255;
    const std::size_t selected_count =
        std::min(kStaticRecordCount, frequency_order.size());
    std::map<Key, std::uint8_t> selected_codes;
    records->clear();
    records->reserve(selected_count);
    for (std::size_t code = 0; code < selected_count; ++code) {
        const UniqueRecord &selected =
            unique_records[frequency_order[code]];
        records->push_back(selected.record);
        selected_codes.emplace(selected.key,
                               static_cast<std::uint8_t>(code));
    }

    codes->resize(expanded.size());
    std::size_t escape_count = 0;
    for (std::size_t index = 0; index < expanded.size(); ++index) {
        const auto selected = selected_codes.find(key_of(expanded[index]));
        if (selected == selected_codes.end()) {
            (*codes)[index] = kEscapeCode;
            ++escape_count;
        } else {
            (*codes)[index] = selected->second;
        }
    }
    return escape_count;
}

bool prepare_factorized_static_records(HPRLP_workspace_gpu *workspace) {
    if (!workspace->windowed_stencil_operator_ready ||
        workspace->x_bound_type == nullptr ||
        workspace->y_bound_type == nullptr) {
        return false;
    }

    std::vector<HPRLP_FLOAT> lower_x(workspace->n);
    std::vector<HPRLP_FLOAT> upper_x(workspace->n);
    std::vector<HPRLP_FLOAT> objective(workspace->n);
    std::vector<std::uint8_t> type_x(workspace->n);
    std::vector<HPRLP_FLOAT> lower_y(workspace->m);
    std::vector<HPRLP_FLOAT> upper_y(workspace->m);
    std::vector<std::uint8_t> type_y(workspace->m);
    CUDA_CHECK(cudaMemcpy(lower_x.data(), workspace->l,
                          lower_x.size() * sizeof(HPRLP_FLOAT),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper_x.data(), workspace->u,
                          upper_x.size() * sizeof(HPRLP_FLOAT),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(objective.data(), workspace->c,
                          objective.size() * sizeof(HPRLP_FLOAT),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(type_x.data(), workspace->x_bound_type,
                          type_x.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lower_y.data(), workspace->AL,
                          lower_y.size() * sizeof(HPRLP_FLOAT),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper_y.data(), workspace->AU,
                          upper_y.size() * sizeof(HPRLP_FLOAT),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(type_y.data(), workspace->y_bound_type,
                          type_y.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));

    std::vector<HPRLP_factorized_x_static_record> expanded_x(workspace->n);
    std::vector<HPRLP_factorized_y_static_record> expanded_y(workspace->m);
    for (int index = 0; index < workspace->n; ++index) {
        expanded_x[index].lower = lower_x[index];
        expanded_x[index].upper = upper_x[index];
        expanded_x[index].objective = objective[index];
        expanded_x[index].bound_type = type_x[index];
    }
    for (int index = 0; index < workspace->m; ++index) {
        expanded_y[index].lower = lower_y[index];
        expanded_y[index].upper = upper_y[index];
        expanded_y[index].bound_type = type_y[index];
    }

    // Certify a reusable state layout independently of model identity.  The
    // windowed-stencil matrix detector has already reconstructed every A and
    // A^T entry; this second gate checks every bound/objective entry consumed
    // by the streamlined normal kernels.
    workspace->windowed_stencil_state_ready = false;
    if (workspace->windowed_stencil_operator_ready) {
        const HPRLPWindowedStencilShape &shape =
            workspace->windowed_stencil_shape;
        const long long grid_count =
            static_cast<long long>(shape.grid_height) *
            shape.grid_width;
        const long long grid_begin = shape.grid_col0;
        const long long grid_end = grid_begin + grid_count;
        const long long observation_count = shape.observation_count;
        const long long stencil_begin = shape.stencil_row0;
        const long long stencil_count =
            static_cast<long long>(shape.grid_height) *
            shape.equation_width;
        const long long stencil_end = stencil_begin + stencil_count;

        bool partition_ok =
            grid_begin >= 0 && grid_end <= workspace->n &&
            grid_count + 1 == workspace->n &&
            shape.dense_col >= 0 && shape.dense_col < workspace->n &&
            (shape.dense_col < grid_begin || shape.dense_col >= grid_end) &&
            observation_count > 0 &&
            2 * observation_count == stencil_begin &&
            stencil_end == workspace->m;
        long long grid_matches = 0;
        long long first_matches = 0;
        long long second_matches = 0;
        long long equality_matches = 0;
        std::uint8_t first_type = 0;
        std::uint8_t second_type = 0;

        if (partition_ok) {
            for (long long col = grid_begin; col < grid_end; ++col) {
                if (type_x[col] == 3 &&
                    float_bits(objective[col]) == 0u) {
                    ++grid_matches;
                }
            }
            first_type = type_y[0];
            second_type = type_y[observation_count];
            const bool first_one_sided =
                first_type == 1 || first_type == 2;
            const bool second_one_sided =
                second_type == 1 || second_type == 2;
            if (first_one_sided) {
                for (long long row = 0; row < observation_count; ++row) {
                    first_matches += type_y[row] == first_type ? 1 : 0;
                }
            }
            if (second_one_sided) {
                for (long long row = observation_count;
                     row < 2 * observation_count; ++row) {
                    second_matches += type_y[row] == second_type ? 1 : 0;
                }
            }
            for (long long row = stencil_begin; row < stencil_end; ++row) {
                if (type_y[row] == 3 &&
                    float_bits(lower_y[row]) == float_bits(upper_y[row])) {
                    ++equality_matches;
                }
            }
        }

        const bool state_certified =
            partition_ok && grid_matches == grid_count &&
            first_matches == observation_count &&
            second_matches == observation_count &&
            equality_matches == stencil_count;
        workspace->windowed_stencil_state_ready = state_certified;
        if (workspace->windowed_stencil_state_ready) {
            workspace->windowed_observation_first_bound_type = first_type;
            workspace->windowed_observation_second_bound_type = second_type;
        }
    }

    std::vector<std::uint8_t> x_codes;
    std::vector<std::uint8_t> y_codes;
    std::vector<HPRLP_factorized_x_static_record> x_records;
    std::vector<HPRLP_factorized_y_static_record> y_records;
    encode_static_records_hybrid<
        HPRLP_factorized_x_static_record, XStaticKey>(
        expanded_x, x_static_key, &x_codes, &x_records);
    encode_static_records_hybrid<
        HPRLP_factorized_y_static_record, YStaticKey>(
        expanded_y, y_static_key, &y_codes, &y_records);

    copy_host_vector_to_device(x_codes,
                               &workspace->factorized_x_static_codes);
    copy_host_vector_to_device(y_codes,
                               &workspace->factorized_y_static_codes);
    copy_host_vector_to_device(x_records,
                               &workspace->factorized_x_static_records);
    copy_host_vector_to_device(y_records,
                               &workspace->factorized_y_static_records);
    workspace->factorized_x_static_record_count =
        static_cast<int>(x_records.size());
    workspace->factorized_y_static_record_count =
        static_cast<int>(y_records.size());
    workspace->factorized_static_records_ready = true;
    return true;
}

}  // namespace

void prepare_unit_operators(HPRLP_workspace_gpu *workspace, const Scaling_info *scaling_info) {
    if (workspace == nullptr || scaling_info == nullptr ||
        workspace->A == nullptr || workspace->AT == nullptr) {
        return;
    }

    const std::size_t nonzeros = static_cast<std::size_t>(workspace->AT->numElements);
    const bool use_unit_x = hprlp_use_unit_operator_x(
        workspace->uniform_unit_sign,
        static_cast<std::size_t>(workspace->m), nonzeros);
    const bool use_unit_y = hprlp_use_unit_operator_y(
        workspace->uniform_unit_sign, nonzeros);
    const bool use_structured = workspace->structured_operator != nullptr;
    const bool use_signed_unit = workspace->signed_unit_operator != nullptr;
    const bool packed_dictionary_x_storage_ready =
        (workspace->packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::PackedU32 &&
         workspace->AT_dictionary_packed_u32 != nullptr) ||
        (workspace->packed_dictionary_storage ==
             HPRLPPackedDictionaryStorage::SeparateU32U16 &&
         workspace->AT_dictionary_indices_u32 != nullptr &&
         workspace->AT_dictionary_codes_u16 != nullptr);
    const bool compact_dictionary = hprlp_use_dictionary_operator_x(
        workspace->coefficient_dictionary != nullptr,
        workspace->all_positive_unit_coefficients,
        workspace->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32,
        static_cast<std::size_t>(workspace->m), nonzeros,
        static_cast<std::size_t>(workspace->coefficient_dictionary_size),
        workspace->max_A_row_nnz, workspace->max_AT_row_nnz) &&
        workspace->AT_dictionary_packed_u32 != nullptr;
    const bool large_dictionary_enabled =
        workspace->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::SeparateU32U16;
    const bool extended_dictionary =
        large_dictionary_enabled &&
        workspace->coefficient_dictionary != nullptr &&
        !workspace->all_positive_unit_coefficients &&
        workspace->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::SeparateU32U16 &&
        packed_dictionary_x_storage_ready && nonzeros > 0 &&
        workspace->m > 0 && workspace->coefficient_dictionary_size > 0 &&
        static_cast<std::size_t>(workspace->coefficient_dictionary_size) <=
            HPRLP_PACKED_DICTIONARY_MAX_VALUES;
    const bool use_dictionary = compact_dictionary || extended_dictionary;
    const bool use_dictionary_y = workspace->dictionary_operator_y_ready;
    const bool use_row_template =
        workspace->row_template_operator != nullptr;
    const bool use_affine_block =
        workspace->affine_block_operator != nullptr;
    const bool use_windowed_stencil =
        workspace->windowed_stencil_operator_ready;
    if (!use_unit_x && !use_unit_y && !use_signed_unit && !use_dictionary &&
        !use_dictionary_y &&
        !use_structured &&
        !use_row_template && !use_affine_block && !use_windowed_stencil) {
        return;
    }

    if (use_unit_x) {
        build_compact_column_indices(
            workspace->AT, workspace->m,
            &workspace->unit_AT_col_index_u16);
    }

    create_zero_vector_device_compressible(workspace->inverse_row_norm, workspace->m);
    create_zero_vector_device_compressible(workspace->inverse_col_norm, workspace->n);
    if (use_unit_x || use_signed_unit || use_dictionary || use_structured ||
        use_row_template || use_windowed_stencil) {
        create_zero_vector_device_compressible(workspace->unit_scaled_y, workspace->m);
    }
    if (use_unit_y || use_signed_unit || use_structured ||
        use_dictionary_y ||
        use_row_template || use_windowed_stencil) {
        create_zero_vector_device_compressible(workspace->unit_scaled_x_hat,
                                               workspace->n);
    }
    if (use_windowed_stencil) {
        create_zero_vector_device_compressible(
            workspace->factorized_stencil_dense_partials,
            workspace->windowed_stencil_shape.observation_height);
        if (workspace->factorized_last_block_enabled) {
            CUDA_CHECK(cudaMalloc(
                &workspace->factorized_stencil_dense_counter,
                sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(
                workspace->factorized_stencil_dense_counter, 0,
                sizeof(unsigned int)));
        }
    }
    if (workspace->unit_coltile_ready) {
        const std::size_t word_count =
            (static_cast<std::size_t>(workspace->n) + 31u) / 32u;
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &workspace->unit_scaled_x_zero_bits,
            word_count * sizeof(std::uint32_t)));
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &workspace->unit_scaled_x_positive_zero_count,
            sizeof(unsigned long long)));
    }
    if (use_signed_unit) {
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &workspace->unit_scaled_x_hat_nonzero,
            static_cast<std::size_t>(workspace->n) * sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMemset(workspace->unit_scaled_x_hat_nonzero, 0,
                              static_cast<std::size_t>(workspace->n) *
                                  sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMalloc(
            &workspace->signed_xhat_positive_zero_count,
            sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(
            workspace->signed_xhat_positive_zero_count, 0,
            sizeof(unsigned long long)));
        CUDA_CHECK(cudaMallocHost(
            &workspace->signed_xhat_positive_zero_count_host,
            sizeof(unsigned long long)));
        *workspace->signed_xhat_positive_zero_count_host = 0;
    }
    workspace->factor_row_norm = scaling_info->row_norm;
    workspace->factor_col_norm = scaling_info->col_norm;

    reciprocal_vector_kernel<<<HPRLP_NUM_BLOCKS(workspace->m), HPRLP_NUM_THREADS, 0, workspace->stream>>>(
        scaling_info->row_norm, workspace->inverse_row_norm, workspace->m);
    reciprocal_vector_kernel<<<HPRLP_NUM_BLOCKS(workspace->n), HPRLP_NUM_THREADS, 0, workspace->stream>>>(
        scaling_info->col_norm, workspace->inverse_col_norm, workspace->n);

    if (use_windowed_stencil) {
        prepare_factorized_static_records(workspace);
    }

    if (use_unit_y) {
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &workspace->unit_A_values, nonzeros * sizeof(HPRLP_FLOAT)));
        set_vector_value_device_kernel<<<
            HPRLP_NUM_BLOCKS(workspace->A->numElements), HPRLP_NUM_THREADS, 0,
            workspace->stream>>>(workspace->unit_A_values,
                                 workspace->A->numElements, 1.0);

        CUSPARSE_CHECK(cusparseCreateCsr(
            &workspace->spmv_A->unit_A_cusparseDescr,
            workspace->m, workspace->n, workspace->A->numElements,
            workspace->A->rowPtr, workspace->A->colIndex,
            workspace->unit_A_values, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(
            &workspace->spmv_A->unit_x_hat_cusparseDescr,
            workspace->n, workspace->unit_scaled_x_hat, CUDA_R_64F));
        CUSPARSE_CHECK(hprlp_prepare_spmvop(
            workspace->spmv_A->cusparseHandle,
            workspace->spmv_A->unit_A_cusparseDescr,
            workspace->spmv_A->unit_x_hat_cusparseDescr,
            workspace->spmv_A->Ax_cusparseDescr,
            workspace->spmv_A->Ax_cusparseDescr,
            workspace->spmv_A->computeType,
            &workspace->spmv_A->unit_operation));
    }
    CUDA_CHECK(cudaStreamSynchronize(workspace->stream));
    workspace->unit_operator_x_ready = use_unit_x;
    workspace->unit_operator_y_ready = use_unit_y;
    workspace->signed_unit_operator_ready = use_signed_unit;
    workspace->dictionary_operator_x_ready = use_dictionary;
    workspace->dictionary_operator_y_ready = use_dictionary_y;
    workspace->structured_operator_ready = use_structured;
    workspace->row_template_operator_ready =
        workspace->row_template_operator != nullptr;
    workspace->affine_block_operator_ready =
        workspace->affine_block_operator != nullptr;
}
