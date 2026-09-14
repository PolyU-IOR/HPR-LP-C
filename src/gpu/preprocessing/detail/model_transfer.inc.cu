namespace {

constexpr HPRLP_FLOAT kInfiniteBoundThreshold = 1e90;

std::uint64_t hprlp_raw_bits(HPRLP_FLOAT value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected scalar size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Predicate>
std::pair<int, int> hprlp_longest_state_run(
    int count, Predicate predicate) {
    int best_begin = 0;
    int best_count = 0;
    int current_begin = 0;
    int current_count = 0;
    for (int index = 0; index < count; ++index) {
        if (predicate(index)) {
            if (current_count == 0) current_begin = index;
            ++current_count;
            if (current_count > best_count) {
                best_begin = current_begin;
                best_count = current_count;
            }
        } else {
            current_count = 0;
        }
    }
    return std::make_pair(best_begin, best_count);
}

HPRLPPackedStatePlan hprlp_build_packed_state_plan(
    const LP_info_cpu *model) {
    HPRLPPackedStatePlan plan{};
    if (model == nullptr || model->m <= 0 || model->n <= 0) return plan;

    const std::pair<int, int> zero_lower_boxed =
        hprlp_longest_state_run(model->n, [&](int col) {
            return hprlp_raw_bits(model->l[col]) == 0 &&
                   model->u[col] < kInfiniteBoundThreshold;
        });
    if (static_cast<long long>(zero_lower_boxed.second) * 4 >= model->n) {
        plan.x_zero_lower_boxed_begin = zero_lower_boxed.first;
        plan.x_zero_lower_boxed_count = zero_lower_boxed.second;
    }

    const std::pair<int, int> objective_zero =
        hprlp_longest_state_run(model->n, [&](int col) {
            return hprlp_raw_bits(model->c[col]) == 0;
        });
    if (static_cast<long long>(objective_zero.second) * 4 >= model->n) {
        plan.x_objective_zero_begin = objective_zero.first;
        plan.x_objective_zero_count = objective_zero.second;
    }

    const auto has_lower = [&](int row) {
        return model->AL[row] > -kInfiniteBoundThreshold;
    };
    const auto has_upper = [&](int row) {
        return model->AU[row] < kInfiniteBoundThreshold;
    };
    const std::pair<int, int> lower_only =
        hprlp_longest_state_run(model->m, [&](int row) {
            return has_lower(row) && !has_upper(row);
        });
    const std::pair<int, int> upper_only =
        hprlp_longest_state_run(model->m, [&](int row) {
            return !has_lower(row) && has_upper(row);
        });
    const std::pair<int, int> equality =
        hprlp_longest_state_run(model->m, [&](int row) {
            return has_lower(row) && has_upper(row) &&
                   hprlp_raw_bits(model->AL[row]) ==
                       hprlp_raw_bits(model->AU[row]);
        });
    const long long covered = static_cast<long long>(lower_only.second) +
                              upper_only.second + equality.second;
    if (covered * 5 >= static_cast<long long>(model->m) * 2) {
        plan.y_lower_only_begin = lower_only.first;
        plan.y_lower_only_count = lower_only.second;
        plan.y_upper_only_begin = upper_only.first;
        plan.y_upper_only_count = upper_only.second;
        plan.y_equality_begin = equality.first;
        plan.y_equality_count = equality.second;
    }
    return plan;
}

bool raw_positive_zero(HPRLP_FLOAT value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected floating type");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits == 0;
}

template <typename T>
void free_device_allocation(T *&ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(hprlp_device_free(ptr));
        ptr = nullptr;
    }
}

void copy_int_vector_to_device(const std::vector<int> &host_values, int **device_values) {
    if (host_values.empty()) {
        *device_values = nullptr;
        return;
    }
    CUDA_CHECK(hprlp_device_malloc_compressible(device_values,
                                                host_values.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*device_values, host_values.data(), static_cast<int>(host_values.size()) * sizeof(int), cudaMemcpyHostToDevice));
}

__global__ void hprlp_build_compact_column_indices_kernel(
    const int *indices, std::size_t count, int column_count,
    std::uint16_t *compact_indices, int *invalid) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const int column = indices[index];
        if (column < 0 || column >= column_count) {
            atomicExch(invalid, 1);
        } else {
            compact_indices[index] = static_cast<std::uint16_t>(column);
        }
    }
}

void build_compact_column_indices(const sparseMatrix *matrix,
                                  int column_count,
                                  std::uint16_t **compact_indices_device) {
    const std::size_t nonzeros =
        static_cast<std::size_t>(matrix->numElements);
    CUDA_CHECK(hprlp_device_malloc_compressible(
        compact_indices_device,
        nonzeros * sizeof(std::uint16_t)));
    int *invalid = nullptr;
    CUDA_CHECK(hprlp_device_malloc_compressible(&invalid, sizeof(int)));
    CUDA_CHECK(cudaMemset(invalid, 0, sizeof(int)));
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<std::size_t>(
        65535, (nonzeros + threads - 1) / threads));
    hprlp_build_compact_column_indices_kernel<<<blocks, threads>>>(
        matrix->colIndex, nonzeros, column_count, *compact_indices_device,
        invalid);
    CUDA_CHECK(cudaGetLastError());
    int invalid_host = 0;
    CUDA_CHECK(cudaMemcpy(&invalid_host, invalid, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(hprlp_device_free(invalid));
    if (invalid_host != 0) {
        free_device_allocation(*compact_indices_device);
        throw std::runtime_error("invalid CSR column index");
    }
}

template <typename T>
void copy_host_vector_to_device(const std::vector<T> &host_values,
                                T **device_values) {
    if (host_values.empty()) {
        *device_values = nullptr;
        return;
    }
    CUDA_CHECK(hprlp_device_malloc_compressible(
        device_values, host_values.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(*device_values, host_values.data(),
                          host_values.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
}

HPRLP_structured_operator_gpu *copy_structured_operator_to_device(
    const HPRLPStructuredOperatorHost &host) {
    HPRLP_structured_operator_gpu *device =
        new HPRLP_structured_operator_gpu;
    try {
        device->coefficient_bias = host.coefficient_bias;
        device->coefficient_has_escape = host.coefficient_has_escape;
        device->coefficient_escape_value = host.coefficient_escape_value;
        device->dense_row_count = static_cast<int>(host.dense_rows.size());
        device->dense_col_count = static_cast<int>(host.dense_cols.size());
        device->sparse_row_count = static_cast<int>(host.sparse_rows.size());
        device->short_AT_output_count =
            static_cast<int>(host.short_AT_output_cols.size());
        device->short_AT_nonzeros =
            static_cast<int>(host.short_AT_rows.size());

        copy_host_vector_to_device(host.dense_rows, &device->dense_rows);
        copy_host_vector_to_device(host.dense_cols, &device->dense_cols);
        copy_host_vector_to_device(host.dense_local_cols,
                                   &device->dense_local_cols);
        copy_host_vector_to_device(host.dense_A_values_u16,
                                   &device->dense_A_values_u16);
        copy_host_vector_to_device(host.dense_AT_values_u16,
                                   &device->dense_AT_values_u16);
        copy_host_vector_to_device(host.sparse_rows, &device->sparse_rows);
        copy_host_vector_to_device(host.sparse_col0, &device->sparse_col0);
        copy_host_vector_to_device(host.sparse_col1, &device->sparse_col1);
        copy_host_vector_to_device(host.sparse_second_sign,
                                   &device->sparse_second_sign);
        copy_host_vector_to_device(host.short_AT_output_cols,
                                   &device->short_AT_output_cols);
        copy_host_vector_to_device(host.short_AT_row_ptr,
                                   &device->short_AT_row_ptr);
        copy_host_vector_to_device(host.short_AT_rows,
                                   &device->short_AT_rows);
        copy_host_vector_to_device(host.short_AT_values_u16,
                                   &device->short_AT_values_u16);
    } catch (...) {
        free_device_allocation(device->dense_rows);
        free_device_allocation(device->dense_cols);
        free_device_allocation(device->dense_local_cols);
        free_device_allocation(device->dense_A_values_u16);
        free_device_allocation(device->dense_AT_values_u16);
        free_device_allocation(device->sparse_rows);
        free_device_allocation(device->sparse_col0);
        free_device_allocation(device->sparse_col1);
        free_device_allocation(device->sparse_second_sign);
        free_device_allocation(device->short_AT_output_cols);
        free_device_allocation(device->short_AT_row_ptr);
        free_device_allocation(device->short_AT_rows);
        free_device_allocation(device->short_AT_values_u16);
        delete device;
        throw;
    }
    return device;
}

void copy_row_template_matrix_to_device(
    const HPRLPRowTemplateHost &host,
    HPRLP_row_template_matrix_gpu *device) {
    device->row_count = host.row_count;
    device->template_count = host.template_count;
    device->encoded_row_count = host.encoded_row_count;
    device->maximum_row_degree = host.maximum_row_degree;
    device->fallback_short_count =
        static_cast<int>(host.fallback_short_rows.size());
    device->fallback_warp_count =
        static_cast<int>(host.fallback_warp_rows.size());
    device->fallback_block_count =
        static_cast<int>(host.fallback_block_rows.size());
    copy_host_vector_to_device(host.row_template_ids,
                               &device->row_template_ids);
    copy_host_vector_to_device(host.row_bases, &device->row_bases);
    copy_host_vector_to_device(host.template_ptr, &device->template_ptr);
    copy_host_vector_to_device(host.template_offsets,
                               &device->template_offsets);
    copy_host_vector_to_device(host.template_values,
                               &device->template_values);
    copy_host_vector_to_device(host.fallback_short_rows,
                               &device->fallback_short_rows);
    copy_host_vector_to_device(host.fallback_warp_rows,
                               &device->fallback_warp_rows);
    copy_host_vector_to_device(host.fallback_block_rows,
                               &device->fallback_block_rows);
    copy_host_vector_to_device(host.fallback_row_ptr,
                               &device->fallback_row_ptr);
    copy_host_vector_to_device(host.fallback_col_indices,
                               &device->fallback_col_indices);
    copy_host_vector_to_device(host.fallback_values,
                               &device->fallback_values);
}

HPRLP_row_template_operator_gpu *copy_row_template_operator_to_device(
    const HPRLPRowTemplateHost &A,
    const HPRLPRowTemplateHost &AT) {
    HPRLP_row_template_operator_gpu *device =
        new HPRLP_row_template_operator_gpu;
    try {
        copy_row_template_matrix_to_device(A, &device->A);
        copy_row_template_matrix_to_device(AT, &device->AT);
    } catch (...) {
        free_device_allocation(device->A.row_template_ids);
        free_device_allocation(device->A.row_bases);
        free_device_allocation(device->A.template_ptr);
        free_device_allocation(device->A.template_offsets);
        free_device_allocation(device->A.template_values);
        free_device_allocation(device->A.fallback_short_rows);
        free_device_allocation(device->A.fallback_warp_rows);
        free_device_allocation(device->A.fallback_block_rows);
        free_device_allocation(device->A.fallback_row_ptr);
        free_device_allocation(device->A.fallback_col_indices);
        free_device_allocation(device->A.fallback_values);
        free_device_allocation(device->AT.row_template_ids);
        free_device_allocation(device->AT.row_bases);
        free_device_allocation(device->AT.template_ptr);
        free_device_allocation(device->AT.template_offsets);
        free_device_allocation(device->AT.template_values);
        free_device_allocation(device->AT.fallback_short_rows);
        free_device_allocation(device->AT.fallback_warp_rows);
        free_device_allocation(device->AT.fallback_block_rows);
        free_device_allocation(device->AT.fallback_row_ptr);
        free_device_allocation(device->AT.fallback_col_indices);
        free_device_allocation(device->AT.fallback_values);
        delete device;
        throw;
    }
    return device;
}

void free_affine_block_matrix_device(HPRLP_affine_block_matrix_gpu *matrix) {
    free_device_allocation(matrix->block_row_begin);
    free_device_allocation(matrix->block_row_count);
    free_device_allocation(matrix->block_entry_ptr);
    free_device_allocation(matrix->entry_base_columns);
    free_device_allocation(matrix->entry_column_strides);
    free_device_allocation(matrix->entry_values);
    free_device_allocation(matrix->fallback_short_rows);
    free_device_allocation(matrix->fallback_warp_rows);
    free_device_allocation(matrix->fallback_block_rows);
    free_device_allocation(matrix->fallback_row_ptr);
    free_device_allocation(matrix->fallback_col_indices);
    free_device_allocation(matrix->fallback_values);
}

void copy_affine_block_matrix_to_device(
    const HPRLPAffineBlockHost &host,
    HPRLP_affine_block_matrix_gpu *device) {
    device->row_count = host.row_count;
    device->block_count = static_cast<int>(host.block_row_begin.size());
    device->encoded_row_count = host.encoded_row_count;
    device->maximum_row_degree = host.maximum_row_degree;
    device->fallback_short_count =
        static_cast<int>(host.fallback_short_rows.size());
    device->fallback_warp_count =
        static_cast<int>(host.fallback_warp_rows.size());
    device->fallback_block_count =
        static_cast<int>(host.fallback_block_rows.size());
    copy_host_vector_to_device(host.block_row_begin,
                               &device->block_row_begin);
    copy_host_vector_to_device(host.block_row_count,
                               &device->block_row_count);
    copy_host_vector_to_device(host.block_entry_ptr,
                               &device->block_entry_ptr);
    copy_host_vector_to_device(host.entry_base_columns,
                               &device->entry_base_columns);
    copy_host_vector_to_device(host.entry_column_strides,
                               &device->entry_column_strides);
    copy_host_vector_to_device(host.entry_values, &device->entry_values);
    copy_host_vector_to_device(host.fallback_short_rows,
                               &device->fallback_short_rows);
    copy_host_vector_to_device(host.fallback_warp_rows,
                               &device->fallback_warp_rows);
    copy_host_vector_to_device(host.fallback_block_rows,
                               &device->fallback_block_rows);
    copy_host_vector_to_device(host.fallback_row_ptr,
                               &device->fallback_row_ptr);
    copy_host_vector_to_device(host.fallback_col_indices,
                               &device->fallback_col_indices);
    copy_host_vector_to_device(host.fallback_values,
                               &device->fallback_values);
}

HPRLP_affine_block_operator_gpu *copy_affine_block_operator_to_device(
    const HPRLPAffineBlockHost &A, const HPRLPAffineBlockHost &AT) {
    HPRLP_affine_block_operator_gpu *device =
        new HPRLP_affine_block_operator_gpu;
    try {
        copy_affine_block_matrix_to_device(A, &device->A);
        copy_affine_block_matrix_to_device(AT, &device->AT);
    } catch (...) {
        free_affine_block_matrix_device(&device->A);
        free_affine_block_matrix_device(&device->AT);
        delete device;
        throw;
    }
    return device;
}

HPRLP_unit_coltile_gpu *copy_unit_coltile_to_device(
    const HPRLPUnitColTileHost &host) {
    HPRLP_unit_coltile_gpu *device = new HPRLP_unit_coltile_gpu;
    try {
        device->rows = host.rows;
        device->columns = host.columns;
        device->tile_cols = host.tile_cols;
        device->tile_count = host.tile_count;
        copy_host_vector_to_device(host.row_tile_offsets,
                                   &device->row_tile_offsets);
        copy_host_vector_to_device(host.local_cols, &device->local_cols);
    } catch (...) {
        free_device_allocation(device->row_tile_offsets);
        free_device_allocation(device->local_cols);
        delete device;
        throw;
    }
    return device;
}

HPRLP_signed_unit_operator_gpu *copy_signed_unit_operator_to_device(
    const HPRLPSignedUnitPackedHost &A,
    const HPRLPSignedUnitPackedHost &AT,
    const HPRLPFixedDegreeRunHost &A_degree2_run,
    const HPRLPFixedDegreeRunHost &AT_degree3_run) {
    HPRLP_signed_unit_operator_gpu *device =
        new HPRLP_signed_unit_operator_gpu;
    try {
        device->A_uses_u16 = A.uses_u16;
        device->AT_uses_u16 = AT.uses_u16;
        device->A_split_u16_ready = A.split_u16_ready;
        device->AT_split_u16_ready = AT.split_u16_ready;
        device->A_degree2_run_row_begin = A_degree2_run.row_begin;
        device->A_degree2_run_row_count = A_degree2_run.row_count;
        device->A_degree2_run_entry_begin = A_degree2_run.entry_begin;
        device->AT_degree3_run_row_begin = AT_degree3_run.row_begin;
        device->AT_degree3_run_row_count = AT_degree3_run.row_count;
        device->AT_degree3_run_entry_begin = AT_degree3_run.entry_begin;
        copy_host_vector_to_device(A.entries_u16, &device->A_entries_u16);
        copy_host_vector_to_device(A.entries_u32, &device->A_entries_u32);
        copy_host_vector_to_device(AT.entries_u16, &device->AT_entries_u16);
        copy_host_vector_to_device(AT.entries_u32, &device->AT_entries_u32);
        copy_host_vector_to_device(A.split_indices_u16,
                                   &device->A_split_indices_u16);
        copy_host_vector_to_device(A.split_negative_u8,
                                   &device->A_split_negative_u8);
        copy_host_vector_to_device(AT.split_indices_u16,
                                   &device->AT_split_indices_u16);
        copy_host_vector_to_device(AT.split_negative_u8,
                                   &device->AT_split_negative_u8);
    } catch (...) {
        free_device_allocation(device->A_entries_u16);
        free_device_allocation(device->A_entries_u32);
        free_device_allocation(device->AT_entries_u16);
        free_device_allocation(device->AT_entries_u32);
        free_device_allocation(device->A_split_indices_u16);
        free_device_allocation(device->A_split_negative_u8);
        free_device_allocation(device->AT_split_indices_u16);
        free_device_allocation(device->AT_split_negative_u8);
        delete device;
        throw;
    }
    return device;
}

struct HPRLP_device_row_bucket_predicate {
    const int *row_ptr;
    int bucket;

    __host__ __device__ bool operator()(int row) const {
        const int nonzeros = row_ptr[row + 1] - row_ptr[row];
        const int current_bucket = nonzeros <= HPRLP_SCALAR_ROW_MAX_NNZ
            ? static_cast<int>(HPRLP_ROW_SCALAR)
            : (nonzeros <= HPRLP_WARP_ROW_MAX_NNZ
                ? static_cast<int>(HPRLP_ROW_WARP)
                : static_cast<int>(HPRLP_ROW_BLOCK));
        return current_bucket == bucket;
    }
};

struct HPRLP_device_row_degree {
    const int *row_ptr;

    __host__ __device__ int operator()(int row) const {
        return row_ptr[row + 1] - row_ptr[row];
    }
};

void build_row_buckets(const sparseMatrix *matrix,
                       int **rows_short, int *num_rows_short,
                       int **rows_medium, int *num_rows_medium,
                       int **rows_long, int *num_rows_long,
                       int *max_row_nnz, cudaStream_t stream) {
    if (matrix == nullptr || matrix->row <= 0 || matrix->rowPtr == nullptr) {
        *rows_short = nullptr;
        *rows_medium = nullptr;
        *rows_long = nullptr;
        *num_rows_short = 0;
        *num_rows_medium = 0;
        *num_rows_long = 0;
        *max_row_nnz = 0;
        return;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(matrix->row) * sizeof(int);
    CUDA_CHECK(hprlp_device_malloc_compressible(rows_short, row_bytes));
    CUDA_CHECK(hprlp_device_malloc_compressible(rows_medium, row_bytes));
    CUDA_CHECK(hprlp_device_malloc_compressible(rows_long, row_bytes));

    const auto policy = thrust::cuda::par.on(stream);
    const auto first = thrust::make_counting_iterator<int>(0);
    const auto last = first + matrix->row;
    const auto short_begin = thrust::device_pointer_cast(*rows_short);
    const auto medium_begin = thrust::device_pointer_cast(*rows_medium);
    const auto long_begin = thrust::device_pointer_cast(*rows_long);
    const auto short_end = thrust::copy_if(
        policy, first, last, short_begin,
        HPRLP_device_row_bucket_predicate{
            matrix->rowPtr, static_cast<int>(HPRLP_ROW_SCALAR)});
    const auto medium_end = thrust::copy_if(
        policy, first, last, medium_begin,
        HPRLP_device_row_bucket_predicate{
            matrix->rowPtr, static_cast<int>(HPRLP_ROW_WARP)});
    const auto long_end = thrust::copy_if(
        policy, first, last, long_begin,
        HPRLP_device_row_bucket_predicate{
            matrix->rowPtr, static_cast<int>(HPRLP_ROW_BLOCK)});
    *num_rows_short = static_cast<int>(short_end - short_begin);
    *num_rows_medium = static_cast<int>(medium_end - medium_begin);
    *num_rows_long = static_cast<int>(long_end - long_begin);
    *max_row_nnz = thrust::transform_reduce(
        policy, first, last, HPRLP_device_row_degree{matrix->rowPtr}, 0,
        thrust::maximum<int>());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void repack_fixed_degree_run_soa_kernel(
    const std::uint32_t *source, std::uint32_t *destination,
    int entry_begin, int row_count, int degree, int entry_count) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= entry_count) return;
    const int position = output_index / row_count;
    const int row_offset = output_index - position * row_count;
    destination[output_index] =
        source[entry_begin + row_offset * degree + position];
}

void build_fixed_degree_run_plan(const sparseMatrix *matrix,
                                 HPRLPFixedDegreeRunPlan *plan,
                                 const std::uint32_t *packed_entries) {
    if (matrix == nullptr || plan == nullptr || matrix->row <= 0 ||
        matrix->rowPtr == nullptr || packed_entries == nullptr) {
        return;
    }

    *plan = HPRLPFixedDegreeRunPlan{};
    std::vector<int> row_ptr(static_cast<std::size_t>(matrix->row) + 1u);
    CUDA_CHECK(cudaMemcpy(row_ptr.data(), matrix->rowPtr,
                          row_ptr.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    constexpr int minimum_run = 32768;
    constexpr int threads = 128;
    constexpr int minimum_nnz_percent = 50;
    plan->threads = threads;
    struct CandidateRun {
        HPRLPFixedDegreeRun run;
        long long covered_nnz = 0;
    };
    std::vector<CandidateRun> candidates;
    for (int row_begin = 0; row_begin < matrix->row;) {
        const int degree = row_ptr[row_begin + 1] - row_ptr[row_begin];
        int row_end = row_begin + 1;
        while (row_end < matrix->row &&
               row_ptr[row_end + 1] - row_ptr[row_end] == degree) {
            ++row_end;
        }
        const int row_count = row_end - row_begin;
        if (degree >= 1 && degree <= HPRLP_SCALAR_ROW_MAX_NNZ &&
            row_count >= minimum_run) {
            CandidateRun candidate;
            candidate.run.row_begin = row_begin;
            candidate.run.row_count = row_count;
            candidate.run.entry_begin = row_ptr[row_begin];
            candidate.run.degree = degree;
            candidate.covered_nnz =
                static_cast<long long>(row_count) * degree;
            candidates.push_back(candidate);
        }
        row_begin = row_end;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const CandidateRun &lhs, const CandidateRun &rhs) {
                  if (lhs.covered_nnz != rhs.covered_nnz) {
                      return lhs.covered_nnz > rhs.covered_nnz;
                  }
                  return lhs.run.row_count > rhs.run.row_count;
              });
    if (candidates.size() > HPRLP_FIXED_DEGREE_MAX_RUNS) {
        candidates.resize(HPRLP_FIXED_DEGREE_MAX_RUNS);
    }

    long long covered_rows = 0;
    long long covered_nnz = 0;
    for (const CandidateRun &candidate : candidates) {
        covered_rows += candidate.run.row_count;
        covered_nnz += candidate.covered_nnz;
    }
    const bool admitted = !candidates.empty() && covered_rows >= minimum_run &&
        covered_nnz * 100 >=
            static_cast<long long>(minimum_nnz_percent) *
                matrix->numElements;
    if (!admitted) {
        return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const CandidateRun &lhs, const CandidateRun &rhs) {
                  return lhs.run.row_begin < rhs.run.row_begin;
              });
    std::vector<std::uint8_t> covered(
        static_cast<std::size_t>(matrix->row), std::uint8_t{0});
    plan->run_count = static_cast<int>(candidates.size());
    plan->covered_rows = covered_rows;
    plan->covered_nnz = covered_nnz;
    for (int run_index = 0; run_index < plan->run_count; ++run_index) {
        plan->runs[run_index] = candidates[run_index].run;
        HPRLPFixedDegreeRun &run = plan->runs[run_index];
        std::fill(covered.begin() + run.row_begin,
                  covered.begin() + run.row_begin + run.row_count,
                  std::uint8_t{1});
        const int entry_count = run.row_count * run.degree;
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &run.packed_entries_soa,
            static_cast<std::size_t>(entry_count) *
                sizeof(std::uint32_t)));
        repack_fixed_degree_run_soa_kernel<<<
            (entry_count + threads - 1) / threads, threads>>>(
            packed_entries, run.packed_entries_soa, run.entry_begin,
            run.row_count, run.degree, entry_count);
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> fallback_short;
    std::vector<int> fallback_warp;
    std::vector<int> fallback_block;
    fallback_short.reserve(matrix->row / 4);
    fallback_warp.reserve(matrix->row / 16);
    fallback_block.reserve(matrix->row / 256);
    for (int row = 0; row < matrix->row; ++row) {
        if (covered[static_cast<std::size_t>(row)] != 0) continue;
        const int degree = row_ptr[row + 1] - row_ptr[row];
        const HPRLPRowBucket bucket = hprlp_row_bucket(degree);
        if (bucket == HPRLP_ROW_SCALAR) {
            fallback_short.push_back(row);
        } else if (bucket == HPRLP_ROW_WARP) {
            fallback_warp.push_back(row);
        } else {
            fallback_block.push_back(row);
        }
    }
    plan->fallback_short_count = static_cast<int>(fallback_short.size());
    plan->fallback_warp_count = static_cast<int>(fallback_warp.size());
    plan->fallback_block_count = static_cast<int>(fallback_block.size());
    copy_int_vector_to_device(fallback_short, &plan->fallback_short_rows);
    copy_int_vector_to_device(fallback_warp, &plan->fallback_warp_rows);
    copy_int_vector_to_device(fallback_block, &plan->fallback_block_rows);
    plan->ready = true;
}

struct HPRLP_device_segmented_row_predicate {
    const int *row_ptr;
    int threshold;
    bool select_segmented;

    __host__ __device__ bool operator()(int row) const {
        const int nonzeros = row_ptr[row + 1] - row_ptr[row];
        if (nonzeros <= HPRLP_WARP_ROW_MAX_NNZ) return false;
        return select_segmented ? nonzeros >= threshold
                                : nonzeros < threshold;
    }
};

struct HPRLP_device_selected_row_degree {
    const int *row_ptr;
    const int *selected_rows;

    __host__ __device__ int operator()(int selected_index) const {
        const int row = selected_rows[selected_index];
        return row_ptr[row + 1] - row_ptr[row];
    }
};

struct HPRLP_device_selected_row_tile_count {
    const int *row_ptr;
    const int *selected_rows;
    int tile_entries;

    __host__ __device__ int operator()(int selected_index) const {
        const int row = selected_rows[selected_index];
        const int nonzeros = row_ptr[row + 1] - row_ptr[row];
        return (nonzeros + tile_entries - 1) / tile_entries;
    }
};

__global__ void hprlp_fill_segment_tiles_kernel(
    const int *row_ptr, const int *segmented_rows,
    const int *row_tile_ptr, int segmented_row_count, int tile_entries,
    int *tile_begin, int *tile_end) {
    const int selected_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (selected_index >= segmented_row_count) return;
    const int row = segmented_rows[selected_index];
    const int row_begin = row_ptr[row];
    const int row_end = row_ptr[row + 1];
    const int output_begin = row_tile_ptr[selected_index];
    const int output_end = row_tile_ptr[selected_index + 1];
    for (int output = output_begin; output < output_end; ++output) {
        const int begin = row_begin +
            (output - output_begin) * tile_entries;
        tile_begin[output] = begin;
        tile_end[output] = min(begin + tile_entries, row_end);
    }
}

void build_segmented_plan_device(
    const sparseMatrix *matrix, int tile_entries, int threshold,
    int **segmented_rows_output, int *segmented_row_count_output,
    int **row_tile_ptr_output, int **tile_begin_output,
    int **tile_end_output, int *tile_count_output,
    int **fallback_rows_output, int *fallback_count_output,
    long long *segmented_nnz_output, HPRLP_FLOAT **partials_output,
    bool *ready_output, cudaStream_t stream) {
    const auto policy = thrust::cuda::par.on(stream);
    const auto first = thrust::make_counting_iterator<int>(0);
    const auto last = first + matrix->row;
    thrust::device_vector<int> segmented_rows(matrix->row);
    thrust::device_vector<int> fallback_rows(matrix->row);
    const auto segmented_end = thrust::copy_if(
        policy, first, last, segmented_rows.begin(),
        HPRLP_device_segmented_row_predicate{
            matrix->rowPtr, threshold, true});
    const auto fallback_end = thrust::copy_if(
        policy, first, last, fallback_rows.begin(),
        HPRLP_device_segmented_row_predicate{
            matrix->rowPtr, threshold, false});
    const int segmented_count =
        static_cast<int>(segmented_end - segmented_rows.begin());
    const int fallback_count =
        static_cast<int>(fallback_end - fallback_rows.begin());
    *segmented_row_count_output = segmented_count;
    *fallback_count_output = fallback_count;
    *tile_count_output = 0;
    *segmented_nnz_output = 0;
    *ready_output = false;
    if (segmented_count == 0) return;

    thrust::device_vector<int> tile_counts(segmented_count);
    thrust::transform(
        policy, first, first + segmented_count, tile_counts.begin(),
        HPRLP_device_selected_row_tile_count{
            matrix->rowPtr,
            thrust::raw_pointer_cast(segmented_rows.data()), tile_entries});
    thrust::device_vector<int> row_tile_ptr(segmented_count + 1);
    CUDA_CHECK(cudaMemsetAsync(
        thrust::raw_pointer_cast(row_tile_ptr.data()), 0, sizeof(int),
        stream));
    thrust::inclusive_scan(
        policy, tile_counts.begin(), tile_counts.end(),
        row_tile_ptr.begin() + 1);
    int tile_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &tile_count,
        thrust::raw_pointer_cast(row_tile_ptr.data()) + segmented_count,
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    const long long segmented_nnz = thrust::transform_reduce(
        policy, first, first + segmented_count,
        HPRLP_device_selected_row_degree{
            matrix->rowPtr,
            thrust::raw_pointer_cast(segmented_rows.data())},
        static_cast<long long>(0), thrust::plus<long long>());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *tile_count_output = tile_count;
    *segmented_nnz_output = segmented_nnz;
    *ready_output = tile_count > segmented_count;
    if (!*ready_output) return;

    CUDA_CHECK(hprlp_device_malloc_compressible(
        segmented_rows_output,
        static_cast<std::size_t>(segmented_count) * sizeof(int)));
    CUDA_CHECK(hprlp_device_malloc_compressible(
        row_tile_ptr_output,
        (static_cast<std::size_t>(segmented_count) + 1) * sizeof(int)));
    CUDA_CHECK(hprlp_device_malloc_compressible(
        tile_begin_output,
        static_cast<std::size_t>(tile_count) * sizeof(int)));
    CUDA_CHECK(hprlp_device_malloc_compressible(
        tile_end_output,
        static_cast<std::size_t>(tile_count) * sizeof(int)));
    if (fallback_count > 0) {
        CUDA_CHECK(hprlp_device_malloc_compressible(
            fallback_rows_output,
            static_cast<std::size_t>(fallback_count) * sizeof(int)));
    }
    CUDA_CHECK(hprlp_device_malloc_compressible(
        partials_output,
        static_cast<std::size_t>(tile_count) * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpyAsync(
        *segmented_rows_output,
        thrust::raw_pointer_cast(segmented_rows.data()),
        static_cast<std::size_t>(segmented_count) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        *row_tile_ptr_output,
        thrust::raw_pointer_cast(row_tile_ptr.data()),
        (static_cast<std::size_t>(segmented_count) + 1) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));
    if (fallback_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            *fallback_rows_output,
            thrust::raw_pointer_cast(fallback_rows.data()),
            static_cast<std::size_t>(fallback_count) * sizeof(int),
            cudaMemcpyDeviceToDevice, stream));
    }
    hprlp_fill_segment_tiles_kernel<<<
        HPRLP_NUM_BLOCKS(segmented_count), HPRLP_NUM_THREADS, 0, stream>>>(
            matrix->rowPtr, *segmented_rows_output, *row_tile_ptr_output,
            segmented_count, tile_entries, *tile_begin_output,
            *tile_end_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void build_segmented_A_plan(const sparseMatrix *matrix,
                            HPRLP_workspace_gpu *workspace) {
    workspace->segmented_A_threshold = 8192;
    workspace->segmented_A_tile_entries = 2048;
    workspace->segmented_A_threads = 512;
    build_segmented_plan_device(
        matrix, workspace->segmented_A_tile_entries,
        workspace->segmented_A_threshold, &workspace->segmented_A_rows,
        &workspace->segmented_A_row_count,
        &workspace->segmented_A_row_tile_ptr,
        &workspace->segmented_A_tile_begin,
        &workspace->segmented_A_tile_end,
        &workspace->segmented_A_tile_count,
        &workspace->segmented_A_fallback_long_rows,
        &workspace->segmented_A_fallback_long_count,
        &workspace->segmented_A_nnz, &workspace->segmented_A_partials,
        &workspace->segmented_A_ready, workspace->stream);
}

void build_segmented_AT_plan(const sparseMatrix *matrix,
                             HPRLP_workspace_gpu *workspace) {
    workspace->segmented_AT_threshold = 8192;
    workspace->segmented_AT_tile_entries = 1024;
    workspace->segmented_AT_threads = 512;
    build_segmented_plan_device(
        matrix, workspace->segmented_AT_tile_entries,
        workspace->segmented_AT_threshold, &workspace->segmented_AT_rows,
        &workspace->segmented_AT_row_count,
        &workspace->segmented_AT_row_tile_ptr,
        &workspace->segmented_AT_tile_begin,
        &workspace->segmented_AT_tile_end,
        &workspace->segmented_AT_tile_count,
        &workspace->segmented_AT_fallback_long_rows,
        &workspace->segmented_AT_fallback_long_count,
        &workspace->segmented_AT_nnz, &workspace->segmented_AT_partials,
        &workspace->segmented_AT_ready, workspace->stream);
}

__global__ void hprlp_build_bound_types_kernel(
    const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper, int count,
    std::uint8_t *bound_type) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (lower[index] <= -kInfiniteBoundThreshold &&
        upper[index] >= kInfiniteBoundThreshold) {
        bound_type[index] = 0;
    } else if (upper[index] >= kInfiniteBoundThreshold) {
        bound_type[index] = 1;
    } else if (lower[index] <= -kInfiniteBoundThreshold) {
        bound_type[index] = 2;
    } else {
        bound_type[index] = 3;
    }
}

void build_bound_types(const HPRLP_FLOAT *lower_dev,
                       const HPRLP_FLOAT *upper_dev, int len,
                       std::uint8_t **bound_type_dev,
                       cudaStream_t stream) {
    if (len <= 0) {
        *bound_type_dev = nullptr;
        return;
    }
    CUDA_CHECK(hprlp_device_malloc_compressible(
        bound_type_dev, static_cast<std::size_t>(len) * sizeof(std::uint8_t)));
    hprlp_build_bound_types_kernel<<<
        HPRLP_NUM_BLOCKS(len), HPRLP_NUM_THREADS, 0, stream>>>(
            lower_dev, upper_dev, len, *bound_type_dev);
    CUDA_CHECK(cudaGetLastError());
}

}

void copy_lpinfo_to_device(const LP_info_cpu *lp_info_cpu, LP_info_gpu *lp_info_gpu) {
    int m = lp_info_cpu->m;
    int n = lp_info_cpu->n;

    lp_info_gpu->m = m;
    lp_info_gpu->n = n;
    lp_info_gpu->obj_constant = lp_info_cpu->obj_constant;
    lp_info_gpu->has_original_coefficient_dictionary = false;
    lp_info_gpu->coefficient_dictionary_size = 0;
    lp_info_gpu->coefficient_dictionary = nullptr;
    lp_info_gpu->AT_value_codes = nullptr;
    lp_info_gpu->packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    lp_info_gpu->packed_dictionary_index_bits = 0;
    lp_info_gpu->packed_dictionary_code_bits = 0;
    lp_info_gpu->AT_dictionary_packed_u32 = nullptr;
    lp_info_gpu->AT_dictionary_indices_u16 = nullptr;
    lp_info_gpu->AT_dictionary_indices_u32 = nullptr;
    lp_info_gpu->AT_dictionary_codes_u8 = nullptr;
    lp_info_gpu->AT_dictionary_codes_u16 = nullptr;
    lp_info_gpu->A_coefficient_dictionary_size = 0;
    lp_info_gpu->A_coefficient_dictionary = nullptr;
    lp_info_gpu->A_packed_dictionary_storage =
        HPRLPPackedDictionaryStorage::None;
    lp_info_gpu->A_packed_dictionary_code_bits = 0;
    lp_info_gpu->A_dictionary_packed_u32 = nullptr;
    lp_info_gpu->A_dictionary_indices_u32 = nullptr;
    lp_info_gpu->A_dictionary_codes_u16 = nullptr;
    lp_info_gpu->packed_state_plan = HPRLPPackedStatePlan{};
    lp_info_gpu->structured_operator = nullptr;
    lp_info_gpu->row_template_operator = nullptr;
    lp_info_gpu->affine_block_operator = nullptr;
    lp_info_gpu->windowed_stencil_operator = false;
    lp_info_gpu->windowed_stencil_shape = HPRLPWindowedStencilShape{};
    lp_info_gpu->grid_slack_laplacian_operator = false;
    lp_info_gpu->grid_slack_laplacian_shape =
        HPRLPGridSlackLaplacianShape{};
    lp_info_gpu->unit_coltile = nullptr;
    lp_info_gpu->signed_unit_operator = nullptr;
    lp_info_gpu->uniform_unit_sign = static_cast<int8_t>(
        hprlp_uniform_unit_sign(
            lp_info_cpu->A->value,
            static_cast<std::size_t>(lp_info_cpu->A->numElements)));
    lp_info_gpu->all_positive_unit_coefficients =
        lp_info_gpu->uniform_unit_sign > 0;
    lp_info_gpu->mixed_signed_unit_coefficients =
        hprlp_mixed_signed_unit(
            lp_info_cpu->A->value,
            static_cast<std::size_t>(lp_info_cpu->A->numElements));
    lp_info_gpu->all_zero_lower_unbounded_variables =
        hprlp_all_zero_lower_unbounded(lp_info_cpu->l, lp_info_cpu->u,
                                       static_cast<std::size_t>(n));

    bool all_x_boxed_zero_lower = n > 0;
    for (int col = 0; col < n; ++col) {
        if (!raw_positive_zero(lp_info_cpu->l[col]) ||
            lp_info_cpu->u[col] >= kInfiniteBoundThreshold) {
            all_x_boxed_zero_lower = false;
            break;
        }
    }
    int x_zero_objective_run_begin = 0;
    int x_zero_objective_run_count = 0;
    int current_x_run_begin = 0;
    int current_x_run_count = 0;
    for (int col = 0; col < n; ++col) {
        if (raw_positive_zero(lp_info_cpu->c[col])) {
            if (current_x_run_count == 0) {
                current_x_run_begin = col;
            }
            ++current_x_run_count;
            if (current_x_run_count > x_zero_objective_run_count) {
                x_zero_objective_run_begin = current_x_run_begin;
                x_zero_objective_run_count = current_x_run_count;
            }
        } else {
            current_x_run_count = 0;
        }
    }
    int y_upper_zero_run_begin = 0;
    int y_upper_zero_run_count = 0;
    int current_y_run_begin = 0;
    int current_y_run_count = 0;
    for (int row = 0; row < m; ++row) {
        const bool upper_zero =
            lp_info_cpu->AL[row] <= -kInfiniteBoundThreshold &&
            raw_positive_zero(lp_info_cpu->AU[row]);
        if (upper_zero) {
            if (current_y_run_count == 0) {
                current_y_run_begin = row;
            }
            ++current_y_run_count;
            if (current_y_run_count > y_upper_zero_run_count) {
                y_upper_zero_run_begin = current_y_run_begin;
                y_upper_zero_run_count = current_y_run_count;
            }
        } else {
            current_y_run_count = 0;
        }
    }
    lp_info_gpu->signed_x_zero_objective_run_begin =
        x_zero_objective_run_begin;
    lp_info_gpu->signed_x_zero_objective_run_count =
        x_zero_objective_run_count;
    lp_info_gpu->signed_y_upper_zero_run_begin =
        y_upper_zero_run_begin;
    lp_info_gpu->signed_y_upper_zero_run_count =
        y_upper_zero_run_count;
    lp_info_gpu->signed_state_plan_ready =
        n >= 1000000 && m >= 1000000 && all_x_boxed_zero_lower &&
        static_cast<long long>(x_zero_objective_run_count) * 2 >= n &&
        static_cast<long long>(y_upper_zero_run_count) * 2 >= m;

    if (hprlp_use_unit_coltile(
            lp_info_gpu->uniform_unit_sign,
            static_cast<std::size_t>(m), static_cast<std::size_t>(n),
            static_cast<std::size_t>(lp_info_cpu->A->numElements))) {
        HPRLPUnitColTileHost unit_coltile_host;
        if (!hprlp_build_unit_coltile(
                m, n, lp_info_cpu->A->numElements,
                lp_info_cpu->A->rowPtr, lp_info_cpu->A->colIndex,
                &unit_coltile_host)) {
            throw std::runtime_error("failed to build unit col-tile operator");
        }
        lp_info_gpu->unit_coltile =
            copy_unit_coltile_to_device(unit_coltile_host);
    }

    // Copy A to GPU
    lp_info_gpu->A = new sparseMatrix;
    transfer_CSR_matrix(lp_info_cpu->A, lp_info_gpu->A);

    // Generate AT on CPU first, then transfer to GPU
    lp_info_gpu->AT = new sparseMatrix;
    sparseMatrix AT_host;
    CSR_transpose_host(*(lp_info_cpu->A), &AT_host);
    transfer_CSR_matrix(&AT_host, lp_info_gpu->AT);

    const HPRLPHostCsrView full_A_view{
        lp_info_cpu->A->row, lp_info_cpu->A->col,
        lp_info_cpu->A->numElements, lp_info_cpu->A->rowPtr,
        lp_info_cpu->A->colIndex, lp_info_cpu->A->value};
    const HPRLPHostCsrView full_AT_view{
        AT_host.row, AT_host.col, AT_host.numElements, AT_host.rowPtr,
        AT_host.colIndex, AT_host.value};

    HPRLPPackedDictionaryHost dictionary_encoding;
    const bool dictionary_encoded = hprlp_build_packed_dictionary_operator(
        AT_host.value, AT_host.colIndex,
        static_cast<std::size_t>(AT_host.numElements),
        static_cast<std::size_t>(m), &dictionary_encoding);
    if (dictionary_encoded) {
        lp_info_gpu->has_original_coefficient_dictionary = true;
        lp_info_gpu->coefficient_dictionary_size = static_cast<int>(
            dictionary_encoding.dictionary.size());
        lp_info_gpu->packed_dictionary_storage =
            dictionary_encoding.storage;
        lp_info_gpu->packed_dictionary_index_bits =
            dictionary_encoding.index_bits;
        lp_info_gpu->packed_dictionary_code_bits =
            dictionary_encoding.code_bits;
        copy_host_vector_to_device(dictionary_encoding.dictionary,
                                   &lp_info_gpu->coefficient_dictionary);
        copy_host_vector_to_device(dictionary_encoding.packed_u32,
                                   &lp_info_gpu->AT_dictionary_packed_u32);
        copy_host_vector_to_device(dictionary_encoding.indices_u16,
                                   &lp_info_gpu->AT_dictionary_indices_u16);
        copy_host_vector_to_device(dictionary_encoding.indices_u32,
                                   &lp_info_gpu->AT_dictionary_indices_u32);
        copy_host_vector_to_device(dictionary_encoding.codes_u8,
                                   &lp_info_gpu->AT_dictionary_codes_u8);
        copy_host_vector_to_device(dictionary_encoding.codes_u16,
                                   &lp_info_gpu->AT_dictionary_codes_u16);
    }

    const bool packed_dictionary_y_enabled =
        lp_info_gpu->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::PackedU32 ||
        lp_info_gpu->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::SeparateU32U16;
    if (packed_dictionary_y_enabled) {
        HPRLPPackedDictionaryHost A_dictionary_encoding;
        const bool A_dictionary_encoded =
            hprlp_build_packed_dictionary_operator(
                lp_info_cpu->A->value, lp_info_cpu->A->colIndex,
                static_cast<std::size_t>(lp_info_cpu->A->numElements),
                static_cast<std::size_t>(n), &A_dictionary_encoding);
        const bool supported_storage =
            A_dictionary_encoding.storage ==
                HPRLPPackedDictionaryStorage::PackedU32 ||
            A_dictionary_encoding.storage ==
                HPRLPPackedDictionaryStorage::SeparateU32U16;
        if (A_dictionary_encoded && supported_storage) {
            lp_info_gpu->A_coefficient_dictionary_size =
                static_cast<int>(A_dictionary_encoding.dictionary.size());
            lp_info_gpu->A_packed_dictionary_storage =
                A_dictionary_encoding.storage;
            lp_info_gpu->A_packed_dictionary_code_bits =
                A_dictionary_encoding.code_bits;
            copy_host_vector_to_device(
                A_dictionary_encoding.dictionary,
                &lp_info_gpu->A_coefficient_dictionary);
            copy_host_vector_to_device(
                A_dictionary_encoding.packed_u32,
                &lp_info_gpu->A_dictionary_packed_u32);
            copy_host_vector_to_device(
                A_dictionary_encoding.indices_u32,
                &lp_info_gpu->A_dictionary_indices_u32);
            copy_host_vector_to_device(
                A_dictionary_encoding.codes_u16,
                &lp_info_gpu->A_dictionary_codes_u16);
        }
    }

    const bool packed_state_enabled =
        lp_info_gpu->packed_dictionary_storage ==
            HPRLPPackedDictionaryStorage::SeparateU32U16;
    if (packed_state_enabled) {
        lp_info_gpu->packed_state_plan =
            hprlp_build_packed_state_plan(lp_info_cpu);
    }

    if (lp_info_gpu->mixed_signed_unit_coefficients) {
        HPRLPSignedUnitPackedHost packed_A;
        HPRLPSignedUnitPackedHost packed_AT;
        if (!hprlp_build_signed_unit_entries(
                n, lp_info_cpu->A->numElements,
                lp_info_cpu->A->colIndex, lp_info_cpu->A->value,
                &packed_A) ||
            !hprlp_build_signed_unit_entries(
                m, AT_host.numElements, AT_host.colIndex, AT_host.value,
                &packed_AT)) {
            throw std::runtime_error(
                "failed to build signed-unit packed operator");
        }
        HPRLPFixedDegreeRunHost A_degree2_run =
            hprlp_find_longest_fixed_degree_run(
                m, lp_info_cpu->A->rowPtr, 2);
        HPRLPFixedDegreeRunHost AT_degree3_run =
            hprlp_find_longest_fixed_degree_run(
                n, AT_host.rowPtr, 3);
        if (!hprlp_use_fixed_degree_run(A_degree2_run, m)) {
            A_degree2_run = HPRLPFixedDegreeRunHost();
        }
        if (!hprlp_use_fixed_degree_run(AT_degree3_run, n)) {
            AT_degree3_run = HPRLPFixedDegreeRunHost();
        }
        lp_info_gpu->signed_unit_operator =
            copy_signed_unit_operator_to_device(
                packed_A, packed_AT, A_degree2_run, AT_degree3_run);
    }

    if (!lp_info_gpu->all_positive_unit_coefficients) {
        HPRLPStructuredOperatorHost structured_host;
        if (hprlp_build_dense_two_sparse_operator(
                full_A_view, full_AT_view, &structured_host)) {
            lp_info_gpu->structured_operator =
                copy_structured_operator_to_device(structured_host);
        }
    }

    HPRLPRowTemplateHost row_template_A;
    HPRLPRowTemplateHost row_template_AT;
    const bool row_template_A_ready = hprlp_build_row_template_operator(
        full_A_view, &row_template_A);
    const bool row_template_AT_ready = hprlp_build_row_template_operator(
        full_AT_view, &row_template_AT);
    if (row_template_A_ready && row_template_AT_ready) {
        lp_info_gpu->row_template_operator =
            copy_row_template_operator_to_device(
                row_template_A, row_template_AT);
    }

    HPRLPAffineBlockHost affine_block_A;
    HPRLPAffineBlockHost affine_block_AT;
    const bool affine_block_A_ready = hprlp_build_affine_block_operator(
        full_A_view, &affine_block_A);
    const bool affine_block_AT_ready = hprlp_build_affine_block_operator(
        full_AT_view, &affine_block_AT);
    if (affine_block_A_ready && affine_block_AT_ready) {
        lp_info_gpu->affine_block_operator =
            copy_affine_block_operator_to_device(affine_block_A,
                                                 affine_block_AT);
    }

    HPRLPWindowedStencilShape windowed_stencil_shape;
    if (hprlp_detect_windowed_stencil_operator(
            full_A_view, full_AT_view, &windowed_stencil_shape) &&
        hprlp_use_windowed_stencil_operator(windowed_stencil_shape)) {
        lp_info_gpu->windowed_stencil_operator = true;
        lp_info_gpu->windowed_stencil_shape = windowed_stencil_shape;
    }

    HPRLPGridSlackLaplacianShape grid_slack_laplacian_shape;
    if (hprlp_detect_grid_slack_laplacian_operator(
            full_A_view, full_AT_view, &grid_slack_laplacian_shape) &&
        hprlp_use_grid_slack_laplacian_operator(
            grid_slack_laplacian_shape)) {
        lp_info_gpu->grid_slack_laplacian_operator = true;
        lp_info_gpu->grid_slack_laplacian_shape =
            grid_slack_laplacian_shape;
    }

    // Free the temporary host AT
    free(AT_host.value);
    free(AT_host.colIndex);
    free(AT_host.rowPtr);

    CUDA_CHECK(hprlp_device_malloc_compressible(&lp_info_gpu->AL, m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->AL, lp_info_cpu->AL, m * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(hprlp_device_malloc_compressible(&lp_info_gpu->AU, m * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->AU, lp_info_cpu->AU, m * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(hprlp_device_malloc_compressible(&lp_info_gpu->l, n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->l, lp_info_cpu->l, n * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
    CUDA_CHECK(hprlp_device_malloc_compressible(&lp_info_gpu->u, n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->u, lp_info_cpu->u, n * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));

    CUDA_CHECK(hprlp_device_malloc_compressible(&lp_info_gpu->c, n * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMemcpy(lp_info_gpu->c, lp_info_cpu->c, n * sizeof(HPRLP_FLOAT), cudaMemcpyHostToDevice));
}
