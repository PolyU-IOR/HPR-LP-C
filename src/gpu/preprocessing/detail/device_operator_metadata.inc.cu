namespace {

struct HPRLPDeviceEqualsValue {
    HPRLP_FLOAT expected;
    __host__ __device__ bool operator()(HPRLP_FLOAT value) const {
        return value == expected;
    }
};

__host__ __device__ inline std::uint64_t hprlp_device_raw_bits(
    HPRLP_FLOAT value) {
    union {
        HPRLP_FLOAT value;
        std::uint64_t bits;
    } converter;
    converter.value = value;
    return converter.bits;
}

struct HPRLPDeviceFlagBreak {
    __host__ __device__ int operator()(int flag) const {
        return flag == 0 ? 1 : 0;
    }
};

template <typename Predicate>
std::pair<int, int> hprlp_device_longest_run(int count,
                                             Predicate predicate) {
    if (count <= 0) return std::make_pair(0, 0);
    thrust::device_vector<int> flags(static_cast<std::size_t>(count));
    thrust::device_vector<int> keys(static_cast<std::size_t>(count));
    thrust::device_vector<int> lengths(static_cast<std::size_t>(count));
    const auto first = thrust::make_counting_iterator<int>(0);
    thrust::transform(first, first + count, flags.begin(), predicate);
    thrust::transform(flags.begin(), flags.end(), keys.begin(),
                      HPRLPDeviceFlagBreak{});
    thrust::inclusive_scan(keys.begin(), keys.end(), keys.begin());
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(), flags.begin(),
                                  lengths.begin());
    const auto best = thrust::max_element(lengths.begin(), lengths.end());
    const int best_count = *best;
    const int best_end = static_cast<int>(best - lengths.begin());
    return std::make_pair(best_end - best_count + 1, best_count);
}

struct HPRLPDeviceRawPositiveZeroAt {
    const HPRLP_FLOAT *values;
    __host__ __device__ int operator()(int index) const {
        return hprlp_device_raw_bits(values[index]) == 0 ? 1 : 0;
    }
};

struct HPRLPDeviceZeroLowerBoxedAt {
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    __host__ __device__ int operator()(int index) const {
        return hprlp_device_raw_bits(lower[index]) == 0 &&
               upper[index] < HPRLP_FLOAT(1e90) ? 1 : 0;
    }
};

struct HPRLPDeviceUpperZeroAt {
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    __host__ __device__ int operator()(int index) const {
        return lower[index] <= HPRLP_FLOAT(-1e90) &&
               hprlp_device_raw_bits(upper[index]) == 0 ? 1 : 0;
    }
};

struct HPRLPDeviceLowerOnlyAt {
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    __host__ __device__ int operator()(int index) const {
        return lower[index] > HPRLP_FLOAT(-1e90) &&
               upper[index] >= HPRLP_FLOAT(1e90) ? 1 : 0;
    }
};

struct HPRLPDeviceUpperOnlyAt {
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    __host__ __device__ int operator()(int index) const {
        return lower[index] <= HPRLP_FLOAT(-1e90) &&
               upper[index] < HPRLP_FLOAT(1e90) ? 1 : 0;
    }
};

struct HPRLPDeviceEqualityAt {
    const HPRLP_FLOAT *lower;
    const HPRLP_FLOAT *upper;
    __host__ __device__ int operator()(int index) const {
        return lower[index] > HPRLP_FLOAT(-1e90) &&
               upper[index] < HPRLP_FLOAT(1e90) &&
               hprlp_device_raw_bits(lower[index]) ==
                   hprlp_device_raw_bits(upper[index]) ? 1 : 0;
    }
};

struct HPRLPDeviceAllZeroLowerUnbounded {
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple &values) const {
        const HPRLP_FLOAT lower = thrust::get<0>(values);
        const HPRLP_FLOAT upper = thrust::get<1>(values);
        return lower == HPRLP_FLOAT(0) && isinf(upper) && upper > 0;
    }
};

struct HPRLPDeviceRowDegreeAt {
    const int *row_ptr;
    int degree;
    __host__ __device__ int operator()(int row) const {
        return row_ptr[row + 1] - row_ptr[row] == degree ? 1 : 0;
    }
};

__global__ void hprlp_pack_signed_unit_kernel(
    const int *indices, const HPRLP_FLOAT *values, int nonzeros,
    int input_count, bool uses_u16, bool split_u16,
    std::uint16_t *entries_u16, std::uint32_t *entries_u32,
    std::uint16_t *split_indices_u16,
    std::uint8_t *split_negative_u8, int *invalid) {
    const int stride = blockDim.x * gridDim.x;
    for (int entry = blockIdx.x * blockDim.x + threadIdx.x;
         entry < nonzeros; entry += stride) {
        const int index = indices[entry];
        const bool negative = values[entry] == HPRLP_FLOAT(-1);
        if (index < 0 || index >= input_count ||
            (!negative && values[entry] != HPRLP_FLOAT(1))) {
            atomicExch(invalid, 1);
            continue;
        }
        if (uses_u16) {
            entries_u16[entry] = static_cast<std::uint16_t>(index) |
                (negative ? HPRLP_SIGNED_U16_SIGN_MASK : 0u);
        } else {
            entries_u32[entry] = static_cast<std::uint32_t>(index) |
                (negative ? HPRLP_SIGNED_U32_SIGN_MASK : 0u);
        }
        if (split_u16) {
            split_indices_u16[entry] = static_cast<std::uint16_t>(index);
            split_negative_u8[entry] = negative ? 1 : 0;
        }
    }
}

template <typename T>
T *hprlp_allocate_device_metadata(std::size_t count) {
    if (count == 0) return nullptr;
    T *output = nullptr;
    CUDA_CHECK(hprlp_device_malloc_compressible(
        &output, count * sizeof(T)));
    return output;
}

void hprlp_build_device_signed_entries(
    const sparseMatrix *matrix, int input_count, bool uses_u16,
    bool split_u16, std::uint16_t **entries_u16,
    std::uint32_t **entries_u32, std::uint16_t **split_indices_u16,
    std::uint8_t **split_negative_u8) {
    const std::size_t count =
        static_cast<std::size_t>(matrix->numElements);
    if (uses_u16) {
        *entries_u16 = hprlp_allocate_device_metadata<std::uint16_t>(count);
    } else {
        *entries_u32 = hprlp_allocate_device_metadata<std::uint32_t>(count);
    }
    if (split_u16) {
        *split_indices_u16 =
            hprlp_allocate_device_metadata<std::uint16_t>(count);
        *split_negative_u8 =
            hprlp_allocate_device_metadata<std::uint8_t>(count);
    }
    int *invalid = hprlp_allocate_device_metadata<int>(1);
    CUDA_CHECK(cudaMemset(invalid, 0, sizeof(int)));
    const int threads = 256;
    const int blocks = std::min(65535,
        (matrix->numElements + threads - 1) / threads);
    hprlp_pack_signed_unit_kernel<<<blocks, threads>>>(
        matrix->colIndex, matrix->value, matrix->numElements, input_count,
        uses_u16, split_u16, *entries_u16, *entries_u32,
        *split_indices_u16, *split_negative_u8, invalid);
    CUDA_CHECK(cudaGetLastError());
    int invalid_host = 0;
    CUDA_CHECK(cudaMemcpy(&invalid_host, invalid, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(hprlp_device_free(invalid));
    if (invalid_host != 0) {
        throw std::runtime_error("invalid signed-unit device encoding");
    }
}

HPRLPFixedDegreeRunHost hprlp_device_signed_fixed_degree_run(
    const sparseMatrix *matrix, int degree) {
    HPRLPFixedDegreeRunHost output;
    const std::pair<int, int> run = hprlp_device_longest_run(
        matrix->row, HPRLPDeviceRowDegreeAt{matrix->rowPtr, degree});
    output.degree = degree;
    output.row_begin = run.first;
    output.row_count = run.second;
    if (!hprlp_use_fixed_degree_run(output, matrix->row)) {
        return HPRLPFixedDegreeRunHost{};
    }
    CUDA_CHECK(cudaMemcpy(&output.entry_begin,
                          matrix->rowPtr + output.row_begin, sizeof(int),
                          cudaMemcpyDeviceToHost));
    return output;
}

__global__ void hprlp_build_coltile_keys_kernel(
    const int *row_ptr, const int *col_index, int rows, int tile_cols,
    int tile_count, std::uint64_t *keys) {
    const int stride = blockDim.x * gridDim.x;
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows; row += stride) {
        const std::uint64_t row_base =
            static_cast<std::uint64_t>(row) * tile_count;
        for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
            const int column = col_index[entry];
            const std::uint64_t pair = row_base + column / tile_cols;
            const std::uint64_t local =
                static_cast<std::uint64_t>(column % tile_cols);
            keys[entry] = (pair << 16) | local;
        }
    }
}

struct HPRLPDeviceColtileQuery {
    __host__ __device__ std::uint64_t operator()(int pair) const {
        return static_cast<std::uint64_t>(pair) << 16;
    }
};

struct HPRLPDeviceColtileLocal {
    __host__ __device__ std::uint16_t operator()(std::uint64_t key) const {
        return static_cast<std::uint16_t>(key & UINT64_C(0xffff));
    }
};

HPRLP_unit_coltile_gpu *hprlp_build_device_unit_coltile(
    const sparseMatrix *matrix) {
    const int tile_cols = HPRLP_UNIT_COLTILE_COLUMNS;
    const int tile_count = static_cast<int>(
        (static_cast<long long>(matrix->col) + tile_cols - 1) / tile_cols);
    const long long pair_count_ll =
        static_cast<long long>(matrix->row) * tile_count;
    if (pair_count_ll <= 0 ||
        pair_count_ll >= std::numeric_limits<int>::max()) {
        return nullptr;
    }
    const int pair_count = static_cast<int>(pair_count_ll);
    const std::size_t nonzeros =
        static_cast<std::size_t>(matrix->numElements);
    thrust::device_vector<std::uint64_t> keys(nonzeros);
    const int threads = 256;
    const int blocks = std::min(65535,
        (matrix->row + threads - 1) / threads);
    hprlp_build_coltile_keys_kernel<<<blocks, threads>>>(
        matrix->rowPtr, matrix->colIndex, matrix->row, tile_cols,
        tile_count, thrust::raw_pointer_cast(keys.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::sort(keys.begin(), keys.end());

    HPRLP_unit_coltile_gpu *output = new HPRLP_unit_coltile_gpu;
    output->rows = matrix->row;
    output->columns = matrix->col;
    output->tile_cols = tile_cols;
    output->tile_count = tile_count;
    output->row_tile_offsets =
        hprlp_allocate_device_metadata<int>(
            static_cast<std::size_t>(pair_count) + 1u);
    output->local_cols =
        hprlp_allocate_device_metadata<std::uint16_t>(nonzeros);

    const auto query_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(0), HPRLPDeviceColtileQuery{});
    thrust::lower_bound(
        keys.begin(), keys.end(), query_begin, query_begin + pair_count + 1,
        thrust::device_pointer_cast(output->row_tile_offsets));
    thrust::transform(keys.begin(), keys.end(),
                      thrust::device_pointer_cast(output->local_cols),
                      HPRLPDeviceColtileLocal{});
    return output;
}

void hprlp_prepare_device_state_metadata(LP_info_gpu *lp) {
    const std::pair<int, int> objective_zero = hprlp_device_longest_run(
        lp->n, HPRLPDeviceRawPositiveZeroAt{lp->c});
    const std::pair<int, int> y_upper_zero = hprlp_device_longest_run(
        lp->m, HPRLPDeviceUpperZeroAt{lp->AL, lp->AU});
    const std::pair<int, int> zero_lower_boxed = hprlp_device_longest_run(
        lp->n, HPRLPDeviceZeroLowerBoxedAt{lp->l, lp->u});

    const auto bounds_begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(lp->l),
        thrust::device_pointer_cast(lp->u)));
    const int zero_lower_unbounded_count = static_cast<int>(thrust::count_if(
        bounds_begin, bounds_begin + lp->n,
        HPRLPDeviceAllZeroLowerUnbounded{}));
    lp->all_zero_lower_unbounded_variables =
        zero_lower_unbounded_count == lp->n;
    lp->signed_x_zero_objective_run_begin = objective_zero.first;
    lp->signed_x_zero_objective_run_count = objective_zero.second;
    lp->signed_y_upper_zero_run_begin = y_upper_zero.first;
    lp->signed_y_upper_zero_run_count = y_upper_zero.second;
    const bool all_x_boxed_zero_lower = zero_lower_boxed.second == lp->n;
    lp->signed_state_plan_ready = lp->n >= 1000000 && lp->m >= 1000000 &&
        all_x_boxed_zero_lower &&
        static_cast<long long>(objective_zero.second) * 2 >= lp->n &&
        static_cast<long long>(y_upper_zero.second) * 2 >= lp->m;

    if (lp->packed_dictionary_storage !=
        HPRLPPackedDictionaryStorage::SeparateU32U16) {
        return;
    }
    HPRLPPackedStatePlan plan{};
    if (static_cast<long long>(zero_lower_boxed.second) * 4 >= lp->n) {
        plan.x_zero_lower_boxed_begin = zero_lower_boxed.first;
        plan.x_zero_lower_boxed_count = zero_lower_boxed.second;
    }
    if (static_cast<long long>(objective_zero.second) * 4 >= lp->n) {
        plan.x_objective_zero_begin = objective_zero.first;
        plan.x_objective_zero_count = objective_zero.second;
    }
    const std::pair<int, int> lower_only = hprlp_device_longest_run(
        lp->m, HPRLPDeviceLowerOnlyAt{lp->AL, lp->AU});
    const std::pair<int, int> upper_only = hprlp_device_longest_run(
        lp->m, HPRLPDeviceUpperOnlyAt{lp->AL, lp->AU});
    const std::pair<int, int> equality = hprlp_device_longest_run(
        lp->m, HPRLPDeviceEqualityAt{lp->AL, lp->AU});
    const long long covered = static_cast<long long>(lower_only.second) +
        upper_only.second + equality.second;
    if (covered * 5 >= static_cast<long long>(lp->m) * 2) {
        plan.y_lower_only_begin = lower_only.first;
        plan.y_lower_only_count = lower_only.second;
        plan.y_upper_only_begin = upper_only.first;
        plan.y_upper_only_count = upper_only.second;
        plan.y_equality_begin = equality.first;
        plan.y_equality_count = equality.second;
    }
    lp->packed_state_plan = plan;
}

struct HPRLPDeviceRawRowDegree {
    const int *row_ptr;
    __host__ __device__ int operator()(int row) const {
        return row_ptr[row + 1] - row_ptr[row];
    }
};

struct HPRLPDeviceFixedDegreeCandidate {
    int row_begin = 0;
    int row_count = 0;
    int entry_begin = 0;
    int degree = 0;
    long long covered_nnz = 0;
};

struct HPRLPDeviceMakeFixedDegreeCandidate {
    const int *degrees;
    const int *run_counts;
    const int *run_begins;
    const int *row_ptr;
    __host__ __device__ HPRLPDeviceFixedDegreeCandidate operator()(
        int run_index) const {
        HPRLPDeviceFixedDegreeCandidate output;
        output.degree = degrees[run_index];
        output.row_count = run_counts[run_index];
        output.row_begin = run_begins[run_index];
        output.entry_begin = row_ptr[output.row_begin];
        output.covered_nnz =
            static_cast<long long>(output.row_count) * output.degree;
        return output;
    }
};

struct HPRLPDeviceAdmissibleFixedDegreeCandidate {
    __host__ __device__ bool operator()(
        const HPRLPDeviceFixedDegreeCandidate &candidate) const {
        return candidate.degree >= 1 &&
               candidate.degree <= HPRLP_SCALAR_ROW_MAX_NNZ &&
               candidate.row_count >= 32768;
    }
};

struct HPRLPDeviceFixedDegreePriority {
    __host__ __device__ bool operator()(
        const HPRLPDeviceFixedDegreeCandidate &left,
        const HPRLPDeviceFixedDegreeCandidate &right) const {
        if (left.covered_nnz != right.covered_nnz) {
            return left.covered_nnz > right.covered_nnz;
        }
        return left.row_count > right.row_count;
    }
};

struct HPRLPDeviceFixedDegreeRowOrder {
    __host__ __device__ bool operator()(
        const HPRLPDeviceFixedDegreeCandidate &left,
        const HPRLPDeviceFixedDegreeCandidate &right) const {
        return left.row_begin < right.row_begin;
    }
};

struct HPRLPDeviceCandidateRows {
    __host__ __device__ long long operator()(
        const HPRLPDeviceFixedDegreeCandidate &candidate) const {
        return candidate.row_count;
    }
};

struct HPRLPDeviceCandidateNnz {
    __host__ __device__ long long operator()(
        const HPRLPDeviceFixedDegreeCandidate &candidate) const {
        return candidate.covered_nnz;
    }
};

__global__ void hprlp_mark_fixed_degree_rows_kernel(
    std::uint8_t *covered,
    const HPRLPDeviceFixedDegreeCandidate *runs, int run_count) {
    const int run_index = blockIdx.y;
    if (run_index >= run_count) return;
    const HPRLPDeviceFixedDegreeCandidate run = runs[run_index];
    const int stride = blockDim.x * gridDim.x;
    for (int offset = blockIdx.x * blockDim.x + threadIdx.x;
         offset < run.row_count; offset += stride) {
        covered[run.row_begin + offset] = 1;
    }
}

struct HPRLPDeviceFallbackRowPredicate {
    const int *row_ptr;
    const std::uint8_t *covered;
    HPRLPRowBucket bucket;
    __host__ __device__ bool operator()(int row) const {
        if (covered[row] != 0) return false;
        const int degree = row_ptr[row + 1] - row_ptr[row];
        const HPRLPRowBucket current =
            degree <= HPRLP_SCALAR_ROW_MAX_NNZ
                ? HPRLP_ROW_SCALAR
                : (degree <= HPRLP_WARP_ROW_MAX_NNZ
                       ? HPRLP_ROW_WARP
                       : HPRLP_ROW_BLOCK);
        return current == bucket;
    }
};

void hprlp_copy_device_ints_to_metadata(
    const thrust::device_vector<int> &source, int count, int **output) {
    if (count <= 0) {
        *output = nullptr;
        return;
    }
    *output = hprlp_allocate_device_metadata<int>(
        static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(
        *output, thrust::raw_pointer_cast(source.data()),
        static_cast<std::size_t>(count) * sizeof(int),
        cudaMemcpyDeviceToDevice));
}

void build_fixed_degree_run_plan_device(
    const sparseMatrix *matrix, HPRLPFixedDegreeRunPlan *plan,
    const std::uint32_t *packed_entries) {
    if (matrix == nullptr || plan == nullptr || matrix->row <= 0 ||
        matrix->rowPtr == nullptr || packed_entries == nullptr) {
        return;
    }
    *plan = HPRLPFixedDegreeRunPlan{};
    constexpr int threads = 128;
    plan->threads = threads;

    const int rows = matrix->row;
    const auto first = thrust::make_counting_iterator<int>(0);
    thrust::device_vector<int> row_degrees(static_cast<std::size_t>(rows));
    thrust::transform(first, first + rows, row_degrees.begin(),
                      HPRLPDeviceRawRowDegree{matrix->rowPtr});
    thrust::device_vector<int> run_degrees(static_cast<std::size_t>(rows));
    thrust::device_vector<int> run_counts(static_cast<std::size_t>(rows));
    const auto run_end = thrust::reduce_by_key(
        row_degrees.begin(), row_degrees.end(),
        thrust::make_constant_iterator<int>(1), run_degrees.begin(),
        run_counts.begin());
    const int run_count =
        static_cast<int>(run_end.first - run_degrees.begin());
    row_degrees.clear();
    row_degrees.shrink_to_fit();
    run_degrees.resize(static_cast<std::size_t>(run_count));
    run_counts.resize(static_cast<std::size_t>(run_count));
    thrust::device_vector<int> run_begins(static_cast<std::size_t>(run_count));
    thrust::exclusive_scan(run_counts.begin(), run_counts.end(),
                           run_begins.begin());

    thrust::device_vector<HPRLPDeviceFixedDegreeCandidate> all_candidates(
        static_cast<std::size_t>(run_count));
    thrust::transform(
        first, first + run_count, all_candidates.begin(),
        HPRLPDeviceMakeFixedDegreeCandidate{
            thrust::raw_pointer_cast(run_degrees.data()),
            thrust::raw_pointer_cast(run_counts.data()),
            thrust::raw_pointer_cast(run_begins.data()), matrix->rowPtr});
    thrust::device_vector<HPRLPDeviceFixedDegreeCandidate> candidates(
        static_cast<std::size_t>(run_count));
    const auto candidate_end = thrust::copy_if(
        all_candidates.begin(), all_candidates.end(), candidates.begin(),
        HPRLPDeviceAdmissibleFixedDegreeCandidate{});
    int candidate_count =
        static_cast<int>(candidate_end - candidates.begin());
    candidates.resize(static_cast<std::size_t>(candidate_count));
    if (candidate_count == 0) return;
    thrust::sort(candidates.begin(), candidates.end(),
                 HPRLPDeviceFixedDegreePriority{});
    candidate_count = std::min(candidate_count,
                               HPRLP_FIXED_DEGREE_MAX_RUNS);
    candidates.resize(static_cast<std::size_t>(candidate_count));
    const long long covered_rows = thrust::transform_reduce(
        candidates.begin(), candidates.end(), HPRLPDeviceCandidateRows{},
        0LL, thrust::plus<long long>());
    const long long covered_nnz = thrust::transform_reduce(
        candidates.begin(), candidates.end(), HPRLPDeviceCandidateNnz{},
        0LL, thrust::plus<long long>());
    if (covered_rows < 32768 ||
        covered_nnz * 100 < 50LL * matrix->numElements) {
        return;
    }
    thrust::sort(candidates.begin(), candidates.end(),
                 HPRLPDeviceFixedDegreeRowOrder{});

    HPRLPDeviceFixedDegreeCandidate host_candidates[
        HPRLP_FIXED_DEGREE_MAX_RUNS]{};
    CUDA_CHECK(cudaMemcpy(
        host_candidates, thrust::raw_pointer_cast(candidates.data()),
        static_cast<std::size_t>(candidate_count) *
            sizeof(HPRLPDeviceFixedDegreeCandidate),
        cudaMemcpyDeviceToHost));
    plan->run_count = candidate_count;
    plan->covered_rows = covered_rows;
    plan->covered_nnz = covered_nnz;
    for (int run_index = 0; run_index < candidate_count; ++run_index) {
        const HPRLPDeviceFixedDegreeCandidate &source =
            host_candidates[run_index];
        HPRLPFixedDegreeRun &run = plan->runs[run_index];
        run.row_begin = source.row_begin;
        run.row_count = source.row_count;
        run.entry_begin = source.entry_begin;
        run.degree = source.degree;
        const int entry_count = run.row_count * run.degree;
        run.packed_entries_soa =
            hprlp_allocate_device_metadata<std::uint32_t>(
                static_cast<std::size_t>(entry_count));
        repack_fixed_degree_run_soa_kernel<<<
            (entry_count + threads - 1) / threads, threads>>>(
            packed_entries, run.packed_entries_soa, run.entry_begin,
            run.row_count, run.degree, entry_count);
    }
    CUDA_CHECK(cudaGetLastError());

    thrust::device_vector<std::uint8_t> covered(
        static_cast<std::size_t>(rows), std::uint8_t{0});
    const int mark_blocks = std::min(65535, (rows + threads - 1) / threads);
    hprlp_mark_fixed_degree_rows_kernel<<<
        dim3(mark_blocks, candidate_count), threads>>>(
        thrust::raw_pointer_cast(covered.data()),
        thrust::raw_pointer_cast(candidates.data()), candidate_count);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_vector<int> fallback(static_cast<std::size_t>(rows));
    auto collect_fallback = [&](HPRLPRowBucket bucket, int *count,
                                int **output) {
        const auto end = thrust::copy_if(
            first, first + rows, fallback.begin(),
            HPRLPDeviceFallbackRowPredicate{
                matrix->rowPtr, thrust::raw_pointer_cast(covered.data()),
                bucket});
        *count = static_cast<int>(end - fallback.begin());
        hprlp_copy_device_ints_to_metadata(fallback, *count, output);
    };
    collect_fallback(HPRLP_ROW_SCALAR, &plan->fallback_short_count,
                     &plan->fallback_short_rows);
    collect_fallback(HPRLP_ROW_WARP, &plan->fallback_warp_count,
                     &plan->fallback_warp_rows);
    collect_fallback(HPRLP_ROW_BLOCK, &plan->fallback_block_count,
                     &plan->fallback_block_rows);
    plan->ready = true;
}

}  // namespace

bool prepare_device_operator_metadata(LP_info_gpu *lp) {
    if (lp == nullptr || lp->A == nullptr || lp->AT == nullptr ||
        lp->A->numElements <= 0) {
        return false;
    }
    const std::size_t nonzeros =
        static_cast<std::size_t>(lp->A->numElements);
    const auto values = thrust::device_pointer_cast(lp->A->value);
    const std::size_t positive = static_cast<std::size_t>(thrust::count(
        values, values + nonzeros, HPRLP_FLOAT(1)));
    const std::size_t negative = static_cast<std::size_t>(thrust::count(
        values, values + nonzeros, HPRLP_FLOAT(-1)));
    lp->uniform_unit_sign = positive == nonzeros ? 1 :
        (negative == nonzeros ? -1 : 0);
    lp->all_positive_unit_coefficients = lp->uniform_unit_sign > 0;
    lp->mixed_signed_unit_coefficients = positive > 0 && negative > 0 &&
        positive + negative == nonzeros;

    hprlp_prepare_device_state_metadata(lp);

    if (hprlp_use_unit_coltile(
            lp->uniform_unit_sign, static_cast<std::size_t>(lp->m),
            static_cast<std::size_t>(lp->n), nonzeros)) {
        lp->unit_coltile = hprlp_build_device_unit_coltile(lp->A);
        if (lp->unit_coltile == nullptr) {
            throw std::runtime_error(
                "failed to build GPU-resident unit col-tile operator");
        }
    }

    if (lp->mixed_signed_unit_coefficients) {
        HPRLP_signed_unit_operator_gpu *op =
            new HPRLP_signed_unit_operator_gpu;
        op->A_uses_u16 = hprlp_signed_index_uses_u16(lp->n);
        op->AT_uses_u16 = hprlp_signed_index_uses_u16(lp->m);
        op->A_split_u16_ready = lp->n <= 65536;
        op->AT_split_u16_ready = lp->m <= 65536;
        hprlp_build_device_signed_entries(
            lp->A, lp->n, op->A_uses_u16, op->A_split_u16_ready,
            &op->A_entries_u16, &op->A_entries_u32,
            &op->A_split_indices_u16, &op->A_split_negative_u8);
        hprlp_build_device_signed_entries(
            lp->AT, lp->m, op->AT_uses_u16, op->AT_split_u16_ready,
            &op->AT_entries_u16, &op->AT_entries_u32,
            &op->AT_split_indices_u16, &op->AT_split_negative_u8);
        const HPRLPFixedDegreeRunHost A_degree2 =
            hprlp_device_signed_fixed_degree_run(lp->A, 2);
        const HPRLPFixedDegreeRunHost AT_degree3 =
            hprlp_device_signed_fixed_degree_run(lp->AT, 3);
        op->A_degree2_run_row_begin = A_degree2.row_begin;
        op->A_degree2_run_row_count = A_degree2.row_count;
        op->A_degree2_run_entry_begin = A_degree2.entry_begin;
        op->AT_degree3_run_row_begin = AT_degree3.row_begin;
        op->AT_degree3_run_row_count = AT_degree3.row_count;
        op->AT_degree3_run_entry_begin = AT_degree3.entry_begin;
        lp->signed_unit_operator = op;
    }
    return true;
}
