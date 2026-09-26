namespace {

__global__ void hprlp_fill_csr_source_rows_kernel(
    const int *row_ptr,
    int rows,
    int *source_rows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    for (int entry = row_ptr[row] + threadIdx.x;
         entry < row_ptr[row + 1]; entry += blockDim.x) {
        source_rows[entry] = row;
    }
}

template <typename ColumnIndex>
__global__ void hprlp_count_transpose_rows64_kernel(
    const ColumnIndex *column_indices, std::int64_t nonzeros,
    std::int64_t *row_offsets) {
    const std::int64_t first =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t entry = first; entry < nonzeros; entry += stride) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                row_offsets + column_indices[entry] + 1),
            1ULL);
    }
}

template <typename ColumnIndex>
__global__ void hprlp_fill_transpose64_kernel(
    const std::int64_t *row_offsets, const ColumnIndex *column_indices,
    const HPRLP_FLOAT *values, int rows, std::int64_t *cursors,
    ColumnIndex *transpose_columns, HPRLP_FLOAT *transpose_values) {
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        for (std::int64_t entry = row_offsets[row] + threadIdx.x;
             entry < row_offsets[row + 1]; entry += blockDim.x) {
            const ColumnIndex destination = column_indices[entry];
            const std::int64_t output = static_cast<std::int64_t>(atomicAdd(
                reinterpret_cast<unsigned long long *>(cursors + destination),
                1ULL));
            transpose_columns[output] = row;
            transpose_values[output] = values[entry];
        }
    }
}

void hprlp_release_sparse_matrix_storage(sparseMatrix *matrix) {
    if (matrix == nullptr) return;
    hprlp_device_free(matrix->rowPtr);
    hprlp_device_free(matrix->rowPtr64);
    hprlp_device_free(matrix->colIndex);
    hprlp_device_free(matrix->colIndex64);
    hprlp_device_free(matrix->value);
    delete matrix;
}

}  // namespace

bool build_stable_device_transpose(const sparseMatrix *matrix,
                                   sparseMatrix **transpose_out) {
    if (matrix == nullptr || transpose_out == nullptr ||
        matrix->row <= 0 || matrix->col <= 0 || matrix->numElements <= 0 ||
        (matrix->rowPtr == nullptr && matrix->rowPtr64 == nullptr) ||
        (matrix->colIndex == nullptr &&
         matrix->colIndex64 == nullptr) ||
        matrix->value == nullptr) {
        return false;
    }
    *transpose_out = nullptr;
    const std::size_t nonzeros =
        static_cast<std::size_t>(matrix->numElements);
    sparseMatrix *transpose = new sparseMatrix{};
    transpose->row = matrix->col;
    transpose->col = matrix->row;
    transpose->numElements = matrix->numElements;

    if (hprlp_sparse_has_64bit_offsets(matrix)) {
        std::int64_t *cursors = nullptr;
        try {
            const std::size_t offset_count =
                static_cast<std::size_t>(transpose->row) + 1;
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &transpose->rowPtr64,
                offset_count * sizeof(std::int64_t)));
            CUDA_CHECK(cudaMemset(transpose->rowPtr64, 0,
                                  offset_count * sizeof(std::int64_t)));
            const int threads = 256;
            const int count_blocks = static_cast<int>(std::min<std::int64_t>(
                65535, (matrix->numElements + threads - 1) / threads));
            if (hprlp_sparse_has_64bit_column_indices(matrix)) {
                hprlp_count_transpose_rows64_kernel<<<count_blocks, threads>>>(
                    matrix->colIndex64, matrix->numElements,
                    transpose->rowPtr64);
            } else {
                hprlp_count_transpose_rows64_kernel<<<count_blocks, threads>>>(
                    matrix->colIndex, matrix->numElements,
                    transpose->rowPtr64);
            }
            CUDA_CHECK(cudaGetLastError());
            thrust::inclusive_scan(
                thrust::device_pointer_cast(transpose->rowPtr64),
                thrust::device_pointer_cast(transpose->rowPtr64) +
                    offset_count,
                thrust::device_pointer_cast(transpose->rowPtr64));

            if (hprlp_sparse_has_64bit_column_indices(matrix)) {
                CUDA_CHECK(hprlp_device_malloc_compressible(
                    &transpose->colIndex64,
                    nonzeros * sizeof(std::int64_t)));
            } else {
                CUDA_CHECK(hprlp_device_malloc_compressible(
                    &transpose->colIndex, nonzeros * sizeof(int)));
            }
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &transpose->value, nonzeros * sizeof(HPRLP_FLOAT)));
            CUDA_CHECK(hprlp_device_malloc_compressible(
                &cursors, static_cast<std::size_t>(transpose->row) *
                    sizeof(std::int64_t)));
            CUDA_CHECK(cudaMemcpy(
                cursors, transpose->rowPtr64,
                static_cast<std::size_t>(transpose->row) *
                    sizeof(std::int64_t), cudaMemcpyDeviceToDevice));
            const int fill_blocks = std::min(matrix->row, 65535);
            if (hprlp_sparse_has_64bit_column_indices(matrix)) {
                hprlp_fill_transpose64_kernel<<<fill_blocks, threads>>>(
                    matrix->rowPtr64, matrix->colIndex64, matrix->value,
                    matrix->row, cursors, transpose->colIndex64,
                    transpose->value);
            } else {
                hprlp_fill_transpose64_kernel<<<fill_blocks, threads>>>(
                    matrix->rowPtr64, matrix->colIndex, matrix->value,
                    matrix->row, cursors, transpose->colIndex,
                    transpose->value);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(hprlp_device_free(cursors));
        } catch (...) {
            if (cursors) hprlp_device_free(cursors);
            hprlp_release_sparse_matrix_storage(transpose);
            throw;
        }
        *transpose_out = transpose;
        return true;
    }

    try {
        // The baseline host transpose visits CSR entries in their original
        // global order and appends each entry to its destination row.  A
        // stable sort by destination column reproduces that ordering exactly
        // while keeping all O(nnz) work on the GPU.
        thrust::device_vector<int> sorted_columns(nonzeros);
        thrust::device_vector<std::uint32_t> sorted_entries(nonzeros);
        thrust::device_vector<int> source_rows(nonzeros);
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(sorted_columns.data()),
            matrix->colIndex, nonzeros * sizeof(int),
            cudaMemcpyDeviceToDevice));
        thrust::sequence(
            sorted_entries.begin(), sorted_entries.end(), std::uint32_t{0});
        hprlp_fill_csr_source_rows_kernel<<<matrix->row, 256>>>(
            matrix->rowPtr, matrix->row,
            thrust::raw_pointer_cast(source_rows.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::stable_sort_by_key(
            sorted_columns.begin(), sorted_columns.end(),
            sorted_entries.begin());

        CUDA_CHECK(hprlp_device_malloc_compressible(
            &transpose->rowPtr,
            (static_cast<std::size_t>(transpose->row) + 1) * sizeof(int)));
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &transpose->colIndex,
            nonzeros * sizeof(int)));
        CUDA_CHECK(hprlp_device_malloc_compressible(
            &transpose->value,
            nonzeros * sizeof(HPRLP_FLOAT)));

        thrust::lower_bound(
            sorted_columns.begin(), sorted_columns.end(),
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(transpose->row + 1),
            thrust::device_pointer_cast(transpose->rowPtr));
        thrust::gather(
            sorted_entries.begin(), sorted_entries.end(),
            source_rows.begin(),
            thrust::device_pointer_cast(transpose->colIndex));
        thrust::gather(
            sorted_entries.begin(), sorted_entries.end(),
            thrust::device_pointer_cast(matrix->value),
            thrust::device_pointer_cast(transpose->value));
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        hprlp_release_sparse_matrix_storage(transpose);
        throw;
    }

    *transpose_out = transpose;
    return true;
}
