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

void hprlp_release_sparse_matrix_storage(sparseMatrix *matrix) {
    if (matrix == nullptr) return;
    hprlp_device_free(matrix->rowPtr);
    hprlp_device_free(matrix->colIndex);
    hprlp_device_free(matrix->value);
    delete matrix;
}

}  // namespace

bool build_stable_device_transpose(const sparseMatrix *matrix,
                                   sparseMatrix **transpose_out) {
    if (matrix == nullptr || transpose_out == nullptr ||
        matrix->row <= 0 || matrix->col <= 0 || matrix->numElements <= 0 ||
        matrix->rowPtr == nullptr || matrix->colIndex == nullptr ||
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
