void prepare_spmv(HPRLP_workspace_gpu *workspace) {
    int n = workspace->n;
    int m = workspace->m;
    workspace->spmv_A = new CUSPARSE_spmvop_A{};
    workspace->spmv_AT = new CUSPARSE_spmvop_AT{};
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    workspace->spmv_A->cusparseHandle = cusparseHandle;
    workspace->spmv_AT->cusparseHandle = cusparseHandle;
    workspace->spmv_A->alpha = 1.0;
    workspace->spmv_A->beta = 0.0;
    workspace->spmv_AT->alpha = 1.0;
    workspace->spmv_AT->beta = 0.0;
    workspace->spmv_A->computeType = CUDA_R_64F;
    workspace->spmv_AT->computeType = CUDA_R_64F;
    cusparseCreateDnVec(&workspace->spmv_A->x_bar_cusparseDescr, n, workspace->x_bar, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_A->x_hat_cusparseDescr, n, workspace->x_hat, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_A->x_temp_cusparseDescr, n, workspace->x_temp, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_AT->y_bar_cusparseDescr, m, workspace->y_bar, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_AT->y_cusparseDescr, m, workspace->y, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_AT->ATy_cusparseDescr, n, workspace->ATy, CUDA_R_64F);
    cusparseCreateDnVec(&workspace->spmv_A->Ax_cusparseDescr, m, workspace->Ax, CUDA_R_64F);

    // CSR Sparse Matrix Descriptor
    cusparseCreateCsr(&workspace->spmv_A->A_cusparseDescr, workspace->m, workspace->n, workspace->A->numElements,
                workspace->A->rowPtr, workspace->A->colIndex, workspace->A->value,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&workspace->spmv_AT->AT_cusparseDescr, workspace->n, workspace->m, workspace->AT->numElements,
                workspace->AT->rowPtr, workspace->AT->colIndex, workspace->AT->value,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);


    CUSPARSE_CHECK(hprlp_prepare_spmvop(
        cusparseHandle, workspace->spmv_A->A_cusparseDescr,
        workspace->spmv_A->x_bar_cusparseDescr,
        workspace->spmv_A->Ax_cusparseDescr,
        workspace->spmv_A->Ax_cusparseDescr,
        workspace->spmv_A->computeType, &workspace->spmv_A->operation));
    CUSPARSE_CHECK(hprlp_prepare_spmvop(
        cusparseHandle, workspace->spmv_AT->AT_cusparseDescr,
        workspace->spmv_AT->y_bar_cusparseDescr,
        workspace->spmv_AT->ATy_cusparseDescr,
        workspace->spmv_AT->ATy_cusparseDescr,
        workspace->spmv_AT->computeType, &workspace->spmv_AT->operation));

    CUSPARSE_CHECK(hprlp_run_spmvop(
        cusparseHandle, workspace->spmv_A->operation,
        &workspace->spmv_A->alpha, &workspace->spmv_A->beta,
        workspace->spmv_A->x_bar_cusparseDescr,
        workspace->spmv_A->Ax_cusparseDescr,
        workspace->spmv_A->Ax_cusparseDescr));
}
void analyze_spmv_pattern(HPRLP_workspace_gpu *workspace, const HPRLP_parameters *param) {
    const int m = workspace->m;
    const int n = workspace->n;

    prepare_spmv(workspace);
    cusparseSetStream(workspace->spmv_A->cusparseHandle, workspace->stream);
    if (workspace->spmv_AT->cusparseHandle != workspace->spmv_A->cusparseHandle) {
         cusparseSetStream(workspace->spmv_AT->cusparseHandle, workspace->stream);
    }
    // Julia allocates bound classifications for the reduced pointwise
    // updates even when the full iteration path is cuSPARSE SpMVOp-only.
    if (!param->CUSPARSE_spmv || param->use_reduced_matrix) {
        build_bound_types(workspace->l, workspace->u, n,
                          &workspace->x_bound_type, workspace->stream);
        build_bound_types(workspace->AL, workspace->AU, m,
                          &workspace->y_bound_type, workspace->stream);
    }
    if (!param->CUSPARSE_spmv) {
        build_row_buckets(workspace->A, &workspace->A_rows_short, &workspace->num_A_rows_short,
                          &workspace->A_rows_medium, &workspace->num_A_rows_medium,
                          &workspace->A_rows_long, &workspace->num_A_rows_long,
                          &workspace->max_A_row_nnz, workspace->stream);
        build_row_buckets(workspace->AT, &workspace->AT_rows_short, &workspace->num_AT_rows_short,
                          &workspace->AT_rows_medium, &workspace->num_AT_rows_medium,
                          &workspace->AT_rows_long, &workspace->num_AT_rows_long,
                          &workspace->max_AT_row_nnz, workspace->stream);
        build_segmented_A_plan(workspace->A, workspace);
        build_segmented_AT_plan(workspace->AT, workspace);
        if (workspace->A_packed_dictionary_storage ==
                HPRLPPackedDictionaryStorage::PackedU32 &&
            workspace->A_dictionary_packed_u32 != nullptr) {
            build_fixed_degree_run_plan_device(
                workspace->A, &workspace->fixed_degree_A_plan,
                workspace->A_dictionary_packed_u32);
        }
        if (workspace->packed_dictionary_storage ==
                HPRLPPackedDictionaryStorage::PackedU32 &&
            workspace->AT_dictionary_packed_u32 != nullptr) {
            build_fixed_degree_run_plan_device(
                workspace->AT, &workspace->fixed_degree_AT_plan,
                workspace->AT_dictionary_packed_u32);
        }
        if (hprlp_all_rows_short(
                m, workspace->num_A_rows_short,
                workspace->num_A_rows_medium,
                workspace->num_A_rows_long) &&
            hprlp_indices_fit_u16(n)) {
            build_compact_column_indices(
                workspace->A, n, &workspace->A_col_index_u16);
        }
    }
}
