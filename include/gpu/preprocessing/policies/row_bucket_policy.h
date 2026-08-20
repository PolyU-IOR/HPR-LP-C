#ifndef HPRLP_ROW_BUCKET_POLICY_H
#define HPRLP_ROW_BUCKET_POLICY_H

enum HPRLPRowBucket {
    HPRLP_ROW_SCALAR,
    HPRLP_ROW_WARP,
    HPRLP_ROW_BLOCK
};

constexpr int HPRLP_SCALAR_ROW_MAX_NNZ = 16;
constexpr int HPRLP_WARP_ROW_MAX_NNZ = 1024;
constexpr int HPRLP_UNIT_SCALAR_ROW_MAX_NNZ = 64;

inline HPRLPRowBucket hprlp_row_bucket(int nnz) {
    if (nnz <= HPRLP_SCALAR_ROW_MAX_NNZ) {
        return HPRLP_ROW_SCALAR;
    }
    if (nnz <= HPRLP_WARP_ROW_MAX_NNZ) {
        return HPRLP_ROW_WARP;
    }
    return HPRLP_ROW_BLOCK;
}

inline bool hprlp_all_rows_short(int total_rows, int short_rows,
                                 int warp_rows, int block_rows) {
    return total_rows > 0 && short_rows == total_rows &&
           warp_rows == 0 && block_rows == 0;
}

inline bool hprlp_indices_fit_u16(int column_count) {
    return column_count > 0 && column_count <= 65536;
}

inline bool hprlp_all_rows_fit_scalar(int total_rows, int max_row_nnz,
                                      int scalar_limit) {
    return total_rows > 0 && max_row_nnz >= 0 &&
           max_row_nnz <= scalar_limit;
}

inline bool hprlp_use_direct_short_rows(int total_rows, int short_rows) {
    constexpr int kDirectShortMinimumRows = 2000000;
    return total_rows >= kDirectShortMinimumRows && short_rows > 0 &&
           static_cast<long long>(short_rows) * 10 >=
               static_cast<long long>(total_rows) * 9;
}

#endif
