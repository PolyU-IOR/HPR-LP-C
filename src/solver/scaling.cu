#include "solver/scaling.h"

#include <cstddef>
#include <cmath>
#include <iostream>

namespace {

constexpr int kScalingCooperativeRowThreshold = 256;
constexpr unsigned kScalingFullWarpMask = 0xffffffffu;

__global__ void curtis_reid_log_update_kernel(int m,
                                              const int *rowPtr,
                                              const int *colIndex,
                                              const HPRLP_FLOAT *value,
                                              const HPRLP_FLOAT *neg_log_abs_value,
                                              const HPRLP_FLOAT *other_log_scale,
                                              HPRLP_FLOAT *result) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int warp_base = threadIdx.x - lane;
    const int start = row < m ? rowPtr[row] : 0;
    const int end = row < m ? rowPtr[row + 1] : 0;
    const int count = end - start;
    const bool cooperative =
        row < m && count > kScalingCooperativeRowThreshold;

    // A warp services only the exceptionally long rows among its 32 ordinary
    // row assignments.  The lanes preload one CSR-order tile, while lane zero
    // performs every addition in the original left-to-right order.  This
    // exposes memory-level parallelism without reassociating the reduction.
    __shared__ HPRLP_FLOAT ordered_terms[HPRLP_NUM_THREADS];
    unsigned cooperative_rows =
        __ballot_sync(kScalingFullWarpMask, cooperative);
    while (cooperative_rows != 0u) {
        const int owner_lane = __ffs(cooperative_rows) - 1;
        const int cooperative_row =
            __shfl_sync(kScalingFullWarpMask, row, owner_lane);
        const int cooperative_start =
            __shfl_sync(kScalingFullWarpMask, start, owner_lane);
        const int cooperative_end =
            __shfl_sync(kScalingFullWarpMask, end, owner_lane);
        const int cooperative_count = cooperative_end - cooperative_start;

        HPRLP_FLOAT sum = 0.0;
        for (int base = cooperative_start;
             base < cooperative_end; base += warpSize) {
            const int idx = base + lane;
            HPRLP_FLOAT ordered_term = 0.0;
            if (idx < cooperative_end) {
                HPRLP_FLOAT term;
                if (neg_log_abs_value != nullptr) {
                    term = neg_log_abs_value[idx];
                } else {
                    const HPRLP_FLOAT abs_value =
                        fmax(fabs(value[idx]), 1e-300);
                    term = -log(abs_value);
                }
                ordered_term =
                    term - other_log_scale[colIndex[idx]];
            }
            ordered_terms[threadIdx.x] = ordered_term;
            __syncwarp(kScalingFullWarpMask);

            if (lane == 0) {
                const int tile_count =
                    cooperative_end - base < warpSize
                        ? cooperative_end - base
                        : warpSize;
                for (int offset = 0; offset < tile_count; ++offset) {
                    sum += ordered_terms[warp_base + offset];
                }
            }
            __syncwarp(kScalingFullWarpMask);
        }
        if (lane == 0) {
            result[cooperative_row] =
                sum / static_cast<HPRLP_FLOAT>(cooperative_count);
        }
        cooperative_rows &= cooperative_rows - 1u;
    }

    if (row >= m || cooperative) {
        return;
    }
    if (count <= 0) {
        result[row] = 0.0;
        return;
    }

    HPRLP_FLOAT sum = 0.0;
    for (int idx = start; idx < end; ++idx) {
        HPRLP_FLOAT term;
        if (neg_log_abs_value != nullptr) {
            term = neg_log_abs_value[idx];
        } else {
            const HPRLP_FLOAT abs_value = fmax(fabs(value[idx]), 1e-300);
            term = -log(abs_value);
        }
        // Cached and uncached execution converge here so the subtraction and
        // row accumulation use one compiled instruction sequence.
        sum += term - other_log_scale[colIndex[idx]];
    }

    result[row] = sum / static_cast<HPRLP_FLOAT>(count);
}

__global__ void curtis_reid_neg_log_abs_kernel(
    int n, const HPRLP_FLOAT *value, HPRLP_FLOAT *neg_log_abs_value) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const HPRLP_FLOAT abs_value = fmax(fabs(value[index]), 1e-300);
        neg_log_abs_value[index] = -log(abs_value);
    }
}

__global__ void exp_clamp_kernel(HPRLP_FLOAT *log_scale, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        log_scale[idx] = fmin(fmax(exp(log_scale[idx]), 1e-30), 1e30);
    }
}

__global__ void apply_curtis_reid_csr_kernel(
    int rows, const int *row_ptr, const int *col_index,
    HPRLP_FLOAT *values, const HPRLP_FLOAT *row_scale,
    const HPRLP_FLOAT *col_scale) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int start = row < rows ? row_ptr[row] : 0;
    const int end = row < rows ? row_ptr[row + 1] : 0;
    const bool cooperative =
        row < rows && end - start > kScalingCooperativeRowThreshold;

    unsigned cooperative_rows =
        __ballot_sync(kScalingFullWarpMask, cooperative);
    while (cooperative_rows != 0u) {
        const int owner_lane = __ffs(cooperative_rows) - 1;
        const int cooperative_row =
            __shfl_sync(kScalingFullWarpMask, row, owner_lane);
        const int cooperative_start =
            __shfl_sync(kScalingFullWarpMask, start, owner_lane);
        const int cooperative_end =
            __shfl_sync(kScalingFullWarpMask, end, owner_lane);
        const HPRLP_FLOAT row_factor = row_scale[cooperative_row];

        for (int index = cooperative_start + lane;
             index < cooperative_end; index += warpSize) {
            values[index] *=
                row_factor * col_scale[col_index[index]];
        }
        __syncwarp(kScalingFullWarpMask);
        cooperative_rows &= cooperative_rows - 1u;
    }

    if (row >= rows || cooperative) {
        return;
    }
    const HPRLP_FLOAT row_factor = row_scale[row];
    for (int index = start; index < end; ++index) {
        values[index] *= row_factor * col_scale[col_index[index]];
    }
}

__global__ void reduce_extrema_kernel(
    const HPRLP_FLOAT *min_input, const HPRLP_FLOAT *max_input, int n,
    HPRLP_FLOAT *min_output, HPRLP_FLOAT *max_output) {
    extern __shared__ HPRLP_FLOAT shared[];
    HPRLP_FLOAT *shared_min = shared;
    HPRLP_FLOAT *shared_max = shared + blockDim.x;
    const int thread = threadIdx.x;
    const int index = 2 * blockIdx.x * blockDim.x + thread;

    HPRLP_FLOAT local_min = INFINITY;
    HPRLP_FLOAT local_max = -INFINITY;
    if (index < n) {
        local_min = min_input[index];
        local_max = max_input[index];
    }
    if (index + blockDim.x < n) {
        local_min = fmin(local_min, min_input[index + blockDim.x]);
        local_max = fmax(local_max, max_input[index + blockDim.x]);
    }
    shared_min[thread] = local_min;
    shared_max[thread] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            shared_min[thread] = fmin(shared_min[thread], shared_min[thread + stride]);
            shared_max[thread] = fmax(shared_max[thread], shared_max[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        min_output[blockIdx.x] = shared_min[0];
        max_output[blockIdx.x] = shared_max[0];
    }
}

void device_extrema(const HPRLP_FLOAT *values, int n,
                    HPRLP_FLOAT *minimum, HPRLP_FLOAT *maximum) {
    const int threads = 256;
    int count = (n + 2 * threads - 1) / (2 * threads);
    HPRLP_FLOAT *min_a = nullptr;
    HPRLP_FLOAT *max_a = nullptr;
    HPRLP_FLOAT *min_b = nullptr;
    HPRLP_FLOAT *max_b = nullptr;
    CUDA_CHECK(cudaMalloc(&min_a, count * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&max_a, count * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&min_b, count * sizeof(HPRLP_FLOAT)));
    CUDA_CHECK(cudaMalloc(&max_b, count * sizeof(HPRLP_FLOAT)));

    const std::size_t shared_bytes = 2 * threads * sizeof(HPRLP_FLOAT);
    reduce_extrema_kernel<<<count, threads, shared_bytes>>>(
        values, values, n, min_a, max_a);
    while (count > 1) {
        const int next_count = (count + 2 * threads - 1) / (2 * threads);
        reduce_extrema_kernel<<<next_count, threads, shared_bytes>>>(
            min_a, max_a, count, min_b, max_b);
        HPRLP_FLOAT *swap_min = min_a;
        HPRLP_FLOAT *swap_max = max_a;
        min_a = min_b;
        max_a = max_b;
        min_b = swap_min;
        max_b = swap_max;
        count = next_count;
    }
    CUDA_CHECK(cudaMemcpy(minimum, min_a, sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(maximum, max_a, sizeof(HPRLP_FLOAT), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(min_a));
    CUDA_CHECK(cudaFree(max_a));
    CUDA_CHECK(cudaFree(min_b));
    CUDA_CHECK(cudaFree(max_b));
}

bool curtis_reid_scale_spread_is_safe(
    const HPRLP_FLOAT *row_log_scale, int m,
    const HPRLP_FLOAT *col_log_scale, int n) {
    HPRLP_FLOAT row_min = 0.0;
    HPRLP_FLOAT row_max = 0.0;
    HPRLP_FLOAT col_min = 0.0;
    HPRLP_FLOAT col_max = 0.0;
    device_extrema(row_log_scale, m, &row_min, &row_max);
    device_extrema(col_log_scale, n, &col_min, &col_max);

    const HPRLP_FLOAT max_log_range = std::log(1.0e8);
    const HPRLP_FLOAT row_range = std::fmax(row_max - row_min, 0.0);
    const HPRLP_FLOAT col_range = std::fmax(col_max - col_min, 0.0);
    const bool accepted = std::isfinite(row_min) && std::isfinite(row_max) &&
        std::isfinite(col_min) && std::isfinite(col_max) &&
        row_range <= max_log_range && col_range <= max_log_range;
    if (!accepted) {
        std::cerr << "Curtis-Reid scaling rejected: candidate scale spread "
                     "exceeds the safety threshold; leaving the LP unchanged "
                     "by Curtis-Reid\n";
    }
    return accepted;
}

void apply_curtis_reid_scaling(LP_info_gpu *lp_info_gpu,
                               HPRLP_FLOAT *rowNormA,
                               HPRLP_FLOAT *colNormA,
                               HPRLP_FLOAT *tempNorm1,
                               HPRLP_FLOAT *tempNorm2) {
    set_vector_value_device(tempNorm1, lp_info_gpu->m, 0.0);
    set_vector_value_device(tempNorm2, lp_info_gpu->n, 0.0);

    const int a_nonzeros = lp_info_gpu->A->numElements;
    const int at_nonzeros = lp_info_gpu->AT->numElements;
    const std::size_t cache_entries =
        static_cast<std::size_t>(a_nonzeros) +
        static_cast<std::size_t>(at_nonzeros);
    HPRLP_FLOAT *neg_log_abs_storage = nullptr;
    CUDA_CHECK(cudaGetLastError());
    const cudaError_t cache_status = cudaMalloc(
        &neg_log_abs_storage, cache_entries * sizeof(HPRLP_FLOAT));
    const bool use_log_cache = cache_status == cudaSuccess;
    if (!use_log_cache) {
        // Preserve the original arithmetic path if the temporary cache does
        // not fit.  Clear only the allocation error before continuing.
        cudaGetLastError();
    }
    HPRLP_FLOAT *neg_log_abs_a = neg_log_abs_storage;
    HPRLP_FLOAT *neg_log_abs_at = use_log_cache
        ? neg_log_abs_storage + a_nonzeros : nullptr;

    // A and AT values do not change during the Curtis--Reid fixed-point
    // iteration.  Cache exactly the same per-entry expression that the
    // original row-ordered loop evaluated on every pass.  The subsequent
    // accumulation order, row/column update order, and scaling applications
    // remain unchanged.
    if (use_log_cache) {
        curtis_reid_neg_log_abs_kernel<<<HPRLP_NUM_BLOCKS(a_nonzeros), HPRLP_NUM_THREADS>>>(
            a_nonzeros, lp_info_gpu->A->value, neg_log_abs_a);
        curtis_reid_neg_log_abs_kernel<<<HPRLP_NUM_BLOCKS(at_nonzeros), HPRLP_NUM_THREADS>>>(
            at_nonzeros, lp_info_gpu->AT->value, neg_log_abs_at);
    }

    for (int i = 0; i < 20; ++i) {
        if (use_log_cache) {
            curtis_reid_log_update_kernel<<<
                HPRLP_NUM_BLOCKS(lp_info_gpu->m), HPRLP_NUM_THREADS>>>(
                lp_info_gpu->A->row,
                lp_info_gpu->A->rowPtr,
                lp_info_gpu->A->colIndex,
                lp_info_gpu->A->value,
                neg_log_abs_a,
                tempNorm2,
                tempNorm1);

            curtis_reid_log_update_kernel<<<
                HPRLP_NUM_BLOCKS(lp_info_gpu->n), HPRLP_NUM_THREADS>>>(
                lp_info_gpu->AT->row,
                lp_info_gpu->AT->rowPtr,
                lp_info_gpu->AT->colIndex,
                lp_info_gpu->AT->value,
                neg_log_abs_at,
                tempNorm1,
                tempNorm2);
        } else {
            curtis_reid_log_update_kernel<<<
                HPRLP_NUM_BLOCKS(lp_info_gpu->m), HPRLP_NUM_THREADS>>>(
                lp_info_gpu->A->row,
                lp_info_gpu->A->rowPtr,
                lp_info_gpu->A->colIndex,
                lp_info_gpu->A->value,
                nullptr,
                tempNorm2,
                tempNorm1);

            curtis_reid_log_update_kernel<<<
                HPRLP_NUM_BLOCKS(lp_info_gpu->n), HPRLP_NUM_THREADS>>>(
                lp_info_gpu->AT->row,
                lp_info_gpu->AT->rowPtr,
                lp_info_gpu->AT->colIndex,
                lp_info_gpu->AT->value,
                nullptr,
                tempNorm1,
                tempNorm2);
        }
    }

    const bool scale_spread_is_safe = curtis_reid_scale_spread_is_safe(
        tempNorm1, lp_info_gpu->m, tempNorm2, lp_info_gpu->n);
    if (use_log_cache) {
        CUDA_CHECK(cudaFree(neg_log_abs_storage));
    }
    if (!scale_spread_is_safe) {
        return;
    }

    exp_clamp_kernel<<<HPRLP_NUM_BLOCKS(lp_info_gpu->m), HPRLP_NUM_THREADS>>>(tempNorm1, lp_info_gpu->m);
    exp_clamp_kernel<<<HPRLP_NUM_BLOCKS(lp_info_gpu->n), HPRLP_NUM_THREADS>>>(tempNorm2, lp_info_gpu->n);

    vector_dot_product(rowNormA, tempNorm1, rowNormA, lp_info_gpu->m, true);
    vector_dot_product(colNormA, tempNorm2, colNormA, lp_info_gpu->n, true);

    // Match Julia's single expression `value *= row_scale * col_scale`.
    // Applying the two factors in separate kernels changes rounding before the
    // first solver iteration on very large models.
    apply_curtis_reid_csr_kernel<<<HPRLP_NUM_BLOCKS(lp_info_gpu->m), HPRLP_NUM_THREADS>>>(
        lp_info_gpu->m, lp_info_gpu->A->rowPtr, lp_info_gpu->A->colIndex,
        lp_info_gpu->A->value, tempNorm1, tempNorm2);
    apply_curtis_reid_csr_kernel<<<HPRLP_NUM_BLOCKS(lp_info_gpu->n), HPRLP_NUM_THREADS>>>(
        lp_info_gpu->n, lp_info_gpu->AT->rowPtr, lp_info_gpu->AT->colIndex,
        lp_info_gpu->AT->value, tempNorm2, tempNorm1);

    vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, lp_info_gpu->m, false);
    vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, lp_info_gpu->m, false);
    vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, lp_info_gpu->n, false);
    vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, lp_info_gpu->n, true);
    vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, lp_info_gpu->n, true);
}

} // namespace


void scaling(LP_info_gpu *lp_info_gpu, Scaling_info* scaling_info, const HPRLP_parameters *param, cublasHandle_t cublasHandle) {
    int m = lp_info_gpu->m;
    int n = lp_info_gpu->n;

    create_zero_vector_device(scaling_info->row_norm, m);
    create_zero_vector_device(scaling_info->col_norm, n);
    create_zero_vector_device(scaling_info->l_org, n);
    create_zero_vector_device(scaling_info->u_org, n);

    HPRLP_FLOAT *rowNormA = scaling_info->row_norm;
    HPRLP_FLOAT *colNormA = scaling_info->col_norm;
    HPRLP_FLOAT *tempNorm1;
    HPRLP_FLOAT *tempNorm2;

    create_zero_vector_device(tempNorm1, m);
    create_zero_vector_device(tempNorm2, n);

    set_vector_value_device(rowNormA, m, 1.0);
    set_vector_value_device(colNormA, n, 1.0);

    vMemcpy_device(scaling_info->l_org, lp_info_gpu->l, n);
    vMemcpy_device(scaling_info->u_org, lp_info_gpu->u, n);

    HPRLP_FLOAT *b;
    CUDA_CHECK(cudaMalloc(&b, m * sizeof(HPRLP_FLOAT)));

    gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

    scaling_info->norm_b_org = 1 + l2_norm(b, m, cublasHandle);
    scaling_info->norm_c_org = 1 + l2_norm(lp_info_gpu->c, n, cublasHandle);

    if (param->use_CR_scaling) {
        apply_curtis_reid_scaling(lp_info_gpu, rowNormA, colNormA, tempNorm1, tempNorm2);
    }

    if (param->use_Ruiz_scaling){

        for (int i = 0; i < 10; ++i) {
            // find the max value of each row of A
            CSR_A_row_norm(lp_info_gpu->A, tempNorm1, 99);
            vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);

            // AL = AL / tempNorm   ;   AU = AU / tempNorm
            vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
            vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

            // find the max value of each column of A which is equivalent to each row of AT
            CSR_A_row_norm(lp_info_gpu->AT, tempNorm2, 99);
            vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

            // A = A / tempNorm, also for AT
            mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
            mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

            // A = A / tempNorm, also for AT
            mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
            mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

            // c = c / tempNorm
            vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

            // l = l * tempNorm, u = u * tempNorm
            vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
            vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
        }
    }

    // Pock and Chambolle scaling
    // compute the sum of each row of A, in tempNorm
    if(param->use_Pock_Chambolle_scaling){
        CSR_A_row_norm(lp_info_gpu->A, tempNorm1, 1);
        vector_dot_product(rowNormA, tempNorm1, rowNormA, m, false);

        // AL = AL / tempNorm   ;   AU = AU / tempNorm
        vector_dot_product(lp_info_gpu->AL, tempNorm1, lp_info_gpu->AL, m, true);
        vector_dot_product(lp_info_gpu->AU, tempNorm1, lp_info_gpu->AU, m, true);

        // compute the sum of each column of A
        CSR_A_row_norm(lp_info_gpu->AT, tempNorm2, 1);
        vector_dot_product(colNormA, tempNorm2, colNormA, n, false);

        // A = A / tempNorm, also for AT
        mul_CSR_A_row(lp_info_gpu->A, tempNorm1, true);
        mul_CSR_AT_row(lp_info_gpu->AT, tempNorm1, true);

        // A = A / tempNorm, also for AT
        mul_CSR_A_row(lp_info_gpu->AT, tempNorm2, true);
        mul_CSR_AT_row(lp_info_gpu->A, tempNorm2, true);

        // c = c / tempNorm
        vector_dot_product(lp_info_gpu->c, tempNorm2, lp_info_gpu->c, n, true);

        // l = l * tempNorm, u = u * tempNorm
        vector_dot_product(lp_info_gpu->l, tempNorm2, lp_info_gpu->l, n, false);
        vector_dot_product(lp_info_gpu->u, tempNorm2, lp_info_gpu->u, n, false);
    }

    if (param->use_bc_scaling){

        gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

        scaling_info->b_scale = 1 + l2_norm(b, m, cublasHandle);
        scaling_info->c_scale = 1 + l2_norm(lp_info_gpu->c, n, cublasHandle);


        // b = b / b_scale, c = c / c_scale
        const HPRLP_FLOAT bs = 1.0 / scaling_info->b_scale;
        const HPRLP_FLOAT cs = 1.0 / scaling_info->c_scale;

        CUBLAS_CHECK(cublasDscal(cublasHandle, m, &bs, lp_info_gpu->AU, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, m, &bs, lp_info_gpu->AL, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, n, &bs, lp_info_gpu->l, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, n, &bs, lp_info_gpu->u, 1));
        CUBLAS_CHECK(cublasDscal(cublasHandle, n, &cs, lp_info_gpu->c, 1));
    }
    else{
        scaling_info->b_scale = 1.0;
        scaling_info->c_scale = 1.0;
    }

    gen_conceptual_b(lp_info_gpu->AL, lp_info_gpu->AU, b, m);

    scaling_info->norm_b = l2_norm(b, m, cublasHandle);
    scaling_info->norm_c = l2_norm(lp_info_gpu->c, n, cublasHandle);

    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(tempNorm1));
    CUDA_CHECK(cudaFree(tempNorm2));
}


void free_scaling_info(Scaling_info *scaling_info) {
    /*
     * Free device memory allocated in scaling function.
     */
    if (!scaling_info) return;

    if (scaling_info->row_norm) cudaFree(scaling_info->row_norm);
    if (scaling_info->col_norm) cudaFree(scaling_info->col_norm);
    if (scaling_info->l_org) cudaFree(scaling_info->l_org);
    if (scaling_info->u_org) cudaFree(scaling_info->u_org);
}
