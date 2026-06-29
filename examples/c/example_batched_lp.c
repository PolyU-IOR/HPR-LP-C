/**
 * @file example_batched_lp.c
 * @brief Example demonstrating batched shared-A LP solves from C.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "HPRLP.h"

int main() {
    printf("\nHPRLP Example: Batched Shared-A LP - C\n\n");

    int m = 2, n = 2, nnz = 4;
    int rowPtr[] = {0, 2, 4};
    int colIndex[] = {0, 1, 0, 1};
    double values[] = {1.0, 2.0, 3.0, 1.0};

    double base_AL[] = {-INFINITY, -INFINITY};
    double base_AU[] = {10.0, 12.0};
    double base_l[] = {0.0, 0.0};
    double base_u[] = {INFINITY, INFINITY};
    double base_c[] = {-3.0, -5.0};

    LP_info_cpu *model = create_model_from_arrays(m, n, nnz,
                                                   rowPtr, colIndex, values,
                                                   base_AL, base_AU, base_l, base_u, base_c,
                                                   false);
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    int B = 3;

    /* Column-major dense inputs: C/l/u are n x B; AL/AU are m x B. */
    double C[] = {
        -3.0, -5.0,
        -2.0, -6.0,
        -4.0, -4.0
    };
    double AL[] = {
        -INFINITY, -INFINITY,
        -INFINITY, -INFINITY,
        -INFINITY, -INFINITY
    };
    double AU[] = {
        10.0, 12.0,
        9.0, 13.0,
        11.0, 11.0
    };
    double l[] = {
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0
    };
    double u[] = {
        INFINITY, INFINITY,
        INFINITY, INFINITY,
        4.0, INFINITY
    };
    double obj_constants[] = {0.0, 0.0, 0.0};

    HPRLP_parameters param;
    param.stop_tol = 1e-8;
    param.max_iter = 200000;
    param.device_number = 0;
    param.use_presolve = false;

    HPRLP_batched_results result = solve_batched(model, B, C, AL, AU, l, u, obj_constants, &param);

    printf("Batch size: %d\n", result.batch_size);
    printf("Total time: %.4f seconds\n", result.time);
    for (int k = 0; k < result.batch_size; ++k) {
        printf("[%d] status=%s obj=%.12e residual=%.6e iter=%d x=[%.6f, %.6f]\n",
               k,
               result.status + 64 * k,
               result.primal_obj[k],
               result.residuals[k],
               result.iter[k],
               result.x[k * n + 0],
               result.x[k * n + 1]);
    }

    free_batched_results(&result);
    free_model(model);
    return 0;
}
