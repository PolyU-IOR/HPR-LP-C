/**
 * @file example_direct_lp.c
 * @brief Example demonstrating solving LP from raw arrays using model-based API
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "HPRLP.h"

int main() {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  HPRLP Example: Direct LP from Arrays - Pure C                   \n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    
    printf("Problem: minimize -3*x1 - 5*x2\n");
    printf("         subject to x1 + 2*x2 <= 10\n");
    printf("                    3*x1 + x2 <= 12\n");
    printf("                    x1, x2 >= 0\n\n");
    
    int m = 2, n = 2, nnz = 4;
    
    int rowPtr[] = {0, 2, 4};
    int colIndex[] = {0, 1, 0, 1};
    double values[] = {1.0, 2.0, 3.0, 1.0};
    
    double AL[] = {-INFINITY, -INFINITY};
    double AU[] = {10.0, 12.0};
    
    double l[] = {0.0, 0.0};
    double u[] = {INFINITY, INFINITY};
    
    double c[] = {-3.0, -5.0};
    
    printf("Creating model from arrays...\n");
    LP_info_cpu* model = create_model_from_arrays(m, n, nnz, 
                                                   rowPtr, colIndex, values,
                                                   AL, AU, l, u, c, false);
    
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }
    
    printf("Model created: %d constraints, %d variables\n\n", model->m, model->n);
    
    HPRLP_parameters param;
    param.stop_tol = 1e-9;
    param.device_number = 0;
    
    HPRLP_results result = solve(model, &param);

    // The following summary is also printed inside the solver, here we print it again for example purposes
    
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  Solution Summary\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("Status: %s\n", result.status);
    printf("Iterations: %d\n", result.iter);
    printf("Time: %.2f seconds\n", result.time);
    printf("Primal Objective: %.12e\n", result.primal_obj);
    printf("Residual: %.12e\n", result.residuals);
    
    if (result.x) {
        printf("\nPrimal solution:\n");
        printf("  x1 = %.6f\n", result.x[0]);
        printf("  x2 = %.6f\n", result.x[1]);
    }
    
    printf("\n══════════════════════════════════════════════════════════════════\n\n");
    
    if (result.x) free(result.x);
    if (result.y) free(result.y);
    free_model(model);
    
    return 0;
}
