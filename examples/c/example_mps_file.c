/**
 * @file example_mps_file.c
 * @brief Example demonstrating solving LP from MPS file using model-based API
 */

#include <stdio.h>
#include <stdlib.h>
#include "HPRLP.h"

int main() {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  HPRLP Example: Solving LP from MPS File - Pure C                \n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    
    const char* mps_path = "../../data/model.mps";
    printf("MPS file: %s\n\n", mps_path);
    
    printf("Loading model from MPS file...\n");
    LP_info_cpu* model = create_model_from_mps(mps_path);
    
    if (!model) {
        fprintf(stderr, "Failed to load model from MPS file\n");
        return 1;
    }
    
    printf("Model loaded: %d constraints, %d variables\n", model->m, model->n);
    printf("Objective constant: %.6f\n\n", model->obj_constant);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Example: Solving with custom parameters (tol=1e-9)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    HPRLP_parameters param1;
    param1.stop_tol = 1e-9;
    param1.device_number = 0;
    
    HPRLP_results result1 = solve(model, &param1);

    // The following summary is also printed inside the solver, here we print it again for example purposes
    
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  Solution Summary (Example 1)\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("Status: %s\n", result1.status);
    printf("Iterations: %d\n", result1.iter);
    printf("Time: %.2f seconds\n", result1.time);
    printf("Primal Objective: %.12e\n", result1.primal_obj);
    printf("Residual: %.12e\n\n", result1.residuals);
    
    if (result1.x) free(result1.x);
    if (result1.y) free(result1.y);
    
    
    free_model(model);
    
    return 0;
}
