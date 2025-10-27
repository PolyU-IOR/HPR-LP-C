/*
 * Example: Direct LP Data Entry with HPRLP
 * 
 * This example demonstrates how to solve LP problems by directly providing
 * raw arrays in CSR (Compressed Sparse Row) format using the model-based API.
 * 
 * This example recreates the problem from model.mps:
 * 
 *   minimize    -3*x1 - 5*x2
 *   subject to  x1 + 2*x2 <= 10
 *               3*x1 + x2 <= 12
 *               x1, x2 >= 0
 * 
 * Expected solution: x1 ≈ 2.8, x2 ≈ 3.6, obj ≈ -26.4
 * 
 * Compile: See Makefile in parent directory
 */

#include <iostream>
#include <iomanip>
#include "HPRLP.h"

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "  HPRLP Example: Direct LP from Arrays - C++                      \n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    std::cout << "Problem: minimize -3*x1 - 5*x2\n";
    std::cout << "         subject to x1 + 2*x2 <= 10\n";
    std::cout << "                    3*x1 + x2 <= 12\n";
    std::cout << "                    x1, x2 >= 0\n";
    std::cout << "\n";
    
    /*
     * Problem from model.mps:
     * 
     * Variables: x1, x2  (n = 2)
     * Constraints: 2 rows (m = 2)
     * 
     * Constraint matrix A (CSR format):
     *   Row 0 (c1):  x1 + 2*x2 <= 10
     *   Row 1 (c2):  3*x1 + x2 <= 12
     * 
     *   A = [[1, 2],
     *        [3, 1]]
     * 
     * nnz = 4
     */
    
    int m = 2;  // rows (constraints)
    int n = 2;  // cols (variables)
    int nnz = 4;  // non-zero elements
    
    // CSR format for constraint matrix A
    int rowPtr[] = {0, 2, 4};  // length m+1 = 3
    int colIndex[] = {0, 1,    // row 0: cols 0, 1 (x1, x2)
                      0, 1};   // row 1: cols 0, 1 (x1, x2)
    HPRLP_FLOAT values[] = {1.0, 2.0,   // row 0: coefficients
                            3.0, 1.0};  // row 1: coefficients
    
    // Constraint bounds: AL <= A*x <= AU
    HPRLP_FLOAT AL[] = {-INFINITY, -INFINITY};  // no lower bound
    HPRLP_FLOAT AU[] = {10.0, 12.0};            // upper bounds (RHS)
    
    // Variable bounds: l <= x <= u
    HPRLP_FLOAT l[] = {0.0, 0.0};           // lower bounds
    HPRLP_FLOAT u[] = {INFINITY, INFINITY}; // upper bounds
    
    // Objective coefficients: minimize c^T x
    HPRLP_FLOAT c[] = {-3.0, -5.0};
    
    // Step 1: Create model from arrays
    std::cout << "Creating model from arrays...\n";
    LP_info_cpu* model = create_model_from_arrays(
        m, n, nnz,
        rowPtr, colIndex, values,
        AL, AU,
        l, u,
        c,
        false  // is_csc=false (input is CSR format)
    );
    
    if (model == nullptr) {
        std::cerr << "Error: Failed to create model\n";
        return 1;
    }
    
    std::cout << "Model created: " << model->m << " constraints, " 
              << model->n << " variables\n\n";
    
    // Step 2: Set solver parameters
    HPRLP_parameters param;
    param.device_number = 0;
    param.stop_tol = 1e-9;
    
    // Step 3: Solve the model
    HPRLP_results result = solve(model, &param);

    // The following summary is also printed inside the solver, here we print it again for example purposes
    
    // Step 4: Print solution
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Solution Summary\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "Status: " << result.status << "\n";
    std::cout << "Iterations: " << result.iter << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(2) << result.time << " seconds\n";
    std::cout << "Primal Objective: " << std::scientific << std::setprecision(12) << result.primal_obj << "\n";
    std::cout << "Residual: " << std::scientific << std::setprecision(12) << result.residuals << "\n";
    std::cout << "\n";
    
    if (result.x != nullptr) {
        std::cout << "Primal solution:\n";
        for (int i = 0; i < n; i++) {
            std::cout << "  x" << (i+1) << " = " << std::fixed << std::setprecision(6) << result.x[i] << "\n";
        }
        std::cout << "\n";
        
        // Clean up result arrays
        free(result.x);
        free(result.y);
    }
    
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    
    // Step 5: Free the model
    free_model(model);

    return 0;
}
