/*
 * Example: Solving MPS files with HPRLP
 * 
 * This example demonstrates the model-based API for loading and solving
 * LP problems from MPS files.
 * 
 * Features demonstrated:
 * 1. Loading model from MPS file
 * 2. Using custom parameters
 * 3. Using default parameters (passing NULL)
 * 4. Solving the same model multiple times with different parameters
 * 5. Proper error handling and memory cleanup
 * 
 * Compile: See Makefile in parent directory
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include "HPRLP.h"

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "  HPRLP Example: Solving LP from MPS File - C++                   \n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    
    // Determine which MPS file to use
    const char* mps_file;
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [mps_file]\n";
            std::cout << "\n";
            std::cout << "If no MPS file is provided, the program will look for ../../data/model.mps\n";
            std::cout << "\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << "\n";
            std::cout << "  " << argv[0] << " problem.mps\n";
            std::cout << "  " << argv[0] << " ../../data/model.mps\n";
            std::cout << "\n";
            return 0;
        }
        mps_file = argv[1];
    } else {
        mps_file = "../../data/model.mps";
    }
    
    std::cout << "MPS file: " << mps_file << "\n\n";
    
    // Step 1: Load model from MPS file
    std::cout << "Loading model from MPS file...\n";
    LP_info_cpu* model = create_model_from_mps(mps_file);
    
    if (model == nullptr) {
        std::cerr << "Error: Failed to load model from MPS file\n";
        return 1;
    }
    
    std::cout << "Model loaded: " << model->m << " constraints, " 
              << model->n << " variables\n";
    std::cout << "Objective constant: " << std::fixed << std::setprecision(6) 
              << model->obj_constant << "\n\n";
    
    // Example 1: Solve with custom parameters
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Example: Solving with custom parameters (tol=1e-9)\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    
    HPRLP_parameters custom_param;
    custom_param.device_number = 0;
    custom_param.stop_tol = 1e-9;      // Higher precision
    
    HPRLP_results result1 = solve(model, &custom_param);
    
    // The following summary is also printed inside the solver, here we print it again for example purposes

    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Solution Summary (Example 1)\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "Status: " << result1.status << "\n";
    std::cout << "Iterations: " << result1.iter << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(2) << result1.time << " seconds\n";
    std::cout << "Primal Objective: " << std::scientific << std::setprecision(12) << result1.primal_obj << "\n";
    std::cout << "Residual: " << std::scientific << std::setprecision(12) << result1.residuals << "\n";
    std::cout << "\n";
    
    // Clean up first result
    if (result1.x != nullptr) {
        free(result1.x);
        free(result1.y);
    }
    
    // Step 5: Free the model
    free_model(model);
    
    return 0;
}
