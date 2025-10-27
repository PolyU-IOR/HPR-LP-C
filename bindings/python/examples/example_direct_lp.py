"""
Example: Solving LP from arrays with HPRLP

This example demonstrates how to solve a linear programming problem
by creating a model from constraint arrays.

Problem:
    minimize    -3*x1 - 5*x2
    subject to   x1 + 2*x2 <= 10
                3*x1 +  x2 <= 12
                 x1, x2 >= 0
"""

import numpy as np
from scipy import sparse
import hprlp


def main():
    print()
    print("=" * 70)
    print("HPRLP Example: Direct LP from Arrays - Python")
    print("=" * 70)
    print()
    
    print("Problem: minimize -3*x1 - 5*x2")
    print("         subject to x1 + 2*x2 <= 10")
    print("                    3*x1 + x2 <= 12")
    print("                    x1, x2 >= 0")
    print()
    
    # Define the LP problem
    # Constraint matrix in CSR format
    A = sparse.csr_matrix([
        [1.0, 2.0],  # x1 + 2*x2 <= 10
        [3.0, 1.0]   # 3*x1 + x2 <= 12
    ])
    
    # Constraint bounds
    AL = np.array([-np.inf, -np.inf])  # Lower bounds
    AU = np.array([10.0, 12.0])         # Upper bounds
    
    # Variable bounds
    l = np.array([0.0, 0.0])           # x >= 0
    u = np.array([np.inf, np.inf])     # Unbounded above
    
    # Objective coefficients
    c = np.array([-3.0, -5.0])
    
    # Step 1: Create model from arrays
    print("Creating model from arrays...")
    model = hprlp.Model.from_arrays(A, AL, AU, l, u, c)
    print(f"Model created: {model.m} constraints, {model.n} variables")
    print()
    
    # Step 2: Set solver parameters
    param = hprlp.Parameters()
    param.stop_tol = 1e-9
    param.device_number = 0
    
    # Step 3: Solve the model
    result = model.solve(param)
    
    # Step 4: Display results
    print()
    print("=" * 70)
    print("Solution Summary")
    print("=" * 70)
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iter}")
    print(f"Time: {result.time:.2f} seconds")
    print(f"Primal Objective: {result.primal_obj:.12e}")
    print(f"Residual: {result.residuals:.12e}")
    print()
    print("Primal solution:")
    print(f"  x1 = {result.x[0]:.6f}")
    print(f"  x2 = {result.x[1]:.6f}")
    print()
    print("=" * 70)
    print()
    
    # Step 5: Free the model
    model.free()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install HPRLP first:")
        print("  cd bindings/python")
        print("  python -m pip install .")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
