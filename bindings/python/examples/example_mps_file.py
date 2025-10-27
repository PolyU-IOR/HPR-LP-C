"""
Example: Solving LP from MPS file with HPRLP

This example demonstrates how to solve a linear programming problem
from an MPS file using the model-based API.
"""

import sys
from pathlib import Path
import hprlp


def main():
    print()
    print("=" * 70)
    print("HPRLP Example: Solving LP from MPS File - Python")
    print("=" * 70)
    print()
    
    # Get MPS file path
    if len(sys.argv) > 1:
        mps_file = sys.argv[1]
    else:
        # Use default example file
        mps_file = "../../../data/model.mps"
    
    mps_path = Path(mps_file)
    
    if not mps_path.exists():
        print(f"Error: MPS file not found: {mps_file}")
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <path_to_mps_file>")
        print()
        return 1
    
    print(f"MPS file: {mps_path.absolute()}")
    print()
    
    # Step 1: Load model from MPS file
    print("Loading model from MPS file...")
    model = hprlp.Model.from_mps(str(mps_path))
    print(f"Model loaded: {model.m} constraints, {model.n} variables")
    print(f"Objective constant: {model.obj_constant:.6f}")
    print()
    
    # Example 1: Solve with custom parameters
    print("=" * 70)
    print("Example 1: Solving with custom parameters (tol=1e-9)")
    print("=" * 70)
    print()
    
    param1 = hprlp.Parameters()
    param1.stop_tol = 1e-9
    param1.device_number = 0

    result1 = model.solve(param1)
    
    print()
    print("=" * 70)
    print("Solution Summary (Example 1)")
    print("=" * 70)
    print(f"Status: {result1.status}")
    print(f"Iterations: {result1.iter}")
    print(f"Time: {result1.time:.2f} seconds")
    print(f"Primal Objective: {result1.primal_obj:.12e}")
    print(f"Residual: {result1.residuals:.12e}")
    print()
    
    # Step 5: Free the model
    model.free()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install HPRLP first:")
        print("  cd bindings/python")
        print("  python -m pip install .")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
