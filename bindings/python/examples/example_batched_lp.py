"""
Example: batched LP solves sharing one sparse matrix A.

Each column of C, AL, AU, l, and u defines one LP instance.
"""

import numpy as np
from scipy import sparse
import hprlp


def main():
    print("=" * 70)
    print("HPRLP Example: Batched Shared-A LP - Python")
    print("=" * 70)

    A = sparse.csr_matrix([
        [1.0, 2.0],
        [3.0, 1.0],
    ])
    B = 3

    C = np.array([
        [-3.0, -2.0, -4.0],
        [-5.0, -6.0, -4.0],
    ], dtype=np.float64)
    AL = np.full((2, B), -np.inf)
    AU = np.array([
        [10.0, 9.0, 11.0],
        [12.0, 13.0, 11.0],
    ], dtype=np.float64)
    l = np.zeros((2, B), dtype=np.float64)
    u = np.full((2, B), np.inf)
    u[0, 2] = 4.0

    params = hprlp.Parameters()
    params.stop_tol = 1e-8
    params.max_iter = 200000
    params.use_presolve = False

    result = hprlp.solve_batched(A, C, AL, AU, l, u, param=params)

    print(f"Batch size: {len(result.status)}")
    print(f"Total time: {result.time:.4f} seconds")
    for k, status in enumerate(result.status):
        print(
            f"[{k}] status={status} obj={result.primal_obj[k]:.12e} "
            f"residual={result.residuals[k]:.6e} iter={result.iter[k]} "
            f"x={result.x[:, k]}"
        )


if __name__ == "__main__":
    main()
