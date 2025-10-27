"""
High-level solver interface for HPRLP
"""
import numpy as np
from scipy import sparse
from typing import Union, Optional, Tuple
from pathlib import Path

from .parameters import Parameters
from .results import Results


def _ensure_contiguous_int32(arr):
    """Ensure array is contiguous int32"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.int32)
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32)
    return np.ascontiguousarray(arr)


def _ensure_contiguous_float64(arr):
    """Ensure array is contiguous float64"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float64)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return np.ascontiguousarray(arr)


class HPRLPSolver:
    """
    High-level interface for HPRLP GPU-accelerated LP solver.
    
    This class provides a convenient object-oriented interface for solving
    linear programming problems of the form:
    
        minimize    c'*x
        subject to  AL <= A*x <= AU
                    l <= x <= u
    
    Parameters
    ----------
    param : Parameters, optional
        Solver parameters. If None, default parameters are used.
    
    Examples
    --------
    >>> import numpy as np
    >>> from hprlp import HPRLPSolver, Parameters
    >>> 
    >>> # Create solver
    >>> solver = HPRLPSolver()
    >>> 
    >>> # Define a simple LP
    >>> A = np.array([[1.0, 2.0], [3.0, 1.0]])
    >>> AL = np.array([-np.inf, -np.inf])
    >>> AU = np.array([10.0, 12.0])
    >>> l = np.array([0.0, 0.0])
    >>> u = np.array([np.inf, np.inf])
    >>> c = np.array([-3.0, -5.0])
    >>> 
    >>> # Solve
    >>> result = solver.solve(A, AL, AU, l, u, c)
    >>> print(f"Status: {result.status}")
    >>> print(f"Optimal value: {result.primal_obj}")
    >>> print(f"Solution: {result.x}")
    """
    
    def __init__(self, param: Optional[Parameters] = None):
        self.param = param if param is not None else Parameters()
        
        # Try to import the C++ core
        try:
            from . import _hprlp_core
            self._core = _hprlp_core
            self._has_core = True
        except ImportError as e:
            self._core = None
            self._has_core = False
            
            # Provide helpful error messages for common issues
            error_msg = str(e)
            
            if 'GLIBC' in error_msg or 'GLIBCXX' in error_msg:
                raise ImportError(
                    f"╔══════════════════════════════════════════════════════════════════╗\n"
                    f"║  HPRLP Import Error: Library Version Mismatch                   ║\n"
                    f"╚══════════════════════════════════════════════════════════════════╝\n\n"
                    f"Error: {error_msg}\n\n"
                    f"CAUSE:\n"
                    f"  The Python module was compiled on a different system with a\n"
                    f"  different GLIBC/C++ library version. Binary Python extensions\n"
                    f"  are NOT portable between systems.\n\n"
                    f"SOLUTION:\n"
                    f"  Rebuild the package on THIS system:\n\n"
                    f"    cd bindings/python\n"
                    f"    rm -rf build/ dist/ *.egg-info hprlp/*.so\n"
                    f"    python -m pip install .\n\n"
                    f"  pip will:\n"
                    f"    • Detect your system's GLIBC version during build\n"
                    f"    • Use compatible compiler settings\n"
                    f"    • Emit a module that matches THIS system\n\n"
                    f"  Do NOT copy .so files between systems - always rebuild!\n"
                )
            else:
                raise ImportError(
                    f"Failed to import compiled HPRLP core module: {error_msg}\n\n"
                    f"Please ensure the package was built locally:\n"
                    f"  cd bindings/python\n"
                    f"  python -m pip install .\n"
                )
    
    def solve(
        self,
        A: Union[np.ndarray, sparse.spmatrix],
        AL: np.ndarray,
        AU: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        param: Optional[Parameters] = None,
    ) -> Results:
        """
        Solve a linear programming problem.
        
        Parameters
        ----------
        A : np.ndarray or scipy.sparse matrix
            Constraint matrix (m x n)
        AL : np.ndarray
            Lower bounds for constraints (length m)
        AU : np.ndarray
            Upper bounds for constraints (length m)
        l : np.ndarray
            Lower bounds for variables (length n)
        u : np.ndarray
            Upper bounds for variables (length n)
        c : np.ndarray
            Objective coefficients (length n)
        param : Parameters, optional
            Solver parameters. If None, uses solver's default parameters.
        
        Returns
        -------
        Results
            Solver results including solution and statistics
        """
        from .model import Model
        
        if not self._has_core:
            raise RuntimeError("HPRLP core module not available")
        
        # Use provided parameters or default
        if param is None:
            param = self.param
        
        # Create model from arrays
        model = Model.from_arrays(A, AL, AU, l, u, c)
        
        # Solve the model
        result = model.solve(param)
        
        # Free the model
        model.free()
        
        return result
    
    def solve_mps(
        self,
        filename: Union[str, Path],
        param: Optional[Parameters] = None,
    ) -> Results:
        """
        Solve a linear program from an MPS file.
        
        Parameters
        ----------
        filename : str or Path
            Path to the MPS file
        param : Parameters, optional
            Solver parameters. If None, uses solver's default parameters.
        
        Returns
        -------
        Results
            Solver results including solution and statistics
        """
        from .model import Model
        
        if not self._has_core:
            raise RuntimeError("HPRLP core module not available")
        
        # Use provided parameters or default
        if param is None:
            param = self.param
        
        # Create model from MPS file
        model = Model.from_mps(filename)
        
        # Solve the model
        result = model.solve(param)
        
        # Free the model
        model.free()
        
        return result


def solve(
    A: Union[np.ndarray, sparse.spmatrix],
    AL: np.ndarray,
    AU: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    c: np.ndarray,
    param: Optional[Parameters] = None,
) -> Results:
    """
    Convenience function to solve an LP without creating a solver object.
    
    Solves:
        minimize    c'*x
        subject to  AL <= A*x <= AU
                    l <= x <= u
    
    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix
        Constraint matrix (m x n)
    AL : np.ndarray
        Lower bounds for constraints (length m)
    AU : np.ndarray
        Upper bounds for constraints (length m)
    l : np.ndarray
        Lower bounds for variables (length n)
    u : np.ndarray
        Upper bounds for variables (length n)
    c : np.ndarray
        Objective coefficients (length n)
    param : Parameters, optional
        Solver parameters. If None, default parameters are used.
    
    Returns
    -------
    Results
        Solver results including solution and statistics
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from hprlp import solve
    >>> 
    >>> # Simple 2D LP
    >>> A = sparse.csr_matrix([[1.0, 2.0], [3.0, 1.0]])
    >>> AL = np.array([-np.inf, -np.inf])
    >>> AU = np.array([10.0, 12.0])
    >>> l = np.array([0.0, 0.0])
    >>> u = np.array([np.inf, np.inf])
    >>> c = np.array([-3.0, -5.0])
    >>> 
    >>> result = solve(A, AL, AU, l, u, c)
    >>> print(result)
    """
    solver = HPRLPSolver(param=param)
    return solver.solve(A, AL, AU, l, u, c)


def solve_mps(
    filename: Union[str, Path],
    param: Optional[Parameters] = None,
) -> Results:
    """
    Convenience function to solve an LP from MPS file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the MPS file
    param : Parameters, optional
        Solver parameters. If None, default parameters are used.
    
    Returns
    -------
    Results
        Solver results including solution and statistics
    
    Examples
    --------
    >>> from hprlp import solve_mps, Parameters
    >>> 
    >>> param = Parameters()
    >>> param.stop_tol = 1e-9
    >>> 
    >>> result = solve_mps("model.mps", param=param)
    >>> print(result)
    """
    solver = HPRLPSolver(param=param)
    return solver.solve_mps(filename)
