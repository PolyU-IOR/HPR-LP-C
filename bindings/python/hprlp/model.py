"""
Model class for HPRLP
"""
import numpy as np
from scipy import sparse
from typing import Union, Optional
from pathlib import Path


class Model:
    """
    LP model for HPRLP solver.
    
    This class wraps an LP_info_cpu structure from the C++ core.
    Models can be created from arrays or MPS files, then solved multiple
    times with different parameters.
    
    The model represents an LP of the form:
        minimize    c'*x
        subject to  AL <= A*x <= AU
                    l <= x <= u
    
    Attributes
    ----------
    m : int
        Number of constraints
    n : int
        Number of variables
    obj_constant : float
        Objective constant term
    
    Examples
    --------
    >>> import numpy as np
    >>> from hprlp import Model, Parameter
    >>> 
    >>> # Create model from arrays
    >>> A = np.array([[1.0, 2.0], [3.0, 1.0]])
    >>> AL = np.array([-np.inf, -np.inf])
    >>> AU = np.array([10.0, 12.0])
    >>> l = np.array([0.0, 0.0])
    >>> u = np.array([np.inf, np.inf])
    >>> c = np.array([-3.0, -5.0])
    >>> 
    >>> model = Model.from_arrays(A, AL, AU, l, u, c)
    >>> print(f"Model: {model.m} constraints, {model.n} variables")
    >>> 
    >>> # Solve with different parameters
    >>> param1 = Parameter(stop_tol=1e-9)
    >>> result1 = model.solve(param1)
    >>> 
    >>> param2 = Parameter(stop_tol=1e-4)
    >>> result2 = model.solve(param2)
    >>> 
    >>> # Don't forget to free
    >>> model.free()
    """
    
    def __init__(self, core_model):
        """
        Initialize Model from C++ core model.
        
        Parameters
        ----------
        core_model : _hprlp_core.Model
            C++ core model object
        """
        self._core_model = core_model
        self._freed = False
    
    @property
    def m(self) -> int:
        """Number of constraints"""
        if self._freed:
            raise RuntimeError("Model has been freed")
        return self._core_model.m
    
    @property
    def n(self) -> int:
        """Number of variables"""
        if self._freed:
            raise RuntimeError("Model has been freed")
        return self._core_model.n
    
    @property
    def obj_constant(self) -> float:
        """Objective constant term"""
        if self._freed:
            raise RuntimeError("Model has been freed")
        return self._core_model.obj_constant
    
    def is_valid(self) -> bool:
        """Check if model is valid (not freed)"""
        return not self._freed and self._core_model.is_valid()
    
    @staticmethod
    def from_arrays(
        A: Union[np.ndarray, sparse.spmatrix],
        AL: np.ndarray,
        AU: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
    ) -> 'Model':
        """
        Create model from constraint matrix and bounds arrays.
        
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
        
        Returns
        -------
        Model
            LP model object
        """
        from . import _hprlp_core
        from .solver import _ensure_contiguous_int32, _ensure_contiguous_float64
        
        # Convert A to CSR format if needed
        if sparse.issparse(A):
            if not sparse.isspmatrix_csr(A):
                A = A.tocsr()
            m, n = A.shape
            nnz = A.nnz
            rowPtr = _ensure_contiguous_int32(A.indptr)
            colIndex = _ensure_contiguous_int32(A.indices)
            values = _ensure_contiguous_float64(A.data)
            is_csc = False
        elif isinstance(A, np.ndarray):
            # Convert dense array to CSR
            A_sparse = sparse.csr_matrix(A)
            m, n = A_sparse.shape
            nnz = A_sparse.nnz
            rowPtr = _ensure_contiguous_int32(A_sparse.indptr)
            colIndex = _ensure_contiguous_int32(A_sparse.indices)
            values = _ensure_contiguous_float64(A_sparse.data)
            is_csc = False
        else:
            raise TypeError("A must be a numpy array or scipy sparse matrix")
        
        # Ensure other arrays are correct type
        AL = _ensure_contiguous_float64(AL)
        AU = _ensure_contiguous_float64(AU)
        l = _ensure_contiguous_float64(l)
        u = _ensure_contiguous_float64(u)
        c = _ensure_contiguous_float64(c)
        
        # Validate dimensions
        if len(AL) != m or len(AU) != m:
            raise ValueError(f"AL and AU must have length {m} (number of constraints)")
        if len(l) != n or len(u) != n or len(c) != n:
            raise ValueError(f"l, u, and c must have length {n} (number of variables)")
        
        # Create model using C++ core
        core_model = _hprlp_core.create_model_from_arrays(
            m, n, nnz,
            rowPtr, colIndex, values,
            AL, AU, l, u, c,
            is_csc
        )
        
        return Model(core_model)
    
    @staticmethod
    def from_mps(filename: Union[str, Path]) -> 'Model':
        """
        Create model from MPS file.
        
        Parameters
        ----------
        filename : str or Path
            Path to the MPS file
        
        Returns
        -------
        Model
            LP model object
        """
        from . import _hprlp_core
        
        # Convert to string
        filename = str(filename)
        
        # Check file exists
        if not Path(filename).exists():
            raise FileNotFoundError(f"MPS file not found: {filename}")
        
        # Create model using C++ core
        core_model = _hprlp_core.create_model_from_mps(filename)
        
        return Model(core_model)
    
    def solve(self, param: Optional['Parameter'] = None) -> 'Results':
        """
        Solve the model.
        
        Parameters
        ----------
        param : Parameter, optional
            Solver parameters. If None, default parameters are used.
        
        Returns
        -------
        Results
            Solver results including solution and statistics
        """
        from . import _hprlp_core
        from .results import Results
        
        if self._freed:
            raise RuntimeError("Cannot solve: model has been freed")
        
        # Convert parameter if provided
        core_param = None
        if param is not None:
            core_param = param.to_core_param()
        
        # Solve using C++ core
        core_results = _hprlp_core.solve(self._core_model, core_param)
        
        # Convert to Python Results
        return Results.from_core_results(core_results)
    
    def free(self):
        """
        Free the model and release memory.
        
        After calling this method, the model cannot be used anymore.
        """
        from . import _hprlp_core
        
        if not self._freed:
            _hprlp_core.free_model(self._core_model)
            self._freed = True
    
    def __del__(self):
        """Destructor - automatically free model when object is garbage collected"""
        if not self._freed:
            self.free()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically free model"""
        self.free()
        return False
    
    def __repr__(self):
        if self._freed:
            return "<HPRLP.Model (freed)>"
        else:
            return f"<HPRLP.Model m={self.m} n={self.n}>"
