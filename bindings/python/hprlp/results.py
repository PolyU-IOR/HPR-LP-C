"""
Results class for HPRLP solver output
"""
import numpy as np
from typing import Optional, Dict, Any


class Results:
    """
    Results from the HPRLP solver.
    
    Attributes
    ----------
    status : str
        Solver status ('OPTIMAL', 'TIME_LIMIT', 'ITER_LIMIT', 'ERROR', etc.)
    x : np.ndarray
        Primal solution vector
    y : np.ndarray
        Dual solution vector
    primal_obj : float
        Primal objective value (c'*x)
    gap : float
        Duality gap
    residuals : float
        Final residuals
    iter : int
        Total number of iterations
    time : float
        Total solve time in seconds
    iter4 : int
        Iterations to reach 1e-4 tolerance
    iter6 : int
        Iterations to reach 1e-6 tolerance
    iter8 : int
        Iterations to reach 1e-8 tolerance
    time4 : float
        Time to reach 1e-4 tolerance
    time6 : float
        Time to reach 1e-6 tolerance
    time8 : float
        Time to reach 1e-8 tolerance
    
    Methods
    -------
    is_optimal()
        Check if solution is optimal
    to_dict()
        Convert results to dictionary
    """
    
    def __init__(self):
        self.status: str = "UNKNOWN"
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.primal_obj: float = 0.0
        self.gap: float = float('inf')
        self.residuals: float = float('inf')
        self.iter: int = 0
        self.time: float = 0.0
        self.iter4: int = 0
        self.iter6: int = 0
        self.iter8: int = 0
        self.time4: float = 0.0
        self.time6: float = 0.0
        self.time8: float = 0.0
    
    def is_optimal(self) -> bool:
        """Check if solution is optimal"""
        return self.status == "OPTIMAL"
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        return self.status in ["OPTIMAL", "TIME_LIMIT", "ITER_LIMIT"]
    
    def __repr__(self):
        if self.x is not None:
            n_vars = len(self.x)
        else:
            n_vars = 0
        
        return (f"Results(status='{self.status}', "
                f"iter={self.iter}, "
                f"time={self.time:.3f}s, "
                f"gap={self.gap:.2e}, "
                f"n_vars={n_vars})")
    
    def __str__(self):
        lines = [
            "HPRLP Solver Results",
            "=" * 50,
            f"Status:          {self.status}",
            f"Primal Obj:      {self.primal_obj:.6e}",
            f"Gap:             {self.gap:.6e}",
            f"Residuals:       {self.residuals:.6e}",
            f"Iterations:      {self.iter}",
            f"Time:            {self.time:.3f} seconds",
        ]
        
        if self.iter4 > 0:
            lines.append(f"Time to 1e-4:    {self.time4:.3f}s ({self.iter4} iters)")
        if self.iter6 > 0:
            lines.append(f"Time to 1e-6:    {self.time6:.3f}s ({self.iter6} iters)")
        if self.iter8 > 0:
            lines.append(f"Time to 1e-8:    {self.time8:.3f}s ({self.iter8} iters)")
        
        if self.x is not None:
            lines.append(f"Variables:       {len(self.x)}")
            lines.append(f"||x||:           {np.linalg.norm(self.x):.6e}")
        
        if self.y is not None:
            lines.append(f"Constraints:     {len(self.y)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'status': self.status,
            'x': self.x.tolist() if self.x is not None else None,
            'y': self.y.tolist() if self.y is not None else None,
            'primal_obj': self.primal_obj,
            'gap': self.gap,
            'residuals': self.residuals,
            'iter': self.iter,
            'time': self.time,
            'iter4': self.iter4,
            'iter6': self.iter6,
            'iter8': self.iter8,
            'time4': self.time4,
            'time6': self.time6,
            'time8': self.time8,
        }
    
    @classmethod
    def from_core_results(cls, core_results):
        """Create Results from C++ results object"""
        results = cls()
        results.status = core_results.status
        results.primal_obj = core_results.primal_obj
        results.gap = core_results.gap
        results.residuals = core_results.residuals
        results.iter = core_results.iter
        results.time = core_results.time
        results.iter4 = core_results.iter4
        results.iter6 = core_results.iter6
        results.iter8 = core_results.iter8
        results.time4 = core_results.time4
        results.time6 = core_results.time6
        results.time8 = core_results.time8
        
        # Convert solution vectors to numpy arrays
        if hasattr(core_results, 'x') and len(core_results.x) > 0:
            results.x = np.array(core_results.x, dtype=np.float64)
        if hasattr(core_results, 'y') and len(core_results.y) > 0:
            results.y = np.array(core_results.y, dtype=np.float64)
        
        return results
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create Results from dictionary"""
        results = cls()
        for key, value in d.items():
            if key in ['x', 'y'] and value is not None:
                setattr(results, key, np.array(value, dtype=np.float64))
            elif hasattr(results, key):
                setattr(results, key, value)
        return results
