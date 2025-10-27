"""
Parameters class for HPRLP solver
"""

class Parameters:
    """
    Configuration parameters for the HPRLP solver.
    
    Attributes
    ----------
    max_iter : int
        Maximum number of iterations (default: 2^31 - 1)
    stop_tol : float
        Stopping tolerance for convergence (default: 1e-4)
    time_limit : float
        Maximum time in seconds (default: 3600.0)
    device_number : int
        CUDA device number to use (default: 0)
    check_iter : int
        Number of iterations between convergence checks (default: 150)
    use_Ruiz_scaling : bool
        Enable Ruiz equilibration scaling (default: True)
    use_Pock_Chambolle_scaling : bool
        Enable Pock-Chambolle scaling (default: True)
    use_bc_scaling : bool
        Enable bound constraint scaling (default: True)
    
    Examples
    --------
    >>> param = Parameters()
    >>> param.max_iter = 10000
    >>> param.stop_tol = 1e-9
    >>> param.device_number = 0
    """
    
    def __init__(self):
        self.max_iter = 2147483647  # INT32_MAX
        self.stop_tol = 1e-4
        self.time_limit = 3600.0
        self.device_number = 0
        self.check_iter = 150
        self.use_Ruiz_scaling = True
        self.use_Pock_Chambolle_scaling = True
        self.use_bc_scaling = True
    
    def __repr__(self):
        return (f"Parameters(max_iter={self.max_iter}, "
                f"stop_tol={self.stop_tol}, "
                f"time_limit={self.time_limit}, "
                f"device_number={self.device_number})")
    
    def to_core_param(self):
        """Convert to C++ Parameters object for pybind11"""
        try:
            from ._hprlp_core import Parameters as CoreParameters
            param = CoreParameters()
            param.max_iter = self.max_iter
            param.stop_tol = self.stop_tol
            param.time_limit = self.time_limit
            param.device_number = self.device_number
            param.check_iter = self.check_iter
            param.use_Ruiz_scaling = self.use_Ruiz_scaling
            param.use_Pock_Chambolle_scaling = self.use_Pock_Chambolle_scaling
            param.use_bc_scaling = self.use_bc_scaling
            return param
        except ImportError:
            return self
    
    @classmethod
    def from_dict(cls, d):
        """Create Parameters from dictionary"""
        param = cls()
        for key, value in d.items():
            if hasattr(param, key):
                setattr(param, key, value)
        return param
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'max_iter': self.max_iter,
            'stop_tol': self.stop_tol,
            'time_limit': self.time_limit,
            'device_number': self.device_number,
            'check_iter': self.check_iter,
            'use_Ruiz_scaling': self.use_Ruiz_scaling,
            'use_Pock_Chambolle_scaling': self.use_Pock_Chambolle_scaling,
            'use_bc_scaling': self.use_bc_scaling,
        }
