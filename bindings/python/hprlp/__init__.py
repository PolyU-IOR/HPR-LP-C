"""
HPRLP Python Package

GPU-accelerated Linear Programming solver using Halpern-Peaceman-Rachford splitting method.
"""

from .solver import HPRLPSolver, solve, solve_mps
from .parameters import Parameters
from .results import Results
from .model import Model
from .modeling import (
    ModelBuilder, Variable, LinearExpression, Constraint, TwoSidedConstraint,
    Sense, ConstraintSense, minimize, maximize, between
)

try:
    from ._hprlp_core import __version__
except ImportError:
    __version__ = "0.1.1"

__all__ = [
    'HPRLPSolver',
    'Model',
    'solve',
    'solve_mps',
    'Parameters',
    'Results',
    '__version__',
    # Modeling interface
    'ModelBuilder',
    'Variable',
    'LinearExpression',
    'Constraint',
    'TwoSidedConstraint',
    'Sense',
    'ConstraintSense',
    'minimize',
    'maximize',
    'between',
]
