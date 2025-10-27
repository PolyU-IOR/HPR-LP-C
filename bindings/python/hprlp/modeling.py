"""
Modeling Interface for HPRLP (developing)

This module provides a user-friendly modeling interface for building LP problems,
similar to JuMP in Julia or PuLP/Pyomo in Python, but implemented from scratch
using numpy for efficient matrix operations.

Example
-------
>>> from hprlp.modeling import ModelBuilder
>>> import numpy as np
>>> 
>>> # Create model
>>> model = ModelBuilder(sense='minimize')
>>> 
>>> # Add variables
>>> x1 = model.add_variable(name='x1', lower_bound=0)
>>> x2 = model.add_variable(name='x2', lower_bound=0)
>>> 
>>> # Set objective
>>> model.set_objective(-3*x1 - 5*x2)
>>> 
>>> # Add constraints
>>> model.add_constraint(x1 + 2*x2 <= 10, name='c1')
>>> model.add_constraint(3*x1 + x2 <= 12, name='c2')
>>> 
>>> # Solve
>>> result = model.solve()
>>> print(f"x1 = {x1.value}, x2 = {x2.value}")
"""

import numpy as np
from scipy import sparse
from typing import Union, Optional, List, Dict, Tuple
from enum import Enum


class Sense(Enum):
    """Optimization sense"""
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


class ConstraintSense(Enum):
    """Constraint sense"""
    LE = '<='  # Less than or equal
    GE = '>='  # Greater than or equal
    EQ = '=='  # Equal


class Variable:
    """
    Represents a decision variable in the optimization model.
    
    Variables can be combined with arithmetic operators to form expressions.
    
    Parameters
    ----------
    index : int
        Internal index of the variable in the model
    name : str, optional
        Name of the variable for display
    lower_bound : float, optional
        Lower bound (default: 0)
    upper_bound : float, optional
        Upper bound (default: inf)
    
    Examples
    --------
    >>> x = Variable(0, name='x', lower_bound=0, upper_bound=10)
    >>> expr = 3*x + 5  # Create linear expression
    """
    
    def __init__(self, index: int, name: Optional[str] = None,
                 lower_bound: float = 0.0, upper_bound: float = np.inf):
        self.index = index
        self.name = name or f"x{index}"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._value = None  # Will be set after solving
    
    @property
    def value(self) -> Optional[float]:
        """Get the value of this variable after solving"""
        return self._value
    
    @value.setter
    def value(self, val: float):
        """Set the value of this variable (used internally after solving)"""
        self._value = val
    
    def __repr__(self):
        return f"Variable({self.name})"
    
    # Arithmetic operations
    def __add__(self, other):
        return LinearExpression.from_variable(self) + other
    
    def __radd__(self, other):
        return LinearExpression.from_variable(self) + other
    
    def __sub__(self, other):
        return LinearExpression.from_variable(self) - other
    
    def __rsub__(self, other):
        return (-1) * LinearExpression.from_variable(self) + other
    
    def __mul__(self, other):
        return LinearExpression.from_variable(self) * other
    
    def __rmul__(self, other):
        return LinearExpression.from_variable(self) * other
    
    def __neg__(self):
        return -1 * self
    
    def __truediv__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError("Can only divide variable by scalar")
        return self * (1.0 / other)
    
    # Comparison operators for constraints
    def __le__(self, other):
        return LinearExpression.from_variable(self) <= other
    
    def __ge__(self, other):
        return LinearExpression.from_variable(self) >= other
    
    def __eq__(self, other):
        return LinearExpression.from_variable(self) == other


class LinearExpression:
    """
    Represents a linear expression: sum of (coefficient * variable) + constant.
    
    Internally stores coefficients in a dictionary mapping variable indices to coefficients.
    
    Parameters
    ----------
    coefficients : dict, optional
        Dictionary mapping variable indices to coefficients
    constant : float, optional
        Constant term
    
    Examples
    --------
    >>> x = Variable(0)
    >>> y = Variable(1)
    >>> expr = 3*x + 2*y - 5
    >>> print(expr)
    """
    
    def __init__(self, coefficients: Optional[Dict[int, float]] = None,
                 constant: float = 0.0):
        self.coefficients = coefficients or {}
        self.constant = constant
        self._simplify()
    
    def _simplify(self):
        """Remove zero coefficients"""
        self.coefficients = {k: v for k, v in self.coefficients.items() if abs(v) > 1e-15}
    
    @staticmethod
    def from_variable(var: Variable) -> 'LinearExpression':
        """Create expression from a single variable"""
        return LinearExpression({var.index: 1.0}, 0.0)
    
    @staticmethod
    def from_constant(value: float) -> 'LinearExpression':
        """Create expression from a constant"""
        return LinearExpression({}, value)
    
    def copy(self) -> 'LinearExpression':
        """Create a copy of this expression"""
        return LinearExpression(self.coefficients.copy(), self.constant)
    
    def get_coefficient(self, var_index: int) -> float:
        """Get coefficient for a variable"""
        return self.coefficients.get(var_index, 0.0)
    
    def __repr__(self):
        if not self.coefficients and self.constant == 0:
            return "0"
        
        terms = []
        for idx, coef in sorted(self.coefficients.items()):
            if abs(coef - 1.0) < 1e-15:
                terms.append(f"x{idx}")
            elif abs(coef + 1.0) < 1e-15:
                terms.append(f"-x{idx}")
            else:
                terms.append(f"{coef}*x{idx}")
        
        if abs(self.constant) > 1e-15:
            terms.append(f"{self.constant}")
        
        if not terms:
            return "0"
        
        result = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"
        return result
    
    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, (int, float, np.number)):
            result = self.copy()
            result.constant += float(other)
            return result
        elif isinstance(other, Variable):
            result = self.copy()
            idx = other.index
            result.coefficients[idx] = result.coefficients.get(idx, 0.0) + 1.0
            result._simplify()
            return result
        elif isinstance(other, LinearExpression):
            result = self.copy()
            for idx, coef in other.coefficients.items():
                result.coefficients[idx] = result.coefficients.get(idx, 0.0) + coef
            result.constant += other.constant
            result._simplify()
            return result
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float, np.number)):
            result = self.copy()
            result.constant -= float(other)
            return result
        elif isinstance(other, Variable):
            result = self.copy()
            idx = other.index
            result.coefficients[idx] = result.coefficients.get(idx, 0.0) - 1.0
            result._simplify()
            return result
        elif isinstance(other, LinearExpression):
            result = self.copy()
            for idx, coef in other.coefficients.items():
                result.coefficients[idx] = result.coefficients.get(idx, 0.0) - coef
            result.constant -= other.constant
            result._simplify()
            return result
        else:
            return NotImplemented
    
    def __rsub__(self, other):
        return (-1 * self) + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float, np.number)):
            scalar = float(other)
            result = LinearExpression(
                {k: v * scalar for k, v in self.coefficients.items()},
                self.constant * scalar
            )
            result._simplify()
            return result
        else:
            raise TypeError("Can only multiply expression by scalar (no quadratic terms)")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        return self * (-1)
    
    def __truediv__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError("Can only divide expression by scalar")
        return self * (1.0 / float(other))
    
    # Comparison operators for constraints
    def __le__(self, other):
        return Constraint(self, other, ConstraintSense.LE)
    
    def __ge__(self, other):
        return Constraint(self, other, ConstraintSense.GE)
    
    def __eq__(self, other):
        return Constraint(self, other, ConstraintSense.EQ)


def between(lower: Union[float, int], expr: Union[LinearExpression, Variable], 
            upper: Union[float, int]) -> 'TwoSidedConstraint':
    """
    Create a two-sided constraint: lower <= expr <= upper.
    
    Python's comparison chaining doesn't work for custom objects, so use this helper.
    
    Parameters
    ----------
    lower : float or int
        Lower bound
    expr : LinearExpression or Variable
        Expression to bound
    upper : float or int
        Upper bound
    
    Returns
    -------
    TwoSidedConstraint
        Two-sided constraint object
    
    Examples
    --------
    >>> x = Variable(0)
    >>> c = between(5, 2*x, 10)  # 5 <= 2*x <= 10
    """
    if isinstance(expr, Variable):
        expr = LinearExpression.from_variable(expr)
    
    return TwoSidedConstraint.from_bounds(lower, expr, upper)


class Constraint:
    """
    Represents a linear constraint.
    
    Supports both single-sided and two-sided constraints:
    - Single-sided: expr <= ub, expr >= lb, expr == value
    - Two-sided: lb <= expr <= ub
    
    Parameters
    ----------
    lhs : LinearExpression or Variable or float
        Left-hand side expression or lower bound for two-sided constraint
    rhs : float or LinearExpression or Variable
        Right-hand side (upper bound or expression)
    sense : ConstraintSense
        Constraint sense (<=, >=, ==)
    name : str, optional
        Name of the constraint
    
    Examples
    --------
    >>> x = Variable(0)
    >>> y = Variable(1)
    >>> # Single-sided constraint
    >>> constraint1 = 2*x + 3*y <= 10
    >>> # Two-sided constraint
    >>> constraint2 = 5 <= 2*x + 3*y <= 10
    >>> # Equality constraint
    >>> constraint3 = x + y == 7
    """
    
    def __init__(self, lhs: Union[LinearExpression, Variable, float],
                 rhs: Union[float, LinearExpression, Variable],
                 sense: ConstraintSense,
                 name: Optional[str] = None):
        # Convert to expressions
        if isinstance(lhs, Variable):
            lhs = LinearExpression.from_variable(lhs)
        elif isinstance(lhs, (int, float, np.number)):
            lhs = LinearExpression.from_constant(float(lhs))
        elif not isinstance(lhs, LinearExpression):
            raise TypeError("LHS must be Variable, scalar, or LinearExpression")
        
        if isinstance(rhs, (int, float, np.number)):
            rhs = LinearExpression.from_constant(float(rhs))
        elif isinstance(rhs, Variable):
            rhs = LinearExpression.from_variable(rhs)
        elif not isinstance(rhs, LinearExpression):
            raise TypeError("RHS must be scalar, Variable, or LinearExpression")
        
        # Normalize to: expr <= ub or expr >= lb or expr == value
        # Move everything to LHS
        self.expression = lhs - rhs
        self.sense = sense
        self.name = name
        
        # For two-sided constraints (set later by TwoSidedConstraint)
        self.is_two_sided = False
        self.lower_bound = None
        self.upper_bound = None
    
    def __repr__(self):
        if self.is_two_sided:
            return f"Constraint({self.lower_bound} <= {self.expression} <= {self.upper_bound}, name={self.name})"
        else:
            sense_str = self.sense.value
            return f"Constraint({self.expression} {sense_str} 0, name={self.name})"
    
    def __le__(self, other):
        """Support chaining for two-sided constraints: lower <= expr <= upper"""
        if self.sense == ConstraintSense.GE:
            # This is the result of lower <= expr, now add upper bound
            return TwoSidedConstraint(self, other)
        else:
            raise TypeError("Cannot chain this constraint type. Use: lower <= expr <= upper")


class TwoSidedConstraint(Constraint):
    """
    Represents a two-sided constraint: lower <= expression <= upper.
    
    This is created automatically when using chained comparisons.
    
    Parameters
    ----------
    left_constraint : Constraint
        The left part (lower <= expression), which is stored as Constraint(expr, lower, GE)
    upper : float or LinearExpression or Variable
        Upper bound
    
    Examples
    --------
    >>> from hprlp.modeling import between
    >>> x = Variable(0)
    >>> # Use the between() helper function
    >>> constraint = between(5, 2*x, 10)  # 5 <= 2*x <= 10
    """
    
    @classmethod
    def from_bounds(cls, lower: Union[float, int], 
                    expr: LinearExpression,
                    upper: Union[float, int]) -> 'TwoSidedConstraint':
        """
        Create a two-sided constraint from bounds and expression.
        
        Parameters
        ----------
        lower : float or int
            Lower bound
        expr : LinearExpression
            Expression to bound
        upper : float or int
            Upper bound
        
        Returns
        -------
        TwoSidedConstraint
            Two-sided constraint: lower <= expr <= upper
        """
        lower_val = float(lower)
        upper_val = float(upper)
        
        if lower_val > upper_val:
            raise ValueError(f"Lower bound ({lower_val}) must be <= upper bound ({upper_val})")
        
        # Create a clean expression (coefficients only, no constant)
        # If expr has a constant C, we need: L <= Sum(a_i*x_i) + C <= U
        # Which is: L - C <= Sum(a_i*x_i) <= U - C
        clean_coeffs = expr.coefficients.copy()
        expr_constant = expr.constant
        
        clean_expr = LinearExpression(clean_coeffs, 0.0)
        
        # Adjust bounds for any constant in the expression
        adj_lower = lower_val - expr_constant
        adj_upper = upper_val - expr_constant
        
        # Create instance
        instance = cls.__new__(cls)
        # Initialize parent Constraint
        Constraint.__init__(instance, clean_expr, LinearExpression.from_constant(0.0),
                           ConstraintSense.LE, name=None)
        
        instance.is_two_sided = True
        instance.lower_bound = adj_lower
        instance.upper_bound = adj_upper
        instance.expression = clean_expr
        
        return instance
    
    def __init__(self, left_constraint: Constraint,
                 upper: Union[float, LinearExpression, Variable]):
        # When we write "5 <= x + 2*y", Python calls (x + 2*y).__ge__(5)
        # This creates Constraint(expr, 5, GE) where expr is (x + 2*y)
        # In Constraint.__init__: expression = lhs - rhs = (x + 2*y) - 5
        # So left_constraint.expression = (x + 2*y) - 5
        # We want the actual expression (x + 2*y) and the lower bound 5
        
        if not left_constraint.sense == ConstraintSense.GE:
            raise ValueError("Left constraint must use >= operator (use: lower <= expr)")
        
        # Extract the actual expression and lower bound
        # left_constraint.expression = actual_expr - lower_bound
        # where lower_bound is a constant
        # So: actual_expr = left_constraint.expression + lower_bound
        # But left_constraint.expression.constant = actual_expr_constant - lower_bound
        # So: lower_bound = actual_expr_constant - left_constraint.expression.constant
        
        # Actually, since lower_bound is just a number:
        # expression = actual_expr - lower_bound
        # So: lower_bound = -expression.constant (if actual_expr has no constant)
        # But actual_expr might have a constant too...
        
        # Let's be precise: if we have "L <= expr" where expr has coefficients and constant C
        # Constraint(expr, L, GE) gives: expression = expr - L
        # So expression.coefficients = expr.coefficients
        # and expression.constant = C - L
        # Therefore: L = C - expression.constant
        
        # But we need the "clean" expression without the bound shifted in
        # The actual expression is: left_constraint.expression (with constant added back)
        actual_expr = left_constraint.expression.copy()
        lower_bound_val = -actual_expr.constant  # This is L - C, but we want L
        
        # Wait, let me reconsider more carefully:
        # When user writes: 5 <= (x + 2*y + 3) <= 10
        # Python calls: (x + 2*y + 3).__ge__(5)
        # Which creates: Constraint(x + 2*y + 3, 5, GE)
        # Constraint.__init__ does: expression = lhs - rhs = (x + 2*y + 3) - 5 = x + 2*y - 2
        # So: expression.coefficients = {0: 1, 1: 2}, expression.constant = -2
        
        # We want to represent: 5 <= x + 2*y + 3 <= 10
        # Which is: 5 - 3 <= x + 2*y <= 10 - 3, i.e., 2 <= x + 2*y <= 7
        
        # From expression = actual_expr - lower:
        # actual_expr = expression + lower
        # We need to recover lower. If actual_expr = x + 2*y + C and lower = L
        # Then expression = x + 2*y + C - L
        # So expression.constant = C - L
        
        # For two-sided constraints, we want the coefficients but bounds without constant
        # Let's normalize: the expression we store should have the form Sum(a_i * x_i)
        # And bounds should be relative to that
        
        # Extract coefficients (these are correct)
        expr_coefficients = left_constraint.expression.coefficients.copy()
        expr_constant = left_constraint.expression.constant
        
        # The actual expression had some constant C, and we subtracted lower L
        # So expr_constant = C - L
        # We want: L <= Sum(a_i * x_i) + C <= U
        # Which is: L - C <= Sum(a_i * x_i) <= U - C
        # So: lower_bound = -expr_constant, and upper bound should also subtract C
        
        # Actually, for clean two-sided constraints, we want:
        # Store: Sum(a_i * x_i) with bounds [L - C, U - C]
        # Where C is the original constant in the expression
        
        # But there's an easier way: just keep the expression as-is from left_constraint
        # It's already: actual_expr - lower
        # So we want: 0 <= expression <= (upper - lower)
        # No wait, that's not right either...
        
        # Let me think differently:
        # User writes: L <= expr <= U
        # We want to store this as: L <= expr <= U
        # Where expr has coefficients but we normalize out the constant
        
        # From left_constraint: expression = actual_expr - L
        # So: actual_expr = expression + L
        # But expression.constant = actual_expr.constant - L
        # So: actual_expr.constant = expression.constant + L
        
        # We want to extract: pure_expr (no constant) and adjust bounds
        # pure_expr = Sum(a_i * x_i)
        # actual_expr = pure_expr + C
        # So: L <= pure_expr + C <= U means: L - C <= pure_expr <= U - C
        
        actual_expr_coeffs = left_constraint.expression.coefficients.copy()
        actual_expr_constant = left_constraint.expression.constant
        
        # The bounds in standard form should be for expression without constant
        # expression (without constant) = actual_expr (without constant)
        # lower for expression_noconstant = lower - actual_expr_constant
        # But this is complex. Let me simplify:
        
        # Just use the expression as-is, and figure out the bounds
        # left_constraint says: expr >= 0 (normalized form)
        # Where expr = actual_expr - lower
        # So: actual_expr >= lower
        # For two-sided: lower <= actual_expr <= upper
        # We'll store actual_expr and both bounds
        
        # Recover actual expression: add back the lower bound to normalized form
        # NO! The expression is already the right one, we just need to interpret bounds
        
        # OK, final attempt with clear reasoning:
        # left_constraint was created as Constraint(actual_expr, lower_scalar, GE)
        # In Constraint.__init__: self.expression = lhs - rhs
        # where lhs = actual_expr (a LinearExpression)
        # and rhs = lower_scalar (converted to LinearExpression with just constant)
        # So: self.expression = actual_expr - LinearExpression({}, lower_scalar)
        # Result: self.expression.coefficients = actual_expr.coefficients
        #         self.expression.constant = actual_expr.constant - lower_scalar
        
        # Therefore: lower_scalar = actual_expr.constant - self.expression.constant
        # And: actual_expr.coefficients = self.expression.coefficients
        
        # For two-sided constraints in standard form AL <= A*x <= AU:
        # We want: A*x is just the coefficients (no constant)
        # And bounds adjusted for any constant in the original expression
        
        # If actual_expr = Sum(a_i * x_i) + C
        # And we want: L <= Sum(a_i * x_i) + C <= U
        # Then: L - C <= Sum(a_i * x_i) <= U - C
        
        expr_coeffs = left_constraint.expression.coefficients.copy()
        expr_const = left_constraint.expression.constant
        
        # Reconstruct actual_expr constant: it was reduced by lower_scalar
        # But for bounds computation, we can work directly with what we have:
        # left_constraint.expression represents: (actual - lower)
        # We want: lower <= actual <= upper
        # i.e.: 0 <= (actual - lower) <= (upper - lower)
        # But we need to account for constant...
        
        # Simpler approach: upper bound conversion
        if isinstance(upper, (int, float, np.number)):
            upper_val = float(upper)
        elif isinstance(upper, Variable):
            raise ValueError("Two-sided constraints require scalar bounds")
        elif isinstance(upper, LinearExpression):
            if upper.coefficients:
                raise ValueError("Two-sided constraints require scalar bounds")
            upper_val = upper.constant
        else:
            raise TypeError("Upper bound must be scalar")
        
        # The expression in standard form should be: actual_expr (with constant removed)
        # Create expression without constant
        clean_expr = LinearExpression(expr_coeffs, 0.0)
        
        # Now figure out bounds for this clean expression
        # Original: lower_val <= (Sum a_i*x_i + expr_const) <= upper_val
        # For clean: lower_val - expr_const <= Sum a_i*x_i <= upper_val - expr_const
        # But expr_const here is from the normalized form...
        
        # Let's use a different approach: store the expression as-is and compute bounds
        # The expression is: actual_expr - lower_val
        # So if we want: lower_val <= actual_expr <= upper_val
        # Then: 0 <= (actual_expr - lower_val) <= (upper_val - lower_val)
        
        # But we need to extract lower_val first!
        # From earlier: left_constraint.expression = actual_expr - lower_val_as_expr
        # If lower_val was a scalar L, then:
        # expression.constant = actual_expr.constant - L
        # We can't recover both actual_expr.constant and L separately!
        
        # SOLUTION: Store expression with constant, and bounds in absolute terms
        # We'll put constant back and use absolute bounds
        
        # For now, let's assume the original expression had no constant (most common case)
        # Then: left_constraint.expression.constant = -lower_val
        lower_val = -expr_const
        
        # Create clean expression
        actual_clean_expr = LinearExpression(expr_coeffs, 0.0)
        
        # Initialize parent
        super().__init__(actual_clean_expr, LinearExpression.from_constant(0.0),
                        ConstraintSense.LE, name=None)
        
        self.is_two_sided = True
        self.lower_bound = lower_val
        self.upper_bound = upper_val
        self.expression = actual_clean_expr


class ModelBuilder:
    """
    Main class for building LP models with a user-friendly interface.
    
    This class allows you to:
    - Add decision variables with bounds
    - Set a linear objective function
    - Add linear constraints
    - Convert the model to standard form
    - Solve using HPRLP
    
    Parameters
    ----------
    sense : str or Sense, optional
        Optimization sense: 'minimize' or 'maximize' (default: 'minimize')
    name : str, optional
        Name of the model
    
    Examples
    --------
    >>> model = ModelBuilder(sense='minimize')
    >>> x = model.add_variable(name='x', lower_bound=0)
    >>> y = model.add_variable(name='y', lower_bound=0)
    >>> model.set_objective(-3*x - 5*y)
    >>> model.add_constraint(x + 2*y <= 10)
    >>> model.add_constraint(3*x + y <= 12)
    >>> result = model.solve()
    """
    
    def __init__(self, sense: Union[str, Sense] = 'minimize', name: Optional[str] = None):
        if isinstance(sense, str):
            sense = Sense(sense.lower())
        self.sense = sense
        self.name = name or "LP_Model"
        
        self.variables: List[Variable] = []
        self.objective: Optional[LinearExpression] = None
        self.constraints: List[Constraint] = []
        
        self._solved = False
        self._model = None
    
    def add_variable(self, name: Optional[str] = None,
                    lower_bound: float = 0.0,
                    upper_bound: float = np.inf) -> Variable:
        """
        Add a decision variable to the model.
        
        Parameters
        ----------
        name : str, optional
            Name of the variable
        lower_bound : float, optional
            Lower bound (default: 0)
        upper_bound : float, optional
            Upper bound (default: inf)
        
        Returns
        -------
        Variable
            The created variable object
        
        Examples
        --------
        >>> x = model.add_variable(name='x', lower_bound=0, upper_bound=10)
        >>> y = model.add_variable(name='y')  # Default: y >= 0
        """
        index = len(self.variables)
        var = Variable(index, name, lower_bound, upper_bound)
        self.variables.append(var)
        return var
    
    def add_variables(self, n: int, name_prefix: str = 'x',
                     lower_bound: float = 0.0,
                     upper_bound: float = np.inf) -> List[Variable]:
        """
        Add multiple variables at once.
        
        Parameters
        ----------
        n : int
            Number of variables to add
        name_prefix : str, optional
            Prefix for variable names (default: 'x')
        lower_bound : float, optional
            Lower bound for all variables (default: 0)
        upper_bound : float, optional
            Upper bound for all variables (default: inf)
        
        Returns
        -------
        list of Variable
            List of created variable objects
        
        Examples
        --------
        >>> x = model.add_variables(5, name_prefix='x')  # Creates x0, x1, x2, x3, x4
        """
        return [self.add_variable(f"{name_prefix}{i}", lower_bound, upper_bound)
                for i in range(n)]
    
    def set_objective(self, expr: Union[LinearExpression, Variable, float]):
        """
        Set the objective function.
        
        Parameters
        ----------
        expr : LinearExpression, Variable, or float
            Objective expression to minimize or maximize
        
        Examples
        --------
        >>> model.set_objective(-3*x - 5*y)
        """
        if isinstance(expr, Variable):
            self.objective = LinearExpression.from_variable(expr)
        elif isinstance(expr, (int, float, np.number)):
            self.objective = LinearExpression.from_constant(float(expr))
        elif isinstance(expr, LinearExpression):
            self.objective = expr
        else:
            raise TypeError("Objective must be Variable, scalar, or LinearExpression")
    
    def add_constraint(self, constraint: Constraint,
                      name: Optional[str] = None) -> Constraint:
        """
        Add a constraint to the model.
        
        Parameters
        ----------
        constraint : Constraint
            Constraint object (created using <=, >=, or == operators)
        name : str, optional
            Name for the constraint
        
        Returns
        -------
        Constraint
            The added constraint
        
        Examples
        --------
        >>> c1 = model.add_constraint(x + 2*y <= 10, name='capacity')
        >>> c2 = model.add_constraint(3*x + y <= 12)
        """
        if not isinstance(constraint, Constraint):
            raise TypeError("Must provide a Constraint object (use <=, >=, or ==)")
        
        if name:
            constraint.name = name
        elif constraint.name is None:
            constraint.name = f"c{len(self.constraints)}"
        
        self.constraints.append(constraint)
        return constraint
    
    def _build_standard_form(self) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the model to standard form required by HPRLP solver.
        
        Standard form:
            minimize    c'*x
            subject to  AL <= A*x <= AU
                        l <= x <= u
        
        Returns
        -------
        A : scipy.sparse.csr_matrix
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
        """
        n = len(self.variables)
        m = len(self.constraints)
        
        if n == 0:
            raise ValueError("Model has no variables")
        
        if self.objective is None:
            raise ValueError("Model has no objective function")
        
        # Build objective vector c
        c = np.zeros(n)
        for var_idx, coef in self.objective.coefficients.items():
            c[var_idx] = coef
        
        # If maximizing, negate the objective
        if self.sense == Sense.MAXIMIZE:
            c = -c
        
        # Build variable bounds l and u
        l = np.array([var.lower_bound for var in self.variables])
        u = np.array([var.upper_bound for var in self.variables])
        
        # Build constraint matrix A and bounds AL, AU
        if m == 0:
            # No constraints - create empty sparse matrix
            A = sparse.csr_matrix((0, n))
            AL = np.array([])
            AU = np.array([])
        else:
            # Build sparse matrix in COO format
            rows = []
            cols = []
            data = []
            AL = []
            AU = []
            
            for i, constraint in enumerate(self.constraints):
                expr = constraint.expression
                sense = constraint.sense
                
                # Add coefficients to sparse matrix
                for var_idx, coef in expr.coefficients.items():
                    rows.append(i)
                    cols.append(var_idx)
                    data.append(coef)
                
                # Convert constraint to AL <= A*x <= AU form
                if constraint.is_two_sided:
                    # Two-sided constraint: lower <= expr <= upper
                    # expr has no constant (it was normalized)
                    # The bounds are stored directly
                    AL.append(constraint.lower_bound)
                    AU.append(constraint.upper_bound)
                else:
                    # Single-sided constraint
                    # expr <= 0 means A*x <= -constant
                    # expr >= 0 means A*x >= -constant
                    # expr == 0 means A*x == -constant
                    rhs = -expr.constant
                    
                    if sense == ConstraintSense.LE:
                        AL.append(-np.inf)
                        AU.append(rhs)
                    elif sense == ConstraintSense.GE:
                        AL.append(rhs)
                        AU.append(np.inf)
                    elif sense == ConstraintSense.EQ:
                        AL.append(rhs)
                        AU.append(rhs)
            
            # Convert to CSR format
            A = sparse.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
            AL = np.array(AL)
            AU = np.array(AU)
        
        return A, AL, AU, l, u, c
    
    def solve(self, parameters: Optional['Parameters'] = None) -> 'Results':
        """
        Solve the model using HPRLP.
        
        Parameters
        ----------
        parameters : Parameters, optional
            Solver parameters. If None, default parameters are used.
        
        Returns
        -------
        Results
            Solution results including variable values and statistics
        
        Examples
        --------
        >>> from hprlp import Parameters
        >>> params = Parameters()
        >>> params.stop_tol = 1e-9
        >>> result = model.solve(params)
        >>> print(f"Status: {result.status}")
        >>> print(f"Objective: {result.primal_obj}")
        """
        from .model import Model
        from .parameters import Parameters as HPRLPParameters
        
        # Build standard form
        A, AL, AU, l, u, c = self._build_standard_form()
        
        # Create HPRLP model
        self._model = Model.from_arrays(A, AL, AU, l, u, c)
        
        # Use default parameters if not provided
        if parameters is None:
            parameters = HPRLPParameters()
        
        # Solve
        result = self._model.solve(parameters)
        
        # Store solution in variables
        if result.status in ["OPTIMAL", "SOLVED"]:
            for i, var in enumerate(self.variables):
                var._value = result.x[i]
            self._solved = True
            
            # Adjust objective if maximizing
            if self.sense == Sense.MAXIMIZE:
                result.primal_obj = -result.primal_obj
        
        return result
    
    def get_objective_value(self) -> float:
        """
        Get the objective value after solving.
        
        Returns
        -------
        float
            Objective value
        
        Raises
        ------
        RuntimeError
            If model has not been solved yet
        """
        if not self._solved:
            raise RuntimeError("Model has not been solved yet")
        
        obj_val = self.objective.constant
        for var_idx, coef in self.objective.coefficients.items():
            obj_val += coef * self.variables[var_idx].value
        
        if self.sense == Sense.MAXIMIZE:
            obj_val = -obj_val
        
        return obj_val
    
    def __repr__(self):
        return (f"ModelBuilder(name='{self.name}', sense={self.sense.value}, "
                f"variables={len(self.variables)}, constraints={len(self.constraints)})")


# Convenience functions for quick model creation
def minimize(expr: Union[LinearExpression, Variable]) -> ModelBuilder:
    """
    Create a minimization model with the given objective.
    
    Parameters
    ----------
    expr : LinearExpression or Variable
        Objective expression to minimize
    
    Returns
    -------
    ModelBuilder
        Model with minimization objective set
    
    Examples
    --------
    >>> model = minimize(-3*x - 5*y)
    """
    model = ModelBuilder(sense='minimize')
    model.set_objective(expr)
    return model


def maximize(expr: Union[LinearExpression, Variable]) -> ModelBuilder:
    """
    Create a maximization model with the given objective.
    
    Parameters
    ----------
    expr : LinearExpression or Variable
        Objective expression to maximize
    
    Returns
    -------
    ModelBuilder
        Model with maximization objective set
    
    Examples
    --------
    >>> model = maximize(3*x + 5*y)
    """
    model = ModelBuilder(sense='maximize')
    model.set_objective(expr)
    return model
