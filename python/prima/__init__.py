# FIXME: Use 4 spaces for indentation instead of 2. 

from ._prima import minimize as _minimize, __version__, PRIMAMessage
from ._nonlinear_constraints import NonlinearConstraint, process_nl_constraints
from ._linear_constraints import LinearConstraint, process_single_linear_constraint, process_multiple_linear_constraints, separate_LC_into_eq_and_ineq
from ._bounds import process_bounds, Bounds
from enum import Enum
import numpy as np
from ._common import _project



class ConstraintType(Enum):
  LINEAR_NATIVE=5
  NONLINEAR_NATIVE=10
  LINEAR_NONNATIVE=15
  NONLINEAR_NONNATIVE=20
  LINEAR_DICT=25
  NONLINEAR_DICT=30


def get_constraint_type(constraint):
  # Make sure the test for native is first, since the hasattr tests will also pass for native constraints
  if   isinstance(constraint, LinearConstraint):                                                                 return ConstraintType.LINEAR_NATIVE
  elif isinstance(constraint, NonlinearConstraint):                                                              return ConstraintType.NONLINEAR_NATIVE
  elif isinstance(constraint, dict) and ('A'   in constraint) and ('lb' in constraint) and ('ub' in constraint): return ConstraintType.LINEAR_DICT
  elif isinstance(constraint, dict) and ('fun' in constraint) and ('lb' in constraint) and ('ub' in constraint): return ConstraintType.NONLINEAR_DICT
  elif hasattr(constraint, 'A')     and hasattr(constraint, 'lb') and hasattr(constraint, 'ub'):                 return ConstraintType.LINEAR_NONNATIVE
  elif hasattr(constraint, 'fun')   and hasattr(constraint, 'lb') and hasattr(constraint, 'ub'):                 return ConstraintType.NONLINEAR_NONNATIVE
  else: raise ValueError('Constraint type not recognized')


def process_constraints(constraints, x0, options):
  # First throw it back if it's None
  if constraints is None:
    return None, None
  # Next figure out if it's a list of constraints or a single constraint
  # If it's a single constraint, make it a list, and then the remaining logic
  # doesn't have to change
  if not isinstance(constraints, list):
    constraints = [constraints]

  # Separate out the linear and nonlinear constraints
  linear_constraints = []
  nonlinear_constraints = []
  for constraint in constraints:
    constraint_type = get_constraint_type(constraint)
    if constraint_type in (ConstraintType.LINEAR_NATIVE, ConstraintType.LINEAR_NONNATIVE):
      linear_constraints.append(constraint)
    elif constraint_type in (ConstraintType.NONLINEAR_NATIVE, ConstraintType.NONLINEAR_NONNATIVE):
      nonlinear_constraints.append(constraint)
    elif constraint_type == ConstraintType.LINEAR_DICT:
      linear_constraints.append(LinearConstraint(constraint['A'], constraint['lb'], constraint['ub']))
    elif constraint_type == ConstraintType.NONLINEAR_DICT:
      nonlinear_constraints.append(NonlinearConstraint(constraint['fun'], constraint['lb'], constraint['ub']))
    else:
      raise ValueError('Constraint type not recognized')
  
  if len(nonlinear_constraints) > 0:
    nonlinear_constraint = process_nl_constraints(x0, nonlinear_constraints, options)
  else:
    nonlinear_constraint = None

  # Determine if we have a multiple linear constraints, just 1, or none, and process accordingly
  if len(linear_constraints) > 1:
    linear_constraint = process_multiple_linear_constraints(linear_constraints)
  elif len(linear_constraints) == 1:
    linear_constraint = process_single_linear_constraint(linear_constraints[0])
  else:
    linear_constraint = None
  
  return linear_constraint, nonlinear_constraint

def minimize(fun, x0, args=(), method=None, bounds=None, constraints=None, callback=None, options=None):

  temp_options = {}
  linear_constraint, nonlinear_constraint = process_constraints(constraints, x0, temp_options)
  if options is None:
    options = temp_options
  else:
    options.update(temp_options)

  if method is None:
    if nonlinear_constraint is not None:
      print("Nonlinear constraints detected, applying COBYLA")
      method = "cobyla"
    elif linear_constraint is not None:
      print("Linear constraints detected without nonlinear constraints, applying LINCOA")
      method = "lincoa"
    elif bounds is not None:
      print("Bounds without linear or nonlinear constraints detected, applying BOBYQA")
      method = "bobyqa"
    else:
      print("No bounds or constraints detected, applying NEWUOA")
      method = "newuoa"
  else:
    # Raise some errors if methods were called with inappropriate options
    method = method.lower()
    if method != "cobyla" and nonlinear_constraint is not None:
      raise ValueError('Nonlinear constraints were provided for an algorithm that cannot handle them')
    if method not in ("cobyla", "lincoa") and linear_constraint is not None:
      raise ValueError('Linear constraints were provided for an algorithm that cannot handle them')
    if method not in ("cobyla", "bobyqa", "lincoa") and bounds is not None:
      raise ValueError('Bounds were provided for an algorithm that cannot handle them')

  try:
    lenx0 = len(x0)
  except TypeError:
    lenx0 = 1

  lb, ub = process_bounds(bounds, lenx0)

  if linear_constraint is not None:
    # FIXME: _project should be applied only if the problem has bounds or linear constraints but
    # no nonlinear constraints. Therefore, ", 'nonlinear': None" is superfluous. 
    x0 = _project(x0, lb, ub, {'linear': linear_constraint, 'nonlinear': None})
    A_eq, b_eq, A_ineq, b_ineq = separate_LC_into_eq_and_ineq(linear_constraint)
  else:
    A_eq = None
    b_eq = None
    A_ineq = None
    b_ineq = None

  if nonlinear_constraint is not None:
    # FIXME: Do the following conversion in _nonlinear_constraints.py. Otherwise, it is 
    # inconsistent with the conversion of the linear constraints.  
    
    # The Python interface receives nonlinear constraints lb <= constraint(x) <= ub, but the Fortran 
    # backend of PRIMA expects that the linear constraints are specified as constr(x) <= 0.  
    # We need to define the constraint function accordingly.

    def constraint_function(x):
      values = np.array(nonlinear_constraint.fun(x), dtype=np.float64)
      return np.concatenate(([vi - ub_i for ub_i, vi in zip(nonlinear_constraint.ub, values) if ub_i < np.inf], [lb_i - vi for lb_i, vi in zip(nonlinear_constraint.lb, values) if lb_i > -np.inf]))

    if options is None:
      options = {}
    # FIXME: 
    # 1. The concept is wrong here. 'm_nlcon', 'nlconstr0', and 'f0' are not options. 
    # They are part of the problem. 
    # 2. 'm_nlcon', 'nlconstr0', and 'f0' should only be defined for COBYLA, not for others. 
    # 3. Due to 2, 'm_nlcon', 'nlconstr0', and 'f0' should not be defined here but by a lower-level function that 
    # invokes COBYLA; for example, see 
    # https://github.com/pdfo/pdfo/blob/c76066dba48dc44abf55c4bbb69575391daef316/python/pdfo/_cobyla.py#L340-L341
    # 4. In the following line, we should write `len(nonlinear_constraint.ub)` rather than `len(nonlinear_constraint.lb)`,
    # even though they should be the same. In addition, we should take the possibility of `ub = inf` into account. Last
    # but not the least, we should write `lb > -inf` rather than `lb != -inf`. The two are different --- the former is 
    # false but the latter is true if lb is NaN. Anyway, we should avoid using `==` or`!=` on floating-point numbers.
    # Hence the revised version is as follows.
    #options['m_nlcon'] = len(nonlinear_constraint.lb) + len([lb_i for lb_i in nonlinear_constraint.lb if lb_i != -np.inf])
    options['m_nlcon'] = len([ub_i for ub_i in nonlinear_constraint.ub if ub_i < np.inf]) + len([lb_i for lb_i in nonlinear_constraint.lb if lb_i > -np.inf])
    options['nlconstr0'] = constraint_function(x0)
    options['nlconstr0'] = np.array(options['nlconstr0'], dtype=np.float64)
    options['f0'] = fun(x0)  
    # FIXME: The following is wrong. f0 should never be an option. 
    # if 'f0' not in options: options['f0'] = fun(x0)
  else:
    constraint_function = None

  return _minimize(fun, x0, args, method, lb, ub, A_eq, b_eq, A_ineq, b_ineq, constraint_function, callback, options)
