from ._prima import minimize as _minimize, __version__
from ._nonlinear_constraints import NonlinearConstraint, process_single_nl_constraint, process_multiple_nl_constraints
from ._linear_constraints import LinearConstraint, process_single_linear_constraint, process_multiple_linear_constraints
from ._bounds import process_bounds
from enum import Enum
import numpy as np



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
  
  # Determine if we have a multiple nl constraints, just 1, or none, and process accordingly
  if len(nonlinear_constraints) > 1:
    nonlinear_constraint = process_multiple_nl_constraints(x0, nonlinear_constraints, options)
  elif len(nonlinear_constraints) == 1:
    nonlinear_constraint = process_single_nl_constraint(x0, nonlinear_constraints[0], options)
  else:
    nonlinear_constraint = None

  # Determine if we have a multiple nl constraints, just 1, or none, and process accordingly
  if len(linear_constraints) > 1:
    linear_constraint = process_multiple_linear_constraints(linear_constraints)
  elif len(linear_constraints) == 1:
    linear_constraint = process_single_linear_constraint(linear_constraints[0])
  else:
    linear_constraint = None
  
  return linear_constraint, nonlinear_constraint

def minimize(fun, x0, args=(), method=None, bounds=None, constraints=None, callback=None, options=None):
  lb, ub = process_bounds(bounds, x0)

  # Since we need f0 and nlconstr0 regardless, we should immediately check if they are provided in options
  # If nlconstr0 is provided and m_nlcon is not, we can infer m_nlcon from the length of nlconstr0

  temp_options = {}
  linear_constraint, nonlinear_constraint = process_constraints(constraints, x0, temp_options)
  if options is None:
    options = temp_options
  else:
    options.update(temp_options)

  if linear_constraint is not None:
    # Two things:
    # 1. PRIMA prefers A <= b as opposed to lb <= A <= ub
    # 2. PRIMA has both A_eq and A_ineq (and b_eq and b_ineq)
    # As such, we must:
    # 1. Convert lb <= A <= ub to A <= b
    # 2. Split A <= b into A_eq and A_ineq
    # Fortunately we can do both at the same time
    A_eq = []
    b_eq = []
    A_ineq = []
    b_ineq = []
    for i in range(len(linear_constraint.lb)):
      if linear_constraint.lb[i] == linear_constraint.ub[i]:
        A_eq.append(linear_constraint.A[i])
        b_eq.append(linear_constraint.lb[i])
      else:
        A_ineq.append(linear_constraint.A[i])
        b_ineq.append(linear_constraint.ub[i])
        # Flip the lb to to take format preferred by PRIMA
        A_ineq.append( - linear_constraint.A[i])
        b_ineq.append( - linear_constraint.lb[i])
    # Convert to numpy arrays, or set to None if empty
    A_eq = np.array(A_eq, dtype=np.float64) if len(A_eq) > 0 else None
    b_eq = np.array(b_eq, dtype=np.float64) if len(b_eq) > 0 else None
    A_ineq = np.array(A_ineq, dtype=np.float64) if len(A_ineq) > 0 else None
    b_ineq = np.array(b_ineq, dtype=np.float64) if len(b_ineq) > 0 else None
  else:
    A_eq = None
    b_eq = None
    A_ineq = None
    b_ineq = None

  if nonlinear_constraint is not None:
    # PRIMA prefers -inf < f(x) <= 0, so we need to modify the nonlinear constraint accordingly
    def constraint_function(x):
      values = np.array(nonlinear_constraint.fun(x), dtype=np.float64)
      return np.concatenate((values - nonlinear_constraint.ub, nonlinear_constraint.lb - values))
    # TODO: I'd like to handle cases where lb is -inf, so that we don't have unnecessary constraints
    # That's probably best done in process_nonlinear_constraints.
    if options is None:
      options = {}
    options['m_nlcon'] = len(nonlinear_constraint.lb)*2
    if 'nlconstr0' in options:
      options['nlconstr0'] = np.concatenate((options['nlconstr0'] - nonlinear_constraint.ub, nonlinear_constraint.lb - options['nlconstr0']))
    else:
      options['nlconstr0'] = constraint_function(x0)
    options['nlconstr0'] = np.array(options['nlconstr0'], dtype=np.float64)
    if 'f0' not in options: options['f0'] = fun(x0)
  else:
    constraint_function = None

  method = method.lower() if method is not None else None
  if method == "cobyla" and constraint_function is None:
    constraint_function = lambda x: 0
    options['m_nlcon'] = 1

  return _minimize(fun, x0, args, method, lb, ub, A_eq, b_eq, A_ineq, b_ineq, constraint_function, callback, options)
