import numpy as np


class LinearConstraint:
  # Defaults for lb/ub are -inf and inf so that only one needs to be provided and the
  # other will have no impact
  def __init__(self, A, lb=-np.inf, ub=np.inf):
    self.A = A
    self.lb = lb
    self.ub = ub
    # A must be a scalar, a list, or a numpy array
    # If it's a list, we assume that it's basically a 1-row matrix
    # If it's a numpy array with only 1 dimension we force it to be a 1-row matrix
    if isinstance(self.A, int) or isinstance(self.A, float):
      self.A = np.array([[self.A]])
    elif isinstance(self.A, list) or isinstance(self.A, np.ndarray):
      self.A = np.atleast_2d(self.A)
    else:
      raise("A must be a scalar, list, or numpy array")
    
    num_constraints = self.A.shape[0]

    # bounds must be a scalar or a 1D list/array of length equal to number of rows of A
    def process_bound(bound, name):
      if isinstance(bound, int) or isinstance(bound, float):
        bound = np.array([bound]*num_constraints)
      elif isinstance(bound, list) or isinstance(bound, np.ndarray):
        bound = np.array(bound)
        assert bound.ndim == 1 and len(bound) == num_constraints, f"{name} must be a scalar or a 1D list/array of length equal to the number of rows of A"
      else:
        raise(f"{name} must be a scalar, list, or numpy array")
      return bound
    
    self.lb = process_bound(self.lb, 'lb')
    self.ub = process_bound(self.ub, 'ub')

def process_single_linear_constraint(constraint):
  # Convert lb and ub to vectors if they are scalars
  num_constraints = constraint.A.shape[0]
  try:
    len_lb = len(constraint.lb)
  except TypeError:
    constraint.lb = [constraint.lb]*num_constraints
    len_lb = num_constraints
  if len_lb != num_constraints and len_lb == 1:
    constraint.lb = [constraint.lb[0]]*num_constraints
  elif len_lb != num_constraints:
    raise ValueError('Length of lb must match number of rows in A')
  # Now ub
  try:
    len_ub = len(constraint.ub)
  except TypeError:
    constraint.ub = [constraint.ub]*num_constraints
    len_ub = num_constraints
  if len_ub != num_constraints and len_ub == 1:
    constraint.ub = [constraint.ub[0]]*num_constraints
  elif len_ub != num_constraints:
    raise ValueError('Length of ub must match number of rows in A')
  return constraint


def process_multiple_linear_constraints(constraints):
  # Need to combine A, and also upgrade lb, ub to vectors if they are scalars
  constraint = process_single_linear_constraint(constraints[0])
  full_A = constraint.A
  full_lb = constraint.lb
  full_ub = constraint.ub
  for constraint in constraints[1:]:
    constraint = process_single_linear_constraint(constraint)
    full_A = np.concatenate((full_A, constraint.A), axis=0)
    full_lb = np.concatenate((full_lb, constraint.lb), axis=0)
    full_ub = np.concatenate((full_ub, constraint.ub), axis=0)
  return LinearConstraint(full_A, full_lb, full_ub)
