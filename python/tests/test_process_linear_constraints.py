import numpy as np
from prima import LinearConstraint, process_multiple_linear_constraints, process_single_linear_constraint

# test upgrading scalar lb/ub to vectors
# test leaving lb/ub alone if they are already vectors

def test_single_linear_constraint_lb_ub_scalars():
    constraint = LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=0, ub=1)
    processed_constraint = process_single_linear_constraint(constraint)
    assert all(processed_constraint.lb == [0, 0])
    assert all(processed_constraint.ub == [1, 1])


def test_single_linear_constraint_lb_ub_vectors():
    constraint = LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=[0, 0], ub=[1, 1])
    processed_constraint = process_single_linear_constraint(constraint)
    assert all(processed_constraint.lb == [0, 0])
    assert all(processed_constraint.ub == [1, 1])


def test_multiple_linear_constraints_lb_ub_scalars():
    constraints = [LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=0, ub=1),
                   LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=0, ub=1)]
    processed_constraint = process_multiple_linear_constraints(constraints)
    assert all(processed_constraint.lb == [0, 0, 0, 0])
    assert all(processed_constraint.ub == [1, 1, 1, 1])


def test_multiple_linear_constraints_lb_ub_vectors():
    constraints = [LinearConstraint(A=np.array([[1, 2], [3, 4]]), lb=[5, 6], ub=[7, 8]),
                   LinearConstraint(A=np.array([[9, 10], [11, 12]]), lb=[13, 14], ub=[15, 16])]
    processed_constraint = process_multiple_linear_constraints(constraints)
    assert (processed_constraint.A == np.array([[1, 2], [3, 4], [9, 10], [11, 12]])).all()
    assert all(processed_constraint.lb == [5, 6, 13, 14])
    assert all(processed_constraint.ub == [7, 8, 15, 16])
