from prima import minimize, NonlinearConstraint as NLC, LinearConstraint as LC
import numpy as np
from objective import fun
import pytest

# test with single nonlinear constraint
# test with multiple nonlinear constraints
# test with nonlinear constraint and equality linear constraint
# test with nonlinear constraint and inequality linear constraint
# test with nonlinear constraint and equality and inequality linear constraints
# test with nonlinear constraint and equality and inequality linear constraints and bounds
# constraint function returning numpy array
# constraint function returning list
# constraint function returning scalar
# test with nonlinear constraint and scalar lb/ub, validate that warning appears
# test providing m_nlcon and not providing and see if the number of function evaluations differs
# test calling COBYLA without providing nonlinear constraints


def test_constraint_function_returns_numpy_array():
    nlc = NLC(lambda x: np.array([x[0], x[1]]), lb=[-np.inf]*2, ub=[10]*2)
    x0 = [0, 0]
    res = minimize(fun, x0, method='COBYLA', constraints=nlc)
    assert fun.result_point_and_value_are_optimal(res)


def test_constraint_function_returns_list():
    nlc = NLC(lambda x: [x[0], x[1]], lb=[-np.inf]*2, ub=[10]*2)
    x0 = [0, 0]
    res = minimize(fun, x0, method='COBYLA', constraints=nlc)
    assert fun.result_point_and_value_are_optimal(res)


def test_constraint_function_returns_scalar():
    nlc = NLC(lambda x: float(np.linalg.norm(x) - np.linalg.norm(fun.optimal_x)), lb=[-np.inf], ub=[0])
    x0 = [0, 0]
    res = minimize(fun, x0, method='COBYLA', constraints=nlc)
    assert fun.result_point_and_value_are_optimal(res)


def test_single_nonlinear_constraint():
    nlc = NLC(lambda x: np.array([x[0], x[1]]), lb=[-np.inf]*2, ub=[10]*2)
    x0 = [0, 0]
    res = minimize(fun, x0, method='COBYLA', constraints=nlc)
    assert fun.result_point_and_value_are_optimal(res)


def test_nonlinear_constraints_not_provided():
    with pytest.raises(ValueError) as e_info:
        x0 = [0, 0]
        minimize(fun, x0, method='COBYLA')
        assert e_info.msg == "Nonlinear constraints must be provided for COBYLA"


def test_nonlinear_constraint_and_equality_constraint_and_inequality_constraint_and_bounds():
    # # All should be active, which means we will need a new objective function since the standard one we
    # # are using for these tests only takes 2 variables.
    # obj_fun = lambda x: (x[0] - 5)**2 + (x[1] - 4)**2 + (x[2] - 3)**2 + (x[3] - 2)**2
    # nlc = NLC(lambda x: np.array([x[0], x[1]]), lb=[-np.inf]*2, ub=[10]*2)
    pass
