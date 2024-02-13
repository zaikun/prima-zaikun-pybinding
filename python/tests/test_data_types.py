from prima import minimize, NonlinearConstraint as NLC
from objective import fun
import numpy as np

def test_x0_as_list():
    x0 = [0.0] * 2
    res = minimize(fun, x0)
    assert fun.result_point_and_value_are_optimal(res)


def test_x0_as_array():
    x0 = np.array([0.0] * 2)
    res = minimize(fun, x0)
    assert fun.result_point_and_value_are_optimal(res)


def test_x0_as_scalar():
    x0 = 0.0
    # We need a custom function since the default objective function we're
    # using for tests expects something that can be unpacked into two variables.
    res = minimize(lambda x: (x-5)**2, x0)
    assert np.isclose(res.x, 5.0, rtol=1e-6)
    assert np.isclose(res.fun, 0.0, rtol=1e-6)


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