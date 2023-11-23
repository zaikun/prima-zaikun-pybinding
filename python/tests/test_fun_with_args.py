from prima import minimize
from objective import fun
import numpy as np
import pytest

# To test:
# - providing a normal function
# - providing a function which accepts arguments


def fun_with_star_args(x, *args):
    return fun(x) + args[0][0]


def fun_with_regular_args(x, args):
    return fun(x) + args[0]


def test_normal_function():
    x0 = [0.0] * 2
    res = minimize(fun, x0, method='NEWUOA')
    assert fun.result_point_and_value_are_optimal(res)


def test_function_with_star_args():
    x0 = np.array([0.0] * 2)
    myargs = (5,)
    res = minimize(fun_with_star_args, x0, args=myargs, method='NEWUOA')
    assert fun.result_point_is_optimal(res)
    assert np.isclose(res.fun, fun.optimal_f + myargs[0], rtol=1e-6)


def test_function_with_regular_args():
    x0 = np.array([0.0] * 2)
    myargs = (6,)
    res = minimize(fun_with_regular_args, x0, args=myargs, method='NEWUOA')
    assert fun.result_point_is_optimal(res)
    assert np.isclose(res.fun, fun.optimal_f + myargs[0], rtol=1e-6)
