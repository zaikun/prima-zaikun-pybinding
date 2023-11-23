from prima import minimize
from objective import fun
import numpy as np
import pytest

# To test:
# On a method which does not accept bounds:
# - Test without bounds
# - Test with bounds and confirm warning appears
# On a method which accepts bounds (BOBYQA or COBYLA):
# - Test without bounds
# - Test with bounds on all variables
# - Test with bounds on some variables
# - Test with bounds that do not affect the solution
# - Test with bounds that do affect the solution
# - Test None as the bounds for a variable
# - Test None as one of the bounds of a variable
# - Test inf as one of the bounds of a variable
# - Test None on all bounds


# On a method which does not accept bounds:

def test_method_without_bounds_runs_correctly():
    x0 = [0.0] * 2
    res = minimize(fun, x0, method='NEWUOA')
    assert fun.result_point_and_value_are_optimal(res)


def test_method_without_bounds_throws_correctly():
    x0 = [0.0] * 2
    bounds = [(0, 2.5), (-6, 6)]
    with pytest.raises(ValueError) as e_info:
        minimize(fun, x0, method='NEWUOA', bounds=bounds)
        assert e_info.msg == "Bounds were provided for an algorithm that cannot handle them"


# On a method which accepts bounds (BOBYQA or COBYLA):


def test_method_with_bounds_runs_without_bounds():
    x0 = [0.0] * 2
    with pytest.raises(ValueError) as e_info:
        minimize(fun, x0, method='BOBYQA')
        assert e_info.msg == "No bounds were provided for an algorithm that requires them"


def test_method_with_bounds_all_variables_bounded():
    x0 = [0.0] * 2
    bounds = [(0, 2.5), (-6, 6)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert np.isclose(res.x[0], 2.5, rtol=1e-6)
    assert np.isclose(res.x[1], fun.optimal_x[1], rtol=1e-6)


def test_method_with_bounds_some_variables_bounded():
    x0 = [0.0] * 2
    bounds = [(0, 2.5)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert np.isclose(res.x[0], 2.5, rtol=1e-6)
    assert np.isclose(res.x[1], fun.optimal_x[1], rtol=1e-6)


def test_method_with_bounds_that_dont_affect_solution():
    x0 = [0.0] * 2
    bounds = [(-6, 6), (-6, 6)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert fun.result_point_and_value_are_optimal(res)


def test_method_with_bounds_that_do_affect_solution():
    x0 = [0.0] * 2
    bounds = [(0, 2.4), (-6, 6)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert np.isclose(res.x[0], 2.4, rtol=1e-6)
    assert np.isclose(res.x[1], fun.optimal_x[1], rtol=1e-6)


def test_method_with_bounds_none_as_bounds():
    x0 = [0.0] * 2
    bounds = [None, (-6, 6)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert fun.result_point_and_value_are_optimal(res)


def test_method_with_bounds_none_as_one_bound():
    x0 = [0.0] * 2
    bounds = [(0, 2.5), (None, 6)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert np.isclose(res.x[0], 2.5, rtol=1e-6)
    assert np.isclose(res.x[1], fun.optimal_x[1], rtol=1e-6)


def test_method_with_bounds_none_as_all_bounds_v1():
    x0 = [0.0] * 2
    bounds = [None, None]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert fun.result_point_and_value_are_optimal(res)


def test_method_with_bounds_none_as_all_bounds_v2():
    x0 = [0.0] * 2
    bounds = [(None, None), (None, None)]
    res = minimize(fun, x0, method='BOBYQA', bounds=bounds)
    assert fun.result_point_and_value_are_optimal(res)
