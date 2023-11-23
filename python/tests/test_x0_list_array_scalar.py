from prima import minimize
from objective import fun
import numpy as np

# To test:
# - providing x0 as a Python list
# - providing x0 as a numpy array
# - providing x0 as a scalar


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
