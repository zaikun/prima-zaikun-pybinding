# On some platforms in CI we are not able to install scipy, and in that
# case we should skip this test. Note that pdfo depends on scipy.
import pytest
scipy = pytest.importorskip("scipy")

import numpy as np
from test_combining_constraints import test_providing_bounds_and_linear_and_nonlinear_constraints

def agnostic_function(minimize, NLC, LC, Bounds):
    nlc = NLC(lambda x: x[0]**2, lb=[36], ub=[100])
    lc = LC(np.array([1,1]), lb=10, ub=15)
    x0 = [0, 0]
    # We have to explicitly specify COBYLA because otherwise SciPy might choose
    # a different optimization function. PRIMA in this case would have chosen
    # COBYLA anyway.
    res = minimize(fun, x0, constraints=[nlc, lc], method="COBYLA")
    assert np.isclose(res.x[0], 6, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.x[1], 4, atol=1e-6, rtol=1e-6)


def test_prima(capfd):
    from prima import minimize, NonlinearConstraint as NLC, LinearConstraint as LC, Bounds
    test_providing_bounds_and_linear_and_nonlinear_constraints(capfd, minimize, NLC, LC, Bounds)


def test_scipy(capfd):
    from scipy.optimize import minimize, NonlinearConstraint as NLC, LinearConstraint as LC, Bounds
    test_providing_bounds_and_linear_and_nonlinear_constraints(None, minimize, NLC, LC, Bounds)
