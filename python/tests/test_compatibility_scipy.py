# On some platforms in CI we are not able to install scipy, and in that
# case we should skip this test. Note that pdfo depends on scipy.
import pytest
scipy = pytest.importorskip("scipy")

from test_combining_constraints import test_providing_bounds_and_linear_and_nonlinear_constraints

def test_prima(capfd):
    from prima import minimize, NonlinearConstraint as NLC, LinearConstraint as LC, Bounds
    test_providing_bounds_and_linear_and_nonlinear_constraints(capfd, minimize, NLC, LC, Bounds)


def test_scipy():
    from scipy.optimize import minimize, NonlinearConstraint as NLC, LinearConstraint as LC, Bounds
    test_providing_bounds_and_linear_and_nonlinear_constraints(None, minimize, NLC, LC, Bounds, "COBYLA")
