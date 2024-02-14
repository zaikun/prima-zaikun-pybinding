# On some platforms in CI we are not able to install scipy, and in that
# case we should skip this test. Note that pdfo depends on scipy.
import pytest
scipy = pytest.importorskip("scipy")

from test_combining_constraints import test_providing_bounds_and_linear_and_nonlinear_constraints

def test_prima(capfd):
    from prima import minimize, NonlinearConstraint as NLC, LinearConstraint as LC, Bounds
    test_providing_bounds_and_linear_and_nonlinear_constraints(capfd, minimize, NLC, LC, Bounds)


# Despite the fact that we are using the pdfo function, we still get this warning because the pdfo
# function itself calls the cobyla function. For now we suppress this warning.
@pytest.mark.filterwarnings("ignore:The `cobyla` function is deprecated. Use the `pdfo` function")
def test_pdfo():
    from pdfo import pdfo
    from scipy.optimize import NonlinearConstraint as NLC, LinearConstraint as LC, Bounds
    test_providing_bounds_and_linear_and_nonlinear_constraints(None, pdfo, NLC, LC, Bounds, package='pdfo')
