# On some platforms in CI we are not able to install scipy, and in that
# case we should skip this test. Note that pdfo depends on scipy.
import pytest
scipy = pytest.importorskip("scipy")

# Import the various minimization functions
from scipy.optimize import minimize as scipy_minimize
from prima import minimize as prima_minimize
from pdfo import pdfo as pdfo_minimize
# Import Linear Constraint class
from scipy.optimize import LinearConstraint as scipy_LC
from prima import LinearConstraint as prima_LC
# PDFO uses SciPy's LinearConstraint class
# Import Nonlinear Constraint class
from scipy.optimize import NonlinearConstraint as scipy_NLC
from prima import NonlinearConstraint as prima_NLC
# PDFO uses SciPy's NonlinearConstraint class

from objective import fun
import numpy as np
import pytest

# TODO: Test callback, options, and bounds.
# For PDFO, test the other algorithms.

@pytest.mark.parametrize('minimize_fn', [scipy_minimize, pdfo_minimize, prima_minimize])
@pytest.mark.parametrize('LC', [scipy_LC])
@pytest.mark.parametrize('NLC', [scipy_NLC, prima_NLC])
def test_cobyla(minimize_fn, LC, NLC):
    x0 = [0.0] * 2
    constraints = NLC(lambda x: x[0]**2, [-np.inf], [9])
    if (minimize_fn in (scipy_minimize, pdfo_minimize) and (NLC == prima_NLC)):
        # First, SciPy does not recognize other types as instances of its own NonlinearConstraint
        # class, nor does it look for the attributes "fun," "lb," and "ub" in the object, it can only
        # accept nonlinear constraints from other sources as a dictionary.

        # Second, SciPy does not respect lb and ub when they come via a dictionary.
        # It assumes that the function that the NLC provides is of the form 0 <= f(x) <= np.inf
        # when it's provided as a dictionary, and so it ignores any lb and ub provided.
        # So we need to convert the NLC into two NLCs, one for lb and another for ub, and we
        # need to provide them as a list of dictionaries.
        captured_constraints = constraints  # We need to copy the constraints so that we can reuse the name later
        # but still refer to the original variables in the lambda below.
        newfun_lb = lambda x: captured_constraints.fun(x) - captured_constraints.lb
        newfun_ub = lambda x: -captured_constraints.fun(x) + captured_constraints.ub
        constraints = [{'type': 'ineq', 'fun': newfun_lb}, {'type': 'ineq', 'fun': newfun_ub}]
    res = minimize_fn(fun, x0, method='COBYLA', constraints=constraints)
    assert np.isclose(res.x[0], 3.0, rtol=1e-3)
    assert np.isclose(res.x[1], 4.0, rtol=1e-3)
    assert np.isclose(res.fun, 4, rtol=1e-3)

