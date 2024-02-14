from prima import minimize as prima_minimize, NonlinearConstraint as prima_NLC, LinearConstraint as prima_LC, Bounds as prima_Bounds
import numpy as np
from objective import fun


def test_providing_linear_and_nonlinear_constraints(capfd):
    nlc = prima_NLC(lambda x: x[0]**2, lb=[25], ub=[100])
    lc = prima_LC(np.array([1,1]), lb=10, ub=15)
    x0 = [0, 0]
    res = prima_minimize(fun, x0, constraints=[nlc, lc])
    assert np.isclose(res.x[0], 5.5, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.x[1], 4.5, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.fun, 0.5, atol=1e-6, rtol=1e-6)
    outerr = capfd.readouterr()
    assert outerr.out == "Nonlinear constraints detected, applying COBYLA\n"
    assert outerr.err == ''


def test_providing_bounds_and_linear_constraints(capfd):
    lc = prima_LC(np.array([1,1]), lb=10, ub=15)
    bounds = prima_Bounds(1, 1)
    x0 = [1, 14]  # Start with a feasible point
    res = prima_minimize(fun, x0, constraints=lc, bounds=bounds)
    assert np.isclose(res.x[0], 1, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.x[1], 9, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.fun, 41, atol=1e-6, rtol=1e-6)
    outerr = capfd.readouterr()
    assert outerr.out == "Linear constraints detected without nonlinear constraints, applying LINCOA\n"
    assert outerr.err == ''


def test_providing_bounds_and_nonlinear_constraints(capfd):
    nlc = prima_NLC(lambda x: x[0]**2, lb=[25], ub=[100])
    bounds = prima_Bounds([None, 1], [None, 1])
    x0 = [0, 0]
    res = prima_minimize(fun, x0, constraints=nlc, bounds=bounds)
    assert np.isclose(res.x[0], 5, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.x[1], 1, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.fun, 9, atol=1e-6, rtol=1e-6)
    outerr = capfd.readouterr()
    assert outerr.out == "Nonlinear constraints detected, applying COBYLA\n"
    assert outerr.err == ''


# This test is re-used for the compatibility tests, hence the extra arguments and their
# default values
def test_providing_bounds_and_linear_and_nonlinear_constraints(capfd, minimize=prima_minimize, NLC=prima_NLC, LC=prima_LC, Bounds=prima_Bounds, method=None):
    # This test needs a 3 variable objective function so that we can check that the
    # bounds and constraints are all active
    def newfun(x):
        return fun(x[0:2]) + (x[2] - 3)**2
    nlc = NLC(lambda x: x[0]**2, lb=[25], ub=[100])
    bounds = Bounds([-np.inf, 1, -np.inf], [np.inf, 1, np.inf])
    lc = LC(np.array([1,1,1]), lb=10, ub=15)
    x0 = [0, 0, 0]
    # macOS seems to stop just short of the optimal solution, so we help it along by
    # taking a larger initial trust region readius and requiring a smaller final radius
    # before stopping
    options = {'rhobeg': 0.1, 'rhoend': 1e-8}
    if minimize != prima_minimize:  # this implies SciPy
         # 'tol' is equivalent to 'rhoend' in the SciPy implementation of COBYLA
        options['tol'] = options['rhoend']
        del options['rhoend']
    res = minimize(newfun, x0, method=method, constraints=[nlc, lc], bounds=bounds, options=options)
    
    assert np.isclose(res.x[0], 5.5, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.x[1], 1, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.x[2], 3.5, atol=1e-6, rtol=1e-6)
    assert np.isclose(res.fun, 9.5, atol=1e-6, rtol=1e-6)
    if minimize == prima_minimize:
        outerr = capfd.readouterr()
        assert outerr.out == "Nonlinear constraints detected, applying COBYLA\n"
        assert outerr.err == ''
