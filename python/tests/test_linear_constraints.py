from prima import minimize, LinearConstraint as LC
from objective import fun
import numpy as np
import pytest

# To test:
# On a method which does not accept constraints:
# - Test without constraints
# - Test with constraints and confirm warning appears
# On a method which accepts linear constraints (LINCOA):
# - Test without constraints
# - Test with constraints on all variables
# - Test with constraints on some variables
# - Test with constraints that do not affect the solution
# - Test with constraints that do affect the solution
# - Test with lb constraints
# - Test with ub constraints
# - Test with ub and lb constraints that do not imply equality
# - Test with lb and ub constraints that do imply equality
# - Test with infeasible starting point
# - Test with A as scalar, A as list
# - Test with lb/ub as scalar, lb/ub as list


# On a method which does not accept constraints:

def test_method_without_constraints_runs_correctly():
    x0 = [0.0] * 2
    res = minimize(fun, x0, method='NEWUOA')
    assert fun.result_point_and_value_are_optimal(res)


def test_method_without_constraints_warns_correctly():
    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # x1 + x2 < 1
    # x1 - x2 < 1
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    b_ineq = np.array([1, 1])

    myLC = LC(A_ineq, ub=b_ineq)
    with pytest.raises(ValueError) as e_info:
        minimize(fun, x0, method='NEWUOA', constraints=myLC)
        assert e_info.msg == "Linear constraints were provided for an algorithm that cannot handle them"


# On a method which accepts linear constraints:
        

def test_method_with_linear_constraints_runs_without_constraints():
    x0 = [0.0] * 2
    res = minimize(fun, x0, method='LINCOA')
    assert fun.result_point_and_value_are_optimal(res)


def test_method_with_linear_constraints_all_variables_constrained():

    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # x1 + x2 < 1
    # x1 - x2 < 1
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    b_ineq = np.array([1, 1])

    myLC = LC(A_ineq, ub=b_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 1.0, rtol=1e-6)
    assert np.isclose(res.x[1], 0.0, rtol=1e-6)


def test_method_with_linear_constraints_some_variables_constrained():

    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # x1 + x2 < 2
    A_ineq = np.array([[1, 1]])
    b_ineq = np.array([2])

    myLC = LC(A_ineq, ub=b_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 1.5, rtol=1e-6)
    assert np.isclose(res.x[1], 0.5, rtol=1e-6)


def test_method_with_linear_constraints_that_dont_affect_solution():
    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # x1 + x2 < 10
    # x1 - x2 < 5
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    b_ineq = np.array([10, 5])

    myLC = LC(A_ineq, ub=b_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert fun.result_point_and_value_are_optimal(res)


def test_method_with_linear_constraints_that_do_affect_solution():
    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # x1 + x2 < 1
    # x1 - x2 < 1
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    b_ineq = np.array([1, 1])

    myLC = LC(A_ineq, ub=b_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 1.0, rtol=1e-6)
    assert np.isclose(res.x[1], 0.0, rtol=1e-6)


def test_method_with_linear_constraints_lb_constraints():
    x0 = [8, 0]
    # The constraints represent the following inequality
    # 3 < x1 + x2
    # 5 < x1 - x2
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    lb_ineq = np.array([3, 5])

    myLC = LC(A_ineq, lb=lb_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 7, rtol=1e-6)
    assert np.isclose(res.x[1], 2, rtol=1e-6)
    assert all(lb_ineq <= A_ineq @ res.x)
    assert res.maxcv == 0


def test_method_with_linear_constraints_ub_constraints():
    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # x1 + x2 < 1
    # x1 - x2 < 1
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    b_ineq = np.array([1, 1])

    myLC = LC(A_ineq, ub=b_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 1, rtol=1e-6)
    assert np.isclose(res.x[1], 0, rtol=1e-6)


def test_method_with_linear_constraints_ub_and_lb_constraints_that_do_not_imply_equality():
    x0 = [6.5, 1.5]
    # The constraints represent the following inequality
    # 8 < x1 + x2 < 10
    # 5 < x1 - x2 < 7
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    lb_ineq = np.array([8, 5])
    ub_ineq = np.array([10, 7])

    myLC = LC(A_ineq, lb=lb_ineq, ub=ub_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 7.0, rtol=1e-6)
    assert np.isclose(res.x[1], 2.0, rtol=1e-6)
    assert all(lb_ineq <= A_ineq @ res.x)
    assert all(A_ineq @ res.x <= ub_ineq)
    assert res.maxcv == 0


def test_method_with_linear_constraints_lb_and_ub_constraints_that_do_imply_equality():
    x0 = [6.5, 1.5]
    # The constraints represent the following inequality
    # 8 < x1 + x2 < 8
    # 5 < x1 - x2 < 5
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    lb_ineq = np.array([8, 5])
    ub_ineq = np.array([8, 5])

    myLC = LC(A_ineq, lb=lb_ineq, ub=ub_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 6.5, rtol=1e-6)
    assert np.isclose(res.x[1], 1.5, rtol=1e-6)
    assert all(lb_ineq <= A_ineq @ res.x)
    assert all(A_ineq @ res.x <= ub_ineq)
    assert res.maxcv == 0

def test_method_with_linear_constraints_and_bad_starting_point(capfd):
    # This test is meant to highlight the fact that if we start with an infeasible
    # starting point, then we might not be able to find a feasible solution, even for
    # a relatively simple low dimensional problem
    x0 = [0.0] * 2
    # The constraints represent the following inequality
    # 8 < x1 + x2
    # 5 < x1 - x2
    A_ineq = np.array([[1,  1],
                       [1, -1]])
    lb_ineq = np.array([8, 5])

    myLC = LC(A_ineq, lb=lb_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert ("Warning: LINCOA: The starting point is infeasible. LINCOA modified "
           "the right-hand sides of the constraints to make it feasible.") in capfd.readouterr().err
    assert np.isclose(res.x[0], 5.7071, rtol=1e-2)
    assert np.isclose(res.x[1], 3.2929, rtol=1e-2)
    assert res.maxcv > 0


def test_method_with_linear_constraints_A_is_scalar_ub_is_scalar():
    # For a scalar linear constraint, we need a scalar obj function
    obj_fun = lambda x: (x - 5)**2
    x0 = 0
    A_ineq = 1
    ub_ineq = np.array([4])  # This constraint implies x <= 4

    myLC = LC(A_ineq, ub=ub_ineq)
    res = minimize(obj_fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x, 4, rtol=1e-2)


def test_method_with_linear_constraints_A_is_list_ub_is_list():
    x0 = [0.0]*2
    A_ineq = [1, 1]
    ub_ineq = [8]  # This constraint implies x1 + x2 <= 8

    myLC = LC(A_ineq, ub=ub_ineq)
    res = minimize(fun, x0, method='LINCOA', constraints=myLC)
    assert np.isclose(res.x[0], 4.5, rtol=1e-2)
    assert np.isclose(res.x[1], 3.5, rtol=1e-2)
