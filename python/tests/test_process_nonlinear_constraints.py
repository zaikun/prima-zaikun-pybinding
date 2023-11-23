import numpy as np
from prima import NonlinearConstraint, process_multiple_nl_constraints, process_single_nl_constraint

# Need to support:
# - test multiple constraints provided with all of them having either lb or ub as list
# - test multiple constraints provided with all of them having either lb or ub as scalar
# - test multiple constraints provided with some of them having lb or ub as list and some as scalar


def test_multiple_nl_constraints_all_provide_lb_ub_as_list():
    nlc1 = NonlinearConstraint(lambda x: x, lb=[-np.inf], ub=[0])
    nlc2 = NonlinearConstraint(lambda x: [x, x], lb=[-np.inf]*2, ub=[0]*2)
    nlcs = [nlc1, nlc2]
    x0 = 0
    nlc = process_multiple_nl_constraints(x0, nlcs, None)  # We can provide options as None since this shouldn't trigger the code path that requires it
    assert all(nlc.lb == [-np.inf, -np.inf, -np.inf])
    assert all(nlc.ub == [0, 0, 0])
    assert all(nlc.fun(0) == [0, 0, 0])


def test_multiple_nl_constraints_some_provide_lb_ub_as_list():
    nlc1 = NonlinearConstraint(lambda x: x, lb=[-np.inf], ub=[0])
    nlc2 = NonlinearConstraint(lambda x: [x, x], lb=-np.inf, ub=0)
    nlcs = [nlc1, nlc2]
    options = {}
    x0 = 0
    nlc = process_multiple_nl_constraints(x0, nlcs, options) 
    assert all(nlc.lb == [-np.inf, -np.inf, -np.inf])
    assert all(nlc.ub == [0, 0, 0])
    assert all(nlc.fun(0) == [0, 0, 0])
    assert options['nlconstr0'] == [0, 0, 0]


def test_multiple_nl_constraints_none_provide_lb_ub_as_list():
    nlc1 = NonlinearConstraint(lambda x: x, lb=-np.inf, ub=0)
    nlc2 = NonlinearConstraint(lambda x: [x, x], lb=-np.inf, ub=0)
    nlcs = [nlc1, nlc2]
    options = {}
    x0 = 0
    nlc = process_multiple_nl_constraints(x0, nlcs, options) 
    assert all(nlc.lb == [-np.inf, -np.inf, -np.inf])
    assert all(nlc.ub == [0, 0, 0])
    assert all(nlc.fun(0) == [0, 0, 0])
    assert options['nlconstr0'] == [0, 0, 0]


def test_single_nl_constraint_provides_lb_as_list():
    num_constraints = 3
    nlc = NonlinearConstraint(lambda x: x, lb=[-np.inf]*num_constraints, ub=0)
    x0 = 0
    processed_nlc = process_single_nl_constraint(x0, nlc, None)
    assert len(processed_nlc.lb) == num_constraints
    assert len(processed_nlc.ub) == num_constraints


def test_single_nl_constraint_provides_lb_as_scalar_with_scalar_constr_function():
    nlc = NonlinearConstraint(lambda x: x, lb=-np.inf, ub=0)
    options = {}
    x0 = 0
    processed_nlc = process_single_nl_constraint(x0, nlc, options)
    assert processed_nlc.lb == [-np.inf]
    assert processed_nlc.ub == [0]
    assert options['nlconstr0'] == 0


def test_single_nl_constraint_provides_lb_as_scalar_with_vector_constr_function():
    nlc = NonlinearConstraint(lambda x: [x, x], lb=-np.inf, ub=0)
    options = {}
    x0 = 2.1
    processed_nlc = process_single_nl_constraint(x0, nlc, options)
    assert all(processed_nlc.lb == [-np.inf, -np.inf])
    assert all(processed_nlc.ub == [0, 0])
    assert options['nlconstr0'] == [2.1, 2.1]
