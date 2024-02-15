import numpy as np
from prima import NonlinearConstraint, process_multiple_nl_constraints, process_single_nl_constraint
import pytest



@pytest.mark.parametrize("lb1", (-np.inf, [-np.inf], np.array([-np.inf])))
@pytest.mark.parametrize("lb2", (-np.inf, [-np.inf]*2, np.array([-np.inf]*2)))
@pytest.mark.parametrize("ub1", (0, [0], np.array([0])))
@pytest.mark.parametrize("ub2", (0, [0]*2, np.array([0]*2)))
def test_multiple_nl_constraints_various_data_types(lb1, ub1, lb2, ub2):
    nlc1 = NonlinearConstraint(lambda x: x, lb=lb1, ub=ub1)
    nlc2 = NonlinearConstraint(lambda x: [x, x], lb=lb2, ub=ub2)
    nlcs = [nlc1, nlc2]
    x0 = 0
    options = {}
    nlc = process_multiple_nl_constraints(x0, nlcs, options)
    assert all(nlc.lb == [-np.inf, -np.inf, -np.inf])
    assert all(nlc.ub == [0, 0, 0])
    assert all(nlc.fun(0) == [0, 0, 0])
    # The proccess_multiple_nl_constraints function will only evaluate the constraint function
    # if it cannot determine the number of constraints from the provided options
    # Only if all of the following fail will the constraint function be evaluated:
    num_constraints = 0
    try: num_constraints = len(lb1) + len(lb2)
    except TypeError: pass
    try: num_constraints = len(lb1) + len(ub2)
    except TypeError: pass
    try: num_constraints = len(ub1) + len(lb2)
    except TypeError: pass
    try: num_constraints = len(ub1) + len(ub2)
    except TypeError: pass
    if num_constraints > 0:
        assert 'nlconstr0' not in options.keys()
    else:
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
