import numpy as np

from warnings import warn  # TODO: Need to determine the text of the warning and actually warn about it


class NonlinearConstraint(object):
    def __init__(self, fun, lb=-np.inf, ub=0):
        self.fun = fun
        self.lb = lb
        self.ub = ub


def get_num_constraints_from_lb_or_ub(nlc):
    # Try to get the number of constraints from lb or ub of nlc
    # If they're scalars or otherwise don't implement len, return None
    try:
        num_constraints = len(nlc.lb)
    except:
        try:
            num_constraints = len(nlc.ub)
        except:
            return None
    return num_constraints


# If the user has provided m_nlcon in the options, do not call this function
def process_single_nl_constraint(x0, nlc, options):
    '''
    In this function we obtain the number of constraints in the nonlinear constraint.
    We first attempt to determine the number from the length of lb, ub. but if it is not
    possible, we call the constraint function in order to determine the number of constraints.

    Since we presume the call to the constraint function is expensive, we want to save the results so that
    COBYLA might make use of them in its initial iteration, and so we save them in the variable nlconstr0.
    However COBYLA needs either both nlconstr0 and f0, or neither, and so we need to call the objective function
    as well so that we can save the result in f0.

    In order to avoid having this function return either 1 or 3 values, we take the options as an input and overwrite
    the nlconstr0 and f0 values in the options object.
    '''
    num_constraints = get_num_constraints_from_lb_or_ub(nlc)
    if num_constraints is None:
        num_constraints, nlconstr0 = get_num_constraints_from_constraint_function(x0, nlc)
        options['nlconstr0'] = nlconstr0
    # Upgrade lb and ub to vectors if they are scalars
    lb = []
    ub = []
    try:
        lb.extend(nlc.lb)
    except TypeError:
        lb.extend([nlc.lb]*num_constraints)

    try:
        ub.extend(nlc.ub)
    except TypeError:
        ub.extend([nlc.ub]*num_constraints)
    # Turn them into numpy arrays to enable arithmetic operations
    lb = np.array(lb)
    ub = np.array(ub)
    
    return NonlinearConstraint(nlc.fun, lb=lb, ub=ub)


def process_multiple_nl_constraints(x0, nlcs, options):
    # First, get the total number of constraints
    num_constraints = []
    evaluate_all = False
    for nlc in nlcs:
        num_constraints_i = get_num_constraints_from_lb_or_ub(nlc)
        if num_constraints_i is None:
            # This means we need to evaluate the constraint function.
            # Since we assume each constraint function is expensive, we want to save the results so that
            # COBYLA might make use of them in its initial iteration. However, it doesn't make sense to evaluate
            # only one constraint function, so we need to evaluate them all.
            # Further, since COBYLA cannot make use of the constraint function results without also knowing the
            # objective function result, we need to evaluate the objective function as well.
            evaluate_all = True
            num_constraints = []  # Reset this since we need to go through the list from the start
            break
        else:
            num_constraints.append(num_constraints_i)

    if evaluate_all:
        nlconstr0 = []
        for nlc in nlcs:
            num_constraints_i, nlconstr0_i = get_num_constraints_from_constraint_function(x0, nlc)
            num_constraints.append(num_constraints_i)
            try:
                nlconstr0.extend(nlconstr0_i)
            except TypeError:
                nlconstr0.append(nlconstr0_i)
        assert len(nlconstr0) == sum(num_constraints), "There is a mismatch in the detected number of constraints and the length of the constraints returned"
        options['nlconstr0'] = nlconstr0

    
    # Upgrade any potentially scalar lb/ub to vectors
    lb = []
    ub = []
    for nlc, num_constraints_i in zip(nlcs, num_constraints):
        try:
            lb.extend(nlc.lb)
        except TypeError:
            lb.extend([nlc.lb]*num_constraints_i)

        try:
            ub.extend(nlc.ub)
        except TypeError:
            ub.extend([nlc.ub]*num_constraints_i)
    # Turn them into numpy arrays to enable arithmetic operations
    lb = np.array(lb)
    ub = np.array(ub)

    # Combine all the constraint functions into a single function
    def nlc_fun(x):
        constraints = []
        for nlc in nlcs:
            constraints_i = nlc.fun(x)
            try:
                constraints.extend(constraints_i)
            except TypeError:
                constraints.append(constraints_i)
        return np.array(constraints)
    
    return NonlinearConstraint(nlc_fun, lb=lb, ub=ub)

    
def get_num_constraints_from_constraint_function(x0, nlc):
    nlconstr0 = nlc.fun(x0)
    try:
        num_constraints = len(nlconstr0)
    except TypeError:
        num_constraints = 1
    return num_constraints, nlconstr0
