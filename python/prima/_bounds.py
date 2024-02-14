import numpy as np

class Bounds:
    def __init__(self, lb=-np.inf, ub=np.inf, **kwargs):
        if 'keep_feasible' in kwargs:
            raise ValueError("PRIMA does not support keep_feasible at this time")
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)
        try:
            res = np.broadcast_arrays(self.lb, self.ub)
            self.lb, self.ub = res
        except ValueError:
            raise ValueError("`lb` and `ub` must be broadcastable.")


def process_bounds(bounds, x0):
    '''
    bounds can either be an object with the properties lb and ub, or a list of tuples
    indicating a lower bound and an upper bound for each variable. If the list contains
    fewer entries than the length of x0, the remaining entries will generated as -/+ infinity.
    Some examples of valid lists of tuple, assuming len(x0) == 3:
    [(0, 1), (2, 3), (4, 5)] -> returns [0, 2, 4], [1, 3, 5]
    [(0, 1), (2, 3), None]   -> returns [0, 2, -inf], [1, 3, inf]
    [(0, 1), (None, 3)]      -> returns [0, -inf, -inf], [1, 3, inf]
    [(0, 1), (-np.inf, 3)]   -> returns [0, -inf, -inf], [1, 3, inf]
    '''
    lb = None
    ub = None
    if bounds is not None:
        # This if statement will handle an object of type scipy.optimize.Bounds or similar
        if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
            return bounds.lb, bounds.ub
        lb = []
        ub = []
        for bound in bounds:
            if bound is None:
                lb.append(-np.inf)
                ub.append(np.inf)
            else:
                if bound[0] is None:
                    lb.append(-np.inf)
                else:
                    lb.append(bound[0])
                if bound[1] is None:
                    ub.append(np.inf)
                else:
                    ub.append(bound[1])
        # If there were fewer bounds than variables, pad the rest with -/+ infinity
        for i in range(len(lb), len(x0)):
            lb.append(-np.inf)
            ub.append(np.inf)
        lb = np.array(lb, dtype=np.float64)
        ub = np.array(ub, dtype=np.float64)
    return lb, ub
