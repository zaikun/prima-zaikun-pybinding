import numpy as np
from prima.common.consts import FUNCMAX, CONSTRMAX

# This is a module evaluating the objective/constraint function with
# Nan/Inf handling.


def moderatex(x):
    np.nan_to_num(x, copy=False, nan=FUNCMAX)
    x = np.clip(x, -np.finfo(float).max, np.finfo(float).max)
    return x

def moderatef(f):
    f = FUNCMAX if np.isnan(f) else f
    return min(FUNCMAX, f)

def moderatec(c):
    np.nan_to_num(c, copy=False, nan=-CONSTRMAX)
    c = np.clip(c, -CONSTRMAX, CONSTRMAX)
    return c


def evaluate(calfun, x):
    """
    This function evaluates calfun at x, setting to f the objective function value.
    Nan/Inf are handled by a moderated extreme barrier
    """

    assert not any(np.isnan(x)), "x contains NaN"

    f, constr = calfun(moderatex(x))
    f = moderatef(f)
    constr = moderatec(constr)
    cstrv = max([*-constr, 0])

    return f, constr, cstrv