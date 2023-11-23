from prima import minimize, NonlinearConstraint as NLC
from objective import fun
import numpy as np

def test_callback():
    x0 = [0.0] * 2
    nlc = NLC(lambda x: np.array([x[0], x[1]]), lb=[-np.inf]*2, ub=[10]*2)
    callback_called = False
    def callback(x, f, nf, tr, cstrv, nlconstr):
        nonlocal callback_called
        callback_called = True
        print(f"best points so far: x={x} f={f} cstrv={cstrv} nlconstr={nlconstr} nf={nf} tr={tr}")
    res = minimize(fun, x0, method="COBYLA", constraints=nlc, callback=callback)
    assert callback_called
    assert fun.result_point_and_value_are_optimal(res)


def test_callback_early_termination():
    x0 = [0.0] * 2
    def callback(x, *args):
        print(x)
        if x[0] > 1:
            return True
    res = minimize(fun, x0, callback=callback)
    # Assert that the results are not optimal since we terminated early
    assert not fun.result_point_and_value_are_optimal(res)
