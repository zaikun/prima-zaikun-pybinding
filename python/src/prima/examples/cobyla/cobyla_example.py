import numpy as np
from prima.cobyla.cobyla import cobyla


def test_chebyquad():
    f, _ = calcfc_chebyquad([1, 2])
    assert np.isclose(f, 91+1/9, atol=1e-6)


def calcfc_chebyquad(x):
    x = np.array(x)
    n = len(x)  # or shape?
    y = np.zeros((n + 1, n + 1))
    y[:n, 0] = 1
    y[:n, 1] = 2*x - 1
    for i in range(1, n):
        y[:n, i+1] = 2*y[:n, 1] * y[:n, i] - y[:n, i - 1]

    f = 0
    for i in range(n+1):
        tmp = sum(y[0:n, i]) / n
        if i % 2 == 0:
            tmp += 1/((i)**2 - 1)
        f += tmp**2
    constr = np.zeros(0)
    return f, constr

def calcfc_hexagon(x):
    # Test problem 10 in Powell's original algorithm
    
    assert len(x) == 9
    
    f = -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])
    constr = np.zeros(14)
    constr[0] = 1 - x[2]**2 - x[3]**2
    constr[1] = 1 - x[8]**2
    constr[2] = 1 - x[4]**2 - x[5]**2
    constr[3] = 1 - x[1-1]**2 - (x[1] - x[8])**2
    constr[4] = 1 - (x[1-1] - x[4])**2 - (x[1] - x[5])**2
    constr[5] = 1 - (x[1-1] - x[6])**2 - (x[1] - x[7])**2
    constr[6] = 1 - (x[2] - x[4])**2 - (x[3] - x[5])**2
    constr[7] = 1 - (x[2] - x[6])**2 - (x[3] - x[7])**2
    constr[8] = 1 - x[6]**2 - (x[7] - x[8])**2
    constr[9] = x[1-1] * x[3] - x[1] * x[2]
    constr[10] = x[2] * x[8]
    constr[11] = -x[4] * x[8]
    constr[12] = x[4] * x[7] - x[5] * x[6]
    constr[13] = x[8]

    return f, constr


if __name__ == "__main__":
    n_chebyquad = 6
    x_chebyquad = np.array([i/(n_chebyquad+1) for i in range(1, n_chebyquad+1)])
    m = 0
    f = 0
    cstrv = np.inf
    x, f = cobyla(calcfc_chebyquad, m, x_chebyquad, f, cstrv)
    print(x, f)
    # evaluate
    f, constr = calcfc_chebyquad(x_chebyquad)
    x_hexagon = np.zeros(9) + 2
    m_hexagon = 14
    f = 0
    cstrv = 0
    initial_f, initial_constr = calcfc_hexagon(x_hexagon)
    print(initial_f, initial_constr)
    x, f = cobyla(calcfc_hexagon, m_hexagon, x_hexagon, f, cstrv)
    print(x, f)