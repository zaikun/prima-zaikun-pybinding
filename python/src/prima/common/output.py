def fmsg(solver, iprint, nf, f, x, cstrv=None, constr=None):
    if abs(iprint) < 3:
        return
    elif iprint > 0:
        output = "stdout"
    else:
        output = "file"
        filename = solver + "_output.txt"

    # Decide whether the problem is truly constrained.
    if constr is None:
        is_constrained = cstrv is not None
    else:
        is_constrained = bool(len(constr))

    # Decide the constraint violation

    # Will finish this later, not important for now


