from prima.common.evaluate import evaluate
from prima.common.consts import EPS, RHOBEG_DEFAULT, RHOEND_DEFAULT, \
    CTOL_DEFAULT, CWEIGHT_DEFAULT, FTARGET_DEFAULT, IPRINT_DEFAULT, \
    MAXFUN_DIM_DEFAULT
from prima.cobyla.cobylb import cobylb
import numpy as np


debugging = True


def cobyla(calcfc, m, x, f, cstrv=0, constr=0, f0=None, constr0=None, nf=None, rhobeg=None,
           rhoend=None, ftarget=FTARGET_DEFAULT, ctol=CTOL_DEFAULT, cweight=CWEIGHT_DEFAULT,
           maxfun=None, iprint=IPRINT_DEFAULT, eta1=None, eta2=None, gamma1=0.5, gamma2=2,
           xhist=False, fhist=False, chist=False, conhist=False, maxhist=0, maxfilt=2000):
    """
    Among all the arguments, only CALCFC, M, X, and F are obligatory. The others are OPTIONAL and you
    can neglect them unless you are familiar with the algorithm. Any unspecified optional input will
    take the default value detailed below. For instance, we may invoke the solver as follows.

    ! First define CALCFC, M, and X, and then do the following.
    call cobyla(calcfc, m, x, f)

    or

    ! First define CALCFC, M, and X, and then do the following.
    call cobyla(calcfc, m, x, f, rhobeg = 0.5D0, rhoend = 1.0D-3, maxfun = 100)

    ! IMPORTANT NOTICE: The user must set M correctly to the number of constraints, namely the size of
    ! CONSTR introduced below. Set M to 0 if there is no constraint.

    See examples/cobyla_exmp.py for a concrete example.

    A detailed introduction to the arguments is as follows.
    N.B.: RP and IK are defined in the module CONSTS_MOD. See consts.F90 under the directory named
    "common". By default, RP = kind(0.0D0) and IK = kind(0), with REAL(RP) being the double-precision
    real, and INTEGER(IK) being the default integer. For ADVANCED USERS, RP and IK can be defined by
    setting PRIMA_REAL_PRECISION and PRIMA_INTEGER_KIND in common/ppf.h. Use the default if unsure.

    CALCFC
    Input, subroutine.
    CALCFC(X, F, CONSTR) should evaluate the objective function and constraints at the given
    REAL(RP) vector X; it should set the objective function value to the REAL(RP) scalar F and the
    constraint value to the REAL(RP) vector CONSTR. It must be provided by the user, and its
    definition must conform to the following interface:
    !-------------------------------------------------------------------------!
        subroutine calcfc(x, f, constr)
        real(RP), intent(in) :: x(:)
        real(RP), intent(out) :: f
        real(RP), intent(out) :: constr(:)
        end subroutine calcfc
    !-------------------------------------------------------------------------!
    Besides, the subroutine should NOT access CONSTR beyond CONSTR(1:M), where M is the second
    compulsory argument (see below), signifying the number of constraints.

    M
    Input, INTEGER(IK) scalar.
    M must be set to the number of constraints, namely the size (length) of CONSTR(X).
    N.B.:
    1. M must be specified correctly, or the program will crash!!!
    2. Why don't we define M as optional and default it to 0 when it is absent? This is because
    we need to allocate memory for CONSTR_LOC according to M. To ensure that the size of CONSTR_LOC
    is correct, we require the user to specify M explicitly.

    X
    Input and output, REAL(RP) vector.
    As an input, X should be an N-dimensional vector that contains the starting point, N being the
    dimension of the problem. As an output, X will be set to an approximate minimizer.

    F
    Output, REAL(RP) scalar.
    F will be set to the objective function value of X at exit.

    CSTRV
    Output, REAL(RP) scalar.
    CSTRV will be set to the constraint violation of X at exit, i.e., MAXVAL([-CONSTR(X), 0]).

    CONSTR
    Output, ALLOCATABLE REAL(RP) vector.
    CONSTR will be set to the constraint value of X at exit.

    F0
    Input, REAL(RP) scalar.
    F0, if present, should be set to the objective function value of the starting X.

    CONSTR0
    Input, REAL(RP) vector.
    CONSTR0, if present, should be set to the constraint value of the starting X; in addition,
    SIZE(CONSTR0) must be M, or the solver will abort.

    NF
    Output, INTEGER(IK) scalar.
    NF will be set to the number of calls of CALCFC at exit.

    RHOBEG, RHOEND
    Inputs, REAL(RP) scalars, default: RHOBEG = 1, RHOEND = 10^-6. RHOBEG and RHOEND must be set to
    the initial and final values of a trust-region radius, both being positive and RHOEND <= RHOBEG.
    Typically RHOBEG should be about one tenth of the greatest expected change to a variable, and
    RHOEND should indicate the accuracy that is required in the final values of the variables.

    FTARGET
    Input, REAL(RP) scalar, default: -Inf.
    FTARGET is the target function value. The algorithm will terminate when a feasible point with a
    function value <= FTARGET is found.

    CTOL
    Input, REAL(RP) scalar, default: machine epsilon.
    CTOL is the tolerance of constraint violation. Any X with MAXVAL(-CONSTR(X)) <= CTOL is
    considered feasible.
    N.B.: 1. CTOL is absolute, not relative. 2. CTOL is used only when selecting the returned X.
    It does not affect the iterations of the algorithm.

    CWEIGHT
    Input, REAL(RP) scalar, default: CWEIGHT_DFT defined in the module CONSTS_MOD in common/consts.F90.
    CWEIGHT is the weight that the constraint violation takes in the selection of the returned X.

    MAXFUN
    Input, INTEGER(IK) scalar, default: MAXFUN_DIM_DFT*N with MAXFUN_DIM_DFT defined in the module
    CONSTS_MOD (see common/consts.F90). MAXFUN is the maximal number of calls of CALCFC.

    IPRINT
    Input, INTEGER(IK) scalar, default: 0.
    The value of IPRINT should be set to 0, 1, -1, 2, -2, 3, or -3, which controls how much
    information will be printed during the computation:
    0: there will be no printing;
    1: a message will be printed to the screen at the return, showing the best vector of variables
        found and its objective function value;
    2: in addition to 1, each new value of RHO is printed to the screen, with the best vector of
        variables so far and its objective function value; each new value of CPEN is also printed;
    3: in addition to 2, each function evaluation with its variables will be printed to the screen;
    -1, -2, -3: the same information as 1, 2, 3 will be printed, not to the screen but to a file
        named COBYLA_output.txt; the file will be created if it does not exist; the new output will
        be appended to the end of this file if it already exists.
    Note that IPRINT = +/-3 can be costly in terms of time and/or space.

    ETA1, ETA2, GAMMA1, GAMMA2
    Input, REAL(RP) scalars, default: ETA1 = 0.1, ETA2 = 0.7, GAMMA1 = 0.5, and GAMMA2 = 2.
    ETA1, ETA2, GAMMA1, and GAMMA2 are parameters in the updating scheme of the trust-region radius
    detailed in the subroutine TRRAD in trustregion.f90. Roughly speaking, the trust-region radius
    is contracted by a factor of GAMMA1 when the reduction ratio is below ETA1, and enlarged by a
    factor of GAMMA2 when the reduction ratio is above ETA2. It is required that 0 < ETA1 <= ETA2
    < 1 and 0 < GAMMA1 < 1 < GAMMA2. Normally, ETA1 <= 0.25. It is NOT advised to set ETA1 >= 0.5.

    XHIST, FHIST, CHIST, CONHIST, MAXHIST
    XHIST: Output, ALLOCATABLE rank 2 REAL(RP) array;
    FHIST: Output, ALLOCATABLE rank 1 REAL(RP) array;
    CHIST: Output, ALLOCATABLE rank 1 REAL(RP) array;
    CONHIST: Output, ALLOCATABLE rank 2 REAL(RP) array;
    MAXHIST: Input, INTEGER(IK) scalar, default: MAXFUN
    XHIST, if present, will output the history of iterates; FHIST, if present, will output the
    history function values; CHIST, if present, will output the history of constraint violations;
    CONHIST, if present, will output the history of constraint values; MAXHIST should be a
    nonnegative integer, and XHIST/FHIST/CHIST/CONHIST will output only the history of the last
    MAXHIST iterations. Therefore, MAXHIST= 0 means XHIST/FHIST/CONHIST/CHIST will output nothing,
    while setting MAXHIST = MAXFUN requests XHIST/FHIST/CHIST/CONHIST to output all the history.
    If XHIST is present, its size at exit will be (N, min(NF, MAXHIST)); if FHIST is present, its
    size at exit will be min(NF, MAXHIST); if CHIST is present, its size at exit will be
    min(NF, MAXHIST); if CONHIST is present, its size at exit will be (M, min(NF, MAXHIST)).

    IMPORTANT NOTICE:
    Setting MAXHIST to a large value can be costly in terms of memory for large problems.
    MAXHIST will be reset to a smaller value if the memory needed exceeds MAXHISTMEM defined in
    CONSTS_MOD (see consts.F90 under the directory named "common").
    Use *HIST with caution!!! (N.B.: the algorithm is NOT designed for large problems).

    MAXFILT
    Input, INTEGER(IK) scalar.
    MAXFILT is a nonnegative integer indicating the maximal length of the filter used for selecting
    the returned solution; default: MAXFILT_DFT (a value lower than MIN_MAXFILT is not recommended);
    see common/consts.F90 for the definitions of MAXFILT_DFT and MIN_MAXFILT.

    INFO
    Output, INTEGER(IK) scalar.
    INFO is the exit flag. It will be set to one of the following values defined in the module
    INFOS_MOD (see common/infos.f90):
    SMALL_TR_RADIUS: the lower bound for the trust region radius is reached;
    FTARGET_ACHIEVED: the target function value is reached;
    MAXFUN_REACHED: the objective function has been evaluated MAXFUN times;
    MAXTR_REACHED: the trust region iteration has been performed MAXTR times (MAXTR = 2*MAXFUN);
    NAN_INF_X: NaN or Inf occurs in X;
    DAMAGING_ROUNDING: rounding errors are becoming damaging.
    !--------------------------------------------------------------------------!
    The following case(s) should NEVER occur unless there is a bug.
    NAN_INF_F: the objective function returns NaN or +Inf;
    NAN_INF_MODEL: NaN or Inf occurs in the model;
    TRSUBP_FAILED: a trust region step failed to reduce the model
    !--------------------------------------------------------------------------!
    """
    if debugging:
        # TODO: need to check how to figure out if python variables are present
        # I guess the way to go is to set the default to None
        assert (f0 is None) == (constr0 is None), "f0 and constr0 must be both present or both absent"

    n = len(x)

    assert len(constr0) == m if constr0 is not None else True, "len(constr0) != m"

    constr_loc = np.zeros(m)
    if f0 is not None and constr0 is not None and all(np.isfinite(x)):
        f = f0
        constr_loc = constr0
    else:
        # Replace all NaNs with 0s and clip all -inf/inf to -floatmax/floatmax
        np.nan_to_num(x, copy=False, neginf=-np.finfo(float).max, posinf=np.finfo(float).max)
        f, constr_loc, cstrv_loc = evaluate(calcfc, x)
    
    if rhobeg is not None:
        rhobeg_loc = rhobeg
    elif rhoend is not None and np.isfinite(rhoend) and rhoend > 0:
        rhobeg_loc = max(10*rhoend, RHOBEG_DEFAULT)
    else:
        rhobeg_loc = RHOBEG_DEFAULT

    if rhoend is not None:
        rhoend_loc = rhoend
    elif rhobeg_loc > 0:
        rhoend_loc = max(EPS, min(0.1*rhobeg_loc, RHOEND_DEFAULT))
    else:
        rhoend_loc = RHOEND_DEFAULT

    maxfun = maxfun if maxfun is not None else MAXFUN_DIM_DEFAULT*n

    if eta1 is not None:
        eta1_loc = eta1
    elif eta2 is not None and 0 < eta2 < 1:
        eta1_loc = max(EPS, eta2/7)
    else:
        eta1_loc = 0.1

    if eta2 is not None:
        eta2_loc = eta2
    else:
        if 0 < eta1_loc < 1:
            eta2_loc = (eta1_loc + 2)/3
        else:
            eta2_loc = 0.7

    if maxhist is not None:
        maxhist_loc = maxhist
    else:
        maxhist_loc = max(maxfun, n+2, MAXFUN_DIM_DEFAULT*n)

    # preprocess

    '''
    ! Further revise MAXHIST_LOC according to MAXHISTMEM, and allocate memory for the history.
    ! In MATLAB/Python/Julia/R implementation, we should simply set MAXHIST = MAXFUN and initialize
    ! CHIST = NaN(1, MAXFUN), CONHIST = NaN(M, MAXFUN), FHIST = NaN(1, MAXFUN), XHIST = NaN(N, MAXFUN)
    ! if they are requested; replace MAXFUN with 0 for the history that is not requested.
    call prehist(maxhist_loc, n, present(xhist), xhist_loc, present(fhist), fhist_loc, &
        & present(chist), chist_loc, m, present(conhist), conhist_loc)
    '''

    # call coblyb, which performs the real calculations
    x, f, nf, info = cobylb(calcfc, iprint, maxfilt, maxfun, ctol, cweight, eta1_loc, eta2_loc,
      ftarget, gamma1, gamma2, rhobeg_loc, rhoend_loc, constr_loc, f, x, 
      cstrv_loc)
    
    # Write the outputs.
    return x, f