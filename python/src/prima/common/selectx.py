'''
This module provides subroutines that ensure the returned X is optimal among all the calculted
points in the sense that no other point achieves both lower function value and lower constraint
violation at the same time. This module is needed only in the constrained case.

Coded by Zaikun ZHANG (www.zhangzk.net).

Python implementation by Nickolai Belakovski
'''

import numpy as np
import numpy.typing as npt
from prima.common.consts import EPS, CONSTRMAX, REALMAX, FUNCMAX

def isbetter(f1: float, c1: float, f2: float, c2: float, ctol: float) -> bool:
    '''
    This function compares whether FC1 = (F1, C1) is (strictly) better than FC2 = (F2, C2), which
    basically means that (F1 < F2 and C1 <= C2) or (F1 <= F2 and C1 < C2).
    It takes care of the cases where some of these values are NaN or Inf, even though some cases
    should never happen due to the moderated extreme barrier.
    At return, BETTER = TRUE if and only if (F1, C1) is better than (F2, C2).
    Here, C means constraint violation, which is a nonnegative number.
    '''
    isbetter = False
    # Even though NaN/+Inf should not occur in FC1 or FC2 due to the moderated extreme barrier, for
    # security and robustness, the code below does not make this assumption.
    isbetter = isbetter or (any(np.isnan([f1, c1])) and not any(np.isnan([f2, c2])))

    isbetter = isbetter or (f1 < f2 and c1 <= c2)
    isbetter = isbetter or (f1 <= f2 and c1 < c2)

    # If C1 <= CTOL and C2 is significantly larger/worse than CTOL, i.e., C2 > MAX(CTOL,CREF),
    # then FC1 is better than FC2 as long as F1 < REALMAX. Normally CREF >= CTOL so MAX(CTOL, CREF)
    # is indeed CREF. However, this may not be true if CTOL > 1E-1*CONSTRMAX.
    cref = 10 * max(EPS, min(ctol, 1.0E-2 * CONSTRMAX))  # The MIN avoids overflow.
    isbetter = isbetter or (f1 < REALMAX and c1 <= ctol and (c2 > max(ctol, cref) or np.isnan(c2)))
    return isbetter


def savefilt(cstrv, ctol, cweight, f, x, nfilt, cfilt, ffilt, xfilt, constr=None, confilt=None):
    '''
    This subroutine saves X, F, and CSTRV in XFILT, FFILT, and CFILT (and CONSTR in CONFILT
    if they are present), unless a vector in XFILT[:, :NFILT] is better than X.
    If X is better than some vectors in XFLIT[:, :NFILT] then these vectors will be
    removed. If X is not better than any of XFILT[:, :NFILT], but NFILT == MAXFILT,
    then we remove a column from XFILT according to the merit function
    PHI = FFILT + CWEIGHT * max(CFILT - CTOL, 0)
    N.B.:
    1. Only XFILT[:, :NFILT] and FFILT[:, :NFILT] etc contains valid information,
    while XFILT[:, NFILT+1:MAXFILT] and FFILT[:, NFILT+1:MAXFILT] etc are not
    initialized yet.
    2. We decide whether and X is better than another by the ISBETTER function
    '''

    # Sizes
    if constr is not None:
        num_constraints = len(constr)
    else:
        num_constraints = 0

    num_vars = len(x)
    maxfilt = len(ffilt)

    # Preconditions TODO

    # Return immeditely if any column of XFILT is better than X. TODO: Lazy evaluation
    if any([isbetter(ffilt_i, cfilt_i, f, cstrv, ctol) for ffilt_i, cfilt_i in zip(ffilt[:nfilt], cfilt[:nfilt])]):
        return nfilt, cfilt, ffilt, xfilt, confilt
    
    # Decide which columns of XFILT to keep.
    keep = np.logical_not([isbetter(f, cstrv, ffilt_i, cfilt_i, ctol) for ffilt_i, cfilt_i in zip(ffilt[:nfilt], cfilt[:nfilt])])
    
    # If NFILT == MAXFILT and X is not better than any column of XFILT, then we remove the worst column
    # of XFILT according to the merit function PHI = FFILT + CWEIGHT * MAX(CFILT - CTOL, ZERO).
    if sum(keep) == maxfilt:  # In this case, NFILT = SIZE(KEEP) = COUNT(KEEP) = MAXFILT > 0.
        # TODO: This code path has not been vetted
        cfilt_shifted = np.maximum(cfilt - ctol, 0)
        if cweight <= 0:
            phi = ffilt
        elif np.isposinf(cweight):
            phi = cfilt_shifted
            # We should not use CFILT here; if MAX(CFILT_SHIFTED) is attained at multiple indices, then
            # we will check FFILT to exhaust the remaining degree of freedom.
        else:
            phi = np.maximum(ffilt, -REALMAX)
            phi = np.nan_to_num(phi, nan=-REALMAX)  # Replace NaN with -REALMAX and +/- inf with large numbers
            phi += cweight * cfilt_shifted
        # We select X to maximize PHI. In case there are multiple maximizers, we take the one with the
        # largest CSTRV_SHIFTED; if there are more than one choices, we take the one with the largest F;
        # if there are several candidates, we take the one with the largest CSTRV; if the last comparison
        # still leads to more than one possibilities, then they are equally bad and we choose the first.
        # N.B.:
        # 1. This process is the opposite of selecting KOPT in SELECTX.
        # 2. In finite-precision arithmetic, PHI_1 == PHI_2 and CSTRV_SHIFTED_1 == CSTRV_SHIFTED_2 do
        # not ensure that F_1 == F_2!
        phimax = max(phi)
        cref = max(cfilt_shifted[phi >= phimax])
        fref = max(ffilt[cfilt_shifted >= cref])
        kworst = np.where(cfilt == max(cfilt[ffilt <= fref]))[0][0]
        if kworst < 0 or kworst >= len(keep):  #  For security. Should not happen.
            kworst = 0
        keep[kworst] = False

    # Keep the good xfilt values and remove all the ones that are strictly worse than the new x.
    nfilt = sum(keep)
    index_to_keep = np.where(keep)[0]
    xfilt[:, :nfilt] = xfilt[:, index_to_keep]
    ffilt[:nfilt] = ffilt[index_to_keep]
    cfilt[:nfilt] = cfilt[index_to_keep]
    if confilt is not None and constr is not None:
        confilt[:, :nfilt] = confilt[:, index_to_keep]

    # Once we have removed all the vectors that are strictly worse than x,
    # we add x to the filter.
    xfilt[:, nfilt] = x
    ffilt[nfilt] = f
    cfilt[nfilt] = cstrv
    if confilt is not None and constr is not None:
        confilt[:, nfilt] = constr
    nfilt += 1  # In Python we need to increment the index afterwards

    return nfilt, cfilt, ffilt, xfilt, confilt


def selectx(fhist: npt.NDArray, chist: npt.NDArray, cweight: float, ctol: float):
    '''
    This subroutine selects X according to FHIST and CHIST, which represents (a part of) history
    of F and CSTRV. Normally, FHIST and CHIST are not the full history but only a filter, e.g. ffilt
    and CFILT generated by SAVEFILT. However, we name them as FHIST and CHIST because the [F, CSTRV]
    in a filter should not dominate each other, but this subroutine does NOT assume such a property.
    N.B.: CTOL is the tolerance of the constraint violation (CSTRV). A point is considered feasible if
    its constraint violation is at most CTOL. Not that CTOL is absolute, not relative.
    '''

    nhist = len(fhist)

    # We select X among the points with F < FREF and CSTRV < CREF.
    # Do NOT use F <= FREF, because F == FREF (FUNCMAX or REALMAX) may mean F == INF in practice!
    if any(np.logical_and(fhist < FUNCMAX, chist < CONSTRMAX)):
        fref = FUNCMAX
        cref = CONSTRMAX
    elif any(np.logical_and(fhist < REALMAX, chist < CONSTRMAX)):
        fref = REALMAX
        cref = CONSTRMAX
    elif any(np.logical_and(fhist < FUNCMAX, chist < REALMAX)):
        fref = FUNCMAX
        cref = REALMAX
    else:
        fref = REALMAX
        cref = REALMAX

    if not any(np.logical_and(fhist < fref, chist < cref)):
        kopt = nhist - 1
    else:
        # Shift the constraint violations by ctol, so that cstrv <= ctol is regarded as no violation.
        chist_shifted = np.maximum(chist - ctol, 0)
        # cmin is the minimal shift constraint violation attained in the history.
        cmin = min(chist_shifted[fhist < fref])
        # We consider only the points whose shifted constraint violations are at most the cref below.
        # N.B.: Without taking max(EPS, .), cref would be 0 if cmin = 0. In that case, asking for
        # cstrv_shift < cref would be WRONG!
        cref = max(EPS, 2*cmin)
        # We use the following phi as our merit function to select X.
        if cweight <= 0:
            phi = fhist
        elif np.isposinf(cweight):
            phi = chist_shifted
            # We should not use chist here; if min(chist_shifted) is attained at multiple indices, then
            # we will check fhist to exhaust the remaining degree of freedom.
        else:
            phi = np.maximum(fhist, -REALMAX) + cweight * chist_shifted
            # max(fhist, -REALMAX) makes sure that phi will not contain NaN (unless there is a bug).
        
        # We select X to minimize phi subject to f < fref and cstrv_shift <= cref (see the comments
        # above for the reason of taking "<" and "<=" in these two constraints). In case there are
        # multiple minimizers, we take the one with the least cstrv_shift; if there is more than one
        # choice, we take the one with the least f; if there are several candidates, we take the one
        # with the least cstrv; if the last comparison still leads to more than one possibility, then
        # they are equally good and we choose the first.
        # N.B.:
        # 1. This process is the opposite of selecting kworst in savefilt
        # 2. In finite-precision arithmetic, phi_2 == phi_2 and cstrv_shift_1 == cstrv_shifted_2 do
        # not ensure thatn f_1 == f_2!
        phimin = min(phi[np.logical_and(fhist < fref, chist_shifted <= cref)])
        cref = min(chist_shifted[np.logical_and(fhist < fref, phi <= phimin)])
        fref = min(fhist[chist_shifted <= cref])
        kopt = np.where(chist == min(chist[fhist <= fref]))[0][0]

    return kopt