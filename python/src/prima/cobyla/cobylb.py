import numpy as np
from prima.common.consts import INFO_DEFAULT, REALMAX, EPS, CONSTRMAX, \
                    MAXTR_REACHED, DAMAGING_ROUNDING, SMALL_TR_RADIUS, \
                    NAN_INF_X, NAN_INF_F, FTARGET_ACHIEVED, MAXFUN_REACHED
from prima.common.evaluate import evaluate, moderatec, moderatef
from prima.cobyla.update import updatepole, findpole, updatexfc
from prima.cobyla.geometry import assess_geo, setdrop_tr, geostep, setdrop_geo
from prima.common.selectx import savefilt , selectx
from prima.cobyla.trustregion import trstlp, redrat, trrad

def fcratio(conmat, fval):
    '''
    This function calculates the ratio between the "typical changre" of F and that of CONSTR.
    See equations (12)-(13) in Section 3 of the COBYLA paper for the definition of the ratio.
    '''
    cmin = np.min(conmat, axis=1)
    cmax = np.max(conmat, axis=1)
    fmin = min(fval)
    fmax = max(fval)
    if any(cmin < 0.5 * cmax) and fmin < fmax:
        denom = np.min(np.maximum(cmax, 0) - cmin, where=cmin < 0.5 * cmax, initial=np.inf)
        # Powell mentioned the following alternative in section 4 of his COBYLA paper. According to a test
        # on 20230610, it does not make much difference to the performance.
        # denom = np.max(max(*cmax, 0) - cmin, mask=(cmin < 0.5 * cmax))
        r = (fmax - fmin) / denom
    else:
        r = 0

    return r

def redrho(rho, rhoend):
    '''
    This function calculates RHO when it needs to be reduced.
    The sceme is shared by UOBYQA, NEWUOA, BOBYQA, LINCOA. For COBYLA, Powell's code reduces RHO by
    'RHO *= 0.5; if RHO <= 1.5 * RHOEND: RHO = RHOEND' as specified in (11) of the COBYLA
    paper. However, this scheme seems to work better, especially after we introduce DELTA.
    '''

    rho_ratio = rho / rhoend

    if rho_ratio > 250:
        rho *= 0.1
    elif rho_ratio <= 16:
        rho = rhoend
    else:
        rho = np.sqrt(rho_ratio) * rhoend  # rho = np.sqrt(rho * rhoend)
    
    return rho

def checkbreak(maxfun, nf, cstrv, ctol, f, ftarget, x):
    '''
        This function checks whether to break the solver in constrained cases.
    '''
    info = INFO_DEFAULT

    # Although X should not contain NaN unless there is a bug, we include the following for security.
    # X can be inf, as finite + finite can be inf numerically.
    if any(np.isnan(x)) or any(np.isinf(x)):
        info = NAN_INF_X

    # Although NAN_INF_F should not happen unless there is a bug, we include the following for security.
    if np.isnan(f) or np.isposinf(f) or np.isnan(cstrv) or np.isposinf(cstrv):
        info = NAN_INF_F
    
    if cstrv <= ctol and f <= ftarget:
        info = FTARGET_ACHIEVED

    if nf >= maxfun:
        info = MAXFUN_REACHED
    return info

def initxfc(calcfc, iprint, maxfun, constr0, ctol, f0, ftarget, rhobeg, x0,
    conmat, cval, fval):
    '''
    This subroutine does the initialization concerning X, function values, and constraints.
    '''

    num_constraints = conmat.shape[0]
    num_vars = x0.size

    # Initialize info to the default value. At return, a value different from this value will
    # indicate an abnormal return
    info = INFO_DEFAULT

    # Initialize the simplex. It will be revised during the initialization.
    sim = np.eye(num_vars, num_vars+1) * rhobeg
    sim[:, -1] = x0

    # evaluated[j] = True iff the function/constraint of SIM[:, j] has been evaluated.
    evaluated = np.zeros(num_vars+1, dtype=bool)

    # Here the FORTRAN code initializes some history values, but apparently only for the benefit of compilers,
    # so we skip it.

    for k in range(num_vars + 1):
        x = sim[:, num_vars].copy()
        # We will evaluate F corresponding to SIM(:, J).
        if k == 0:
            j = num_vars
            f = moderatef(f0)
            constr = moderatec(constr0)
            cstrv = max(max(-constr), 0)
        else:
            j = k - 1
            x[j] += rhobeg
            f, constr, cstrv = evaluate(calcfc, x)
        
        # Print a message about the function/constraint evaluation according to IPRINT.
        # TODO: Finish implementing fmsg, if decided that its worth it
        # fmsg(solver, iprint, k, f, x, cstrv, constr)

        # Save X, F, CONSTR, CSTRV into the history.
        # TODO: Implement history (maybe we need it for the iterations?)
        # savehist(k, x, xhist, f, fhist, cstrv, chist, constr, conhist)

        # Save F, CONSTR, and CSTRV to FVAL, CONMAT, and CVAL respectively.
        evaluated[j] = True
        fval[j] = f
        conmat[:, j] = constr
        cval[j] = cstrv

        # Check whether to exit.
        # subinfo = checkexit(maxfun, k, cstrv, ctol, f, ftarget, x)
        # if subinfo != INFO_DEFAULT:
        #     info = subinfo
        #     break

        # Exchange the new vertex of the initial simplex with the optimal vertex if necessary.
        # This is the ONLY part that is essentially non-parallel.
        if j < num_vars and fval[j] < fval[num_vars]:
            fval[j], fval[num_vars] = fval[num_vars], fval[j]
            cval[j], cval[num_vars] = cval[num_vars], cval[j]
            conmat[:, [j, num_vars]] = conmat[:, [num_vars, j]]
            sim[:, num_vars] = x
            sim[j, :j+1] = -rhobeg

    nf = np.count_nonzero(evaluated)

    if evaluated.all():
        # Initialize SIMI to the inverse of SIM[:, :num_vars]
        simi = np.linalg.inv(sim[:, :num_vars])
    
    return evaluated, sim, simi, nf, info


def initfilt(conmat, ctol, cweight, cval, fval, sim, evaluated, cfilt, confilt, ffilt, xfilt):
    '''
    This function initializes the filter (XFILT, etc) that will be used when selecting
    x at the end of the solver.
    N.B.:
    1. Why not initialize the filters using XHIST, etc? Because the history is empty if
    the user chooses not to output it.
    2. We decouple INITXFC and INITFILT so that it is easier to parallelize the former
    if needed.
    '''

    # Sizes
    num_constraints = conmat.shape[0]
    num_vars = sim.shape[0]
    maxfilt = len(ffilt)

    # Precondictions
    assert num_constraints >= 0
    assert num_vars >= 1
    assert maxfilt >= 1
    assert confilt.shape == (num_constraints, maxfilt)
    assert cfilt.shape == (maxfilt,)
    assert xfilt.shape == (num_vars, maxfilt)
    assert ffilt.shape == (maxfilt,)
    assert conmat.shape == (num_constraints, num_vars+1)
    # TODO: Need to finish these preconditions


    nfilt = 0
    for i in range(num_vars+1):
        if evaluated[i]:
            if i < num_vars:
                x = sim[:, i] + sim[:, num_vars]
            else:
                x = sim[:, i]  # i == num_vars, i.e. the last column
            nfilt, cfilt, ffilt, xfilt, confilt = savefilt(cval[i], ctol, cweight, fval[i], x, nfilt, cfilt, ffilt, xfilt, conmat[:, i], confilt)


    return nfilt


def cobylb(calcfc, iprint, maxfilt, maxfun, ctol, cweight, eta1, eta2, ftarget,
           gamma1, gamma2, rhobeg, rhoend, constr, f, x, 
           cstrv):
    '''
    This subroutine performs the actual computations of COBYLA.
    '''

    # A[:, :num_constraints] contains the approximate gradient for the constraints, and A[:, num_constraints] is minus the
    # approximate gradient for the objective function.
    A = np.zeros((len(x), len(constr) + 1))
    
    # CPENMIN is the minimum of the penalty parameter CPEN for the L-infinity constraint violation in
    # the merit function. Note that CPENMIN = 0 in Powell's implementation, which allows CPEN to be 0.
    # Here, we take CPENMIN > 0 so that CPEN is always positive. This avoids the situation where PREREM
    # becomes 0 when PREREF = 0 = CPEN. It brings two advantages as follows.
    # 1. If the trust-region subproblem solver works correctly and the trust-region center is not
    # optimal for the subproblem, then PREREM > 0 is guaranteed. This is because, in theory, PREREC >= 0
    # and MAX(PREREC, PREREF) > 0 , and the definition of CPEN in GETCPEN ensures that PREREM > 0.
    # 2. There is no need to revise ACTREM and PREREM when CPEN = 0 and F = FVAL(N+1) as in lines
    # 312--314 of Powell's cobylb.f code. Powell's code revises ACTREM to CVAL(N + 1) - CSTRV and PREREM
    # to PREREC in this case, which is crucial for feasibility problems.
    cpenmin = EPS
    # FACTOR_ALPHA, FACTOR_BETA, FACTOR_GAMMA, and FACTOR_DELTA are four factors that COBYLB uses
    # when managing the simplex. Note the following.
    # 1. FACTOR_ALPHA < FACTOR_GAMMA < 1 < FACTOR_DELTA <= FACTOR_BETA.
    # 2. FACTOR_DELTA has nothing to do with DELTA, which is the trust-region radius.
    # 3. FACTOR_GAMMA has nothing to do with GAMMA1 and GAMMA2, which are the contracting/expanding
    # factors for updating the trust-region radius DELTA.
    factor_alpha = 0.25  # The factor alpha in the COBYLA paper
    factor_beta = 2.1  # The factor beta in the COBYLA paper
    factor_delta = 1.1  # The factor delta in the COBYLA paper
    factor_gamma = 0.5  # The factor gamma in the COBYLA paper
    num_constraints = len(constr)
    num_vars = len(x)
    # maxxhist = xhist.shape[1]
    # maxfhist = len(fhist)
    # maxconhist = conhist.shape[1]
    # maxchist = len(chist)
    # maxhist = max(maxxhist, maxfhist, maxconhist, maxchist)
    conmat = np.zeros((num_constraints, num_vars+1)) - REALMAX
    cval = np.zeros(num_vars+1) + REALMAX
    fval = np.zeros(num_vars+1) + REALMAX
    
    subinfo = 0
    evaluated, sim, simi, nf, subinfo = initxfc(calcfc, iprint, maxfun, constr, ctol, f, ftarget, rhobeg, x,
      conmat, cval, fval)
    # TODO: Need to initialize history with f and x and constr
    
    # Initialize the filter, including xfilt, ffilt, confiolt, cfilt, and nfilt.
    # N.B.: The filter is used only when selecting which iterate to return. It does not
    # interfere with the iterations. COBYLA is NOT a filter method but a trust-region method
    # based on an L-inifinity merit function. Powell's implementation does not use a
    # filter to select the iterate, possibly returning a suboptimal iterate.
    cfilt = np.zeros(min(max(maxfilt, 1), maxfun))
    confilt = np.zeros((len(constr), len(cfilt)))
    ffilt = np.zeros(len(cfilt))
    xfilt = np.zeros((len(x), len(cfilt)))
    nfilt = initfilt(conmat, ctol, cweight, cval, fval, sim, evaluated, cfilt, confilt, ffilt, xfilt)

    # TODO: Initfilt and savefilt appear to be working. It did not get tested with a rising edge in the
    # keep array (i.e. a False followed by a True), but it looks like that shouldn't be a problem.
    # For now we should move on to the rest of the algorithm so that we can update the PR with a milestone.

    # Check whether to return due to abnormal cases that may occur during the initialization.
    if subinfo != INFO_DEFAULT:
        info = subinfo
        # Return the best calculated values of the variables
        # N.B: Selectx and findpole choose X by different standards, one cannot replace the other
        kopt = selectx(ffilt[:nfilt], cfilt[:nfilt], cweight, ctol)
        x = xfilt[:, kopt]
        f = ffilt[kopt]
        constr = confilt[:, kopt]
        cstrv = cfilt[kopt]
        # Arrange chist, conhist, fhist, and xhist so that they are in chronological order.
        # TODO: Implement me
        # rangehist(nf, xhist, fhist, chist, conhist)
        # print a return message according to IPRINT.
        # TODO: Implement me
        # retmsg("COBYLA", info, iprint, nf, f, x, cstrv, constr)


    # Set some more initial values.
    # We must initialize ACTREM and PREREM. Otherwise, when SHORTD = TRUE, compilers may raise a
    # run-time error that they are undefined. But their values will not be used: when SHORTD = FALSE,
    # they will be overwritten; when SHORTD = TRUE, the values are used only in BAD_TRSTEP, which is
    # TRUE regardless of ACTREM or PREREM. Similar for PREREC, PREREF, PREREM, RATIO, and JDROP_TR.
    # No need to initialize SHORTD unless MAXTR < 1, but some compilers may complain if we do not do it.
    # Our initialization of CPEN differs from Powell's in two ways. First, we use the ratio defined in
    # (13) of Powell's COBYLA paper to initialize CPEN. Second, we impose CPEN >= CPENMIN > 0. Powell's
    # code simply initializes CPEN to 0.
    rho = rhobeg
    delta = rhobeg
    cpen = max(cpenmin, min(1.0E3, fcratio(conmat, fval)))  # Powell's code: CPEN = ZERO
    prerec = -REALMAX
    preref = -REALMAX
    prerem = -REALMAX
    actrem = -REALMAX  # TODO: Is it necessary to define this and prerem here, per the above comment?
    shortd = False
    trfail = False
    ratio = -1
    jdrop_tr = 0
    jdrop_geo = 0

    # If DELTA <= GAMMA3*RHO after an update, we set DELTA to RHO. GAMMA3 must be less than GAMMA2. The
    # reason is as follows. Imagine a very successful step with DNORM = the un-updated DELTA = RHO.
    # The TRRAD will update DELTA to GAMMA2*RHO. If GAMMA3 >= GAMMA2, then DELTA will be reset to RHO,
    # which is not reasonable as D is very successful. See paragraph two of Sec 5.2.5 in
    # T. M. Ragonneau's thesis: "Model-Based Derivative-Free Optimization Methods and Software."
    # According to test on 20230613, for COBYLA, this Powellful updating scheme of DELTA works slightly
    # better than setting directly DELTA = max(NEW_DELTA, RHO).
    gamma3 = max(1, min(0.75 * gamma2, 1.5))

    # MAXTR is the maximal number of trust region iterations. Each trust-region iteration takes 1 or 2
    # function evaluations unless the trust-region step is short or the trust-region subproblem solver
    # fails but the geometry step is not invoked. Thus the following MAXTR is unlikely to be reached
    # 
    # 
    # Normally each trust-region
    # iteration takes 1 or 2 function evaluations except for the following cases:
    # 1. The update of cpen alters the optimal vertex
    # 2. The trust-region step is short or fails to reduce either the
    #    linearized objective or the linearized constraint violation but
    #    the geometry step is not invoked.
    # The following maxtr is unlikely to be reached.
    maxtr = max(maxfun, 2*maxfun)  # max: precaution against overflow, which will make 2*MAXFUN < 0
    info = MAXTR_REACHED

    # Begin the iterative procedure
    # After solving a trust-region subproblem, we use three boolean variables to control the workflow.
    # SHORTD - Is the trust-region trial step too short to invoke a function evaluation?
    # IMPROVE_GEO - Will we improve the model after the trust-region iteration? If yes, a geometry
    # step will be taken, corresponding to the "Branch (Delta)" in the COBYLA paper.
    # REDUCE_RHO - Will we reduce rho after the trust-region iteration?
    # COBYLA never sets IMPROVE_GEO and REDUCE_RHO to True simultaneously.
    
    for tr in range(maxtr):
        # Increase the penalty parameter CPEN, if needed, so that PREREM = PREREF + CPEN * PREREC > 0.
        # This is the first (out of two) update of CPEN, where CPEN increases or remains the same.
        # N.B.: CPEN and the merit function PHI = FVAL + CPEN*CVAL are used in three places only.
        # 1. In FINDPOLE/UPDATEPOLE, deciding the optimal vertex of the current simplex.
        # 2. After the trust-region trial step, calculating the reduction ratio.
        # 3. In GEOSTEP, deciding the direction of the geometry step.
        # They do not appear explicitly in the trust-region subproblem, though the trust-region center
        # (i.e. the current optimal vertex) is defined by them.
        cpen = getcpen(conmat, cpen, cval, delta, fval, rho, sim, simi)

        # Switch the best vertex of the current simplex to SIM[:, NUM_VARS].
        conmat, cval, fval, sim, simi, subinfo = updatepole(cpen, conmat, cval, fval, sim, simi)
        # Check whether to exit due to damaging rounding in UPDATEPOLE.
        if subinfo == DAMAGING_ROUNDING:
            info = subinfo
            break  # Better action to take? Geometry step, or simply continue?

        # Does the interpolation set have acceptable geometry? It affects improve_geo and reduce_rho
        adequate_geo = assess_geo(delta, factor_alpha, factor_beta, sim, simi)

        # Calculate the linear approximations to the objective and constraint functions, placing
        # minus the objective function gradient after the constraint gradients in the array A.
        # N.B.: TRSTLP accesses A mostly by colums, so it is more reasonable to save A instead of A.T
        A[:, :num_constraints] = ((conmat[:, :num_vars] - np.tile(conmat[:, num_vars], (num_vars, 1)).T)@simi).T
        A[:, num_constraints] = (fval[num_vars] - fval[:num_vars])@simi
        # Theoretically, but not numerically, the last entry of b does not affect the result of TRSTLP
        # We set it to -fval[num_vars] following Powell's code.
        b = np.hstack((-conmat[:, num_vars], -fval[num_vars]))

        # Calculate the trust-region trial step d. Note that d does NOT depend on cpen.
        d = trstlp(A, b, delta)
        dnorm = min(delta, np.linalg.norm(d))

        # Is the trust-region trial step short? N.B.: we compare DNORM with RHO, not DELTA.
        # Powell's code especially defines SHORTD by SHORTD = (DNORM < 0.5 * RHO). In our tests
        # 1/10 seems to work better than 1/2 or 1/4, especially for linearly constrained
        # problems. Note that LINCOA has a slightly more sophisticated way of defining SHORTD,
        # taking into account whether D casues a change to the active set. Should we try the same here?
        shortd = (dnorm < 0.1 * rho)

        # Predict the change to F (PREREF) and to the constraint violation (PREREC) due to D.
        # We have the following in precise arithmetic. They may fail to hold due to rounding errors.
        # 1. B[:NUM_CONSTRAINTS] = -CONMAT[:, NUM_VARS] and hence max(max(B[:NUM_CONSTRAINTS] - D@A[:, :NUM_CONSTRAINTS]), 0) is the
        # L-infinity violation of the linearized constraints corresponding to D. When D=0, the
        # violation is max(max(B[:NUM_CONSTRAINTS]), 0) = CVAL[NUM_VARS]. PREREC is the reduction of this 
        # violation achieved by D, which is nonnegative in theory; PREREC = 0 iff B[:NUM_CONSTRAINTS] <= 0, i.e. the
        # trust-region center satisfies the linearized constraints.
        # 2. PREREF may be negative or 0, but it is positive when PREREC = 0 and shortd is False
        # 3. Due to 2, in theory, max(PREREC, PREREF) > 0 if shortd is False.
        prerec = cval[num_vars] - max(max(b[:num_constraints] - d@A[:, :num_constraints]), 0)
        preref = np.dot(d, A[:, num_constraints])  # Can be negative

        # Evaluate PREREM, which is the predicted reduction in the merit function.
        # In theory, PREREM >= 0 and it is 0 iff CPEN = 0 = PREREF. This may not be true numerically.
        prerem = preref + cpen * prerec
        trfail = prerem < 1e-5 * min(cpen, 1) * rho**2 or np.isnan(prerem)  # PREREM is tiny/negative or NaN.

        if shortd or trfail:
            # Reduce DELTA if D is short or if D fails to render PREREM > 0. The latter can only happen due to
            # rounding errors. This seems quite important for performance
            delta *= 0.1
            if delta <= gamma3 * rho:
                delta = rho  # set delta to rho when it is close to or below
        else:
            x = sim[:, num_vars] + d
            # Evaluate the objective and constraints at X, taking care of possible inf/nan values.
            f, constr, cstrv = evaluate(calcfc, x)
            nf += 1

            # Print a message about the function/constraint evaluation accoring to iprint
            # TODO: Implement me
            # fmsg(solver, iprint, nf, f, x, cstrv, constr)
            # Save X, F, CONSTR, CSTRV into the history.
            # TODO: Implement me
            # savehist(nf, x, xhist, f, fhist, cstrv, chist, constr, conhist)
            # Save X, F, CONSTR, CSTRV into the filter.
            nfilt, cfilt, ffilt, xfilt, confilt = savefilt(cstrv, ctol, cweight, f, x, nfilt, cfilt, ffilt, xfilt, constr, confilt)

            # Evaluate ACTREM, which is the actual reduction in the merit function
            actrem = (fval[num_vars] + cpen * cval[num_vars]) - (f + cpen * cstrv)

            # Calculate the reduction  ratio by redrat, which hands inf/nan carefully
            ratio = redrat(actrem, prerem, eta1)

            # Update DELTA. After this, DELTA < DNORM may hold.
            # N.B.:
            # 1. Powell's code uses RHO as the trust-region radius and updates it as follows:
            #    Reduce RHO to GAMMA1*RHO if ADEQUATE_GEO is TRUE and either SHORTD is TRUE or RATIO < ETA1,
            #    and then revise RHO to RHOEND if its new value is not more than GAMMA3*RHOEND; RHO remains
            #    unchanged in all other cases; in particular, RHO is never increased.
            # 2. Our implementation uses DELTA as the trust-region radius, while using RHO as a lower
            #    bound for DELTA. DELTA is updated in a way that is typical for trust-region methods, and
            #    it is revised to RHO if its new value is not more than GAMMA3*RHO. RHO reflects the current
            #    resolution of the algorithm; its update is essentially the same as the update of RHO in
            #    Powell's code (see the definition of REDUCE_RHO below). Our implementation aligns with
            #    UOBYQA/NEWUOA/BOBYQA/LINCOA and improves the performance of COBYLA.
            # 3. The same as Powell's code, we do not reduce RHO unless ADEQUATE_GEO is TRUE. This is
            #    also how Powell updated RHO in UOBYQA/NEWUOA/BOBYQA/LINCOA. What about we also use
            #    ADEQUATE_GEO == TRUE as a prerequisite for reducing DELTA? The argument would be that the
            #    bad (small) value of RATIO may be because of a bad geometry (and hence a bad model) rather
            #    than an improperly large DELTA, and it might be good to try improving the geometry first
            #    without reducing DELTA. However, according to a test on 20230206, it does not improve the
            #    performance if we skip the update of DELTA when ADEQUATE_GEO is FALSE and RATIO < 0.1.
            #    Therefore, we choose to update DELTA without checking ADEQUATE_GEO.

            delta = trrad(delta, dnorm, eta1, eta2, gamma1, gamma2, ratio)
            if delta <= gamma3*rho:
                delta = rho  # Set delta to rho when it is close to or below.

            # Is the newly generated X better than the current best point?
            ximproved = actrem > 0  # If ACTREM is NaN, then XIMPROVED should and will be False

            # Set JDROP_TR to the index of the vertex to be replaced with X. JDROP_TR = 0 means there
            # is no good point to replace, and X will not be included into the simplex; in this case,
            # the geometry of the simplex likely needs improvement, which will be handled below.
            jdrop_tr = setdrop_tr(ximproved, d, delta, rho, sim, simi)

            # Update SIM, SIMI, FVAL, CONMAT, and CVAL so that SIM[:, JDROP_TR] is replaced with D.
            # UPDATEXFC does nothing if JDROP_TR = 0, as the algorithm decides to discard X.
            sim, simi, fval, conmat, cval, subinfo = updatexfc(jdrop_tr, constr, cpen, cstrv, d, f, conmat, cval, fval, sim, simi)
            # Check whether to break due to dmaging rounding in UPDATEXFC
            if subinfo == DAMAGING_ROUNDING:
                info = subinfo
                break  # Better action to take? Geometry step, or a RESCUE as in BOBYQA?

            # Check whether to break due to maxfun, ftarget, etc.
            subinfo = checkbreak(maxfun, nf, cstrv, ctol, f, ftarget, x)
            if subinfo != INFO_DEFAULT:
                info = subinfo
                break
        # End of if SHORTD or TRFAIL. The normal trust-region calculation ends.
    
        # Before the next trust-region iteration, we possibly improve the geometry of the simplex or
        # reduce RHO according to IMPROVE_GEO and REDUCE_RHO. Now we decide these indicators.
        # N.B.: We must ensure that the algorithm does not set IMPROVE_GEO = True at infinitely many
        # consecutive iterations without moving SIM[:, NUM_VARS] or reducing RHO. Otherwise, the algorithm
        # will get stuck in repetitive invocations of GEOSTEP. This is ensured by the following facts:
        # 1. If an iteration sets IMPROVE_GEO to True, it must also reduce DELTA or set DELTA to RHO.
        # 2. If SIM[:, NUM_VARS] and RHO remain unchanged, then ADEQUATE_GEO will become True after at
        # most NUM_VARS invocations of GEOSTEP.

        # BAD_TRSTEP: Is the last trust-region step bad?
        bad_trstep = shortd or trfail or ratio <= 0 or jdrop_tr == None
        # IMPROVE_GEO: Should we take a geometry step to improve the geometry of the interpolation set?
        improve_geo = bad_trstep and not adequate_geo
        # REDUCE_RHO: Should we enhance the resolution by reducing rho?
        reduce_rho = bad_trstep and adequate_geo and max(delta, dnorm) <= rho

        # COBYLA never sets IMPROVE_GEO and REDUCE_RHO to True simultaneously.
        # assert not (IMPROVE_GEO and REDUCE_RHO), 'IMPROVE_GEO or REDUCE_RHO are not both TRUE, COBYLA'
    
        # If SHORTD or TRFAIL is True, then either IMPROVE_GEO or REDUCE_RHO is True unless ADEQUATE_GEO
        # is True and max(DELTA, DNORM) > RHO.
        assert not (shortd or trfail) or (improve_geo or reduce_rho or (adequate_geo and max(delta, dnorm) > rho)), \
            'If SHORTD or TRFAIL is TRUE, then either IMPROVE_GEO or REDUCE_RHO is TRUE unless ADEQUATE_GEO is TRUE and MAX(DELTA, DNORM) > RHO'

        # Comments on BAD_TRSTEP:
        # 1. Powell's definition of BAD_TRSTEP is as follows. The one used above seems to work better,
        # especially for linearly constrained problems due to the factor TENTH (= ETA1).
        # !bad_trstep = (shortd .or. actrem <= 0 .or. actrem < TENTH * prerem .or. jdrop_tr == 0)
        # Besides, Powell did not check PREREM > 0 in BAD_TRSTEP, which is reasonable to do but has
        # little impact upon the performance.
        # 2. NEWUOA/BOBYQA/LINCOA would define BAD_TRSTEP, IMPROVE_GEO, and REDUCE_RHO as follows. Two
        # different thresholds are used in BAD_TRSTEP. It outperforms Powell's version.
        # !bad_trstep = (shortd .or. trfail .or. ratio <= eta1 .or. jdrop_tr == 0)
        # !improve_geo = bad_trstep .and. .not. adequate_geo
        # !bad_trstep = (shortd .or. trfail .or. ratio <= 0 .or. jdrop_tr == 0)
        # !reduce_rho = bad_trstep .and. adequate_geo .and. max(delta, dnorm) <= rho
        # 3. Theoretically, JDROP_TR > 0 when ACTREM > 0 (guaranteed by RATIO > 0). However, in Powell's
        # implementation, JDROP_TR may be 0 even RATIO > 0 due to NaN. The modernized code has rectified
        # this in the function SETDROP_TR. After this rectification, we can indeed simplify the
        # definition of BAD_TRSTEP by removing the condition JDROP_TR == 0. We retain it for robustness.

        # Comments on REDUCE_RHO:
        # When SHORTD is TRUE, UOBYQA/NEWUOA/BOBYQA/LINCOA all set REDUCE_RHO to TRUE if the recent
        # models are sufficiently accurate according to certain criteria. See the paragraph around (37)
        # in the UOBYQA paper and the discussions about Box 14 in the NEWUOA paper. This strategy is
        # crucial for the performance of the solvers. However, as of 20221111, we have not managed to
        # make it work in COBYLA. As in NEWUOA, we recorded the errors of the recent models, and set
        # REDUCE_RHO to true if they are small (e.g., ALL(ABS(MODERR_REC) <= 0.1 * MAXVAL(ABS(A))*RHO) or
        # ALL(ABS(MODERR_REC) <= RHO**2)) when SHORTD is TRUE. It made little impact on the performance.


        # Since COBYLA never sets IMPROVE_GEO and REDUCE_RHO to TRUE simultaneously, the following
        # two blocks are exchangeable: IF (IMPROVE_GEO) ... END IF and IF (REDUCE_RHO) ... END IF.

        # Improve the geometry of the simplex by removing a point and adding a new one.
        # If the current interpolation set has acceptable geometry, then we skip the geometry step.
        # The code has a small difference from Powell's original code here: If the current geometry
        # is acceptable, then we will continue with a new trust-region iteration; however, at the
        # beginning of the iteration, CPEN may be updated, which may alter the pole point SIM(:, N+1)
        # by UPDATEPOLE; the quality of the interpolation point depends on SIM(:, N + 1), meaning
        # that the same interpolation set may have good or bad geometry with respect to different
        # "poles"; if the geometry turns out bad with the new pole, the original COBYLA code will
        # take a geometry step, but our code here will NOT do it but continue to take a trust-region
        # step. The argument is this: even if the geometry step is not skipped in the first place, the
        # geometry may turn out bad again after the pole is altered due to an update to CPEN; should
        # we take another geometry step in that case? If no, why should we do it here? Indeed, this
        # distinction makes no practical difference for CUTEst problems with at most 100 variables
        # and 5000 constraints, while the algorithm framework is simplified.
        if improve_geo and not assess_geo(delta, factor_alpha, factor_beta, sim, simi):
            # Before the geometry step, updatepole has been called either implicitly by UPDATEXFC or
            # explicitly after CPEN is updated, so that SIM[:, :NUM_VARS] is the optimal vertex.

            # Decide a vertex to drop from the simplex. It will be replaced with SIM[:, NUM_VARS] + D to
            # improve acceptability of the simplex. See equations (15) and (16) of the COBYLA paper.
            # N.B.: COBYLA never sets JDROP_GEO = num_vars.
            jdrop_geo = setdrop_geo(delta, factor_alpha, factor_beta, sim, simi)

            # The following JDROP_GEO comes from UOBYQA/NEWUOA/BOBYQA/LINCOA. It performs poorly!
            # jdrop_geo = np.argmax(np.sum(sim[:, :num_vars]**2, axis=0), axis=0)

            # JDROP_GEO is between 0 and NUM_VARS unless SIM and SIMI contain NaN, which should not happen
            # at this point unless there is a bug. Nevertheless, for robustness, we include the
            # following instruction to break when JDROP_GEO == None (if JDROP_GEO does become None, then a
            # TypeError will occur if we continue, as JDROP_GEO will be used as an index of arrays.)
            if jdrop_geo is None:
                info = DAMAGING_ROUNDING
                break

            # Calculate the geometry step D.
            # In NEWUOA, GEOSTEP takes DELBAR = max(min(0.1 * sqrt(max(DISTSQ)), 0.% * DELTA), RHO)
            # rather than DELTA. This should not be done here, because D should improve the geometry of
            # the simplex when SIM[:, JDROP] is replaced with D; the quality of the geometry is defined
            # by DELTA instead of DELBAR as in (14) of the COBYLA paper. See GEOSTEP for more detail.
            d = geostep(jdrop_geo, cpen, conmat, cval, delta, fval, factor_gamma, simi)

            x = sim[:, num_vars] + d
            # Evaluate the objective and constraints at X, taking care of possible inf/nan values.
            f, constr, cstrv = evaluate(calcfc, x)
            nf += 1

            # Print a message about the function/constraint evaluation accoring to iprint
            # TODO: Implement me
            # fmsg(solver, iprint, nf, f, x, cstrv, constr)
            # Save X, F, CONSTR, CSTRV into the history.
            # TODO: Implement me
            # savehist(nf, x, xhist, f, fhist, cstrv, chist, constr, conhist)
            # Save X, F, CONSTR, CSTRV into the filter.
            nfilt, cfilt, ffilt, xfilt, confilt = savefilt(cstrv, ctol, cweight, f, x, nfilt, cfilt, ffilt, xfilt, constr, confilt)

            subinfo = updatexfc(jdrop_geo, constr, cpen, cstrv, d, f, conmat, cval, fval, sim, simi)
            if subinfo == DAMAGING_ROUNDING:
                info = subinfo
                break

            # Check whether to break due to maxfun, ftarget, etc.
            subinfo = checkbreak(maxfun, nf, cstrv, ctol, f, ftarget, x)
            if subinfo != INFO_DEFAULT:
                info = subinfo
                break
        # end of if improve_geo. The procedure of improving the geometry ends.

        # The calculations with the current RHO are complete. Enhance the resolution of the algorithm
        # by reducing RHO; update DELTA and CPEN at the same time.
        if reduce_rho:
            if rho <= rhoend:
                info = SMALL_TR_RADIUS
                break
            delta = max(0.5 * rho, redrho(rho, rhoend))
            rho = redrho(rho, rhoend)
            # THe second (out of two) updates of CPEN, where CPEN decreases or remains the same.
            # Powell's code: cpen = min(cpen, fcratio(fval, conmat)), which may set CPEN to 0.
            cpen = max(cpenmin, min(cpen, fcratio(conmat, fval)))
            # Print a message about the reduction of rho according to iprint
            # TODO: implement me!
            #call rhomsg(solver, iprint, nf, fval(n + 1), rho, sim(:, n + 1), cval(n + 1), conmat(:, n + 1), cpen)
            conmat, cval, fval, sim, simi, subinfo = updatepole(cpen, conmat, cval, fval, sim, simi)
            # Check whether to break due to damaging rounding detected in updatepole
            if subinfo == DAMAGING_ROUNDING:
                info = subinfo
                break  # Better action to take? Geometry step, or simply continue?
        # End of if reduce_rho. The procedure of reducing RHO ends.
    # End of for loop. The iterative procedure ends

    # Return from the calculation, after trying the last trust-region step if it has not been tried yet.
    if info == SMALL_TR_RADIUS and shortd and nf < maxfun:
        # Zaikun 20230615: UPDATEXFC or UPDATEPOLE is not called since the last trust-region step. Hence
        # SIM[:, NUM_VARS] remains unchanged. Otherwise SIM[:, NUM_VARS] + D would not make sense.
        x = sim[:, num_vars] + d
        f, constr, cstrv = evaluate(calcfc, x)
        nf += 1
        # TODO: msg, savehist
        nfilt, cfilt, ffilt, xfilt, confilt = savefilt(cstrv, ctol, cweight, f, x, nfilt, cfilt, ffilt, xfilt, constr, confilt)

    # Return the best calculated values of the variables
    # N.B.: SELECTX and FINDPOLE choose X by different standards, one cannot replace the other.
    kopt = selectx(ffilt[:nfilt], cfilt[:nfilt], max(cpen, cweight), ctol)
    x = xfilt[:, kopt]
    f = ffilt[kopt]
    constr = confilt[:, kopt]
    cstrv = cfilt[kopt]

    # Arrange CHIST, CONHIST, FHIST, and XHIST so that they are in the chronological order.
    # TODO: Implement me
    # call rangehist(nf, xhist, fhist, chist, conhist)

    # Print a return message according to IPRINT.
    # TODO: Implement me
    #call retmsg(solver, info, iprint, nf, f, x, cstrv, constr)
    return x, f, nf, info

        

def getcpen(conmat, cpen, cval, delta, fval, rho, sim, simi):
    '''
    This function gets the penalty parameter CPEN so that PREREM = PREREF + CPEN * PREREC > 0.
    See the discussions around equation (9) of the COBYLA paper.
    '''

    num_constraints = conmat.shape[0]
    num_vars = sim.shape[0]

    A = np.zeros((num_vars, num_constraints + 1))

    # Initialize INFO, PREREF, and PREREC, which are needed in the postconditions
    info = INFO_DEFAULT
    preref = 0
    prerec = 0

    # Increase CPEN if neccessary to ensure PREREM > 0. Branch back for the next loop if this change
    # alters the optimal vertex of the current simplex. Note the following:
    # 1. In each loop, CPEN is changed only if PREREC > 0 > PREREF, in which case PREREM is guaranteed
    #    positive after the update. Note that PREREC >= 0 and max(PREREC, PREREF) > 0 in theory. If this
    #    holds numerically as well then CPEN is not changed only if PREREC = 0 or PREREF >= 0, in which
    #    case PREREM is currently positive, explaining why CPEN needs no update.
    # 2. Even without an upper bound for the loop counter, the loop can occur at most NUM_VARS+1 times. This
    #    is because the update of CPEN does not decrease CPEN, and hence it can make vertex J (J <= NUM_VARS) become
    #    the new optimal vertex only if CVAL[J] is less than CVAL[NUM_VARS], which can happen at most NUM_VARS times.
    #    See the paragraph below (9) in the COBYLA paper. After the "correct" optimal vertex is found,
    #    one more loop is needed to calculate CPEN, and hence the loop can occur at most NUM_VARS+1 times.
    for iter in range(num_vars + 1):
        # Switch the best vertex of the current simplex to SIM[:, NUM_VARS]
        conmat, cval, fval, sim, simi, info = updatepole(cpen, conmat, cval, fval, sim, simi)
        # Check whether to exit due to damaging rounding in UPDATEPOLE
        if info == DAMAGING_ROUNDING:
            break

        # Calculate the linear approximations to the objective and constraint functions, placing minus
        # the objective function gradient after the constraint gradients in the array A.
        A[:, :num_constraints] = ((conmat[:, :num_vars] - np.tile(conmat[:, num_vars], (num_vars, 1)).T)@simi).T
        A[:, num_constraints] = (fval[num_vars] - fval[:num_vars])@simi
        b = np.hstack((-conmat[:, num_vars], -fval[num_vars]))

        # Calculate the trust-region trial step D. Note that D does NOT depend on CPEN.
        d = trstlp(A, b, delta)

        # Predict the change to F (PREREF) and to the constraint violation (PREREC) due to D.
        prerec = cval[num_vars] - max(max(b[:num_constraints] - d@A[:, :num_constraints]), 0)
        preref = np.dot(d, A[:, num_constraints])  # Can be negative

        if not (prerec > 0 and preref < 0):  # PREREC <= 0 or PREREF >=0 or either is NaN
            break

        # Powell's code defines BARMU = -PREREF / PREREC, and CPEN is increased to 2*BARMU if and
        # only if it is currently less than 1.5*BARMU, a very "Powellful" scheme. In our
        # implementation, however, we seet CPEN directly to the maximum between its current value and
        # 2*BARMU while handling possible overflow. The simplifies the scheme without worsening the
        # performance of COBYLA.
        cpen = max(cpen, min(-2 * preref / prerec, REALMAX))

        if findpole(cpen, cval, fval) == num_vars:
            break

    return cpen