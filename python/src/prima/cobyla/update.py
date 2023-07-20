from prima.common.consts import DAMAGING_ROUNDING, INFO_DEFAULT
import numpy as np


def updatexfc(jdrop, constr, cpen, cstrv, d, f, conmat, cval, fval, sim, simi):
    '''
    This function revises the simplex by updating the elements of SIM, SIMI, FVAL, CONMAT, and CVAL
    '''

    num_constraints = len(constr)
    num_vars = sim.shape[0]

    # Do nothing when JDROP is None. This can only happen after a trust-region step.
    if jdrop is None:  # JDROP is None is impossible if the input is correct.
        return conmat, cval, fval, sim, simi, INFO_DEFAULT
    
    sim_old = sim
    simi_old = simi
    if jdrop < num_vars:
        sim[:, jdrop] = d
        simi_jdrop = simi[jdrop, :] / np.dot(simi[jdrop, :], d)
        simi -= np.outer(simi@d, simi_jdrop)
        simi[jdrop, :] = simi_jdrop
    else:  # jdrop == num_vars
        sim[:, num_vars] += d
        sim[:, :num_vars] -= np.tile(d, (num_vars, 1)).T
        simid = simi@d
        sum_simi = np.sum(simi, axis=0)
        simi += np.outer(simid, sum_simi / (1 - sum(simid)))

    # Check whether SIMI is a poor approximation to the inverse of SIM[:, :NUM_VARS]
    # Calculate SIMI from scratch if the current one is damaged by rounding errors.
    itol = 1
    erri = np.max(abs(simi@sim[:, :num_vars] - np.eye(num_vars)))  # np.max returns NaN if any input is NaN
    if erri > 0.1 * itol or np.isnan(erri):
        simi_test = np.linalg.inv(sim[:, :num_vars])
        erri_test = np.max(abs(simi_test@sim[:, :num_vars] - np.eye(num_vars)))
        if erri_test < erri or (np.isnan(erri) and not np.isnan(erri_test)):
            simi = simi_test
            erri = erri_test

    # If SIMI is satisfactory, then update FVAL, CONMAT, CVAL, and the pole position. Otherwise restore
    # SIM and SIMI, and return with INFO = DAMAGING_ROUNDING.
    if erri <= itol:
        fval[jdrop] = f
        conmat[:, jdrop] = constr
        cval[jdrop] = cstrv
        # Switch the best vertex to the pole position SIM[:, NUM_VARS] if it is not there already
        conmat, cval, fval, sim, simi, info = updatepole(cpen, conmat, cval, fval, sim, simi)
    else:
        info = DAMAGING_ROUNDING
        sim = sim_old
        simi = simi_old
    
    
    return sim, simi, fval, conmat, cval, info

def findpole(cpen, cval, fval):
    # This subroutine identifies the best vertex of the current simplex with respect to the merit
    # function PHI = F + CPEN * CSTRV.
    n = len(fval) - 1  # used for debugging which I have not implemented yet

    # Identify the optimal vertex of the current simplex
    jopt = len(fval) - 1
    phi = fval + cpen * cval
    phimin = min(phi)
    joptcandidate = np.argmin(phi)
    if phi[joptcandidate] < phi[jopt]:
        jopt = joptcandidate
    if cpen <= 0 and any(np.logical_and(cval < cval[jopt], phi <= phimin)):
        # jopt is the index of the minimum value of cval
        # on the set of cval values where phi <= phimin
        jopt = np.where(cval == min(cval[phi <= phimin]))[0][0]
    return jopt


def updatepole(cpen, conmat, cval, fval, sim, simi):
    #--------------------------------------------------------------------------------------------------!
    # This subroutine identifies the best vertex of the current simplex with respect to the merit
    # function PHI = F + CPEN * CSTRV, and then switch this vertex to SIM[:, NUM_VARS], which Powell called
    # the "pole position" in his comments. CONMAT, CVAL, FVAL, and SIMI are updated accordingly.
    #
    # N.B. 1: In precise arithmetic, the following two procedures produce the same results:
    # 1) apply UPDATEPOLE to SIM twice, first with CPEN = CPEN1 and then with CPEN = CPEN2;
    # 2) apply UPDATEPOLE to SIM with CPEN = CPEN2.
    # In finite-precision arithmetic, however, they may produce different results unless CPEN1 = CPEN2.
    #
    # N.B. 2: When JOPT == N+1, the best vertex is already at the pole position, so there is nothing to
    # switch. However, as in Powell's code, the code below will check whether SIMI is good enough to
    # work as the inverse of SIM(:, 1:N) or not. If not, Powell's code would invoke an error return of
    # COBYLB; our implementation, however, will try calculating SIMI from scratch; if the recalculated
    # SIMI is still of poor quality, then UPDATEPOLE will return with INFO = DAMAGING_ROUNDING,
    # informing COBYLB that SIMI is poor due to damaging rounding errors.
    #
    # N.B. 3: UPDATEPOLE should be called when and only when FINDPOLE can potentially returns a value
    # other than N+1. The value of FINDPOLE is determined by CPEN, CVAL, and FVAL, the latter two being
    # decided by SIM. Thus UPDATEPOLE should be called after CPEN or SIM changes. COBYLA updates CPEN at
    # only two places: the beginning of each trust-region iteration, and when REDRHO is called;
    # SIM is updated only by UPDATEXFC, which itself calls UPDATEPOLE internally. Therefore, we only
    # need to call UPDATEPOLE after updating CPEN at the beginning of each trust-region iteration and
    # after each invocation of REDRHO.

    num_constraints = conmat.shape[0]
    num_vars = sim.shape[0]
    info = INFO_DEFAULT

    # Identify the optimal vertex of the current simplex.
    jopt = findpole(cpen, cval, fval)

    # Switch the best vertex to the pole position SIM[:, NUM_VARS] if it is not there already and update
    # SIMI. Before the update, save a copy of SIM and SIMI. If the update is unsuccessful due to
    # damaging rounding errors, we restore them and return with INFO = DAMAGING_ROUNDING.
    sim_old = sim.copy()
    simi_old = simi.copy()
    if 0 <= jopt < num_vars:
        # Unless there is a bug in FINDPOLE it is guaranteed that JOPT >= 0
        # When JOPT == NUM_VARS, there is nothing to switch; in addition SIMI[JOPT, :] will be illegal.
        # fval[[jopt, -1]] = fval[[-1, jopt]]
        # conmat[:, [jopt, -1]] = conmat[:, [-1, jopt]] # Exchange CONMAT[:, JOPT] AND CONMAT[:, -1]
        # cval[[jopt, -1]] = cval[[-1, jopt]]
        sim[:, num_vars] += sim[:, jopt]
        sim_jopt = sim[:, jopt].copy()
        sim[:, jopt] = 0  # np.zeros(num_constraints)?
        sim[:, :num_vars] -= np.tile(sim_jopt, (num_vars, 1)).T
        # The above update is equivalent to multiplying SIM[:, :NUM_VARS] from the right side by a matrix whose
        # JOPT-th row is [-1, -1, ..., -1], while all the other rows are the same as those of the
        # identity matrix. It is easy to check that the inverse of this matrix is itself. Therefore,
        # SIMI should be updated by a multiplication with this matrix (i.e. its inverse) from the left
        # side, as is done in the following line. The JOPT-th row of the updated SIMI is minus the sum
        # of all rows of the original SIMI, whereas all the other rows remain unchanged.
        simi[jopt, :] = -np.sum(simi, axis=0)

    # Check whether SIMI is a poor approximation to the inverse of SIM[:, :NUM_VARS]
    # Calculate SIMI from scratch if the current one is damaged by rounding errors.
    erri = np.max(abs(simi@sim[:, :num_vars] - np.eye(num_vars)))  # np.max returns NaN if any input is NaN
    itol = 1
    if erri > 0.1 * itol or np.isnan(erri):
        simi_test = np.linalg.inv(sim[:, :num_vars])
        erri_test = np.max(abs(simi_test@sim[:, :num_vars] - np.eye(num_vars)))
        if erri_test < erri or (np.isnan(erri) and not np.isnan(erri_test)):
            simi = simi_test
            erri = erri_test


    # If SIMI is satisfactory, then update FVAL, CONMAT, and CVAL. Otherwiae restore SIM and SIMI, and
    # return with INFO = DAMAGING_ROUNDING.
    if erri <= itol:
        if 0 <= jopt < num_vars:
            fval[[jopt, num_vars]] = fval[[num_vars, jopt]]
            conmat[:, [jopt, num_vars]] = conmat[:, [num_vars, jopt]]
            cval[[jopt, num_vars]] = cval[[num_vars, jopt]]
    else:  # erri > itol or erri is NaN
        info = DAMAGING_ROUNDING
        sim = sim_old
        simi = simi_old

    return conmat, cval, fval, sim, simi, info