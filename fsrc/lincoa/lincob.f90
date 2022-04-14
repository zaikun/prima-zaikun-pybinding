module lincob_mod
!--------------------------------------------------------------------------------------------------!
! This module performs the major calculations of LINCOA.
!
! Coded by Zaikun ZHANG (www.zhangzk.net) based on Powell's Fortran 77 code and the paper
!
! M. J. D. Powell, On fast trust region methods for quadratic models with linear constraints,
! Math. Program. Comput., 7:237--267, 2015
!
! Dedicated to late Professor M. J. D. Powell FRS (1936--2015).
!
! Started: February 2022
!
! Last Modified: Friday, April 15, 2022 AM12:27:12
!--------------------------------------------------------------------------------------------------!

implicit none
private
public :: lincob


contains


subroutine lincob(calfun, iprint, maxfilt, maxfun, npt, A_orig, amat, b_orig, bvec, eta1, eta2, &
    & ftarget, gamma1, gamma2, rhobeg, rhoend, x, nf, chist, cstrv, f, fhist, xhist, info)
!--------------------------------------------------------------------------------------------------!
! This subroutine performs the actual calculations of LINCOA. The arguments IPRINT, MAXFILT, MAXFUN,
! MAXHIST, NPT, CTOL, CWEIGHT, ETA1, ETA2, FTARGET, GAMMA1, GAMMA2, RHOBEG, RHOEND, X, NF, F, XHIST,
! FHIST, CHIST, CSTRV and INFO are identical to the corresponding arguments in subroutine LINCOA.
!--------------------------------------------------------------------------------------------------!

! Generic models
use, non_intrinsic :: consts_mod, only : RP, IK, ZERO, ONE, TWO, HALF, TENTH, MIN_MAXFILT, DEBUGGING
use, non_intrinsic :: debug_mod, only : assert
use, non_intrinsic :: evaluate_mod, only : evaluate
use, non_intrinsic :: history_mod, only : savehist, rangehist
use, non_intrinsic :: infnan_mod, only : is_nan, is_posinf
use, non_intrinsic :: info_mod, only : NAN_INF_X, NAN_INF_F, NAN_MODEL, FTARGET_ACHIEVED, INFO_DFT, &
    & MAXFUN_REACHED, DAMAGING_ROUNDING!, SMALL_TR_RADIUS, MAXTR_REACHED
use, non_intrinsic :: linalg_mod, only : matprod, inprod, maximum, eye, r1update
use, non_intrinsic :: pintrf_mod, only : OBJ
use, non_intrinsic :: powalg_mod, only : calquad, omega_col, omega_mul

! Solver-specific modules
use, non_intrinsic :: geometry_mod, only : geostep
use, non_intrinsic :: initialize_mod, only : initialize
use, non_intrinsic :: shiftbase_mod, only : shiftbase
use, non_intrinsic :: trustregion_mod, only : trstep
use, non_intrinsic :: update_mod, only : update

implicit none

! Inputs
procedure(OBJ) :: calfun  ! N.B.: INTENT cannot be specified if a dummy procedure is not a POINTER
integer(IK), intent(in) :: iprint
integer(IK), intent(in) :: maxfilt
integer(IK), intent(in) :: maxfun
integer(IK), intent(in) :: npt
real(RP), intent(in) :: A_orig(:, :)  ! A_ORIG(N, M) ; Better names? necessary?
real(RP), intent(in) :: amat(:, :)  ! AMAT(N, M) ; Better names? necessary?
real(RP), intent(in) :: b_orig(:) ! B_ORIG(M) ; Better names? necessary?
real(RP), intent(in) :: bvec(:)  ! BVEC(M) ; Better names? necessary?
real(RP), intent(in) :: eta1
real(RP), intent(in) :: eta2
real(RP), intent(in) :: ftarget
real(RP), intent(in) :: gamma1
real(RP), intent(in) :: gamma2
real(RP), intent(in) :: rhobeg
real(RP), intent(in) :: rhoend

! In-outputs
real(RP), intent(inout) :: x(:)  ! X(N)

! Outputs
integer(IK), intent(out) :: info
integer(IK), intent(out) :: nf
real(RP), intent(out) :: chist(:)  ! CHIST(MAXCHIST)
real(RP), intent(out) :: cstrv
real(RP), intent(out) :: f
real(RP), intent(out) :: fhist(:)  ! FHIST(MAXFHIST)
real(RP), intent(out) :: xhist(:, :)  ! XHIST(N, MAXXHIST)

! Local variables
character(len=*), parameter :: srname = 'LINCOB'
integer(IK) :: iact(size(bvec))
integer(IK) :: m
integer(IK) :: maxchist
integer(IK) :: maxfhist
integer(IK) :: maxhist
integer(IK) :: maxxhist
integer(IK) :: n
real(RP) :: b(size(bvec))
real(RP) :: bmat(size(x), npt + size(x))
real(RP) :: fval(npt)
real(RP) :: gopt(size(x))
real(RP) :: hq(size(x), size(x))
real(RP) :: pq(npt)
real(RP) :: pqw(npt + size(x))  ! Note that the size is npt + N instead of npt; Isn't it VLAG in NEWUOA??? Better name?
real(RP) :: qfac(size(x), size(x))
real(RP) :: rescon(size(bvec))
real(RP) :: rfac(size(x), size(x))
real(RP) :: xsxpt(2 * npt)
real(RP) :: step(size(x))
real(RP) :: xbase(size(x))
real(RP) :: xnew(size(x))
real(RP) :: xopt(size(x))
real(RP) :: xpt(size(x), npt)
real(RP) :: xsav(size(x))
real(RP) :: zmat(npt, npt - size(x) - 1)
real(RP) :: del, delsav, delta, dffalt, diff, &
&        distsq, xdsq(npt), fopt, fsave, ratio,     &
&        rho, snorm, ssq, summ, temp, vqalt,   &
&        qred, xdiff
integer(IK) :: i, idz, ifeas, imprv, itest, j, k,    &
&           knew, kopt, ksave, nact,      &
&           nvala, nvalb, ngetact
real(RP) :: w(max(int(size(bvec), IK) + 3_IK * int(size(x), IK), 2_IK * int(size(bvec), IK) + int(size(x), IK), 2_IK * npt))



! Sizes.
m = int(size(bvec), kind(m))
n = int(size(x), kind(n))
maxxhist = int(size(xhist, 2), kind(maxxhist))
maxfhist = int(size(fhist), kind(maxfhist))
maxchist = int(size(chist), kind(maxchist))
maxhist = int(max(maxxhist, maxfhist, maxchist), kind(maxhist))

! Preconditions
if (DEBUGGING) then
    call assert(abs(iprint) <= 3, 'IPRINT is 0, 1, -1, 2, -2, 3, or -3', srname)
    call assert(m >= 0, 'M >= 0', srname)
    call assert(n >= 1, 'N >= 1', srname)
    call assert(npt >= n + 2, 'NPT >= N+2', srname)
    call assert(maxfun >= npt + 1, 'MAXFUN >= NPT+1', srname)
    call assert(size(A_orig, 1) == n .and. size(A_orig, 2) == m, 'SIZE(A_ORIG) == [N, M]', srname)
    call assert(size(b_orig) == m, 'SIZE(B_ORIG) == M', srname)
    call assert(size(amat, 1) == n .and. size(amat, 2) == m, 'SIZE(AMAT) == [N, M]', srname)
    call assert(rhobeg >= rhoend .and. rhoend > 0, 'RHOBEG >= RHOEND > 0', srname)
    call assert(eta1 >= 0 .and. eta1 <= eta2 .and. eta2 < 1, '0 <= ETA1 <= ETA2 < 1', srname)
    call assert(gamma1 > 0 .and. gamma1 < 1 .and. gamma2 > 1, '0 < GAMMA1 < 1 < GAMMA2', srname)
    call assert(maxfilt >= min(MIN_MAXFILT, maxfun) .and. maxfilt <= maxfun, &
        & 'MIN(MIN_MAXFILT, MAXFUN) <= MAXFILT <= MAXFUN', srname)
    call assert(maxhist >= 0 .and. maxhist <= maxfun, '0 <= MAXHIST <= MAXFUN', srname)
    call assert(size(xhist, 1) == n .and. maxxhist * (maxxhist - maxhist) == 0, &
        & 'SIZE(XHIST, 1) == N, SIZE(XHIST, 2) == 0 or MAXHIST', srname)
    call assert(maxfhist * (maxfhist - maxhist) == 0, 'SIZE(FHIST) == 0 or MAXHIST', srname)
    call assert(maxchist * (maxchist - maxhist) == 0, 'SIZE(CHIST) == 0 or MAXHIST', srname)
end if

qfac = eye(n)
rfac = ZERO
!
!     The arguments N, NPT, M, X, RHOBEG, RHOEND, IPRINT and MAXFUN are
!       identical to the corresponding arguments in SUBROUTINE LINCOA.
!     AMAT is a matrix whose columns are the constraint gradients, scaled
!       so that they have unit length.
!     B contains on entry the right hand sides of the constraints, scaled
!       as above, but later B is modified for variables relative to XBASE.
!     XBASE holds a shift of origin that should reduce the contributions
!       from rounding errors to values of the model and Lagrange functions.
!     XPT contains the interpolation point coordinates relative to XBASE.
!     FVAL holds the values of F at the interpolation points.
!     XSAV holds the best feasible vector of variables so far, without any
!       shift of origin.
!     XOPT is set to XSAV-XBASE, which is the displacement from XBASE of
!       the feasible vector of variables that provides the least calculated
!       F so far, this vector being the current trust region centre.
!     GOPT holds the gradient of the quadratic model at XSAV = XBASE+XOPT.
!     HQ holds the explicit second derivatives of the quadratic model.
!     PQ contains the parameters of the implicit second derivatives of the
!       quadratic model.
!     BMAT holds the last N columns of the big inverse matrix H.
!     ZMAT holds the factorization of the leading NPT by NPT submatrix
!       of H, this factorization being ZMAT times Diag(DZ) times ZMAT^T,
!       where the elements of DZ are plus or minus ONE, as specified by IDZ.
!     NDIM is the second dimension of BMAT and has the value NPT+N.
!     STEP is employed for trial steps from XOPT. It is also used for working
!       space when XBASE is shifted and in PRELIM.
!     XSXPT is reserved for the scalar products XOPT^T XPT(K,.), K=1,2,...,NPT,
!       followed by STEP^T XPT(K,.), K=1,2,...,NPT.
!     XNEW is the displacement from XBASE of the vector of variables for
!       the current calculation of F, except that SUBROUTINE TRSTEP uses it
!       for working space.
!     IACT is an integer array for the indices of the active constraints.
!     RESCON holds useful information about the constraint residuals. Every
!       nonnegative RESCON(J) is the residual of the J-th constraint at the
!       current trust region centre. Otherwise, if RESCON(J) is negative, the
!       J-th constraint holds as a strict inequality at the trust region
!       centre, its residual being at least |RESCON(J)|; further, the value
!       of |RESCON(J)| is at least the current trust region radius DELTA.
!     QFAC is the orthogonal part of the QR factorization of the matrix of
!       active constraint gradients, these gradients being ordered in
!       accordance with IACT. When NACT is less than N, columns are added
!       to QFAC to complete an N by N orthogonal matrix, which is important
!       for keeping calculated steps sufficiently close to the boundaries
!       of the active constraints.
!     RFAC is the upper triangular part of this QR factorization, beginning
!       with the first diagonal element, followed by the two elements in the
!       upper triangular part of the second column and so on.
!     PQW is used for working space, mainly for storing second derivative
!       coefficients of quadratic functions. Its length is NPT+N.
!     The array W is also used for working space. The required number of
!       elements, namely MAX[M+3*N,2*M+N,2*NPT], is set in LINCOA.
!
!     Set some constants.
!
b = bvec
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Zaikun 15-08-2019
! See the comments below line number 210
imprv = 0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     Set the elements of XBASE, XPT, FVAL, XSAV, XOPT, GOPT, HQ, PQ, BMAT,
!       ZMAT and XSXPT for the first iteration. An important feature is that,
!       if the interpolation point XPT(K,.) is not feasible, where K is any
!       integer from [1,NPT], then a change is made to XPT(K,.) if necessary
!       so that the constraint violation is at least 0.2*RHOBEG. Also KOPT
!       is set so that XPT(KOPT,.) is the initial trust region centre.
!
call initialize(calfun, iprint, A_orig, amat, b_orig, ftarget, rhobeg, x, b, &
    & idz, kopt, nf, bmat, chist, cstrv, f, fhist, fval, gopt, hq, pq, rescon, xsxpt, &
    & step, pqw, xbase, xhist, xopt, xpt, xsav, zmat)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if (is_nan(f) .or. is_posinf(f)) then
    fopt = fval(kopt)
    info = NAN_INF_F
    goto 600
end if
! Note that we should NOT compare F and FTARGET, because X may not be feasible.
if (fval(kopt) <= ftarget) then
    f = fval(kopt)
    x = xsav
    info = FTARGET_ACHIEVED
    goto 616
end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     Begin the iterative procedure.
!
nf = npt
fopt = fval(kopt)
rho = rhobeg
delta = rho
ifeas = 0
nact = 0
itest = 3
10 knew = 0
nvala = 0
nvalb = 0

20 continue

fsave = fopt

! Shift XBASE if XOPT may be too far from XBASE.
if (sum(xopt**2) >= 1.0E4_RP * delta**2) then
    xsxpt(1:npt) = ZERO
    b = b - matprod(xopt, amat)
    call shiftbase(xbase, xopt, xpt, zmat, bmat, pq, hq, idz)
end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Zaikun 21-03-2020: Exit if BMAT or ZMAT contians NaN
if (is_nan(sum(abs(bmat)) + sum(abs(zmat)))) then
    info = NAN_MODEL
    goto 600
end if

!     In the case KNEW=0, generate the next trust region step by calling
!       TRSTEP, where SNORM is the current trust region radius initially.
!       The final value of SNORM is the length of the calculated step,
!       except that SNORM is ZERO on return if the projected gradient is
!       unsuitable for starting the conjugate gradient iterations.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Zaikun 2019-08-29: For ill-conditiONEd problems, NaN may occur in the
! models. In such a case, we terminate the code. Otherwise, the behavior
! of TRSTEP or QMSTEP is not predictable, and Segmentation Fault or
! infinite cycling may happen. This is because any equality/inequality
! comparison involving NaN returns FALSE, which can lead to unintended
! behavior of the code, including uninitialized indices, which can lead
! to segmentation faults.
if (is_nan(sum(abs(gopt)) + sum(abs(hq)) + sum(abs(pq)))) then
    info = NAN_MODEL
    goto 600
end if

delsav = delta
ksave = knew
if (knew == 0) then
    call trstep(amat, delta, gopt, hq, pq, rescon, xpt, iact, nact, qfac, rfac, ngetact, snorm, step)
!
!     A trust region step is applied whenever its length, namely SNORM, is at
!       least HALF*DELTA. It is also applied if its length is at least 0.1999
!       times DELTA and if a line search of TRSTEP has caused a change to the
!       active set. Otherwise there is a branch below to label 530 or 560.
!
    temp = HALF * delta
    if (ngetact > 1) temp = 0.1999_RP * delta
    if (snorm <= temp) then
        delta = HALF * delta
        if (delta <= 1.4_RP * rho) delta = rho
        nvala = nvala + 1
        nvalb = nvalb + 1
        temp = snorm / rho
        if (delsav > rho) temp = ONE
        if (temp >= HALF) nvala = 0
        if (temp >= TENTH) nvalb = 0
        if (delsav > rho) goto 530
        if (nvala < 5 .and. nvalb < 3) goto 530
        if (snorm > ZERO) ksave = -1
        goto 560
    end if
    nvala = 0
    nvalb = 0

!     Alternatively, KNEW is positive. Then the model step is calculated
!       within a trust region of radius DEL, after setting the gradient at
!       XBASE and the second derivative parameters of the KNEW-th Lagrange
!       function in W(1) to W(N) and in PQW(1) to PQW(NPT), respectively.
!
else
    del = max(TENTH * delta, rho)

    if (is_nan(sum(abs(bmat(:, knew))))) then  ! Necessary?
        info = NAN_MODEL
        goto 600
    end if

    pqw(1:npt) = omega_col(idz, zmat, knew)
    if (is_nan(sum(abs(pqw(1:npt))))) then
        info = NAN_MODEL
        goto 600
    end if

    call geostep(iact, knew, kopt, nact, amat, del, bmat(:, knew), pqw(1:npt), qfac, rescon, xopt, xpt, ifeas, step)
end if

!
!     Set VQUAD to the change to the quadratic model when the move STEP is
!       made from XOPT. If STEP is a trust region step, then VQUAD should be
!       negative. If it is nonnegative due to rounding errors in this case,
!       there is a branch to label 530 to try to improve the model.
!-------------------------------------------------------------------------------------------!
! Zaikun 20220405: The improvement does not exist in NEWUOA/BOBYQA, which should do the same.
!-------------------------------------------------------------------------------------------!
!
xsxpt(npt + 1:2 * npt) = matprod(step, xpt)
qred = calquad(step, gopt, hq, pq, xpt)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Zaikun 15-08-2019
! Although very rarely, with the original code, an infinite loop can occur
! in the following scenario.
! Suppose that, at an certain iteration,
! KNEW = 0, SNORM > 0.5*DELTA > RHO, VQUAD >= 0, and
! summ_{K=1}^NPT ||XPT(K,:)-XOPT(:)||^2 < DELTA^2
! (i.e., DELTA is large and SNORM is not small, yet VQUAD >= 0 due to
! rounding errors and XPT are not far from XOPT).
! Then the program will goto 530 and then goto 20, where XBASE may be
! shifted to the current best point, in the hope of reducing rounding
! errors and 'improve' the model. Afterwards, another trust region step
! is produced by the 'improved' model. Note that DELTA remains unchanged
! in this process. If the new trust region step turns out to satisfy
! SNORM > 0.5*DELTA and VQUAD >= 0 again (i.e., the 'improved' model
! still suffers from rounding errors), then the program will goto 530
! and then goto 20, where shifting will not happen because either XBASE
! was already shifted to the current best point in last step, or XBASE
! is close to the current best point. Consequently, the model will
! remain unchanged, and produce the same trust region step again. This
! leads to an infinite loop.
! The infinite loop did happen when the MATLAB interface was applied to
! min atan(x+100) s.t. x<=-99 (x0=-99, npt=3, rhobeg=1, rhoend=1e-6).
! The problem does not exist in NEWUOA or BOBYQA, where the program will
! exit immediately when VQUAD >= 0.
! To prevent such a loop, here we use IMPRV to record whether the path
! 530 --> 20 has already happened for last trust region step. IMPRV=1
! implies that last trust region step satisfies VQUAD >= 0 and followed
! 530 --> 20. With IMPRV=1, if VQUAD is again nonnegative for the new trust
! region step, we should not goto 530 but goto 560, where IMPRV will be
! set to 0 and DELTA will be reduced. Otherwise, an infinite loop would happen.
!      IF (KSAVE .EQ. 0 .AND. VQUAD .GE. ZERO) GOTO 530
if (ksave == 0 .and. .not. (qred > ZERO)) then
    if (imprv == 1) then
        goto 560
    else
        imprv = 1
        goto 530
    end if
else
    imprv = 0
end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     Calculate the next value of the objective function. The difference
!       between the actual new value of F and the value predicted by the
!       model is recorded in DIFF.
!
220 nf = nf + 1
if (nf > maxfun) then
    nf = nf - 1
    info = MAXFUN_REACHED
    goto 600
end if
xnew = xopt + step
x = xbase + xnew
xdiff = sqrt(sum((x - xsav)**2))
if (ksave == -1) xdiff = rho
if (.not. (xdiff > TENTH * rho .and. xdiff < delta + delta)) then
    ifeas = 0
    info = DAMAGING_ROUNDING
    goto 600
end if
if (ksave <= 0) ifeas = 1
f = real(ifeas, RP)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if (is_nan(sum(abs(x)))) then
    f = sum(x)  ! Set F to NaN
    if (nf == 1) then
        fopt = f
        xopt = ZERO
    end if
    info = NAN_INF_X
    goto 600
end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
call evaluate(calfun, x, f)
cstrv = maximum([ZERO, matprod(x, A_orig) - b_orig])
call savehist(nf, x, xhist, f, fhist, cstrv, chist)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     By Tom (on 04-06-2019):
if (is_nan(f) .or. is_posinf(f)) then
    if (nf == 1) then
        fopt = f
        xopt = ZERO
    end if
    info = NAN_INF_F
    goto 600
end if
if (ksave == -1) then
    info = INFO_DFT !!??
    goto 600
end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
diff = f - fopt + qred
!
!     If X is feasible, then set DFFALT to the difference between the new
!       value of F and the value predicted by the alternative model.
!
if (ifeas == 1 .and. itest < 3) then
    w(1:npt) = fval - fval(kopt)
    pqw(1:npt) = omega_mul(idz, zmat, w(1:npt))
    !-----------------------------------------------------------------------------------------!
    ! The following evaluates Q_alt(XOPT+STEP) - Q_alt(XOPT), which should be done by CALQUAD.
    vqalt = ZERO
    do k = 1, npt
        vqalt = vqalt + inprod(step, bmat(:, k)) * w(k)
        vqalt = vqalt + pqw(k) * xsxpt(npt + k) * (HALF * xsxpt(npt + k) + xsxpt(k))
    end do
    !-----------------------------------------------------------------------------------------!
    dffalt = f - fopt - vqalt
end if
if (itest == 3) then
    dffalt = diff
    itest = 0
end if
!
!     Pick the next value of DELTA after a trust region step.
!
if (ksave == 0) then
    ratio = (fopt - f) / qred
    if (ratio <= TENTH) then
        delta = HALF * delta
    else if (ratio <= 0.7_RP) then
        delta = max(HALF * delta, snorm)
    else
        temp = sqrt(TWO) * delta
        delta = max(HALF * delta, snorm + snorm)
        delta = min(delta, temp)
    end if
    if (delta <= 1.4_RP * rho) delta = rho
end if
!
!     Update BMAT, ZMAT and IDZ, so that the KNEW-th interpolation point
!       can be moved. If STEP is a trust region step, then KNEW is ZERO at
!       present, but a positive value is picked by subroutine UPDATE.
!
call update(kopt, step, xpt, idz, knew, bmat, zmat, pqw)
if (knew == 0) then
    info = DAMAGING_ROUNDING
    goto 600
end if

! Zaikun 19-03-2020: Exit if BMAT or ZMAT contians NaN
if (is_nan(sum(abs(bmat)) + sum(abs(zmat)))) then
    info = NAN_MODEL
    goto 600
end if

!
!     If ITEST is increased to 3, then the next quadratic model is the
!       ONE whose second derivative matrix is least subject to the new
!       interpolation conditions. Otherwise the new model is constructed
!       by the symmetric Broyden method in the usual way.
!
if (ifeas == 1) then
    itest = itest + 1
    if (abs(dffalt) >= TENTH * abs(diff)) itest = 0
end if
!
!     Update the second derivatives of the model by the symmetric Broyden
!       method, using PQW for the second derivative parameters of the new
!       KNEW-th Lagrange function. The contribution from the old parameter
!       PQ(KNEW) is included in the second derivative matrix HQ. W is used
!       later for the gradient of the new KNEW-th Lagrange function.
!
if (itest < 3) then
    call r1update(hq, pq(knew), xpt(:, knew))
    pq(knew) = ZERO
    pqw(1:npt) = omega_col(idz, zmat, knew)
    pq = pq + diff * pqw(1:npt)
    w(1:n) = bmat(:, knew)
end if
!
!     Include the new interpolation point with the corresponding updates of
!       XSXPT. Also make the changes of the symmetric Broyden method to GOPT at
!       the old XOPT if ITEST is less than 3.
!
fval(knew) = f
xsxpt(knew) = xsxpt(kopt) + xsxpt(npt + kopt)
! XOPT'*XPT(:, KOPT) + STEP'*XPT(:, KOPT) = (XOPT+STEP)'*XPT(:, KOPT) = XNEW'*XPT(:, KOPT) = XNEW'*XOPT
!!xsxpt(knew) = inprod(xopt, xnew)

xpt(:, knew) = xnew
ssq = sum(step**2)
xsxpt(npt + knew) = xsxpt(npt + kopt) + ssq
! STEP'*XPT(:, KNEW) + STEP'*STEP = STEP'*[XPT(:, KOPT) + STEP] = STEP'*XNEW
!!xsxpt(npt + knew) = inprod(step, xnew)
if (itest < 3) then
    do k = 1, npt
        w(1:n) = w(1:n) + pqw(k) * xsxpt(k) * xpt(:, k)
    end do
    gopt = gopt + diff * w(1:n)
end if
!
!     Update FOPT, XSAV, XOPT, KOPT, RESCON and XSXPT if the new F is the
!       least calculated value so far with a feasible vector of variables.
!
if (f < fopt .and. ifeas == 1) then
    fopt = f
    xsav = x
    xopt = xnew
    kopt = knew
    if (fopt <= ftarget) then
        info = FTARGET_ACHIEVED
        goto 616
    end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    snorm = sqrt(ssq)

    ! RESCON holds useful information about the constraint residuals.
    ! 1. RESCON(J) = B(J) - AMAT(:, J)^T*XOPT if and only if B(J) - AMAT(:, J)^T*XOPT <= DELTA.
    ! 2. Otherwise, RESCON(J) is a value such that B(J) - AMAT(:, J)^T*XOPT >= RESCON(J) >= DELTA.
    ! RESCON can be updated without evaluating the constraints that are far from being active:
    !
    !!RESCON = MAX(RESCON - SNORM, DELTA)
    !!WHERE (.NOT. RESCON >= DELTA + SNORM)
    !!    RESCON = MAX(B - MATPROD(XOPT, AMAT), ZERO)
    !!END WHERE
    !!!MATLAB: mask = ~(rescon >= delta+snorm); rescon(mask) = max(b(mask) - (xopt'*amat(:, mask))', 0);
    ! Powell set RESCON to the negative of the above value when B(J) - AMAT(:, J)^T*XOPT > DELTA.

    where (abs(rescon) >= snorm + delta)
        rescon = min(-abs(rescon) + snorm, -delta)
    elsewhere
        rescon = max(b - matprod(xopt, amat), ZERO)  ! Calculation changed
        where (rescon >= delta)
            rescon = -rescon
        end where
    end where

    xsxpt(1:npt) = xsxpt(1:npt) + xsxpt(npt + 1:2 * npt)
    !!xsxpt(1:npt) = matprod(xopt, xpt)
!
!     Also revise GOPT when symmetric Broyden updating is applied.
!
    if (itest < 3) then
        do j = 1, n
            do i = 1, j
                if (i < j) gopt(j) = gopt(j) + hq(i, j) * step(i)
                gopt(i) = gopt(i) + hq(i, j) * step(j)
            end do
        end do
        !gopt = gopt + matprod(hq, step)
        do k = 1, npt
            gopt = gopt + pq(k) * xsxpt(npt + k) * xpt(:, k)
        end do
    end if
end if
!
!     Replace the current model by the least Frobenius norm interpolant if
!       this interpolant gives substantial reductions in the predictions
!       of values of F at feasible points.
!
if (itest == 3) then
    w(1:npt) = fval - fval(kopt)
    pq = omega_mul(idz, zmat, w(1:npt))
    gopt = matprod(bmat(:, 1:npt), w(1:npt))
    do k = 1, npt
        gopt = gopt + pq(k) * xsxpt(k) * xpt(:, k)
    end do
    hq = ZERO
end if
!
!     If a trust region step has provided a sufficient decrease in F, then
!       branch for another trust region calculation. Every iteration that
!       takes a model step is followed by an attempt to take a trust region
!       step.
!
knew = 0
if (ksave > 0) goto 20
if (ratio >= TENTH) goto 20

530 continue

! Alternatively, find out if the interpolation points are close enough to the best point so far.
distsq = max(delta * delta, 4.0_RP * rho * rho)
xopt = xpt(:, kopt)
xdsq = sum((xpt - spread(xopt, dim=2, ncopies=npt))**2, dim=1)
! MATLAB: xdsq = sum((xpt - xopt).^2)  % xopt should be a column!! Implicit expansion
knew = maxloc([distsq, xdsq], dim=1) - 1_IK

! If KNEW is positive, then branch back for the next iteration, which will generate a "model step".
! Otherwise, if the current iteration has reduced F, or if DELTA was above its lower bound when the
! last trust region step was calculated, then try a "trust region" step instead.
if (knew > 0) goto 20
knew = 0
if (fopt < fsave) goto 20
if (delsav > rho) goto 20
!
!     The calculations with the current value of RHO are complete.
!       Pick the next value of RHO.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Zaikun 15-08-2019
! See the comments below line number 210
!  560 IF (RHO .GT. RHOEND) THEN
560 imprv = 0
if (rho > rhoend) then
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    delta = HALF * rho
    if (rho > 250.0_RP * rhoend) then
        rho = TENTH * rho
    else if (rho <= 16.0_RP * rhoend) then
        rho = rhoend
    else
        rho = sqrt(rho * rhoend)
    end if
    delta = max(delta, rho)
    goto 10
end if
!
!     Return from the calculation, after branching to label 220 for another
!       Newton-Raphson step if it has not been tried before.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
info = INFO_DFT  !!info = SMALL_TR_RADIUS !!??
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if (ksave == -1) goto 220
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  600 IF (FOPT .LE. F .OR. IFEAS .EQ. 0) THEN
600 continue
if (fopt <= f .or. ifeas == 0 .or. is_nan(f)) then
    x = xsav
    f = fopt
end if
616 continue
cstrv = maximum([ZERO, matprod(x, A_orig) - b_orig])
!w(1) = f
!w(2) = real(nf, RP) + HALF


! Arrange CHIST, FHIST, and XHIST so that they are in the chronological order.
call rangehist(nf, xhist, fhist, chist)

!close (16)

end subroutine lincob


end module lincob_mod
