# ------------------------------------------------------------
# Authors (see AUTHORS file for details): DALV
# ------------------------------------------------------------
# Multipole interpolation:
#                         - analytical solution for 1-3 poles
#                         - Linear solver for n poles*
#                         - Pade-Thiele solver for n poles*
#
# Failure condition for the position of the poles*
#
# *DA. Leon et al, PRB 104, 115157 (2021)
#
# Notes:
#
#   1) X(w) is approximated as a sum of poles
#   2) Form of one pole: -P/(w**2-Q) = 2*E*R/(w**2-E**2)
#   3) The input are two w and X(w) for each pole
#   4) The output are E and R coefficients
#   5) Use real(R), imaginary(I) or complex(C) w
#
# **The module works for scalar polarizabilities, so if
#   one wants the solution for matrix element X(G,G',q)
#   then RQ_solver should be called for each G, G' and q.
# ------------------------------------------------------------

from __future__ import annotations
from gpaw.typing import Array1D
import numpy as np
from numpy.linalg import eigvals


#-------------------------------------------------------------
# New restructured code
#-------------------------------------------------------------
def fit_residue(npr_GG, omega_w, X_wGG, E_pGG):
    print('X_wGG.shape', X_wGG.shape)
    npols = len(E_pGG)
    nw = len(omega_w)
    A_GGwp = np.zeros((*E_pGG.shape[1:], nw, npols), dtype=np.complex128)
    b_GGw = np.zeros((*E_pGG.shape[1:], nw), dtype=np.complex128)
    for w in range(nw):
        A_GGwp[:, :, w, :] = (2 * E_pGG / (omega_w[w]**2 - E_pGG**2)).transpose((1, 2, 0))
        b_GGw[:, :, w] = X_wGG[w, :, :]

    for p in range(npols):
        print((p>=npr_GG).shape)
        print(npr_GG.shape, 'nprGGshape')
        print('A_GGwp', A_GGwp.shape)
        for w in range(A_GGwp.shape[2]):
            A_GGwp[:, :, w, p][p >= npr_GG] = 0.0

    print('A matrix new', A_GGwp[0,0])
    temp_GGp = np.einsum('GHwp,GHw->GHp', A_GGwp.conj(), b_GGw)
    XTX_GGpp = np.einsum('GHwp,GHwo->GHpo', A_GGwp.conj(), A_GGwp)
    
    if XTX_GGpp.shape[2] == 1:
        # 1D matrix, invert the number
        XTX_GGpp = 1 / XTX_GGpp
    else:
        XTX_GGpp = np.linalg.pinv(XTX_GGpp)

    R_GGp = np.einsum('GHpo,GHo->GHp', XTX_GGpp, temp_GGp)
    return R_GGp

class Solver:
    def __init__(self, omega_w):
        assert len(omega_w) % 2 == 0
        self.omega_w = omega_w
        self.npoles = len(omega_w) // 2

    def solve(self, X_wG):
        raise NotImplementedError

class SinglePoleSolver(Solver):
    def __init__(self, omega_w):
        Solver.__init__(self, omega_w)
        self.threshold = 1e-5
        self.epsilon = 1e-8

    def solve(self, X_wGG):
        assert len(X_wGG) == 2
        omega_w = self.omega_w
        E_GG = (X_wGG[0,:, :] * omega_w[0]**2 - X_wGG[1, :, :] * omega_w[1]**2) / (X_wGG[0,:, :] - X_wGG[1,:, :])
        def branch_sqrt_inplace(E_GG):
            E_GG.real = np.abs(E_GG.real)  # physical pole
            E_GG.imag = -np.abs(E_GG.imag)  # correct time ordering
            E_GG[:] = np.emath.sqrt(E_GG)

        branch_sqrt_inplace(E_GG)
        mask = E_GG < self.threshold  # null pole
        E_GG[mask] = self.threshold - 1j * self.epsilon 
        #E_GG *= np.sign(E_GG)

        R_GG = fit_residue(np.zeros_like(E_GG)+1, omega_w, X_wGG, E_GG.reshape((1, *E_GG.shape)))[:, :, 0]
        return E_GG, R_GG


def Pade_solver(X_wGG, z_w):
    nw, nG1, nG2 = X_wGG.shape
    npols = nw // 2 
    M = npols + 1
    b_GGm = np.zeros((nG1, nG2, M), dtype=np.complex128)
    bm1_GGm = b_GGm
    b_GGm[..., 0] = 1.0
    c_GGw = X_wGG.transpose((1,2,0)).copy()

    for i in range(1, 2 * npols):
        cm1_GGw = np.copy(c_GGw)
        c_GGw[..., i:] = (cm1_GGw[..., i - 1][..., np.newaxis] - cm1_GGw[..., i:]) / ((z_w[i:] - z_w[i - 1]) * cm1_GGw[..., i:])
        bm2_GGm = np.copy(bm1_GGm)
        bm1_GGm = np.copy(b_GGm)
        b_GGm = bm1_GGm - z_w[i - 1] * c_GGw[..., i][..., np.newaxis] * bm2_GGm
        bm2_GGm[..., npols:0:-1] = c_GGw[..., i][..., np.newaxis] * bm2_GGm[..., npols - 1::-1]
        b_GGm[..., 1:] = b_GGm[..., 1:] + bm2_GGm[..., 1:]



    #def mpa_E_solver_Pade(npols, z, x):
    #b_m1 = b = np.zeros(npols + 1, dtype='complex64')
    #b[0] = 1
    #c = np.copy(x)
    #for i in range(1, 2 * npols):
    #    c_m1 = np.copy(c)
    #    c[i:] = (c_m1[i - 1] - c_m1[i:]) / ((z[i:] - z[i - 1]) * c_m1[i:])
    #    b_m2 = np.copy(b_m1)
    #    b_m1 = np.copy(b)
    #    b = b_m1 - z[i - 1] * c[i] * b_m2
    #    b_m2[npols:0:-1] = c[i] * b_m2[npols - 1::-1]
    #    b[1:] = b[1:] + b_m2[1:]

    companion_GGmm = np.empty((nG1, nG2, npols, npols), dtype=np.complex128)
    for i in range(nG1):
        for j in range(nG2):
            companion_GGmm[i,j] = np.polynomial.polynomial.polycompanion(b_GGm[i,j, :npols + 1])

    print(companion_GGmm.shape)
    E_GGm = eigvals(companion_GGmm)

    npr_GG = np.zeros((nG1, nG2), dtype=np.int32)
    for i in range(nG1):
        for j in range(nG2):
            E_GGm[i,j], npr_GG[i,j], PPcond = mpa_cond(npols, z_w, E_GGm[i,j])

    return E_GGm, npr_GG, PPcond


class MultipoleSolver(Solver):
    def __init__(self, omega_w):
        Solver.__init__(self, omega_w)
        self.threshold = 1e-5
        self.epsilon = 1e-8

    def solve(self, X_wGG):
        assert len(X_wGG) == 2*self.npoles

        # DALV: Pade-Thiele solver (mpa_sol='PT')
        E_GGp, npr_GG, PPcond = Pade_solver(X_wGG, self.omega_w**2)
        print(npr_GG, 'npr_GG')
        #R_GGp = mpa_R_fit(npols, npr, w, x, E)
        E_pGG = E_GGp.transpose((2,0,1))
        R_pGG = fit_residue(npr_GG, self.omega_w, X_wGG, E_pGG)
        #R_pGG = np.zeros_like(E_pGG)
        #print('Passing E_pGG', E_pGG)
        #print('Passing npr_GG', npr_GG)
        #print('omega_w', self.omega_w)
        #print('X_wGG', X_wGG)
        #R_pGG[:, 0,0] = mpa_R_fit(self.npoles, npr_GG[0,0], self.omega_w, X_wGG[:, 0,0], E_pGG[:,0,0])

        return E_pGG, R_pGG
        
        #E, npr, PPcond = mpa_E_solver_Pade(npols, w**2, x)
        #R = mpa_R_fit(npols, npr, w, x, E)


def RESolver(omega_w):
    assert len(omega_w) % 2 == 0
    npoles = len(omega_w) / 2
    assert npoles > 0
    if npoles == 1:
        return SinglePoleSolver(omega_w)
    else:
        return MultipoleSolver(omega_w)

#-------------------------------------------------------------
# Old reference code
#-------------------------------------------------------------

null_pole_thr = 1e-5
pole_resolution = 1e-5
epsilon = 1e-8  # SP

# ####### 1 pole: ########################


def mpa_cond1(z: tuple[complex, complex] | Array1D,
              E2: tuple[complex] | Array1D) -> tuple[[complex], [float]] | Array2D:  
    # complex(SP), intent(in)     :: z(2)
    # complex(SP), intent(inout)  :: E
    # real(SP),    intent(out)    :: PPcond_rate

     
    PPcond_rate = 0
    if abs(E2) < null_pole_thr:  # need to check also NAN(abs(E))
        #E = complex(abs(z[0]), -epsilon)
        PPcond_rate = 1
    elif np.real(E2) > 0.:
        pass
        #E = np.emath.sqrt(E)
    else:
        #E = np.emath.sqrt(-np.conj(E2))  # note: PPA uses E = 1._SP
        PPcond_rate = 1

    # DALV: since MPA uses complex poles we need to guarantee the time ordering
    #if np.real(E) < 0.:
    #    E = -E
    
    
    E2 = complex(abs(E2.real), -abs(E2.imag))
    E = np.emath.sqrt(E2)

    #if np.imag(E) > -epsilon:
    #    E = np.conj(E) # complex(np.real(E), -epsilon)

    return E, PPcond_rate


def mpa_E_1p_solver(z, x):
    #
    # DALV: analytical form of the E position of the 1 pole solution
    #
    # complex(SP), intent(in)   :: z(2)
    # complex(SP), intent(in)   :: x(2)
    # complex(SP), intent(out)  :: E
    # real(SP),    intent(out)  :: PPcond_rate

    E2 = (x[0] * z[0]**2 - x[1] * z[1]**2) / (x[0] - x[1])
    E, PPcond_rate = mpa_cond1(z, E2)
    return E, PPcond_rate


# ######### any number of poles #############################################

def pole_is_out(i, wmax, thr, E):  # we need to modify E inside the function
    # integer ,    intent(in)     :: i
    # real(SP),    intent(in)     :: wmax, thr
    # complex(SP), intent(inout)  :: E(:)

    # integer     :: j
    # logical     :: pole_is_out

    is_out = False

    if np.real(E[i]) > wmax:
        is_out = True

    j = 0
    while j < i and not is_out:
        if abs(np.real(E[i]) - np.real(E[j])) < thr:
            is_out = True
            if abs(np.real(E[j])) > max(abs(np.imag(E[j])),
               abs(np.imag(E[i]))):
                E[j] = np.mean([np.real(E[j]), np.real(E[i])]) - 1j * \
                    max(abs(np.imag(E[j])), abs(np.imag(E[i])))
            else:
                E[j] = np.mean([np.real(E[j]), np.real(E[i])]) - 1j * \
                    min(abs(np.imag(E[j])), abs(np.imag(E[i])))
        j = j + 1

    return is_out


def mpa_cond(npols, z, E):
    # integer,     intent(in)     :: np
    # integer,     intent(out)    :: npr
    # logical,     intent(out)    :: PPcond(np)
    # complex(SP), intent(in)     :: z(2*np)
    # complex(SP), intent(inout)  :: E(np)

    # integer     :: i, j
    # complex(SP) :: Eaux(np)
    # real(SP)    :: wmax, thr=0.00001_SP

    PPcond = np.full(npols, False)
    npr = npols
    wr = np.real(np.emath.sqrt(z))
    wmax = np.max(wr) * 1.5  # DALV: we use 1.5* the extreme of the interval
    Eaux = np.emath.sqrt(E)

    i = 0
    while i < npr:
        Eaux[i] = max(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i]))) - 1j * \
            min(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i])))
        is_out = pole_is_out(i, wmax, pole_resolution, Eaux)

        if is_out:
            Eaux[i] = np.emath.sqrt(E[npr - 1])
            Eaux[i] = max(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i]))) - 1j \
                * min(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i])))
            PPcond[npr - 1] = True
            npr = npr - 1
        else:
            i = i + 1

    E[:npr] = Eaux[:npr]
    if npr < npols:
        E[npr:npols] = complex(1, -epsilon)
        PPcond[npr:npols] = True

    return E, npr, PPcond


def mpa_R_1p_fit(npols, npr, w, x, E):
    # Transforming the problem into a 2* larger least square with real numbers:
    A = np.zeros((4, 2), dtype='complex64')
    b = np.zeros((4), dtype='complex64')
    for k in range(2):
        b[2 * k] = np.real(x[k])
        b[2 * k + 1] = np.imag(x[k])
        A[2 * k][0] = 2. * np.real(E / (w[k]**2 - E**2))
        A[2 * k][1] = -2. * np.imag(E / (w[k]**2 - E**2))
        A[2 * k + 1][0] = 2. * np.imag(E / (w[k]**2 - E**2))
        A[2 * k + 1][1] = 2. * np.real(E / (w[k]**2 - E**2))
    #print('A', A, 'b', b)

    Rri = np.linalg.lstsq(A, b, rcond=None)[0]

    R = Rri[0] + 1j * Rri[1]

    return R


"""
def residuals(R,A,x):
    return np.abs(np.dot(A,R.view(np.complex))-x.view(np.complex))
"""



def mpa_R_fit(npols, npr, w, x, E):
    # integer,     intent(in)  :: np, npr
    # complex(SP), intent(in)  :: w(2*np)
    # complex(SP), intent(in)  :: x(2*np), E(np)
    # complex(SP), intent(out) :: R(np)

    # complex()            :: A(2*np,npr), B(2*np)
    # integer              :: i, k, info, rank, lwork, lwmax
    # parameter            (lwmax=1000)
    # real(SP)             :: rcond
    # integer              :: iwork(3*npr*0+11*npr)
    # real(SP)             :: S(npr), rwork(10*npr+2*npr*25+8*npr*0+3*25+26*26)
    # complex(SP)          :: work(lwmax)

    """
    A = np.zeros((2*npols,npr),dtype='complex64')
    for k in range(2*npols):
      #B[k] = x[k]
      for i in range(npr):
        A[k][i] = 2.*E[i]/(w[k]**2 -E[i]**2)
    """

#    rcond = -1._SP
# #ifdef _DOUBLE
#      call zgelsd(2*np,npr,1,A,2*np,B,2*np,S,rcond,rank,work,-1,rwork,iwork,
#                  info)
#      lwork=min(lwmax,int(work(1)))
#      call zgelsd(2*np,npr,1,A,2*np,B,2*np,S,rcond,rank,work,lwork,rwork,
#                  iwork,info)
# #else
#      call cgelsd(2*np,npr,1,A,2*np,B,2*np,S,rcond,rank,work,-1,rwork,iwork,
#                  info)
#      lwork=min(lwmax,int(work(1)))
#      call cgelsd(2*np,npr,1,A,2*np,B,2*np,S,rcond,rank,work,lwork,rwork,
#                   iwork,info)
# #endif

    # Failed attempts to do a linear least square with complex numbers:
    """
    R = np.linalg.lstsq(A, x, rcond='warn')[0]
    init = np.zeros(len(x))
    R = leastsq(residuals,init,args=(A, x))[0]
    """

    # Transforming the problem into a 2* larger least square with real numbers:
    A = np.zeros((2 * npols * 2, npr * 2), dtype='complex64')
    b = np.zeros((2 * npols * 2), dtype='complex64')
    for k in range(2 * npols):
        b[2 * k] = np.real(x[k])
        b[2 * k + 1] = np.imag(x[k])
        for i in range(npr):
            A[2 * k][2 * i] = 2. * np.real(E[i] / (w[k]**2 - E[i]**2))
            A[2 * k][2 * i + 1] = -2. * np.imag(E[i] / (w[k]**2 - E[i]**2))
            A[2 * k + 1][2 * i] = 2. * np.imag(E[i] / (w[k]**2 - E[i]**2))
            A[2 * k + 1][2 * i + 1] = 2. * np.real(E[i] / (w[k]**2 - E[i]**2))

    print('A matrix old', A)
    Rri = np.linalg.lstsq(A, b, rcond=None)[0]

    R = np.zeros(npols, dtype='complex64')
    R[:npr] = Rri[::2] + 1j * Rri[1::2]

    return R

def mpa_E_solver_Pade_new(npols,
                          omega_w, # used to be z
                          X_wGG):  # used to be x
    b = np.zeros(npols + 1, dtype=np.complex128) 

    #b_m1 = b = np.zeros(npols + 1, dtype='complex64')
    #b_m1[0] = b[0] = 1
    #c = np.copy(x)

    X_GGw = X_wGG.transpose((2,0,1))

    for i in range(1, 2 * npols):
        #c_m1 = np.copy(c)
        c[i:] = (X_GGw[:, :, i-1] - X_GGw[:, :, i:]) / ((z[i:] - z[i - 1]) * c_m1[i:])

        b_m2 = np.copy(b_m1)
        b_m1 = np.copy(b)

        b = b_m1 - z[i - 1] * c[i] * b_m2

        b_m2[npols:0:-1] = c[i] * b_m2[npols - 1::-1]

        b[1:] = b[1:] + b_m2[1:]

    Companion = np.polynomial.polynomial.polycompanion(b[:npols + 1])
    E = eigvals(Companion)
    E, npr, PPcond = mpa_cond(npols, z, E)
    return E, npr, PPcond

def mpa_E_solver_Pade(npols, z, x):
    # integer,     intent(in)   :: np
    # integer,     intent(out)  :: npr
    # complex(SP), intent(in)   :: z(2*np)
    # complex(SP), intent(in)   :: X(2*np)
    # complex(SP), intent(out)  :: E(np)
    # logical,     intent(out)  :: PPcond(np)

    # complex(SP) :: c(2*np), b(np+1), Companion(np,np)
    # complex(SP) :: c_m1(2*np), b_m1(np+1), b_m2(np+1)
    # complex(SP) :: rwork(2*np),work(2*np),VR(np),VL(np)
    # integer     :: i, j, info
    # real(SP)    :: rcond, anorm, Wm

    # PPcond[:] = True
    b_m1 = b = np.zeros(npols + 1, dtype='complex64')
    b_m1[0] = b[0] = 1
    c = np.copy(x)

    for i in range(1, 2 * npols):

        c_m1 = np.copy(c)
        c[i:] = (c_m1[i - 1] - c_m1[i:]) / ((z[i:] - z[i - 1]) * c_m1[i:])

        b_m2 = np.copy(b_m1)
        b_m1 = np.copy(b)

        # for j in range(npols+1):
        #   b[j] = b_m1[j]-z[i-1]*c[i]*b_m2[j]
        b = b_m1 - z[i - 1] * c[i] * b_m2

        # for j in range(npols):
        #   b_m2[npols-j] = c[i]*b_m2[npols-1-j]
        b_m2[npols:0:-1] = c[i] * b_m2[npols - 1::-1]

        # for j in range(1,npols+1):
        #   b[j] = b[j] + b_m2[j]
        b[1:] = b[1:] + b_m2[1:]

    Companion = np.polynomial.polynomial.polycompanion(b[:npols + 1])
    # DALV: /b[npols] it is carried inside

# #ifdef _DOUBLE
#      call zgeev( 'N', 'N', np, Companion, np, E, VL, 1, VR, 1, work, 2*np,
#                rwork, info )
# #else
#      call cgeev( 'N', 'N', np, Companion, np, E, VL, 1, VR, 1, work, 2*np,
#                rwork, info )
# #endif

    # E = np.linalg.eigvals(Companion) #DALV: the companion matrix isn't normal
    E = eigvals(Companion)
    # E = np.roots(np.flip(b[:npols+1])) #/b[npols] it is carried inside
    # E = zgeev(Companion)[0] #with low level lapack

    # DALV: here we need to force real(E) to be positive.
    # This is because of the way the residue integral is performed, later.
    E, npr, PPcond = mpa_cond(npols, z, E)

    return E, npr, PPcond


def mpa_RE_solver(npols, w, x):
    # integer,      intent(in)   :: np
    # complex(SP),  intent(in)   :: w(2*np), x(2*np)
    # complex(SP),  intent(out)  :: R(np), E(np)
    # character(2), intent(in)   :: mpa_sol
    # logical,      intent(out)  :: MPred
    # real(SP),     intent(out)  :: PPcond_rate,MP_err

    # integer  :: i,npr
    # logical  :: PPcond(np)
    # real(SP) :: cond_num(2) #DALV: only LA solver

    if npols == 1:  # DALV: we could particularize the solution for 2-3 poles
        E, PPcond_rate = mpa_E_1p_solver(w, x)
        R = mpa_R_1p_fit(1, 1, w, x, E)
        # DALV: if PPcond_rate=0, R = x[1]*(z[1]**2-E**2)/(2*E)
        MPred = 0
    else:
        # DALV: Pade-Thiele solver (mpa_sol='PT')
        E, npr, PPcond = mpa_E_solver_Pade(npols, w**2, x)
        R = mpa_R_fit(npols, npr, w, x, E)

        Rnew = fit_residue(np.array([[npr]]), w, x.reshape((-1, 1, 1)), E.reshape((-1, 1, 1)))
        print(R, Rnew)
        assert np.allclose(R, Rnew)

        # if(npr < npols): MPred = True

        # PPcond_rate = 0.
        # for i in range(npols):
        #  if(not PPcond[i]): PPcond_rate = PPcond_rate+abs(R[i])

        # PPcond_rate = PPcond_rate/sum(abs(R))
        PPcond_rate = 1
        MPred = 1

    # for later: MP_err = err_func_X(np, R, E, w, x)

    return R, E, MPred, PPcond_rate  # , MP_err
