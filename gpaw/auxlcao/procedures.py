import numpy as np
from gpaw.lfc import LFC
from gpaw.utilities.tools import tri2full
from gpaw.auxlcao.utilities import safe_inv


from gpaw.auxlcao.generatedcode import generated_W_LL

sqrt3 = 3**0.5
sqrt5 = 5**0.5
sqrt15 = 15**0.5

"""

    Methods helping with slicing of local arrays

"""

def reduce_list(lst):
    return list(dict.fromkeys(lst))

def get_L_slices(lst, S):
    return [ slice(a*S, (a+1)*S) for a in lst ]

def get_M_slices(lst, M_a):
    return [ slice(M_a[a], M_a[a+1]) for a in lst ]
    
def grab_local_W_LL(W_LL, alst, lmax):
    S = (lmax+1)**2
    Lslices = get_L_slices(reduce_list(alst),S)
    return np.block( [ [ W_LL[slice1, slice2] for slice2 in Lslices ] for slice1 in Lslices ] ), Lslices

def get_A_a(auxt_aj):
    A_a = []
    A = 0
    for a, auxt_j in enumerate(auxt_aj):
        A_a.append(A)
        A += sum([ 2*auxt.l+1 for auxt in auxt_j ])
    A_a.append(A)
    return A_a

def create_local_M_a(alst, M_a):
    M = 0
    Mloc_a = [ 0 ]
    for a in alst:
        Msize = M_a[a+1] - M_a[a]
        M += Msize
        Mloc_a.append(M)
    return Mloc_a

def calculate_local_I_LMM(matrix_elements, alst, lmax):
    M_a = matrix_elements.M_a
    gMslices = get_M_slices(alst, M_a)

    M1 = M_a[alst[0]+1] - M_a[alst[0]]
    M2 = M_a[alst[1]+1] - M_a[alst[1]]

    alstred = reduce_list(alst)
    loc_alstred = range(len(alstred))

    S = (lmax+1)**2
    Iloc_LMM = np.zeros( (S*len(alstred), M1, M2) )

    a2 = alstred[0]
    a3 = alstred[-1]

    for a1, Lslice in zip(alstred, get_L_slices(loc_alstred, S)):
        Iloc_LMM[Lslice, :, :] = matrix_elements.evaluate_3ci_LMM(a1,a2,a3)

    lLslices = get_L_slices(range(len(alst)), S)
    gLslices = get_L_slices(alstred, S)
    gMslices = get_M_slices(alst, matrix_elements.M_a)

    slicing_internals = lLslices, gLslices, gMslices
    return Iloc_LMM, slicing_internals

def add_to_global_P_LMM(gP_LMM, lP_LMM, slicing_internals):
    lLslices, gLslices, gMslices = slicing_internals
    for lLslice, gLslice in zip(lLslices, gLslices):
        #for lMslice1, gMslice1 in zip(lMslices, gMslices):
        #    for lMslice2, gMslice2 in zip(lMslices, gMslices):
        gP_LMM[gLslice, gMslices[0], gMslices[1]] += \
          lP_LMM[lLslice, :, :]

def get_W_LL_diagonals_from_setups(W_LL, lmax, setups):
    S = (lmax+1)**2
    for a, setup in enumerate(setups):
        W_LL[a*S:(a+1)*S:,a*S:(a+1)*S] = setup.W_LL[:S, :S]

def calculate_W_LL_offdiagonals_multipole_screened(cell_cv, spos_ac, pbc_c, ibzk_qc, dtype, lmax, coeff = 4*np.pi):
    self.a1a2 = AtomPairRegistry(cutoff_a, pbc_c, cell_cv, spos_ac)

    TODO

def calculate_W_LL_offdiagonals_multipole(cell_cv, spos_ac, pbc_c, ibzk_qc, dtype, lmax, coeff = 4*np.pi):
    if np.any(pbc_c):
        raise NotImplementedError('Periodic boundary conditions')

    if lmax == 2:
        S = (lmax+1)**2
        Na = len(spos_ac)

        R_av = np.dot(spos_ac, cell_cv)
      
        dx = R_av[:, None, 0] - R_av[None, :, 0]
        dy = R_av[:, None, 1] - R_av[None, :, 1]
        dz = R_av[:, None, 2] - R_av[None, :, 2]

        # Diagonals will be done separately, just avoid division by zero here
        dx = dx + 1000*np.eye(Na)
        dy = dy + 1000*np.eye(Na)
        dz = dz + 1000*np.eye(Na)

        d2 = dx**2 + dy**2 + dz**2
        d = d2**0.5

        W_LL = np.zeros((Na*S, Na*S))
        generated_W_LL(W_LL, d, dx, dy, dz)
        return coeff * W_LL
    else:
        return calculate_W_LL_offdiagonals_multipole_old(cell_cv, spos_ac, pbc_c, ibzk_qc, dtype, lmax, coeff)


def calculate_W_LL_offdiagonals_multipole_old(cell_cv, spos_ac, pbc_c, ibzk_qc, dtype, lmax, coeff = 4*np.pi):
    if np.any(pbc_c):
        raise NotImplementedError('Periodic boundary conditions')

    assert lmax < 3

    S = (lmax+1)**2
    Na = len(spos_ac)

    R_av = np.dot(spos_ac, cell_cv)
      
    dx = R_av[:, None, 0] - R_av[None, :, 0]
    dy = R_av[:, None, 1] - R_av[None, :, 1]
    dz = R_av[:, None, 2] - R_av[None, :, 2]

    # Diagonals will be done separately, just avoid division by zero here
    dx = dx + 1000*np.eye(Na)
    dy = dy + 1000*np.eye(Na)
    dz = dz + 1000*np.eye(Na)

    d2 = dx**2 + dy**2 + dz**2
    d = d2**0.5
    d3 = d**3
    d4 = d2**2
    d5 = d2*d3
    d7 = d3*d4
    d9 = d4*d5
    dx2 = dx**2
    dy2 = dy**2
    dz2 = dz**2
    dx4 = dx2**2
    dy4 = dy2**2
    dz4 = dz2**2

    W_LL = np.zeros((Na*S, Na*S))

    # s-s
    W_LL[0::S, 0::S] = 1/d

    if lmax == 0:
        return coeff * W_LL


    # s-p
    W_LL[0::S, 1::S] = (sqrt3*dy)/(3*d3)
    W_LL[0::S, 2::S] = (sqrt3*dz)/(3*d3)
    W_LL[0::S, 3::S] = (sqrt3*dx)/(3*d3)
    W_LL[1::S, 0::S] = -(sqrt3*dy)/(3*d3)
    W_LL[2::S, 0::S] = -(sqrt3*dz)/(3*d3)
    W_LL[3::S, 0::S] = -(sqrt3*dx)/(3*d3)

    # p-p
    W_LL[1::S, 1::S] = (d2 - 3*dy2)/(3*d5)
    W_LL[1::S, 2::S] = -(dy*dz)/d5
    W_LL[1::S, 3::S] = -(dx*dy)/d5

    W_LL[2::S, 1::S] = -(dy*dz)/d5
    W_LL[2::S, 2::S] = (d2 - 3*dz2)/(3*d5)
    W_LL[2::S, 3::S] = -(dx*dz)/d5

    W_LL[3::S, 1::S] = -(dx*dy)/d5
    W_LL[3::S, 2::S] = -(dx*dz)/d5
    W_LL[3::S, 3::S] = (d2 - 3*dx2)/(3*d5)

    if lmax == 1:
        return coeff * W_LL

    # s-d
    W_LL[0::S, 4::S] = (sqrt15*dx*dy)/(5*d5)
    W_LL[0::S, 5::S] = (sqrt15*dy*dz)/(5*d5)
    W_LL[0::S, 6::S] = -(sqrt5*(dx2 + dy2 - 2*dz2))/(10*d5)
    W_LL[0::S, 7::S] = (sqrt15*dx*dz)/(5*d5)
    W_LL[0::S, 8::S] = (sqrt15*(dx2 - dy2))/(10*d5)

    W_LL[4::S, 0::S] = (sqrt15*dx*dy)/(5*d5)
    W_LL[5::S, 0::S] = (sqrt15*dy*dz)/(5*d5)
    W_LL[6::S, 0::S] = -(sqrt5*(dx2 + dy2 - 2*dz2))/(10*d5)
    W_LL[7::S, 0::S] = (sqrt15*dx*dz)/(5*d5)
    W_LL[8::S, 0::S] = (sqrt15*(dx2 - dy2))/(10*d5)

    # p-d
    W_LL[1::S, 4::S] = (sqrt5*dx*(d2 - 5*dy2))/(5*d7)
    W_LL[1::S, 5::S] = (sqrt5*dz*(d2 - 5*dy2))/(5*d7)
    W_LL[1::S, 6::S] = -(sqrt15*dy*(2*d2 - 5*dx2 - 5*dy2 + 10*dz2))/(30*d7)
    W_LL[1::S, 7::S] = -(sqrt5*dx*dy*dz)/d7
    W_LL[1::S, 8::S] = -(sqrt5*dy*(2*d2 + 5*dx2 - 5*dy2))/(10*d7)

    W_LL[2::S, 4::S] = -(sqrt5*dx*dy*dz)/d7
    W_LL[2::S, 5::S] = (sqrt5*dy*(d2 - 5*dz2))/(5*d7)
    W_LL[2::S, 6::S] = (sqrt15*dz*(4*d2 + 5*dx2 + 5*dy2 - 10*dz2))/(30*d7)
    W_LL[2::S, 7::S] = (sqrt5*dx*(d2 - 5*dz2))/(5*d7)
    W_LL[2::S, 8::S] = -(sqrt5*dz*(dx2 - dy2))/(2*d7)

    W_LL[3::S, 4::S] = (sqrt5*dy*(d2 - 5*dx2))/(5*d7)
    W_LL[3::S, 5::S] = -(sqrt5*dx*dy*dz)/d7
    W_LL[3::S, 6::S] = -(sqrt15*dx*(2*d2 - 5*dx2 - 5*dy2 + 10*dz2))/(30*d7)
    W_LL[3::S, 7::S] = (sqrt5*dz*(d2 - 5*dx2))/(5*d7)
    W_LL[3::S, 8::S] = (sqrt5*dx*(2*d2 - 5*dx2 + 5*dy2))/(10*d7)

    W_LL[4::S, 1::S] = -(sqrt5*dx*(d2 - 5*dy2))/(5*d7)
    W_LL[4::S, 2::S] = (sqrt5*dx*dy*dz)/d7
    W_LL[4::S, 3::S] = -(sqrt5*dy*(d2 - 5*dx2))/(5*d7)

    W_LL[5::S, 1::S] = -(sqrt5*dz*(d2 - 5*dy2))/(5*d7)
    W_LL[5::S, 2::S] = -(sqrt5*dy*(d2 - 5*dz2))/(5*d7)
    W_LL[5::S, 3::S] = (sqrt5*dx*dy*dz)/d7

    W_LL[6::S, 1::S] = (sqrt15*dy*(2*d2 - 5*dx2 - 5*dy2 + 10*dz2))/(30*d7)
    W_LL[6::S, 2::S] = -(sqrt15*dz*(4*d2 + 5*dx2 + 5*dy2 - 10*dz2))/(30*d7)
    W_LL[6::S, 3::S] = (sqrt15*dx*(2*d2 - 5*dx2 - 5*dy2 + 10*dz2))/(30*d7)

    W_LL[7::S, 1::S] = (sqrt5*dx*dy*dz)/d7
    W_LL[7::S, 2::S] = -(sqrt5*dx*(d2 - 5*dz2))/(5*d7)
    W_LL[7::S, 3::S] = -(sqrt5*dz*(d2 - 5*dx2))/(5*d7)

    W_LL[8::S, 1::S] = (sqrt5*dy*(2*d2 + 5*dx2 - 5*dy2))/(10*d7)
    W_LL[8::S, 2::S] = (sqrt5*dz*(dx2 - dy2))/(2*d7)
    W_LL[8::S, 3::S] = -(sqrt5*dx*(2*d2 - 5*dx2 + 5*dy2))/(10*d7)

    # d-d
    W_LL[4::S, 4::S] = (d4 - 5*d2*dx2 - 5*d2*dy2 + 35*dx2*dy2)/(5*d9)
    W_LL[4::S, 5::S] = -(dx*dz*(d2 - 7*dy2))/d9
    W_LL[4::S, 6::S] = (sqrt3*dx*dy*(4*d2 - 7*dx2 - 7*dy2 + 14*dz2))/(6*d9)
    W_LL[4::S, 7::S] = -(dy*dz*(d2 - 7*dx2))/d9
    W_LL[4::S, 8::S] = (7*dx*dy*(dx2 - dy2))/(2*d9)

    W_LL[5::S, 4::S] = -(dx*dz*(d2 - 7*dy2))/d9
    W_LL[5::S, 5::S] = (d4 - 5*d2*dy2 - 5*d2*dz2 + 35*dy2*dz2)/(5*d9)
    W_LL[5::S, 6::S] = -(sqrt3*dy*dz*(2*d2 + 7*dx2 + 7*dy2 - 14*dz2))/(6*d9)
    W_LL[5::S, 7::S] = -(dx*dy*(d2 - 7*dz2))/d9
    W_LL[5::S, 8::S] = (dy*dz*(2*d2 + 7*dx2 - 7*dy2))/(2*d9)

    W_LL[6::S, 4::S] = (sqrt3*dx*dy*(4*d2 - 7*dx2 - 7*dy2 + 14*dz2))/(6*d9)
    W_LL[6::S, 5::S] = -(sqrt3*dy*dz*(2*d2 + 7*dx2 + 7*dy2 - 14*dz2))/(6*d9)
    W_LL[6::S, 6::S] = (12*d4 + 35*dx4 + 35*dy4 + 140*dz4 - 20*d2*dx2 - 20*d2*dy2 - 80*d2*dz2 + 70*dx2*dy2 - 140*dx2*dz2 - 140*dy2*dz2)/(60*d9)
    W_LL[6::S, 7::S] = -(sqrt3*dx*dz*(2*d2 + 7*dx2 + 7*dy2 - 14*dz2))/(6*d9)
    W_LL[6::S, 8::S] = (sqrt3*(dx2 - dy2)*(4*d2 - 7*dx2 - 7*dy2 + 14*dz2))/(12*d9)

    W_LL[7::S, 4::S] = -(dy*dz*(d2 - 7*dx2))/d9
    W_LL[7::S, 5::S] = -(dx*dy*(d2 - 7*dz2))/d9
    W_LL[7::S, 6::S] = -(sqrt3*dx*dz*(2*d2 + 7*dx2 + 7*dy2 - 14*dz2))/(6*d9)
    W_LL[7::S, 7::S] = (d4 - 5*d2*dx2 - 5*d2*dz2 + 35*dx2*dz2)/(5*d9)
    W_LL[7::S, 8::S] = -(dx*dz*(2*d2 - 7*dx2 + 7*dy2))/(2*d9)

    W_LL[8::S, 4::S] = (7*dx*dy*(dx2 - dy2))/(2*d9)
    W_LL[8::S, 5::S] = (dy*dz*(2*d2 + 7*dx2 - 7*dy2))/(2*d9)
    W_LL[8::S, 6::S] = (sqrt3*(dx2 - dy2)*(4*d2 - 7*dx2 - 7*dy2 + 14*dz2))/(12*d9)
    W_LL[8::S, 7::S] = -(dx*dz*(2*d2 - 7*dx2 + 7*dy2))/(2*d9)
    W_LL[8::S, 8::S] = (4*d4 + 35*dx4 + 35*dy4 - 20*d2*dx2 - 20*d2*dy2 - 70*dx2*dy2)/(20*d9)

    if lmax == 2:
        return coeff * W_LL

    raise NotImplementedError('lmax=%d for multipole interaction' % lmax)

"""

    A reference implementation of

            /    /                1 
     W    = | dr | dr' phi (r) --------  phi (r').
      AA    /    /        A    | r-r' |     A'

    This function separately places each phi (r),
                                            A
    to grid and evaluates it's Coulomb solution, and
    integrates using LocaliedFunctions object.
    Only to be used for testing.
    
"""


def reference_W_AA(density, poisson, auxt_aj, spos_ac):
    # Obtain local (per atom) coordinates for auxiliary functions,
    # and allocate the W_AA array.
    A_a = get_A_a(auxt_aj)
    Atot = A_a[-1]
    W_AA = np.zeros( (Atot, Atot) )

    gd = density.finegd
    aux_lfc = LFC(gd, auxt_aj)
    aux_lfc.set_positions(spos_ac)

    for a, (Astart, Aend) in enumerate(zip(A_a[:-1], A_a[1:])):
        Aloctot = Aend - Astart
        for Aloc, Aglob in enumerate(range(Astart, Aend)):
            print(a,Aloc)
            # Add a single function to the grid
            Q_aA = aux_lfc.dict(zero=True)
            Q_aA[a][Aloc] = 1.0
            auxt_g = gd.zeros()
            aux_lfc.add(auxt_g, Q_aA)

            # Solve its Poisson equation
            wauxt_g = gd.zeros()
            poisson.solve(wauxt_g, auxt_g, charge=None)

            # Integrate wrt. all other auxiliary functions
            W_aM = aux_lfc.dict(zero=True)
            aux_lfc.integrate(wauxt_g, W_aM)

            # Fill a single row of the matrix
            for a2, (A2start, A2end) in enumerate(zip(A_a[:-1], A_a[1:])):
                W_M = W_aM[a2]
                W_AA[Aglob, A2start:A2end] = W_aM[a2]
    return W_AA



"""

    Production implementation of

             /    /      a1      1       a2     a3 
     I     = | dr | dr' φ (r) --------  φ (r') φ (r'),
      AMM'   /    /      A    | r-r' |   M      M'

    where a2 and a3 can be anything, and a1 ∈  (a2, a3).


"""

def calculate_I_AMM(matrix_elements):
    A_a = matrix_elements.A_a
    M_a = matrix_elements.M_a
    Atot = A_a[-1]
    nao = M_a[-1]
    I_AMM = np.zeros( (Atot, nao, nao) )
    I_AMM[:] = np.nan
    for a1, (A1start, A1end) in enumerate(zip(A_a[:-1], A_a[1:])):
        for a2, (M1start, M1end) in enumerate(zip(M_a[:-1], M_a[1:])):
            for a3, (M2start, M2end) in enumerate(zip(M_a[:-1], M_a[1:])):
                if a1 != a2 and a1 != a3:
                    continue
                I_AMM[A1start:A1end, M1start:M1end, M2start:M2end] = \
                      matrix_elements.evaluate_3ci_AMM(a1, a2, a3)

    return I_AMM


"""

    A reference implementation of

             /    /              1 
     I     = | dr | dr' φ (r) --------  φ (r') φ (r')
      AMM'   /    /      A    | r-r' |   M      M'

    This function separately places each phi (r),
                                            A
    to grid and evaluates it's Coulomb solution, and integrates
    with phi products using the LocalizedFunctions object.
    Only to be used for testing.

"""

def reference_I_AMM(wfs, density, hamiltonian, poisson, auxt_aj, spos_ac):
    # Obtain local (per atom) coordinates for auxiliary functions,
    # and allocate the W_AA array.
    A_a = get_A_a(auxt_aj)
    Atot = A_a[-1]
    nao = wfs.setups.nao

    I_AMM = np.zeros( (Atot, nao, nao) )

    gd = density.finegd
    aux_lfc = LFC(gd, auxt_aj)
    aux_lfc.set_positions(spos_ac)

    for a, (Astart, Aend) in enumerate(zip(A_a[:-1], A_a[1:])):
        Aloctot = Aend - Astart
        for Aloc, Aglob in enumerate(range(Astart, Aend)):
            print(a,Aloc)
            # Add a single function to the grid
            Q_aA = aux_lfc.dict(zero=True)
            Q_aA[a][Aloc] = 1.0
            auxt_g = gd.zeros()
            aux_lfc.add(auxt_g, Q_aA)

            # Solve its Poisson equation
            wauxt_g = gd.zeros()
            poisson.solve(wauxt_g, auxt_g, charge=None)

            wauxt_G = hamiltonian.gd.zeros()
            hamiltonian.restrict(wauxt_g, wauxt_G)
            V_MM = wfs.basis_functions.calculate_potential_matrices(wauxt_G)[0]
            tri2full(V_MM)
            I_AMM[Aglob] = (V_MM+V_MM.T)/2

    return I_AMM


"""
 Production implementation of compensation charge projection

                              
               __   a      /  a    |       \  /  a  |       \
 P           = \   Δ       | p (r) | φ (r) |  | p   | φ (r) |
  (L,a)M1M2    /_   Li1i2  \  i1   |  M1   /  \  i2 |  M2   /
               ai1i2


"""

def calculate_P_LMM(matrix_elements, setups, atomic_correction):
    L_a = matrix_elements.L_a
    M_a = matrix_elements.M_a
    Ltot = L_a[-1]
    nao = M_a[-1]
    P_LMM = np.zeros( (Ltot, nao, nao) )

    for a, (setup, Lstart, Lend) in enumerate(zip(setups, L_a[:-1], L_a[1:])):
        P_Mi = atomic_correction.P_aqMi[a][0]
        P_LMM[Lstart:Lend, :, :] = np.einsum('ijL,Mi,Nj->LMN', setup.Delta_iiL, P_Mi, P_Mi, optimize=True)
    return P_LMM 

"""
 Production implementation of two center auxiliary RI-V projection.

                             -1
           /       ||       \  /       ||             \
 P       = | φ (r) || φ (r) |  | φ (r) || φ (r) φ (r) |
  AM1M2    \  A    ||  A'   /  \  A'   ||  M1    M2   /

 Where A and A' ∈  a(M1) ∪ a(M2).


"""

def calculate_P_AMM(matrix_elements, W_AA):
    A_a = matrix_elements.A_a
    M_a = matrix_elements.M_a
    Atot = A_a[-1]
    nao = M_a[-1]
    P_AMM = np.zeros( (Atot, nao, nao) )

    # Single center projections
    for a, (Astart, Aend, Mstart, Mend) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):
        iW_AA = safe_inv(W_AA[Astart:Aend, Astart:Aend])
        I_AMM = matrix_elements.evaluate_3ci_AMM(a, a, a)
        P_AMM[Astart:Aend, Mstart:Mend, Mstart:Mend] += \
           np.einsum('AB,Bij->Aij', iW_AA, I_AMM)

    # Two center projections
    for a1, (A1start, A1end, M1start, M1end) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):
        for a2, (A2start, A2end, M2start, M2end) in enumerate(zip(A_a[:-1], A_a[1:], M_a[:-1], M_a[1:])):
            if a1 == a2:
                continue
            locW_AA = np.block( [ [ W_AA[A1start:A1end, A1start:A1end], W_AA[A1start:A1end, A2start:A2end] ],
                                  [ W_AA[A2start:A2end, A1start:A1end], W_AA[A2start:A2end, A2start:A2end] ] ] )
            iW_AA = safe_inv(locW_AA)
            I_AMM = [matrix_elements.evaluate_3ci_AMM(a1, a1, a2),
                     matrix_elements.evaluate_3ci_AMM(a2, a1, a2) ]
            if I_AMM[0] is None or I_AMM[1] is None:
                continue
            I_AMM = np.vstack(I_AMM)
            Ploc_AMM = np.einsum('AB,Bij', iW_AA, I_AMM, optimize=True)
            P_AMM[A1start:A1end, M1start:M1end, M2start:M2end] += Ploc_AMM[:A1end-A1start]
            P_AMM[A2start:A2end, M1start:M1end, M2start:M2end] += Ploc_AMM[A1end-A1start:]

    return P_AMM

"""

       /    /               1
 W   = | dr | dr' φ (r) ---------- φ (r')
  AA'  /    /      A    | r - r' |  A'

The integrals are screened

          lr.     sr.
 φ (r) = φ (r) + φ (r) 
  A       A       A

       lr                                         sr
Where φ (r) is a generalized gaussian, such that φ (r) has no multipole moment.

Let

       /    /      sr.      1       sr.
 S   = | dr | dr' φ (r) ---------- φ (r')
  AA'  /    /      A    | r - r' |  A'


We can evaluate this with the Fourier-Bessel overlap code,

             sr
by defining V (r), which is short ranged Poisson solution,
             A'
    sr.
to φ (r) due to missing multipoles.
    A'


       /     sr.   sr.
 S   = | dr φ (r) V (r)
  AA'  /     A     A'


       /     lr.   sr.
 M   = | dr φ (r) V (r)
  AA'  /     A     A'

                            T
 W    = V   + S   + M    + M
  AA     AA    AA    AA'    AA'

 V    =M_A  M   ( R_a - R_a' ) M_A'
  AA'        L L
              A A'


 
"""

"""
                                M_AL            L_AL = M_A W_(L_A)L

    /       ||       \   /  sr.  ||       \   /  lr.  ||       \
    | φ (r) || g (r) | = | φ (r) || g (r) | + | φ (r) || g (r) |
    \  A    ||  L    /   \  A    ||  L    /   \  A    ||  L    /

     M
      A_


"""

def calculate_M_AL(matrix_elements):
    A_a = matrix_elements.A_a
    L_a = matrix_elements.L_a
    Atot = A_a[-1]
    Ltot = L_a[-1]
    M_AL = np.zeros( (Atot, Ltot) )
    for a1, (Astart, Aend) in enumerate(zip(A_a[:-1], A_a[1:])):
        A2 = 0
        for a2, (Lstart, Lend) in enumerate(zip(L_a[:-1], L_a[1:])):
            Mloc_AL = matrix_elements.evaluate_2ci_M_AL(a1, a2)
            if Mloc_AL is None:
                continue
            M_AL[Astart:Aend, Lstart:Lend] = Mloc_AL
    return M_AL

def calculate_W_AL(matrix_elements, auxt_aj, M_aj, W_LL):
    W_AL = calculate_M_AL(matrix_elements)
    A_a = matrix_elements.A_a
    L_a = matrix_elements.L_a
    Atot = A_a[-1]
    M_AA = np.zeros( (Atot, Atot) )
    
    A = 0
    for a1, (A1start, A1end) in enumerate(zip(A_a[:-1], A_a[1:])):
        for auxt, M in zip(auxt_aj[a1], M_aj[a1]):
            for m in range(2*auxt.l+1):
                locL = auxt.l**2 + m
                W_AL[A, :] += M*W_LL[L_a[a1]+locL, :]
                A += 1
    return W_AL


def calculate_V_AA(auxt_aj, M_aj, W_LL, lmax):
    Lmax = (lmax+1)**2

    # Obtain local (per atom) coordinates for auxiliary functions,
    # and allocate the W_AA array.
    A_a = get_A_a(auxt_aj)
    Atot = A_a[-1]
    W_AA = np.zeros( (Atot, Atot) )
    A1 = 0
    for a1, (A1start, A1end) in enumerate(zip(A_a[:-1], A_a[1:])):
        for M1, auxt1 in zip(M_aj[a1], auxt_aj[a1]):
            for m1 in range(2*auxt1.l+1):
                L1 = auxt1.l**2 + m1
                A2 = 0
                for a2, (A2start, A2end) in enumerate(zip(A_a[:-1], A_a[1:])):
                    for M2, auxt2 in zip(M_aj[a2], auxt_aj[a2]):
                        for m2 in range(2*auxt2.l+1):
                            L2 = auxt2.l**2 + m2
                            W_AA[A1, A2] = M1*M2*W_LL[a1*Lmax + L1, a2*Lmax + L2]
                            A2 += 1
                A1 += 1

    return W_AA

def calculate_S_AA(matrix_elements):
    A_a = matrix_elements.A_a
    Atot = A_a[-1]
    S_AA = np.zeros( (Atot, Atot) )
    
    for a1, (A1start, A1end) in enumerate(zip(A_a[:-1], A_a[1:])):
        for a2, (A2start, A2end) in enumerate(zip(A_a[:-1], A_a[1:])):
            loc_S_AA = matrix_elements.evaluate_2ci_S_AA(a1, a2)
            if loc_S_AA is None:
                continue
            S_AA[A1start:A1end, A2start:A2end] = loc_S_AA
    return S_AA

def calculate_M_AA(matrix_elements, auxt_aj, M_aj, lmax):
    A_a = matrix_elements.A_a
    Atot = A_a[-1]
    M_AA = np.zeros( (Atot, Atot) )
    Lmax = (lmax+1)**2
    
    for a1, (A1start, A1end) in enumerate(zip(A_a[:-1], A_a[1:])):
        A2 = 0
        for a2, (A2start, A2end) in enumerate(zip(A_a[:-1], A_a[1:])):
            M_AL = matrix_elements.evaluate_2ci_M_AL(a1, a2)
            if M_AL is None:
                continue
            for M2, auxt2 in zip(M_aj[a2], auxt_aj[a2]):
                for m2 in range(2*auxt2.l+1):
                    L2 = auxt2.l**2 + m2
                    M_AA[A1start:A1end, A2] = M2*M_AL[:, L2]
                    A2 += 1

    return M_AA
