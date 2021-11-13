import numpy as np

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
    return A

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

def calculate_W_LL_offdiagonals_multipole(cell_cv, spos_ac, pbc_c, ibzk_qc, dtype, lmax, coeff = 4*np.pi):
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

def reference_W_AA(density, auxt_aj):
    aux_lfc = LFC(density.finegd, auxt_aj)
    aux_lfc.set_positions(spos_ac)
    A_a = get_A_a(auxt_aj)

    
