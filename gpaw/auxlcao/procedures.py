import numpy as np


sqrt3 = 3**0.5
sqrt5 = 5**0.5
sqrt15 = 15**0.5

def get_L_range(as, lmax):
    # Remove duplicates and order
    as = list(dict.fromkeys(as))
    S = (lmax+1)**2
    Lspan = []
    for a in as:
        Lspan.append(np.arange(a*S, (a+1)*S))
    return np.block(Lspan)

def get_M_range(as, M_a):
    # Remove duplicates and order
    as = list(dict.fromkeys(as))
    Mspan = []
    for a in as:
        Mspan.append(np.arange(M_a[a], M_a[a+1]))
    return np.block(Mspan)
    
def grab_local_W_LL(W_LL, a1, a2, lmax)
    Lspan = get_L_range([a1,a2], lmax)
    return Wloc_LL[Lspan,Lspan], Lspan

def calculate_local_I_LMM(matrix_elements, a1, a2, lmax):
    as = [ a1, a2 ]
    Lspan = get_L_range(as, lmax)
    Ispan = get_I_range(as, matrix_elements.M_a)

    as = list(dict.fromkeys(as))
    for a in as:
        atomLspan = get_L_range([a], lmax)
        for b in as:
            atomMspanb = get_M_range([b], M_a)
            for c in cs:
                atomMspanc = get_L_range([c], M_a)
                Iloc_LMM[atomLspan, atomMspanb] = 
    return Iloc_LMM, Lspan, Mspan

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
