from ase.neighborlist import PrimitiveNeighborList
import numpy as np

def calculate_W_qLL(setups, cell_cv, spos_ac, pbc_c, kd, dtype, lcomp, coeff = 4*np.pi, omega=None):
    assert lcomp == 2

    S = (lcomp+1)**2
    Na = len(spos_ac)
    W_LL = np.zeros((Na*S, Na*S))
    bzq_qc = kd.get_bz_q_points()
    nq = len(bzq_qc)
    Wq_LL = np.zeros( (nq, Na*S, Na*S) )

    # Calculate displacement vectors
    R_av = np.dot(spos_ac, cell_cv)
    dx = R_av[:, None, 0] - R_av[None, :, 0]
    dy = R_av[:, None, 1] - R_av[None, :, 1]
    dz = R_av[:, None, 2] - R_av[None, :, 2]

    # Use ASE neighbour list to enumerate the supercells which need to be enumerated
    cutoff = 3.0 / omega
    nl = PrimitiveNeighborList([ cutoff ], skin=0, 
                               self_interaction=True,
                               use_scaled_positions=True)

    nl.update(pbc=pbc_c, cell=cell_cv, coordinates=np.array([[0.0, 0.0, 0.0]]))
    a_a, offsets_ac = nl.get_neighbors(0)

    for offsets_c in offsets_ac:
        zero_disp = np.all(offsets_c == 0)
        print(np.dot(offsets_c, cell_cv))
        disp_v = np.dot(offsets_c, cell_cv)

        # Diagonals will be done separately, just avoid division by zero here
        if zero_disp:
            dx1 = dx + 1000*np.eye(Na)
            dy1 = dy + 1000*np.eye(Na)
            dz1 = dz + 1000*np.eye(Na)
        else:
            dx1 = dx + disp_v[0]
            dy1 = dy + disp_v[1]
            dz1 = dz + disp_v[2]

        d2 = dx1**2 + dy1**2 + dz1**2
        d = d2**0.5

        generated_W_LL_screening(W_LL, d, dx1, dy1, dz1, omega)
        W_LL *= coeff
        if zero_disp:
            get_W_LL_diagonals_from_setups(W_LL, lcomp, setups)

        phase_q = np.exp(2j*np.pi*np.dot(bzq_qc, offset_c))
        W_qLL += phase_q[:, None, None] * W_LL[None, :, :]

    return W_qLL


