import numpy as np




def correct_density(n_sg, gd, setups, spos_ac, rcut_factor=0.8):
    if not hasattr(setups[0], "calculate_pseudized_atomic_density"):
        return n_sg
        
    if len(n_sg) != 1:
        raise NotImplementedError
            
    dens = n_sg[0].copy()

    for a, setup in enumerate(setups):
        spos_ac_indices = list(filter(lambda x: x[1] == setup, enumerate(setups)))
        spos_ac_indices = [x[0] for x in spos_ac_indices]
        spos_ac = spos_ac[spos_ac_indices]
        t = setup.calculate_pseudized_atomic_density(spos_ac, rcut_factor)
        t.add(dens)
        
    return np.array([dens])


def get_ni_grid(rank, size, endval, pts_per_rank, grid_fct=None, return_full_size=False):
    assert rank >= 0 and rank < size
    # Make an interface that allows for testing
    
    # Algo:
    # Define a global grid. We want a grid such that each rank has not too many
    num_pts = pts_per_rank *size
    if grid_fct is None:
        fulln_i = np.linspace(0, endval, num_pts)
    else:
        fulln_i = grid_fct(0, endval, num_pts)
    # Split it up evenly
    my_start = rank * pts_per_rank
    my_n_i = fulln_i[my_start:my_start+pts_per_rank]

    # Each ranks needs to know the closest pts outside its own range
    if rank == 0:
        my_lower = 0
        my_upper = fulln_i[min(num_pts - 1, pts_per_rank)]
    elif rank == size - 1:
        my_lower = fulln_i[my_start - 1]
        my_upper = fulln_i[-1]
    else:
        my_lower = fulln_i[my_start - 1]
        my_upper = fulln_i[my_start + pts_per_rank]
    
    if return_full_size:
        return my_n_i, my_lower, my_upper, len(fulln_i), fulln_i
    else:
        return my_n_i, my_lower, my_upper 

def get_ni_grid_w_min(rank, size, startval, endval, pts_per_rank, grid_fct=None, return_full_size=False):
    assert rank >= 0 and rank < size
    # Make an interface that allows for testing
    
    # Algo:
    # Define a global grid. We want a grid such that each rank has not too many
    num_pts = pts_per_rank *size
    if grid_fct is None:
        fulln_i = np.linspace(startval, endval, num_pts)
    else:
        fulln_i = grid_fct(startval, endval, num_pts)
    # Split it up evenly
    my_start = rank * pts_per_rank
    my_n_i = fulln_i[my_start:my_start+pts_per_rank]

    # Each ranks needs to know the closest pts outside its own range
    if rank == 0:
        my_lower = fulln_i[0]
        my_upper = fulln_i[min(num_pts - 1, pts_per_rank)]
    elif rank == size - 1:
        my_lower = fulln_i[my_start - 1]
        my_upper = fulln_i[-1]
    else:
        my_lower = fulln_i[my_start - 1]
        my_upper = fulln_i[my_start + pts_per_rank]

    if return_full_size:
        return my_n_i, my_lower, my_upper, len(fulln_i)
    else:
        return my_n_i, my_lower, my_upper


def get_K_K(gd):
    from gpaw.utilities.tools import construct_reciprocal
    from ase.units import Bohr
    K2_K, _ = construct_reciprocal(gd)
    K2_K[0, 0, 0] = 0
    return K2_K**(1 / 2)


def build_splines(nb_i, gd):
    from scipy.interpolate import interp1d
    from scipy.integrate import quad
    from gpaw.atom.radialgd import fsbt
    na = np.newaxis
    C_i, dC_i = get_C(nb_i)

    lambd_i, dlambd_i = get_lambd(nb_i)
    Gsplines_i = []
    Grsplines_i = []

    for i, nb in enumerate(nb_i):
        C, dC = get_C(nb)
        lambd, dlambd = get_lambd(nb)
        assert lambd > 0
        K_K = get_K_K(gd)
        kmax = 1.2 * np.max(K_K)

        dr = 0.01
        rmax = lambd * 500
        r_j = np.arange(dr, rmax, dr)

        nks = 900
        k_k = np.exp(np.linspace(0, np.log(kmax), nks)) - 1

        assert np.allclose(np.min(k_k), 0)

        G_j = C * (1 - np.exp(- (lambd / r_j)**5))
        integral_k = 2 * np.pi * fsbt(0, G_j, r_j, k_k)
        
        interf = lambda k, G: interp1d(k, G,
                                       kind='cubic', fill_value='extrapolate')
        Gspline = interf(k_k, integral_k)
        assert np.allclose(Gspline(k_k), integral_k)
        assert np.allclose(Gspline(0) * nb, -1)

        Gr_j = C * (1 - np.exp(- (lambd / r_j)**5)) / r_j

        integral_k = 2 * np.pi * fsbt(0, Gr_j, r_j, k_k)

        Grspline = interf(k_k, integral_k)

        Gsplines_i.append(Gspline)
        Grsplines_i.append(Grspline)

    return Gsplines_i, Grsplines_i    

def get_lambd(n):
    from scipy.special import gamma
    dn = 0.00000001
    if np.allclose(n, 0):
        n = 1e-10
    exc = get_lda_xc(n, 0)
    dexc = (get_lda_xc(n + dn, 0) - exc) / dn
    
    lamb = - 3 * gamma(3 / 4) / (2 * gamma(2 / 5) * exc)

    dlambd = 3 * gamma(3 / 4) / (2 * gamma(2/5)) \
             * (1 / exc**2 * dexc)
    return lamb, dlambd


def get_C(n):
    from scipy.special import gamma
    if np.allclose(n, 0):
        n = 1e-10
    lamb, dlambd = get_lambd(n)
    C = - 3 / (4 * np.pi * gamma(2 / 5) * n * lamb**3)

    dC = 3 / (4 * np.pi * gamma(2 / 5)) \
         * (1 / (n**2 * lamb**3) + 3 / (n * lamb**4) * dlambd)
    return C, dC

def get_lda_xc(n, spin):
    if np.allclose(n, 0):
        return 0

    from gpaw.xc.lda import lda_c, lda_x
    narr = np.array([n])
    earr = np.zeros_like(narr)
    varr = np.zeros_like(narr)

    lda_x(spin, earr, n, varr)
    zeta = 0
    lda_c(spin, earr, narr, varr, zeta)

    return earr / n

















