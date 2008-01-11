from math import pi, sqrt
import numpy as npy
from gpaw.utilities import pack, wignerseitz

def print_projectors(nucleus):
    """Print information on the projectors of input nucleus object"""
    n_j = nucleus.setup.n_j
    l_j = nucleus.setup.l_j
    angular = [['1'],
               ['y', 'z', 'x'],
               ['xy', 'yz', '3z^2-r^2', 'xz', 'x^2-y^2'],
               ['3x^2y-y^3', 'xyz', '5yz^2-yr^2', '5z^3-3zr^2',
                '5xz^2-xr^2', 'x^2z-y^2z', 'x^3-3xy^2'],
               ]
    print ' i n l m'
    print '--------'
    i = 0
    for n, l in zip(n_j, l_j):
        for m in range(2*l+1):
            if n == -1:
                n = '*'
            print '%2s %s %s_%s' % (i, n, 'spdf'[l], angular[l][m])
            i += 1

def get_angular_projectors(nucleus, angular, type='bound'):
    """Determine the projector indices which have specified angula
    quantum number.

    angular can be s, p, d, f, or a list of these.
    If type is 'bound', only bound state projectors are considered, otherwize
    all projectors are included.
    """
    # Get the number of relevant j values
    if type == 'bound':
        nj = 0
        while nucleus.setup.n_j[nj] != -1: nj += 1
    else:
        nj = len(nucleus.setup.n_j)
            

    # Choose the relevant projectors
    projectors = []
    i = j = 0
    for j in range(nj):
        m = 2 * nucleus.setup.l_j[j] + 1
        if 'spdf'[nucleus.setup.l_j[j]] in angular:
            projectors.extend(range(i, i + m))
        j += 1
        i += m

    return projectors

def delta(x, x0, width):
    """Return a gaussian of given width centered at x0."""
    return npy.exp(npy.clip(-((x - x0) / width)**2,
                            -100.0, 100.0)) / (sqrt(pi) * width)

def fold(energies, weights, npts, width):
    """Take a list of energies and weights, and sum a delta function
    for each."""
    emin = min(energies) - 5 * width
    emax = max(energies) + 5 * width
    step = (emax - emin) / (npts - 1)
    e = npy.arange(emin, emax + 1e-7, step, dtype=float)
    ldos_e = npy.zeros(npts, dtype=float)
    for e0, w in zip(energies, weights):
        ldos_e += w * delta(e, e0, width)
    return e, ldos_e

def raw_orbital_LDOS(paw, a, spin, angular='spdf'):
    """Return a list of eigenvalues, and their weight on the specified atom.

    angular can be s, p, d, f, or a list of these.
    If angular is None, the raw weight for each projector is returned"""
    w_k = paw.weight_k
    nk = len(w_k)
    nb = paw.nbands
    nucleus = paw.nuclei[a]

    energies = npy.empty(nb * nk)
    weights_xi = npy.empty((nb * nk, nucleus.setup.ni))
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = paw.get_eigenvalues(k=k, s=spin)
        u = spin * nk + k
        weights_xi[x:x + nb, :] = w * npy.absolute(nucleus.P_uni[u])**2
        x += nb

    if angular is None:
        return energies, weights_xi
    else:
        projectors = get_angular_projectors(nucleus, angular, type='bound')
        weights = npy.sum(npy.take(weights_xi,
                                   indices=projectors, axis=1), axis=1)
        return energies, weights

def raw_wignerseitz_LDOS(paw, a, spin):
    """Return a list of eigenvalues, and their weight on the specified atom"""
    gd = paw.gd
    atom_index = gd.empty(dtype=int)
    atom_ac = npy.array([n.spos_c * gd.N_c for n in paw.nuclei])
    wignerseitz(atom_index, atom_ac, gd.beg_c, gd.end_c)

    w_k = paw.weight_k
    nk = len(w_k)
    nb = paw.nbands
    nucleus = paw.nuclei[a]

    energies = npy.empty(nb * nk)
    weights = npy.empty(nb * nk)
    x = 0
    for k, w in enumerate(w_k):
        u = spin * nk + k
        energies[x:x + nb] = paw.get_eigenvalues(k=k, s=spin)
        for n, psit_G in enumerate(paw.kpt_u[u].psit_nG):
            P_i = nucleus.P_uni[u, n]
            P_p = pack(npy.outer(P_i, P_i))
            Delta_p = sqrt(4 * pi) * nucleus.setup.Delta_pL[:, 0]
            weights[x + n] = w * (gd.integrate(npy.absolute(
                npy.where(atom_index == a, psit_G, 0.0))**2)
                                  + npy.dot(Delta_p, P_p))
        x += nb
    return energies, weights
