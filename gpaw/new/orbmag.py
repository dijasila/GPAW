r"""This module calculates the orbital magnetization vector for each atom.

The orbital magnetization is calculated in the atom-centred approximation
where only the PAW correction to the wave function is assumed to contribute.
This leads to the equation (presented in SI units):

::

                 ===  ===
   a         e   \    \         / a   \*   a   a
  M      = - --  /    /    f   | P     | P    L
   orb,v     2m  ===  ===   kn  \ knsi/  knsj  vij
                 kn   sij

with L_vij containing the matrix elements of the angular momentum operator
between two partial waves.

NB: As a convention, the orbital magnetization is returned without the sign of
the negative electronic charge q = -e.

The orbital magnetization is returned in units of μ_B as an array, orbmag_av,
 where a and v are indices for atoms and Cartesian axes, respectively.
"""

import numpy as np

from gpaw.spinorbit import get_L_vlmm

L_vlmm = get_L_vlmm()


def get_orbmag_from_calc(calc):
    "Return orbital magnetization vectors calculated from scf spinors."
    if not calc.density.ncomponents == 4:
        raise AssertionError('Collinear calculations require spin-orbit '
                             'coupling for nonzero orbital magnetization.')
    if not calc.params.soc:
        import warnings
        warnings.warn('Non-collinear calculation was performed without spin'
                      '-orbit coupling. Orbital magnetization may not be '
                      'accurate.')
    assert calc.wfs.bd.comm.size == 1 and calc.wfs.gd.comm.size == 1

    orbmag_av = np.zeros([len(calc.atoms), 3])
    for wfs in calc.wfs.kpt_u:
        f_n = wfs.f_n
        for (a, P_nsi), setup in zip(wfs.P_ani.items(), calc.setups):
            orbmag_av[a] += calculate_orbmag_1k(f_n, P_nsi, setup.l_j)

    calc.wfs.kd.comm.sum(orbmag_av)

    return orbmag_av


def get_orbmag_from_soc_eigs(soc):
    "Return orbital magnetization vectors calculated from nscf spinors."
    assert soc.bcomm.size == 1 and soc.domain_comm.size == 1

    orbmag_av = np.zeros([len(soc.l_aj), 3])
    for wfs, weight in zip(soc.wfs.values(), soc.weights()):
        f_n = wfs.f_m * weight
        for a, l_j in soc.l_aj.items():
            orbmag_av[a] += calculate_orbmag_1k(f_n, wfs.projections[a], l_j)

    soc.kpt_comm.sum(orbmag_av)

    return orbmag_av


def calculate_orbmag_1k(f_n, P_nsi, l_j):
    """Calculate contribution to orbital magnetization for a single k-point.

    Only partial waves with the same radial function (j index) may yield
    nonzero contributions, so the sum can be limited to diagonal blocks
    of shape [2 * l_j + 1, 2 * l_j +1] where l_j is the angular momentum
    quantum number of the j'th radial function."""
    orbmag_v = np.zeros(3)
    Ni = 0
    for l in l_j:
        Nm = 2 * l + 1
        for v in range(3):
            orbmag_v[v] += np.einsum('nsi,nsj,n,ij->',
                                     P_nsi[:, :, Ni:Ni + Nm].conj(),
                                     P_nsi[:, :, Ni:Ni + Nm],
                                     f_n, L_vlmm[v][l]).real
        Ni += Nm

    return orbmag_v
