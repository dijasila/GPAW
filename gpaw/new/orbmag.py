r"""This module calculates the orbital magnetic moment vector for each atom.

The orbital magnetic moment is calculated in the atom-centred approximation
where only the PAW correction to the wave function is assumed to contribute.
This leads to the equation (presented in SI units):

::

                 ===
   a         e   \     a    a
  m      = - --  /    D    L
   orb,v     2m  ===   ii'  vii'
                 ii'

with L^a_vii' containing the matrix elements of the angular momentum operator
between two partial waves centred at atom a.

The orbital magnetic moments are returned in units of μ_B without the sign of
the negative electronic charge, q = - e.
"""

from itertools import chain

from ase.parallel import parprint
from ase.units import Ha
from ase.utils.timing import Timer
import numpy as np

from gpaw.new import zips
from gpaw.spinorbit import get_L_vlmm
from gpaw.utilities.progressbar import ProgressBar

L_vlmm = get_L_vlmm()


def calculate_orbmag_from_density(D_asii, n_aj, l_aj):
    """Returns orbital magnetic moment vectors for each atom a
    calculated from its respective atomic density matrix.

    This method assumes that D_asii is on every rank and not parallelised.

    Parameters
    ----------
    D_asii : AtomArrays or dictionary
        Atomic density matrix for each atom a. The i-index refers to the
        partial waves of an atom and the s-index refers to 0, x, y, and z.
    n_aj : List of lists of integers
        Principal quantum number for each radial partial wave j
    l_aj : List of lists of integers
        Angular momentum quantum number for each radial partial wave j

    NB: i is an index for all partial waves for one atom and j is an index for
    only the radial wave function which is used to build all of the partial
    waves. i and j do not refer to the same kind of index.

    Only pairs of partial waves with the same radial function may yield
    nonzero contributions. The sum can therefore be limited to diagonal blocks
    of shape [2 * l_j + 1, 2 * l_j + 1] where l_j is the angular momentum
    quantum number of the j'th radial function.

    Partials with unbounded radial functions (negative n_j) are skipped.
    """

    orbmag_av = np.zeros([len(n_aj), 3])
    for (a, D_sii), n_j, l_j in zips(D_asii.items(), n_aj, l_aj):
        assert D_sii.shape[0] == 4
        D_ii = D_sii[0]  # Only the electron density

        Ni = 0
        for n, l in zips(n_j, l_j):
            Nm = 2 * l + 1
            if n < 0:
                Ni += Nm
                continue
            for v in range(3):
                orbmag_av[a, v] += np.einsum('ij,ij->',
                                             D_ii[Ni:Ni + Nm, Ni:Ni + Nm],
                                             L_vlmm[v][l]).real
            Ni += Nm
    return orbmag_av


def modern_theory(mmedata, bands=None, fermishift=None):
    """
    Calculates the orbital magnetization vector for the unit cell
    in units of Bohr magnetons / Bohr radii^3.

    Parameters
    ----------
    mmedata
        Data object of class NLOData. Contains energies wrt. Fermi level,
        occupancies and momentum matrix elements.
    bands
        Range of band indices over which the summation is performed.
    fermishift
        Shift of the Fermi energy in eV which has been set to zero during
        matix element calculations.
    """

    # Start timer
    timer = Timer()

    # Convert input in eV to Ha
    if fermishift is None:
        E_F = 0
    else:
        E_F = fermishift * Ha
        raise NotImplementedError()

    # Load the required data
    comm = mmedata.comm
    master = (comm.rank == 0)
    with timer('Distributing matrix elements'):
        data_k = mmedata.distribute()
        if data_k:
            if bands is None:
                nb = len(list(data_k.values())[0][1])
                bands = range(0, nb)

    parprint('Calculating orbital magnetization vector ' +
             f'(in {comm.size:d} cores).')
    orbmag_v = np.zeros([3])

    with timer('Performing Brillouin zone integral'):
        # Initial call to print 0 % progress
        if master:
            count = 0
            ncount = len(data_k)
            pb = ProgressBar()

        with timer('Summing over bands'):
            for dk, f_n, E_n, p_vnn in data_k.values():
                px_nn = p_vnn[0]
                py_nn = p_vnn[1]
                pz_nn = p_vnn[2]

                E_n -= E_F
                occ_n = f_n * dk

                for n1 in bands:
                    occ = occ_n[n1]
                    En1 = E_n[n1]
                    for n2 in chain(range(bands.start, n1),
                                    range(n1 + 1, bands.stop)):
                        factor = occ * (E_n[n2] + En1) / (E_n[n2] - En1)**2

                        orbmag_v[0] += factor * np.imag(
                            py_nn[n1, n2] * pz_nn[n2, n1] -
                            pz_nn[n1, n2] * py_nn[n2, n1])
                        orbmag_v[1] += factor * np.imag(
                            pz_nn[n1, n2] * px_nn[n2, n1] -
                            px_nn[n1, n2] * pz_nn[n2, n1])
                        orbmag_v[2] += factor * np.imag(
                            px_nn[n1, n2] * py_nn[n2, n1] -
                            py_nn[n1, n2] * px_nn[n2, n1])

                # Update progress bar
                if master:
                    pb.update(count / ncount)
                    count += 1

            orbmag_v /= 8 * np.pi**3

        if master:
            pb.finish()
        with timer('Summing over cores'):
            comm.sum(orbmag_v)

    if master:
        timer.write()

    return orbmag_v
