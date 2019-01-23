import numpy as np

from ase.units import Hartree, Bohr

from gpaw.xc.hybrid import HybridXCBase


def calculate_forces(wfs, dens, ham, log=None):
    """Return the atomic forces."""

    assert not isinstance(ham.xc, HybridXCBase)

    if hasattr(wfs.eigensolver, 'odd'):
        odd_name = getattr(wfs.eigensolver.odd, "name", None)
    else:
        odd_name = None
    if odd_name == 'PZ_SIC':
        for kpt in wfs.kpt_u:
            kpt.rho_MM = \
                wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)

    natoms = len(wfs.setups)

    # Force from projector functions (and basis set):
    F_wfs_av = np.zeros((natoms, 3))
    wfs.calculate_forces(ham, F_wfs_av)
    wfs.gd.comm.sum(F_wfs_av, 0)

    F_ham_av = np.zeros((natoms, 3))

    try:
        # ODD functionals need force corrections for each spin
        correction = ham.xc.setup_force_corrections
    except AttributeError:
        pass
    else:
        correction(F_ham_av)

    ham.calculate_forces(dens, F_ham_av)

    F_av = F_ham_av + F_wfs_av
    wfs.world.broadcast(F_av, 0)

    if odd_name == 'PZ_SIC':
        F_av += wfs.eigensolver.odd.get_odd_corrections_to_forces(wfs,
                                                                  dens)

        for kpt in wfs.kpt_u:
            # need to re-set rho_MM otherwise it will be used
            # it's probably better to in wfs.reset, but
            # when position changes wfs.reset is not called
            kpt.rho_MM = None

    F_av = wfs.kd.symmetry.symmetrize_forces(F_av)

    if log:
        log('\nForces in eV/Ang:')
        c = Hartree / Bohr
        for a, setup in enumerate(wfs.setups):
            log('%3d %-2s %10.5f %10.5f %10.5f' %
                ((a, setup.symbol) + tuple(F_av[a] * c)))
        log()

    return F_av
