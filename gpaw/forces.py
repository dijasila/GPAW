import numpy as np

from ase.units import Hartree, Bohr

from gpaw.xc.hybrid import HybridXCBase

X = [[], [], []]


def calculate_forces(wfs, dens, ham, log=None):
    """Return the atomic forces."""

    assert not isinstance(ham.xc, HybridXCBase)

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

    nt_sG = dens.gd.zeros(1)
    wfs.calculate_density_contribution(nt_sG)
    nt_sG += dens.nct_G
    _, nt_Q = dens.pd2.interpolate(nt_sG[0], dens.pd3)
    vt_av = ham.vt.dict(derivative=True)
    ham.vt.derivative(dens.nt_Q - nt_Q, vt_av)
    for a, dF_v in vt_av.items():
        pass#F_ham_av[a] -= dF_v[0]

    F_av = F_ham_av + F_wfs_av
    print(vt_av[0][0, 2], F_av[0, 2] - -0.12088694004903458,
          F_av[0, 2] - -0.12088694004903458 - vt_av[0][0, 2])
    X[0].append(vt_av[0][0, 2])
    X[1].append(F_av[0, 2] - -0.12088694004903458)
    X[2].append(F_av[0, 2] - -0.12088694004903458 - vt_av[0][0, 2])
    wfs.world.broadcast(F_av, 0)

    F_av = wfs.kd.symmetry.symmetrize_forces(F_av)

    if log:
        log('\nForces in eV/Ang:')
        c = Hartree / Bohr
        for a, setup in enumerate(wfs.setups):
            log('%3d %-2s %10.5f %10.5f %10.5f' %
                ((a, setup.symbol) + tuple(F_av[a] * c)))
        log()
        import matplotlib.pyplot as plt
        plt.plot(X[0])
        plt.plot(X[1])
        plt.plot(X[2])
        plt.show()
    return F_av
