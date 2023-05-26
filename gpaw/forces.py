import numpy as np

from ase.units import Hartree, Bohr

from gpaw.xc.hybrid import HybridXCBase

from ase.parallel import paropen

def write_data(file_data,mat):
    for j in np.arange(0,len(mat[:,0])):
        for k in np.arange(0,len(mat[0,:])):
            file_data.write('%14.8f ' % (mat[j,k]))
    file_data.write('\n')
    file_data.flush()
    file_data.close()

def calculate_forces(wfs, dens, ham, log=None):
    """Return the atomic forces."""

    assert not isinstance(ham.xc, HybridXCBase)
    assert not ham.xc.name.startswith('GLLB')

    natoms = len(wfs.setups)

    force_wfs = paropen('force_wfs.dat',"a")
    force_ham = paropen('force_ham.dat',"a")
    force_F = paropen('force_F.dat',"a")
    force_F_sym = paropen('force_F_sym.dat',"a")

    # Force from projector functions (and basis set):
    F_wfs_av = np.zeros((natoms, 3))
    wfs.calculate_forces(ham, F_wfs_av)
    wfs.gd.comm.sum(F_wfs_av, 0)
    write_data(force_wfs,F_wfs_av)
    F_ham_av = np.zeros((natoms, 3))

    #write_data(force_wfs,F_wfs_av)

    try:
        # ODD functionals need force corrections for each spin
        correction = ham.xc.setup_force_corrections
    except AttributeError:
        pass
    else:
        correction(F_ham_av)

    ham.calculate_forces(dens, F_ham_av)

    write_data(force_ham,F_ham_av)

    F_av = F_ham_av + F_wfs_av
    wfs.world.broadcast(F_av, 0)

    write_data(force_F,F_av)

    F_av = wfs.kd.symmetry.symmetrize_forces(F_av)

    write_data(force_F_sym,F_av)

    if log:
        log('\nForces in eV/Ang:')
        c = Hartree / Bohr
        for a, setup in enumerate(wfs.setups):
            log('%3d %-2s %10.5f %10.5f %10.5f' %
                ((a, setup.symbol) + tuple(F_av[a] * c)))
        log()

    return F_av
