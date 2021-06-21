import numpy as np

from ase.phonons import Phonons
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.mpi import world

# XXX DELETE
def run_elph(atoms, calc, delta=0.01, calculate_forces=True):
    """
    Finds the forces and effective potential at different atomic positions used
    to calculate the change in the effective potential.

    Use calculate_forces=False, if phonons are calculated separately.

    This is an optional wrapper. Use ElectronPhononCoupling directly,
    if you want to change further parameters, or want to use a supercell.

    Parameters
    ----------
    atoms: Atoms
        Equilibrium geometry
    calc: GPAW
        Calculator for finite displacements
    delta: float
        Displacement increment (default 0.01)
    calculate_forces: bool
        Whether to include phonon calculation (default True)
    """

    # Calculate the forces and effective potential at different nuclear
    # positions
    elph = ElectronPhononCoupling(atoms, calc, supercell=(1, 1, 1),
                                  delta=delta,
                                  calculate_forces=calculate_forces)
    elph.run()


# DELETE
def calculate_supercell_matrix(atoms, calc, dump=1):
    """
    Calculate elph supercell matrix.

    This is a necessary intermediary step before calculating the electron-
    phonon matrix.

    This is an optional wrapper. Use ElectronPhononCoupling directly,
    if you want to change further parameters, or want to use a supercell.

    Parameters
    ----------
    atoms: Atoms
        Equilibrium geometry
    calc: GPAW
        Converged ground-state calculation. Same grid as before.
    dump: int
        Whether to write elph supercell matrix in one file(1), several
        files (2) or not at all (0).
    """
    assert dump in (0, 1, 2)
    elph = ElectronPhononCoupling(atoms, supercell=(1, 1, 1))
    elph.set_lcao_calculator(calc)
    elph.calculate_supercell_matrix(dump=dump, include_pseudo=True)
    return elph


def get_elph_matrix(atoms, calc, elphcache='elph', phononcache='elph', dump=1,
                    load_gx_as_needed=False, elph=None):
    """
    Evaluates the dipole transition matrix elements.

    Note: This part is currently NOT parallelised properly. Use serial only!

    Parameters:

    atoms: Atoms
        Equilibrium geometry
    calc: GPAW
        Converged ground-state calculation.
    elphcache: str
        String used for elph cache
    elphcache: str
        String used for phonon cache
    dump: (0, 1, 2)
        Whether to elph matrix was written in one file(1), several files (2) or
        not at all (0).
    load_gx_as_needed: bool
        If dump=2 allows to load elph elements as needed, instead of the whole
        matrix. Recommended for large systems.
    elph: ElectronPhononCoupling object
        If dump=0 an ElectronPhononCoupling onject containing the supercell
        matrix must be supplied. Only recommend for smallest of systems.
    """

    kpts = calc.get_ibz_k_points()
    qpts = [[0, 0, 0], ]

    # Read previous phonon calculation.
    # This only looks at gamma point phonons
    ph = Phonons(atoms=atoms, supercell=(1, 1, 1), name=phononcache)
    ph.read()
    frequencies, modes = ph.band_structure(qpts, modes=True)

    # Find el-ph matrix in the LCAO basis
    if elph is None:
        elph = ElectronPhononCoupling(atoms, name=elphcache,
                                      supercell=(1, 1, 1))
        elph.set_lcao_calculator(calc)
        basis = calc.parameters['basis']
    if not load_gx_as_needed:
        elph.load_supercell_matrix(basis=basis, dump=dump)

    # Find the bloch expansion coefficients
    g_sqklnn = []
    for s in range(calc.wfs.nspins):
        # c_kn = np.zeros((nk, nbands, nbands), dtype=complex)
        c_kn = []
        for k in range(calc.wfs.kd.nibzkpts):
            C_nM = calc.wfs.collect_array('C_nM', k, s)
            # if world.rank == 0:
            c_kn.append(C_nM)
        c_kn = np.array(c_kn)

        # And we finally find the electron-phonon coupling matrix elements!
        if not load_gx_as_needed:
            elph.g_xNNMM = elph.g_xsNNMM[:, s]
        g_qklnn = elph.bloch_matrix(kpts, qpts, c_kn, u_ql=modes,
                                    spin=s, basis=basis,
                                    load_gx_as_needed=load_gx_as_needed)
        g_sqklnn.append(g_qklnn)

    if world.rank == 0:
        np.save("gsqklnn.npy", np.array(g_sqklnn))
    return g_sqklnn
