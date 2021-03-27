import numpy as np

from ase.phonons import Phonons
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.mpi import world


def calculate_elph(atoms, calc, delta=0.01, calculate_forces=False):
    """
    Finds the forces and effective potential at different atomic positions used
    to calculate the change in the effective potential.

    Note: Raman is measuring zone-centre phonons only.

    Note: Use calculate_forces=False, if phonons are calculated separately

    Input
    ----------
    atoms : Atoms object
    calc  : Calculator object

    Output
    ----------
    elph.*.pckl files with the effective potential
    phonons.*.pckl files with the atomic forces.
    """

    # Calculate the forces and effective potential at different nuclear
    # positions
    elph = ElectronPhononCoupling(atoms, calc, supercell=(1, 1, 1),
                                  delta=delta,
                                  calculate_forces=calculate_forces)
    elph.run()


def get_elph_elements(atoms, calc, basename=None, load_file=False,
                      dump=0, load_gx_as_needed=False):
    """
    Evaluates the dipole transition matrix elements.

    Note: This part is currently NOT parallelised properly. Use serial only!

    Input
    ----------
    atoms : Atoms object
    calc  : Calculator object (ground state)
    basename  : If you want give a specific name (gqklnn_{}.pckl)

    Output
    ----------
    gqklnn.pckl, the electron-phonon matrix elements
    """

    kpts = calc.get_ibz_k_points()
    qpts = [[0, 0, 0], ]

    # Read previous phonon calculation.
    # This only looks at gamma point phonons
    ph = Phonons(atoms=atoms, name="phonons", supercell=(1, 1, 1))
    ph.read()
    frequencies, modes = ph.band_structure(qpts, modes=True)
    if world.rank == 0:
        print("Phonon frequencies are loaded.")

    # Find el-ph matrix in the LCAO basis
    elph = ElectronPhononCoupling(atoms, calc=calc, supercell=(1, 1, 1))
    elph.set_lcao_calculator(calc)
    basis = calc.parameters['basis']
    if load_file and not load_gx_as_needed:
        elph.load_supercell_matrix(basis=basis, dump=dump)
        if world.rank == 0:
            print("Supercell matrix is loaded")
    elif not load_file:
        # not parellel
        elph.calculate_supercell_matrix(dump=dump, include_pseudo=True)
        if world.rank == 0:
            print("Supercell matrix is calculated")

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
                                    load_gx_as_needed=load_gx_as_needed,
                                    spin=s, basis=basis)
        g_sqklnn.append(g_qklnn)
    if world.rank == 0:
        print("Saving the elctron-phonon coupling matrix")
        if basename is None:
            np.save("gsqklnn.npy", np.array(g_sqklnn))
        else:
            np.save("gsqklnn_{}.npy".format(basename), np.array(g_sqklnn))
