import numpy as np

from ase.phonons import Phonons

from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling

calc = GPAW("scf.gpw")

kpts = calc.get_ibz_k_points()
qpts = [[0, 0, 0], ]

# Phonon calculation, We'll read the forces from the elph.run function
# This only looks at gamma point phonons
ph = Phonons(atoms=calc.atoms, supercell=(1, 1, 1))
ph.read()
frequencies, modes = ph.band_structure(qpts, modes=True)

# Find el-ph matrix in the LCAO basis
elph = ElectronPhononCoupling(calc.atoms, calc=None, calculate_forces=False)
elph.set_lcao_calculator(calc)
elph.load_supercell_matrix()

# Find the bloch expansion coefficients
g_sqklnn = []
for s in range(calc.wfs.nspins):
    c_kn = []
    for k in range(calc.wfs.kd.nibzkpts):
        C_nM = calc.wfs.collect_array('C_nM', k, s)
        c_kn.append(C_nM)
    c_kn = np.array(c_kn)

    # And we finally find the electron-phonon coupling matrix elements!
    # elph.g_xNNMM = elph.g_xsNNMM[:, s]
    g_qklnn = elph.bloch_matrix(kpts, qpts, c_kn, u_ql=modes, spin=s)
    g_sqklnn.append(g_qklnn)
    # np.save("gsqklnn.npy", np.array(g_sqklnn))
