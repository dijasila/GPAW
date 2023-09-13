from gpaw import GPAW
from gpaw.elph import ElectronPhononMatrix

calc = GPAW("scf.gpw")
atoms = calc.atoms

elph = ElectronPhononMatrix(atoms, 'supercell', 'elph')
q = [[0., 0., 0.], ]
g_sqklnn = elph.bloch_matrix(calc, k_qc=q,
                             savetofile=True, prefactor=False)
