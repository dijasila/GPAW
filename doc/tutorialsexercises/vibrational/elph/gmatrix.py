from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.elph import ElectronPhononMatrix

atoms = bulk('Si', 'diamond', a=5.431)

calc = GPAW(mode='lcao', h=0.18, basis='dzp',
            kpts=(11, 11, 11),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            convergence={'energy': 2e-5, 'density': 1e-5},
            txt='scf.txt')
atoms.calc = calc
atoms.get_potential_energy()

elph = ElectronPhononMatrix(atoms, 'supercell', 'elph')
q = [[0., 0., 0.], ]
g_sqklnn = elph.bloch_matrix(calc, k_qc=q,
                             savetofile=True, prefactor=True)
