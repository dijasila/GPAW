from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.elph.electronphonon import ElectronPhononCoupling

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode='lcao',
            basis='dzp',
            kpts=(5, 5, 5),
            xc='PBE',
            occupations=FermiDirac(0.01),
            symmetry={'point_group': False},
            convergence={'bands':'nao'},
            )
atoms.calc = calc

elph = ElectronPhononCoupling(atoms, atoms.calc, calculate_forces=False)
#elph.set_lcao_calculator(atoms.calc)
elph.load_supercell_matrix(multiple=False)

# NOT FINISHED YET

elph.bloch_matrix(kpts=(5,5,5), qpts=[], c_kn, u_ql,
                     omega_ql=None)
