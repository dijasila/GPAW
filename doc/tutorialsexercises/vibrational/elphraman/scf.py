from ase.io import read
from gpaw import GPAW, FermiDirac


atoms = read("MoS2_2H_relaxed_PBE.json")

calc = GPAW(mode='lcao',
            parallel={'sl_auto': True, 'augment_grids': True},
            xc='PBE',
            h=0.14,
            kpts=(6, 6, 3),
            occupations=FermiDirac(0.02),
            convergence={'energy': 1e-6,
                         'density': 1e-6,
                         'bands': 34},
            symmetry='off',
            )

atoms.calc = calc
atoms.get_potential_energy()

calc.write("scf.gpw", 'all')
