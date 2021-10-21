from ase.io import read
from gpaw import GPAW

atoms = read('r-methyloxirane.xyz')
atoms.center(vacuum=8.0)

calc = GPAW(mode='fd',
            h=0.2,
            xc='PBE',
            nbands=16,
            poissonsolver={'remove_moment': 1 + 3 + 5},
            convergence={'density': 1e-12},
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
