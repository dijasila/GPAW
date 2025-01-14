import numpy as np
from ase.build import molecule
from gpaw import GPAW, setup_paths
setup_paths.insert(0, '.')

atoms = molecule('H2O')

h = 0.2

for L in np.arange(4, 14, 2) * 8 * h:
    atoms.set_cell((L, L, L))
    atoms.center()
    calc = GPAW(mode='fd',
                xc='PBE',
                h=h,
                nbands=-40,
                eigensolver='cg',
                setups={'O': 'hch1s'})
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    calc.write(f'h2o_hch_{L:.1f}.gpw')
