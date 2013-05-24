from ase.io import read
from gpaw import GPAW, ConvergenceError
from gpaw.mixer import Mixer
from gpaw.utilities import compiled_with_sl

# the system loosing magnetic moment

slab = read('ScSZ.xyz')
slab.set_cell([[  7.307241,   0.,         0.,      ],
               [  0.,        12.656514,   0.,      ],
               [  0.,         0.,        19.,      ]],
              scale_atoms=False)
slab.center(axis=2)
magmoms = [0.0 for n in range(len(slab))]
for n, a in enumerate(slab):
    if a.symbol == 'Ni':
        magmoms[n] = 0.6
slab.set_initial_magnetic_moments(magmoms)

slab.pbc = (True, True, False)

calc = GPAW(h=0.20,
            kpts=(2,1,1),
            xc='PBE',
            width=0.1,
            maxiter=100,
            txt='ScSZ.txt',
            )
if compiled_with_sl():
    if 1:  # only with rmm-diis
        calc.set(parallel={'sl_auto': True})

slab.set_calculator(calc)

try:
    slab.get_potential_energy()
except ConvergenceError:
    pass

assert not calc.scf.converged
