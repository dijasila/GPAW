from ase.io import read
from gpaw import GPAW, ConvergenceError
from gpaw.poisson import PoissonSolver
from gpaw.dipole_correction import DipoleCorrection

slab = read('Pt_H2O.xyz')
slab.set_cell([[  8.527708,   0.,         0.,      ],
               [  0.,         4.923474,   0.,      ],
               [  0.,         0.,        16.,      ]],
              scale_atoms=False)
slab.center(axis=2)

slab.pbc = (True, True, False)

calc = GPAW(h=0.20,
            kpts=(2,4,1),
            xc='RPBE',
            poissonsolver=DipoleCorrection(PoissonSolver(),2),
            basis='dzp',
            maxiter=200,
            width=0.1,
            txt='Pt_H2O.txt',
            )

slab.set_calculator(calc)
try:
    slab.get_potential_energy()
except ConvergenceError:
    pass

assert not calc.scf.converged
