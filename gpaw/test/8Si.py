from ase import Atoms
from ase.calculators.test import numeric_force
from gpaw import GPAW, FermiDirac, PoissonSolver
from gpaw.test import equal

a = 5.404
bulk = Atoms(symbols='Si8',
             positions=[(0, 0, 0.1 / a),
                        (0, 0.5, 0.5),
                        (0.5, 0, 0.5),
                        (0.5, 0.5, 0),
                        (0.25, 0.25, 0.25),
                        (0.25, 0.75, 0.75),
                        (0.75, 0.25, 0.75),
                        (0.75, 0.75, 0.25)],
             pbc=True)
bulk.set_cell((a, a, a), scale_atoms=True)
n = 20
calc = GPAW(gpts=(n, n, n),
            nbands=8*3,
            occupations=FermiDirac(width=0.01),
            poissonsolver=PoissonSolver(nn='M', relax='J'),
            kpts=(2, 2, 2),
            convergence={'energy': 1e-7}
            )
bulk.set_calculator(calc)
f1 = bulk.get_forces()[0, 2]
e1 = bulk.get_potential_energy()

f2 = numeric_force(bulk, 0, 2)
print f1,f2,f1-f2
equal(f1, f2, 0.005)

# Volume per atom:
vol = a**3 / 8
de = calc.get_electrostatic_corrections() / vol
equal(de[0], -2.0, 0.1)
