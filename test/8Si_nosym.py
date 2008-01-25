from gpaw import Calculator
from ase import *
from ase.calculators import numeric_force
from gpaw.utilities import equal

E = []

for xc in ['GLLB','LDA']:
    for flag in [True, False]:
        a = 5.404
        bulk = Atoms(symbols='Si8',
                     positions=[(0, 0, 0),
                                (0, 0.5, 0.5),
                                (0.5, 0, 0.5),
                                (0.5, 0.5, 0),
                                (0.25, 0.25, 0.25),
                                (0.25, 0.75, 0.75),
                                (0.75, 0.25, 0.75),
                                (0.75, 0.75, 0.25)],
                               pbc=True)
        bulk.set_cell((a, a, a))
        n = 20
        calc = Calculator(gpts=(n, n, n),
                          nbands=8*3,
                          width=0.01, xc = xc, 
                          kpts=(2, 2, 2), usesymm=flag, 
                          convergence = {'energy': 0.00005, # eV
                                        'density': 1.0e-6,
                                        'eigenstates': 1.0e-13,
                                        'bands': 'occupied'})
        bulk.set_calculator(calc)
        E.append(bulk.get_potential_energy())

equal(E[0], E[1], 0.0001)
equal(E[2], E[3], 0.0001)

