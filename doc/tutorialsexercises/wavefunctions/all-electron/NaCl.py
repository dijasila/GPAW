# web-page: all_electron.csv
import numpy as np

from ase.build import molecule
from gpaw import GPAW
from ase.parallel import paropen

unitcell = np.array([6.5, 6.6, 9.])
gridrefinement = 2

f = paropen('all_electron.csv', 'w')

for formula in ('Na', 'Cl', 'NaCl',):
    if formula in ['Na', 'Cl']:
        hund = True
    else:
        hund = False
    calc = GPAW(mode='fd',
                xc='PBE',
                h=0.18,
                hund=hund,
                convergence={'eigenstates': 1e-8},
                txt=formula + '.txt')

    sys = molecule(formula, cell=unitcell, calculator=calc)
    sys.center()
    sys.get_potential_energy()

    # Get densities
    nt = calc.get_pseudo_density()
    n = calc.get_all_electron_density(gridrefinement=gridrefinement)

    # Get integrated values
    dv = sys.get_volume() / calc.get_number_of_grid_points().prod()
    It = nt.sum() * dv
    I = n.sum() * dv / gridrefinement**3
    print('%-4s,%4.2f,%5.2f' % (formula, It, I), file=f)

f.close()
