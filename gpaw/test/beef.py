from ase import Atoms
from ase.dft.bee import BEEFEnsemble
from gpaw import GPAW
from gpaw.test import equal, gen
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('MGGA_X_MBEEF') is not None

c = {'energy': 0.001, 'eigenstates': 1, 'density': 1}
d = 0.75

gen('H', xcname='PBEsol')

for xc, E0, dE0 in [('mBEEF', 4.86, 0.16),
                    ('BEEF-vdW', 5.13, 0.20),
                    ('mBEEF-vdW', 4.74, 0.36)]:
    print(xc)
    if not newlibxc and xc[0] == 'm':
        print('Skipped')
        continue
        
    # H2 molecule:
    h2 = Atoms('H2', [[0, 0, 0], [0, 0, d]])
    h2.center(vacuum=2)
    h2.calc = GPAW(txt='H2-' + xc + '.txt', convergence=c)
    h2.get_potential_energy()
    h2.calc.set(xc=xc)
    e_h2 = h2.get_potential_energy()
    f = h2.get_forces()
    ens = BEEFEnsemble(h2.calc)
    de_h2 = ens.get_ensemble_energies()

    # H atom:
    h = Atoms('H', cell=h2.cell, magmoms=[1])
    h.center()
    h.calc = GPAW(txt='H-' + xc + '.txt', convergence=c)
    h.get_potential_energy()
    h.calc.set(xc=xc)
    e_h = h.get_potential_energy()
    ens = BEEFEnsemble(h.calc)
    de_h = ens.get_ensemble_energies()

    # binding energy
    E = 2 * e_h - e_h2
    dE = (2 * de_h - de_h2).std()
    print(E, dE)
    equal(E, E0, 0.01)
    equal(dE, dE0, 0.01)
