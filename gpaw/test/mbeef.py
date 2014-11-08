from ase.structure import molecule
from ase.dft.bee import BEEF_Ensemble
from ase.parallel import rank, barrier
from gpaw import GPAW
from gpaw.test import equal
import numpy as np

xc = 'mBEEF'
h = 0.18
conv  = {'eigenstates':1.e2, 'density':1.e-2, 'energy':1.e-4}
list  = ['N2', 'N']
tol1 = 1.e-4
tol2 = 1.e-1

for i,a in enumerate(list):
    # ASE
    atoms = molecule(a)
    atoms.center(vacuum=3.0)

    # scf
    calc = GPAW(xc='PBE',
                h=h,
                convergence=conv,
                nbands=-2)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    atoms.calc.set(xc=xc)
    atoms.get_potential_energy()

    # forces
    if i == 0:
        f = atoms.get_forces()
        f0 = f[0].sum()
        f1 = f[1].sum()
        equal(f0, -f1, tol1)

    # BEE ensemble
    ens = BEEF_Ensemble(calc)
    ens.get_ensemble_energies()
    ens.write(a)

    del atoms, calc, ens

# evaluate binding energy and ensemble error estimate
barrier
if rank == 0:
    ens = BEEF_Ensemble()
    e1, de1 = ens.read(list[0])
    e2, de2 = ens.read(list[1])
    e = 2*e2 - e1
    de = np.std(2*de2 - de1)
    equal(e, 9.8, tol2)
    equal(de, 0.4, tol2)
