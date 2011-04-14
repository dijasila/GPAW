import numpy as np
from ase import Atom, Atoms
from ase.structure import bulk
from ase.units import Hartree, Bohr
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE


GS = 1
bse = 1
check = 1

if GS:

    a = 5.431 # From PRB 73,045112 (2006)
    atoms = bulk('Si', 'diamond', a=a)
    calc = GPAW(h=0.2,
                kpts=(2,2,2),
                occupations=FermiDirac(0.001),
                nbands=8,
                convergence={'band':'all'})

    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Si.gpw','all')

if bse:

    eshift = 1.224
    
    bse = BSE('Si.gpw',w=np.linspace(0,20,201),
              q=np.array([0,0,0.0001]),optical_limit=True,ecut=10.,
              nc=np.array([4,6]), nv=np.array([2,4]), eshift=eshift,
              nbands=8,positive_w=True)
    
    bse.get_dielectric_function('Si_bse.dat')

if check:
    
    d = np.loadtxt('Si_bse.dat')

    Nw1 = 37
    Nw2 = 47
    if d[Nw1, 2] > d[Nw1-1, 2] and d[Nw1, 2] > d[Nw1+1, 2] \
            and d[Nw2, 2] > d[Nw2-1, 2] and d[Nw2, 2] > d[Nw2+1, 2]:
        pass
    else:
        raise ValueError('Absorption peak not correct ! ')

    if (np.abs(d[Nw1, 2] - 70.1307082035) > 1e-5
        or np.abs(d[Nw2, 2] -  50.958188969) > 1e-5):
        print d[Nw1, 2], d[Nw2, 2]
        raise ValueError('Please check spectrum strength ! ')

