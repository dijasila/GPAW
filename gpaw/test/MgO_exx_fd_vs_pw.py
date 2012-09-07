import numpy as np
from ase.units import Ha
from ase.dft import monkhorst_pack
from ase.parallel import paropen
from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.mpi import size
from gpaw.xc.hybridk import HybridXC
from ase.io import read


nk = 4
for mode in ('pw', 'fd'):
    kpts = monkhorst_pack((nk,nk,nk))
    kshift = 1./(2*nk)
    kpts += np.array([kshift, kshift, kshift])

    atoms = bulk('MgO', 'rocksalt', a = 4.212)

    if mode == 'pw':
        calc = GPAW(mode=PW(600.),basis='dzp', xc='PBE', maxiter=300,
        kpts=kpts,parallel={'band':1, 'domain':1},
                    occupations=FermiDirac(0.01))
        atoms.set_calculator(calc)
        E1 = atoms.get_potential_energy()
        exx = HybridXC('EXX', acdf=True)
        E_hf1 = E1 + calc.get_xc_difference(exx)
        
    else:
        calc = GPAW(h=0.16,
            basis='dzp', kpts=kpts, xc='PBE',
            parallel={'domain':1, 'band':1},
            occupations=FermiDirac(0.01)
              )
        atoms.set_calculator(calc)
        E2 = atoms.get_potential_energy()
        exx = HybridXC('EXX', acdf=True)
        E_hf2 = E2 + calc.get_xc_difference(exx)

print E1, E2, E_hf1, E_hf2
assert np.abs(E1 - E2) < 0.1
assert np.abs(E_hf1 - E_hf2) < 0.5
        
    

