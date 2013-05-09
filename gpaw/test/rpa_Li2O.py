import numpy as np
from time import time
from ase import Atoms
from ase.dft import monkhorst_pack
from ase.parallel import paropen
from ase.structure import bulk, molecule
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.xc.rpa_correlation_energy import RPACorrelation
from gpaw.mpi import serial_comm, size, rank
from ase.io import read
from ase.lattice.spacegroup import crystal

# Atoms
positions = \
np.array([[ 0.        ,  0.        ,  0.        ],
         [ 4.86346904,  2.81526974,  1.99265222],
         [ 1.61961769,  0.93753258,  0.66358699]])

atoms = Atoms(symbols='OLi2', positions=positions, cell=[[3.2479150024913865, 0.0, 0.0], [1.6175859840445548, 2.8164460114302714, 0.0], [1.6175857471665958, 0.9363563151454799, 2.6562392137523734]], pbc=[True, True, True])

# GS 
kpts = monkhorst_pack((2,2,2))
kpts += np.array([1/4., 1/4., 1/4.])

calc = GPAW(mode=PW(700),dtype=complex, basis='dzp', xc='PBE', maxiter=300,
            txt='gs_occ.txt', kpts=kpts,parallel={'band':1}, eigensolver='cg',
                        occupations=FermiDirac(0.01))
atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('gs.gpw', 'all')


# RPA
calc = GPAW('gs.gpw', communicator=serial_comm, txt=None)

rpa = RPACorrelation(calc, txt='rpa_all.txt',cuda=True, sync=False, nmultix=50)

if rank == 0 :
    t0 = time()
E_e = rpa.get_rpa_correlation_energy(ecutlist=[250, 280, 300, ],
                                     directions=[[0, 1.]],
                                     kcommsize=size,
                                     dfcommsize=size)

saved_data = np.array([
[-12.54230069, -12.78240637, -12.9350376 ],
[-11.44054596, -11.70858729, -11.75403433],
[-11.44034345, -11.70842109, -11.75383464],
[-11.34151943, -11.57236843, -11.83442544],
[-11.4403452 , -11.70842484, -11.75383638],
[-11.34152019, -11.57236919, -11.83442622],
[-11.34133587, -11.57218439, -11.83424205],
[-11.43291855, -11.70175245, -11.7470508 ],
])

assert np.abs(rpa.E_qe - saved_data).sum() < 1e-7

if rank == 0 :
    t1 = time() - t0
    if t1 > 250: # 4 mins
        print 'Takes too much time ! '



