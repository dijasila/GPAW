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

for cuda in (True, False):
    rpa = RPACorrelation(calc, txt='rpa_all.txt',cuda=cuda, sync=False, nmultix=50)
    
    if rank == 0 :
        t0 = time()
    E_e = rpa.get_rpa_correlation_energy(ecutlist=[250, 280, 300],
                                         directions=[[0, 1.]],
                                         kcommsize=size,
                                         dfcommsize=size)
    
    saved_data = np.array([
            [-12.5422912133, -12.7824025558, -12.9350374962],
            [-11.4404771161, -11.7085244832, -11.75397295  ], 
            [-11.4402748241, -11.7083585036, -11.7537734758], 
            [-11.3414419581, -11.5722960916, -11.8343593908], 
            [-11.4402765766, -11.7083622588, -11.7537752176], 
            [-11.3414427223, -11.5722968542, -11.8343601755], 
            [-11.3412583181, -11.5721119723, -11.8341759124], 
            [-11.4328498446, -11.7016898246, -11.7469895487], 
            ])

    if cuda is False: # CPU used finite difference method
        saved_data[0,0] = -11.5398763793
        saved_data[0,1] = -11.7905922944
        saved_data[0,2] = -11.9306422869
    
    assert np.abs(rpa.E_qe - saved_data).sum() < 1e-7

    if rank == 0:
        print 'Cuda is %s, and takes %s seconds on %s cores' %(cuda, time()-t0, size)
    
    if cuda is True and rank == 0 :
        t1 = time() - t0
        if t1 > 250: # 4 mins
            print 'Takes too much time ! '



