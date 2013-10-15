import numpy as np
from ase.units import Ha
from ase.dft import monkhorst_pack
from ase.parallel import paropen
from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.mpi import size, serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation
from gpaw.test import equal

kpts = monkhorst_pack((4,4,4))
kpts += np.array([1/8., 1/8., 1/8.])

bulk = bulk('Na', 'bcc', a=4.23)

ecut = 350
calc = GPAW(mode=PW(ecut),dtype=complex, basis='dzp', kpts=kpts, 
            parallel={'domain': 1}, txt='gs_occ_pw.txt', nbands=4,
              occupations=FermiDirac(0.01),
            setups={'Na':'1'},
              )
bulk.set_calculator(calc)
bulk.get_potential_energy()
calc.write('gs_occ_pw.gpw')

calc = GPAW('gs_occ_pw.gpw',txt='gs_pw.txt', parallel={'band': 1})
calc.diagonalize_full_hamiltonian(nbands=520)
calc.write('gs_pw.gpw', 'all')

calc = GPAW('gs_pw.gpw', communicator=serial_comm, txt=None)
rpa = RPACorrelation(calc, txt='rpa_%s.txt' %(ecut), cuda=True, nmultix=1)
E = rpa.get_rpa_correlation_energy(ecutlist=[120,],
                                   skip_gamma=False,
                                   directions=[[0,1.0]],
                                   kcommsize=size,
                                   dfcommsize=size)

equal(E, -1.106, 0.005)
