import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import world

class FragileG0W0(G0W0):
    def calculate_q(self, *args, **kwargs):
        if not hasattr(self, 'doom'):
            self.doom = 0
        self.doom += 1  # Advance doom
        if self.doom == 4:
            raise ValueError('Cthulhu awakens')
        G0W0.calculate_q(self, *args, **kwargs)

from gpaw import GPAW, PW
from ase import Atoms
atoms = Atoms('He', cell=[3,3,3], pbc=1)
calc = GPAW(mode=PW(ecut=150), kpts=dict(size=[2,2,2], gamma=True), nbands=10)
atoms.calc = calc
atoms.get_potential_energy()
calc.write('wfs.gpw', mode='all')

kwargs = dict(bands=(2, 4),
              nbands=10,
              ecut=40,
              kpts=[0])

# Do in a single run
gw = G0W0('wfs.gpw', 'singlerun', **kwargs)
gw.calculate()

# # Do it two calls to calculate
# while True:
#     try:
#         gw = FragileG0W0('wfs.gpw', 'restarted',  **kwargs)
#         gw.calculate()
#         break
#     except ValueError as e:
#         print(e)
#         assert str(e) == 'Cthulhu awakens'


