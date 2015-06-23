import numpy as np

from matplotlib import pyplot as plt

from ase import Atoms
from ase.units import Hartree
from ase.lattice import bulk

from gpaw import GPAW
from gpaw.bztools import tesselate_brillouin_zone

from gpaw.response.jdos import JDOS


# Make simple gs calc
atoms = bulk('Na')
atoms.calc = GPAW(mode='pw', kpts={'size': (4, 4, 4)})
atoms.get_potential_energy()
atoms.calc.write('gs.gpw', 'all')

# Make refined kpoint grid
rk_kc = tesselate_brillouin_zone('gs.gpw', density=5.0)
responseGS = GPAW('gs.gpw', fixdensity=True,
                  kpts=rk_kc, nbands=10)

responseGS.get_potential_energy()
responseGS.write('gsresponse.gpw', 'all')

jdos = JDOS('gsresponse.gpw')
dos = jdos.calculate()

plt.figure()
ax = plt.gca()
plt.plot(jdos.omega_w * Hartree, dos, label='JDOS')
plt.xlabel('Frequency (eV)')
plt.legend()

plt.show()
