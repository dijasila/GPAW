from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal
import numpy as np

bulk = Atoms('Li', pbc=True)
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2)#, txt=None)
bulk.set_calculator(calc)
a = np.linspace(2.6, 2.8, 5)
e = []
for x in a:
    bulk.set_cell((x, x, x))
    e1 = bulk.get_potential_energy()
    e.append(e1)

try:
    from gpaw.io.etsf import ETSFWriter
except ImportError:
    pass  # Scientific.IO.NetCDF was not installed
else:
    if calc.wfs.world.size == 1:
        ETSFWriter().write(calc)

fit = np.polyfit(a, e, 2)
a0 = np.roots(np.polyder(fit, 1))[0]
e0 = np.polyval(fit, a0)
print 'a,e =', a0, e0
equal(a0, 2.66, 0.01)
