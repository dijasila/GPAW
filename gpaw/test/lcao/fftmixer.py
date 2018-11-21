from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import FFTMixer
from gpaw import LCAO
from gpaw.test import equal

bulk = Atoms('Li', pbc=True,
             cell=[2.6, 2.6, 2.6])
k = 4
bulk.calc = GPAW(mode=LCAO(),
                 kpts=(k, k, k),
                 mixer=FFTMixer())
e = bulk.get_potential_energy()
equal(e, -1.7103633745906508, 1.0e-6)
