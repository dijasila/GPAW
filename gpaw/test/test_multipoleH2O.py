
import numpy as np

from ase.build import molecule
from ase.parallel import parprint

from gpaw import GPAW
from gpaw.analyse.multipole import Multipole
from gpaw.cluster import Cluster
from gpaw.test import equal


def test_multipoleH2O(in_tmp_dir):
    h = 0.3

    s = Cluster(molecule('H2O'))
    s.minimal_box(3., h)

    gpwname = 'H2O_h' + str(h) + '.gpw'
    s.calc = GPAW(h=h, charge=0, txt=None)
    s.get_potential_energy()
    s.calc.write(gpwname)

    dipole_c = s.get_dipole_moment()
    parprint('Dipole', dipole_c)

    center = np.array([1, 1, 1]) * 50.
    mp = Multipole(center, s.calc, lmax=2)
    q_L = mp.expand(-s.calc.density.rhot_g)
    parprint('Multipole', q_L)

    # The dipole moment is independent of the center
    equal(dipole_c[2], q_L[2], 1e-10)

    mp.to_file(s.calc, mode='w')
