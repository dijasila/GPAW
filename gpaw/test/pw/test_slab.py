from ase import Atoms
from ase.optimize import BFGS

from gpaw import GPAW
from gpaw import PW
from gpaw.mpi import world


def test_pw_slab():
    a = 2.65
    slab = Atoms('Li2',
                 [(0, 0, 0), (0, 0, a)],
                 cell=(a, a, 3 * a),
                 pbc=True)
    k = 4
    calc = GPAW(mode=PW(200),
                eigensolver='rmm-diis',
                parallel={'band': min(world.size, 4)},
                kpts=(k, k, 1))
    slab.calc = calc
    BFGS(slab).run(fmax=0.01)
    assert abs(slab.get_distance(0, 1) - 2.46) < 0.01
