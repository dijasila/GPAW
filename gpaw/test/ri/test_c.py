import pytest
import numpy as np
from gpaw.mpi import world


@pytest.mark.skipif(world.size > 1, reason='Not parallelized')
def test_diamond():
    import subprocess
    print(subprocess.run("gpaw -T dataset -f PBE C -w --omega=0.11".split()))

    from gpaw import setup_paths
    setup_paths.insert(0, '.')
    from ase.build import bulk
    from gpaw import GPAW
    atoms = bulk('C', 'diamond')
    atoms.calc = GPAW(kpts={'size': (2, 2, 2), 'gamma': True},
                      mode='lcao', basis='dzp',
                      xc='HSE06WIP:backend=ri')

    # NOTE: HSE06 does not yet work. This is just a placeholder for
    # integration test.
    assert np.allclose(atoms.get_potential_energy(), +5.532720)


if __name__ == "__main__":
    test_diamond()
