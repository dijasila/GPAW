import pytest
import numpy as np
from gpaw.mpi import world
from gpaw.atom.generator2 import generate, get_parameters
from argparse import Namespace


@pytest.mark.skipif(world.size > 1, reason='Not parallelized')
def test_diamond(in_tmp_dir):
    ns = Namespace(traceback=False, command='dataset', parallel=None,
                   symbol='C', xc_functional='PBE', configuration=None,
                   projectors=None, radius=None, zero_potential=None,
                   pseudo_core_density_radius=None, pseudize=None,
                   plot=False, logarithmic_derivatives=None, write=True,
                   scalar_relativistic=False, no_check=False, tag=None,
                   alpha=None, gamma=0.0, create_basis_set=False,
                   nlcc=False, core_hole=None, electrons=None,
                   ri=None, omega=0.11)

    gen = generate(**get_parameters('C', ns))
    setup = gen.make_paw_setup()
    setup.write_xml()

    from gpaw import setup_paths
    setup_paths.insert(0, '.')

    from ase.build import bulk
    from gpaw import GPAW
    atoms = bulk('C', 'diamond')
    atoms.calc = GPAW(kpts={'size': (2, 2, 2), 'gamma': True},
                      mode='lcao', basis='dzp',
                      xc='HSE06WIP:backend=ri')

    del setup_paths[0]
    # NOTE: HSE06 does not yet work. This is just a placeholder for
    # integration test.
    assert np.allclose(atoms.get_potential_energy(), +5.315192)
