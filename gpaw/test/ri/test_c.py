import pytest
import numpy as np
from gpaw.mpi import world
from gpaw.atom.generator2 import generate


@pytest.mark.skipif(world.size > 1, reason='Not parallelized')
def test_diamond(in_tmp_dir, add_cwd_to_setup_paths):
    parameters = {'symbol': 'C', 'Z': 6, 'xc': 'PBE',
                  'projectors': '2s,s,2p,p,d', 'radii': [1.2],
                  'scalar_relativistic': False, 'r0': 1.2,
                  'v0': None,
                  'nderiv0': 2, 'pseudize': ('poly', 4),
                  'omega': 0.11}

    gen = generate(**parameters)
    setup = gen.make_paw_setup()
    setup.write_xml()

    from ase.build import bulk
    from gpaw import GPAW
    atoms = bulk('C', 'diamond')
    atoms.calc = GPAW(kpts={'size': (2, 2, 2), 'gamma': True},
                      mode='lcao', basis='dzp',
                      xc='HSE06WIP:backend=ri')

    # NOTE: HSE06 does not yet work. This is just a placeholder for
    # integration test.
    assert np.allclose(atoms.get_potential_energy(), +5.315192)

    calc = atoms.calc

    assert np.allclose(calc.density.setups[0].M_pp, calc.density.setups[0].M_wpp[0.11], rtol=1e-2, atol=1e-4)
    
