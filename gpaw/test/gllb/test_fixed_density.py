import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.mpi import world, serial_comm

from gpaw.test.lcaotddft.test_molecule import only_on_master

pytestmark = [pytest.mark.usefixtures('module_tmp_path'),
              pytest.mark.gllb,
              pytest.mark.libxc
              ]


@pytest.fixture(scope='module')
@only_on_master(world)
def ground_state_calculation():
    comm = serial_comm
    atoms = molecule('H2')
    atoms.center(vacuum=4)

    gs_calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.4,
                   xc='GLLBSC',
                   txt='gs.out',
                   communicator=comm,
                   )
    atoms.calc = gs_calc
    atoms.get_potential_energy()
    gs_calc.write('gs.gpw', mode='all')


def test_fixed_density(ground_state_calculation, module_tmp_path, in_tmp_dir):
    gs_calc = GPAW(module_tmp_path / 'gs.gpw', txt=None)
    bs_calc = gs_calc.fixed_density(parallel={'band': min(2, world.size)},
                                    txt='unocc.out',
                                    )
    bs_calc.write('unocc.gpw', mode='all')
