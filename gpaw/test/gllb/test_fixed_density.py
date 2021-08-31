import pytest
import numpy as np
from ase.build import molecule
from gpaw import GPAW
from gpaw.mpi import world, serial_comm, broadcast_exception

from gpaw.test.lcaotddft.test_molecule import only_on_master

pytestmark = [pytest.mark.usefixtures('module_tmp_path'),
              pytest.mark.gllb,
              pytest.mark.libxc
              ]


def check_asp(ref_asp, D_asp, atol=0):
    assert ref_asp is not D_asp, \
        'Trying to compare same objects. Is the test broken?'
    assert np.allclose(ref_asp.toarray(), D_asp.toarray(), atol=atol, rtol=0)


@pytest.fixture(scope='module')
@only_on_master(world)
def ground_state_calculation():
    comm = serial_comm
    atoms = molecule('H2O')
    atoms.center(vacuum=4)

    gs_calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.4,
                   xc='GLLBSC',
                   txt='gs.out',
                   communicator=comm,
                   )
    atoms.calc = gs_calc
    atoms.get_potential_energy()
    gs_calc.write('gs.gpw', mode='all')
    return gs_calc.hamiltonian.xc.response


def test_read(ground_state_calculation, module_tmp_path):
    ref_response = ground_state_calculation

    # Read response and collect in master
    gs_calc = GPAW(module_tmp_path / 'gs.gpw', txt=None)
    response = gs_calc.hamiltonian.xc.response
    for D_asp in [response.D_asp, response.Dresp_asp]:
        D_asp.redistribute(D_asp.partition.as_serial())
    response.vt_sG = gs_calc.wfs.gd.collect(response.vt_sG)

    with broadcast_exception(world):
        if world.rank == 0:
            check_asp(ref_response.D_asp, response.D_asp)
            check_asp(ref_response.Dresp_asp, response.Dresp_asp)
            np.allclose(ref_response.vt_sG, response.vt_sG, rtol=0, atol=0)


def test_fixed_density(ground_state_calculation, module_tmp_path, in_tmp_dir):
    gs_calc = GPAW(module_tmp_path / 'gs.gpw', txt=None)
    bs_calc = gs_calc.fixed_density(parallel={'band': min(2, world.size)},
                                    txt='unocc.out',
                                    )
    bs_calc.write('unocc.gpw', mode='all')
