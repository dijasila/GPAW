import numpy as np
import pytest

from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import world, serial_comm
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter
from gpaw.utilities import compiled_with_sl

from .test_ksdecomp import only_on_master, calculate_error


pytestmark = pytest.mark.usefixtures('module_tmp_path')


parallel_i = [{}]
if compiled_with_sl():
    parallel_i.append({'sl_auto': True})
    if world.size > 1:
        parallel_i.append({'band': 2})
        parallel_i.append({'sl_auto': True, 'band': 2})


def check_mm(ref_fpath, data_fpath, atol):
    ref = np.loadtxt(ref_fpath)
    data = np.loadtxt(data_fpath)
    err = calculate_error(data, ref)
    assert err < atol


@pytest.fixture(scope='module')
@only_on_master(world)
def initialize_system():
    comm = serial_comm

    atoms = Atoms('LiNaNaNa',
                  positions=[[0.0, 0.0, 0.0],
                             [2.0, 1.0, 0.0],
                             [4.0, 0.0, 1.0],
                             [6.0, -1.0, 0.0]])
    atoms.center(vacuum=4.0)

    calc = GPAW(nbands=2,
                h=0.4,
                setups={'Na': '1'},
                basis='sz(dzp)',
                mode='lcao',
                convergence={'density': 1e-12},
                communicator=comm,
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    td_calc = LCAOTDDFT('gs.gpw',
                        communicator=comm,
                        txt='td.out')
    MagneticMomentWriter(td_calc, 'mm.dat')
    MagneticMomentWriter(td_calc, 'mm_grid.dat', calculate_on_grid=True)
    # MagneticMomentWriter(td_calc, 'mm.dat', origin_shift=[1.0, 2.0, 3.0])
    td_calc.absorption_kick([1e-5, 0., 0.])
    td_calc.propagate(100, 5)


def test_magnetic_moment_grid_evaluation(initialize_system, module_tmp_path):
    dpath = module_tmp_path
    check_mm(dpath / 'mm.dat', dpath / 'mm_grid.dat', atol=2e-8)


@pytest.mark.parametrize('parallel', parallel_i)
def test_magnetic_moment_parallel(initialize_system, module_tmp_path, parallel,
                                  in_tmp_dir):
    td_calc = LCAOTDDFT(module_tmp_path / 'gs.gpw',
                        parallel=parallel,
                        txt='td.out')
    MagneticMomentWriter(td_calc, 'mm.dat')
    MagneticMomentWriter(td_calc, 'mm_grid.dat', calculate_on_grid=True)
    td_calc.absorption_kick([1e-5, 0., 0.])
    td_calc.propagate(100, 5)

    check_mm(module_tmp_path / 'mm.dat', 'mm.dat', atol=3e-14)
    check_mm(module_tmp_path / 'mm_grid.dat', 'mm_grid.dat', atol=3e-14)
