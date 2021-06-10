import numpy as np
import pytest

from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import world, serial_comm
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter
from gpaw.utilities import compiled_with_sl

from .test_molecule import only_on_master, calculate_error


pytestmark = pytest.mark.usefixtures('module_tmp_path')


parallel_i = [{}]
if compiled_with_sl():
    parallel_i.append({'sl_auto': True})
    if world.size > 1:
        parallel_i.append({'band': 2})
        parallel_i.append({'sl_auto': True, 'band': 2})


def check_mm(ref_fpath, data_fpath, atol):
    world.barrier()
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
    dmat = DensityMatrix(td_calc)
    MagneticMomentWriter(td_calc, 'mm.dat', dmat=dmat)
    MagneticMomentWriter(td_calc, 'mm_grid.dat', calculate_on_grid=True)
    # MagneticMomentWriter(td_calc, 'mm.dat', origin_shift=[1.0, 2.0, 3.0])
    td_calc.absorption_kick([1e-5, 0., 0.])
    td_calc.propagate(100, 5)


def test_magnetic_moment_values(initialize_system, module_tmp_path,
                                in_tmp_dir):
    with open('mm_ref.dat', 'w') as f:
        f.write('''
# MagneticMomentWriter[version=4](origin='COM')
# origin_v = [7.634300, 5.000000, 4.302858]
#            time                    cmx                    cmy                    cmz
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
# Kick = [    1.000000000000e-05,     0.000000000000e+00,     0.000000000000e+00]; Time = 0.00000000
          0.00000000     8.192189793082e-06     1.038446327373e-05    -2.730498071751e-05
          4.13413733     7.838837723234e-06     1.000765310013e-05    -2.573300722038e-05
          8.26827467     6.809084660174e-06     8.879683492897e-06    -2.128890950807e-05
         12.40241200     5.175350632237e-06     7.009694921954e-06    -1.462938416394e-05
         16.53654934     3.058296873929e-06     4.443905967036e-06    -6.697375210691e-06
         20.67068667     6.247451722277e-07     1.298788405738e-06     1.460017881082e-06
'''.strip())  # noqa: E501

    check_mm(module_tmp_path / 'mm.dat', 'mm_ref.dat', atol=2e-14)


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
