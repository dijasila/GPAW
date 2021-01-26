import numpy as np
import pytest

from ase.build import fcc111

from gpaw import GPAW
from gpaw.mpi import world, serial_comm
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter, WaveFunctionReader
from gpaw.utilities import compiled_with_sl

from .test_molecule import only_on_master, calculate_error

pytestmark = pytest.mark.usefixtures('module_tmp_path')


# Generate different parallelization options
parallel_i = [{}]
if compiled_with_sl():
    parallel_i.append({'sl_auto': True})
    if world.size > 1:
        parallel_i.append({'sl_auto': True, 'band': 2})


@pytest.fixture(scope='module')
@only_on_master(world)
def initialize_system():
    comm = serial_comm

    # Ground-state calculation
    atoms = fcc111('Al', size=(1, 1, 2), vacuum=4.0)
    atoms.symbols[0] = 'Li'
    calc = GPAW(nbands=4,
                h=0.4,
                kpts={'size': (3, 3, 1)},
                basis='sz(dzp)',
                mode='lcao',
                convergence={'density': 1e-8},
                symmetry={'point_group': False},
                communicator=comm,
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    td_calc = LCAOTDDFT('gs.gpw',
                        communicator=comm,
                        txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    WaveFunctionWriter(td_calc, 'wf.ulm')
    td_calc.absorption_kick([0, 0, 1e-5])
    td_calc.propagate(20, 3)


def test_propagated_wave_function(initialize_system, module_tmp_path):
    wfr = WaveFunctionReader(module_tmp_path / 'wf.ulm')
    coeff = wfr[-1].wave_functions.coefficients
    coeff = coeff[np.ix_([0], [0, 1], [1, 3], [0, 1, 2])]
    ref = [[[[-5.4119034398864430e-01 - 4.6958807325576735e-01j,
              5.8836045927143954e-01 + 5.1047688429408378e-01j,
              6.5609314466400698e-06 + 5.8109609173527947e-06j],
             [1.6425837099429430e-06 - 1.4779657236004961e-06j,
              -8.7230715222772428e-07 + 8.9374679369814926e-07j,
              3.1300283337601806e+00 - 2.7306795126551076e+00j]],
            [[-1.9820345503468246e+00 - 1.0562314330323577e+00j,
              1.5008623926242098e-01 - 4.5817475674967340e-01j,
              4.8385783015916195e-01 + 5.3676335879786385e-01j],
             [-2.4227856141643818e+00 - 3.7767002050641824e-01j,
              2.6174901880264838e+00 - 1.9885717875694848e+00j,
              -7.2641847473298660e-01 - 1.6020733667409095e+00j]]]]
    err = calculate_error(coeff, ref)
    assert err < 1e-12


@pytest.mark.parametrize('parallel', parallel_i)
def test_propagation(initialize_system, module_tmp_path, parallel, in_tmp_dir):
    td_calc = LCAOTDDFT(module_tmp_path / 'gs.gpw',
                        parallel=parallel,
                        txt='td.out')
    WaveFunctionWriter(td_calc, 'wf.ulm')
    td_calc.absorption_kick([0, 0, 1e-5])
    td_calc.propagate(20, 3)
    world.barrier()

    wfr_ref = WaveFunctionReader(module_tmp_path / 'wf.ulm')
    wfr = WaveFunctionReader('wf.ulm')
    assert len(wfr) == len(wfr_ref)
    for i in range(1, len(wfr)):
        ref = wfr_ref[i].wave_functions.coefficients
        coeff = wfr[i].wave_functions.coefficients
        err = calculate_error(coeff, ref)
        atol = 1e-12
        assert err < atol, f'error at i={i}'
