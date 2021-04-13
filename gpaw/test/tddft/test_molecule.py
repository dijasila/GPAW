import pytest

from ase.build import molecule

from gpaw import GPAW
from gpaw.tddft import TDDFT
from gpaw.mpi import world, serial_comm
from gpaw.utilities import compiled_with_sl

from ..lcaotddft.test_molecule import only_on_master


pytestmark = pytest.mark.usefixtures('module_tmp_path')


def calculate_time_propagation(gpw_fpath, *,
                               iterations=3,
                               kick=[1e-5, 1e-5, 1e-5],
                               propagator='SICN',
                               communicator=world,
                               write=False,
                               parallel={}):
    td_calc = TDDFT(gpw_fpath,
                    propagator=propagator,
                    communicator=communicator,
                    parallel=parallel,
                    txt='td.out')
    if kick is not None:
        td_calc.absorption_kick(kick)
    td_calc.propagate(20, iterations, 'dm.dat')
    if write:
        td_calc.write('td.gpw', mode='all')
        td_calc.propagate(20, iterations, 'dm2.dat')
    communicator.barrier()


def check_dm(ref_fpath, fpath, rtol=1e-8, atol=1e-12):
    from gpaw.tddft.spectrum import read_dipole_moment_file

    world.barrier()
    _, time_ref_t, _, dm_ref_tv = read_dipole_moment_file(ref_fpath)
    _, time_t, _, dm_tv = read_dipole_moment_file(fpath)
    assert time_t == pytest.approx(time_ref_t, abs=0)
    assert dm_tv == pytest.approx(dm_ref_tv, rel=rtol, abs=atol)


# Generate different parallelization options
parallel_i = [{}]
if world.size > 1:
    parallel_i.append({'band': 2})
if compiled_with_sl():
    parallel_i.append({'sl_auto': True})
    if world.size > 1:
        parallel_i.append({'sl_auto': True, 'band': 2})


@pytest.fixture(scope='module')
@only_on_master(world)
def ground_state():
    atoms = molecule('SiH4')
    atoms.center(vacuum=4.0)

    calc = GPAW(nbands=6, h=0.4,
                convergence={'density': 1e-8},
                communicator=serial_comm,
                xc='LDA',
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')


@pytest.fixture(scope='module')
@only_on_master(world)
def time_propagation_reference(ground_state):
    calculate_time_propagation('gs.gpw',
                               communicator=serial_comm,
                               write=True)


def test_dipole_moment_values(time_propagation_reference,
                              module_tmp_path, in_tmp_dir):
    with open('dm.dat', 'w') as fd:
        fd.write('''
# Kick = [    1.000000000000e-05,     1.000000000000e-05,     1.000000000000e-05]
#            time            norm                    dmx                    dmy                    dmz
          0.00000000       6.92701356e-16    -3.798602757097e-08    -3.850923113536e-10    -2.506988148420e-10
          0.82682747      -3.11611967e-16     6.011432043389e-05     6.015251317290e-05     6.015179500177e-05
          1.65365493       1.71405522e-15     1.075009677567e-04     1.075385921602e-04     1.075337414463e-04
'''.strip())  # noqa: E501

    with open('dm2.dat', 'w') as fd:
        fd.write('''
# Kick = [    1.000000000000e-05,     1.000000000000e-05,     1.000000000000e-05]
#            time            norm                    dmx                    dmy                    dmz
          2.48048240       1.55070479e-15     1.388363880650e-04     1.388662733804e-04     1.388701173331e-04
          3.30730987      -1.85697397e-16     1.528174640313e-04     1.528352677280e-04     1.528428543052e-04
          4.13413733      -6.23799730e-17     1.497979692345e-04     1.498097215567e-04     1.498150038226e-04
'''.strip())  # noqa: E501

    rtol = 4e-4
    atol = 1e-8
    check_dm('dm.dat', module_tmp_path / 'dm.dat', rtol=rtol, atol=atol)
    check_dm('dm2.dat', module_tmp_path / 'dm2.dat', rtol=rtol, atol=atol)


@pytest.mark.parametrize('parallel', parallel_i)
@pytest.mark.parametrize('propagator', [
    'SICN', 'ECN', 'ETRSCN', 'SITE', 'SIKE'])
def test_propagation(time_propagation_reference,
                     parallel, propagator,
                     module_tmp_path, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'gs.gpw',
                               propagator=propagator,
                               parallel=parallel)
    if propagator == 'SICN':
        # This is the same propagator as the reference;
        # error comes only from parallelization
        rtol = 1e-8
        if 'band' in parallel:
            rtol = 6e-4
    elif propagator == 'SITE':
        # This propagator seems to be unstable?
        rtol = 8e-1
    else:
        # Other propagators match qualitatively
        rtol = 5e-2
    check_dm(module_tmp_path / 'dm.dat', 'dm.dat', rtol=rtol)


@pytest.mark.parametrize('parallel', parallel_i)
def test_restart(time_propagation_reference,
                 parallel,
                 module_tmp_path, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'td.gpw',
                               kick=None,
                               parallel=parallel)
    rtol = 1e-8
    if 'band' in parallel:
        rtol = 5e-4
    check_dm(module_tmp_path / 'dm2.dat', 'dm.dat', rtol=rtol)
