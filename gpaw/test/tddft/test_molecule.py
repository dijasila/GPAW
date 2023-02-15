import pytest

from ase.build import molecule

from gpaw import GPAW
from gpaw.tddft import TDDFT, DipoleMomentWriter
from gpaw.mpi import world, serial_comm
from gpaw.utilities import compiled_with_sl

from ..lcaotddft.test_molecule import only_on_master


pytestmark = pytest.mark.usefixtures('module_tmp_path')


def calculate_time_propagation(gpw_fpath, *,
                               iterations=3,
                               kick=[1e-5, 1e-5, 1e-5],
                               propagator='SICN',
                               communicator=world,
                               write_and_continue=False,
                               force_new_dm_file=False,
                               parallel={}):
    td_calc = TDDFT(gpw_fpath,
                    propagator=propagator,
                    communicator=communicator,
                    parallel=parallel,
                    txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat',
                       force_new_file=force_new_dm_file)
    if kick is not None:
        td_calc.absorption_kick(kick)
    td_calc.propagate(20, iterations)
    if write_and_continue:
        td_calc.write('td.gpw', mode='all')
        # Switch dipole moment writer and output
        td_calc.observers.pop()
        dm = DipoleMomentWriter(td_calc, 'dm2.dat', force_new_file=True)
        dm._update(td_calc)
        td_calc.propagate(20, iterations)
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
                               write_and_continue=True)


@pytest.mark.later
def test_dipole_moment_values(time_propagation_reference,
                              module_tmp_path, in_tmp_dir):
    with open('dm.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000      -8.56088154e-17     4.296381447369e-08     1.289379568857e-09    -1.233204211725e-09
# Kick = [    1.000000000000e-05,     1.000000000000e-05,     1.000000000000e-05]; Time = 0.00000000
          0.00000000       4.25434810e-16     1.805954997902e-08    -2.385463813123e-08    -2.583325884784e-08
          0.82682747       1.27894097e-15     6.018910259134e-05     6.012624112181e-05     6.013666678706e-05
          1.65365493      -8.19328094e-17     1.075705106312e-04     1.075209970999e-04     1.075282128476e-04
          2.48048240       2.59437532e-15     1.388924710065e-04     1.388577267811e-04     1.388634386442e-04
'''.strip())  # noqa: E501

    with open('dm2.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
          2.48048240       2.59437532e-15     1.388924710065e-04     1.388577267811e-04     1.388634386442e-04
          3.30730987      -1.73239889e-15     1.528374994804e-04     1.528206041482e-04     1.528241519002e-04
          4.13413733      -3.76976990e-16     1.497881512265e-04     1.497877814862e-04     1.497895148696e-04
          4.96096480       2.28911833e-15     1.323974599858e-04     1.324119052030e-04     1.324118569132e-04
'''.strip())  # noqa: E501

    rtol = 5e-4
    atol = 1e-8
    check_dm('dm.dat', module_tmp_path / 'dm.dat', rtol=rtol, atol=atol)
    check_dm('dm2.dat', module_tmp_path / 'dm2.dat', rtol=rtol, atol=atol)


@pytest.mark.later
@pytest.mark.parametrize('parallel', parallel_i)
@pytest.mark.parametrize('propagator', [
    'SICN', 'ECN', 'ETRSCN', 'SIKE'])
def test_propagation(time_propagation_reference,
                     parallel, propagator,
                     module_tmp_path, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'gs.gpw',
                               propagator=propagator,
                               parallel=parallel)
    atol = 1e-12
    if propagator == 'SICN':
        # This is the same propagator as the reference;
        # error comes only from parallelization
        rtol = 1e-8
        if 'band' in parallel:
            # XXX band parallelization is inaccurate!
            rtol = 7e-4
            atol = 5e-8
    else:
        # Other propagators match qualitatively
        rtol = 5e-2
        if 'band' in parallel:
            # XXX band parallelization is inaccurate!
            atol = 5e-8
    check_dm(module_tmp_path / 'dm.dat', 'dm.dat', rtol=rtol, atol=atol)


@pytest.mark.later
@pytest.mark.parametrize('parallel', parallel_i)
def test_restart(time_propagation_reference,
                 parallel,
                 module_tmp_path, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'td.gpw',
                               kick=None,
                               force_new_dm_file=True,
                               parallel=parallel)
    rtol = 1e-8
    if 'band' in parallel:
        rtol = 5e-4
    check_dm(module_tmp_path / 'dm2.dat', 'dm.dat', rtol=rtol)
