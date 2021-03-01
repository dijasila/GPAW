from functools import wraps
import numpy as np
import pytest

from ase.build import molecule

from gpaw import GPAW
from gpaw.mpi import world, serial_comm, broadcast_float, broadcast
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.tddft.folding import frequencies
from gpaw.utilities import compiled_with_sl

pytestmark = pytest.mark.usefixtures('module_tmp_path')


def only_on_master(comm, broadcast=None):
    """Decorator for executing the function only on the rank 0.

    Parameters
    ----------
    comm
        communicator
    broadcast
        function for broadcasting the return value or
        `None` for no broadcasting
    """
    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if comm.rank == 0:
                ret = func(*args, **kwargs)
            else:
                ret = None
            comm.barrier()
            if broadcast is not None:
                ret = broadcast(ret, comm=comm)
            return ret
        return wrapped_func
    return wrap


def calculate_error(a, ref_a):
    if world.rank == 0:
        err = np.abs(a - ref_a).max()
        print()
        print('ERR', err)
    else:
        err = np.nan
    err = broadcast_float(err, world)
    return err


# Generate different parallelization options
parallel_i = [{}]
if compiled_with_sl():
    if world.size == 1:
        # Choose BLACS grid manually as the one given by sl_auto
        # doesn't work well for the small test system and 1 process
        parallel_i.append({'sl_default': (1, 1, 8)})
    else:
        parallel_i.append({'sl_auto': True})
        parallel_i.append({'sl_auto': True, 'band': 2})


@pytest.fixture(scope='module')
@only_on_master(world)
def initialize_system():
    comm = serial_comm

    # Ground-state calculation
    atoms = molecule('NaCl')
    atoms.center(vacuum=4.0)
    calc = GPAW(nbands=6,
                h=0.4,
                setups=dict(Na='1'),
                basis='dzp',
                mode='lcao',
                poissonsolver=PoissonSolver(eps=1e-16),
                convergence={'density': 1e-8},
                communicator=comm,
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    td_calc = LCAOTDDFT('gs.gpw',
                        communicator=comm,
                        txt='td.out')
    dmat = DensityMatrix(td_calc)
    ffreqs = frequencies(range(0, 31, 5), 'Gauss', 0.1)
    fdm = FrequencyDensityMatrix(td_calc, dmat, frequencies=ffreqs)
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)
    fdm.write('fdm.ulm')

    # Calculate ground state with full unoccupied space
    unocc_calc = calc.fixed_density(nbands='nao',
                                    communicator=comm,
                                    txt='unocc.out')
    unocc_calc.write('unocc.gpw', mode='all')
    return unocc_calc, fdm


@pytest.fixture(scope='module')
@only_on_master(world, broadcast=broadcast)
def dipole_moment_reference(initialize_system):
    from gpaw.tddft.spectrum import \
        read_dipole_moment_file, calculate_fourier_transform

    unocc_calc, fdm = initialize_system
    _, time_t, _, dm_tv = read_dipole_moment_file('dm.dat')
    dm_wv = calculate_fourier_transform(time_t, dm_tv,
                                        fdm.foldedfreqs_f[0])
    return dm_wv


@pytest.fixture(scope='module')
@only_on_master(world)
def ksd_reference(initialize_system):
    unocc_calc, fdm = initialize_system
    ksd = KohnShamDecomposition(unocc_calc)
    ksd.initialize(unocc_calc)
    return ksd, fdm


def ksd_transform_fdm(ksd, fdm):
    rho_iwp = np.empty((2, len(fdm.freq_w), len(ksd.w_p)), dtype=complex)
    rho_iwp[:] = np.nan + 1j * np.nan
    for i, rho_wuMM in enumerate([fdm.FReDrho_wuMM, fdm.FImDrho_wuMM]):
        for w in range(len(fdm.freq_w)):
            rho_uMM = rho_wuMM[w]
            rho_up = ksd.transform(rho_uMM)
            rho_iwp[i, w, :] = rho_up[0]
    return rho_iwp


@pytest.fixture(scope='module')
@only_on_master(world, broadcast=broadcast)
def ksd_transform_reference(ksd_reference):
    ksd, fdm = ksd_reference
    ref_rho_iwp = ksd_transform_fdm(ksd, fdm)
    return ref_rho_iwp


@pytest.fixture(scope='module', params=parallel_i)
def build_ksd(initialize_system, request):
    calc = GPAW('unocc.gpw', parallel=request.param, txt=None)
    ksd = KohnShamDecomposition(calc)
    ksd.initialize(calc)
    ksd.write('ksd.ulm')


@pytest.fixture(scope='module', params=parallel_i)
def load_ksd(build_ksd, request):
    calc = GPAW('unocc.gpw', parallel=request.param, txt=None)
    # Initialize positions in order to calculate density
    calc.initialize_positions()
    ksd = KohnShamDecomposition(calc, 'ksd.ulm')
    dmat = DensityMatrix(calc)
    fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')
    return ksd, fdm


@pytest.fixture(scope='module')
def ksd_transform(load_ksd):
    ksd, fdm = load_ksd
    rho_iwp = ksd_transform_fdm(ksd, fdm)
    return rho_iwp


def test_ksd_transform(ksd_transform, ksd_transform_reference):
    ref_iwp = ksd_transform_reference
    rho_iwp = ksd_transform
    err = calculate_error(rho_iwp, ref_iwp)
    atol = 1e-18
    assert err < atol


def test_ksd_transform_real_only(load_ksd, ksd_transform_reference):
    ksd, fdm = load_ksd
    ref_iwp = ksd_transform_reference
    rho_iwp = np.empty((2, len(fdm.freq_w), len(ksd.w_p)), dtype=complex)
    rho_iwp[:] = np.nan + 1j * np.nan
    for i, rho_wuMM in enumerate([fdm.FReDrho_wuMM, fdm.FImDrho_wuMM]):
        for w in range(len(fdm.freq_w)):
            rho_uMM = rho_wuMM[w]
            rho_p = ksd.transform([rho_uMM[0].real], broadcast=True)[0] \
                + 1j * ksd.transform([rho_uMM[0].imag], broadcast=True)[0]
            rho_iwp[i, w, :] = rho_p
    err = calculate_error(rho_iwp, ref_iwp)
    atol = 1e-18
    assert err < atol


def test_dipole_moment_from_ksd(ksd_transform, load_ksd,
                                dipole_moment_reference):
    ksd, fdm = load_ksd
    dm_wv = np.empty((len(fdm.freq_w), 3), dtype=complex)
    dm_wv[:] = np.nan + 1j * np.nan
    rho_wp = ksd_transform[0]
    for w in range(len(fdm.freq_w)):
        dm_v = ksd.get_dipole_moment([rho_wp[w]])
        dm_wv[w, :] = dm_v

    ref_wv = dipole_moment_reference
    err = calculate_error(dm_wv, ref_wv)
    atol = 1e-7
    assert err < atol


def get_density_fdm(ksd, fdm, kind):
    assert kind in ['dmat', 'ksd']
    rho_wg = fdm.dmat.density.finegd.empty(len(fdm.freq_w), dtype=complex)
    rho_wg[:] = np.nan + 1j * np.nan
    for w in range(len(fdm.freq_w)):
        rho_uMM = fdm.FReDrho_wuMM[w]
        if kind == 'dmat':
            rho_g = fdm.dmat.get_density([rho_uMM[0].real]) \
                + 1j * fdm.dmat.get_density([rho_uMM[0].imag])
        elif kind == 'ksd':
            rho_up = ksd.transform(rho_uMM, broadcast=True)
            rho_g = ksd.get_density(fdm.dmat.wfs, [rho_up[0].real]) \
                + 1j * ksd.get_density(fdm.dmat.wfs, [rho_up[0].imag])
        rho_wg[w, :] = rho_g
    return rho_wg


@pytest.fixture(scope='module')
@only_on_master(world, broadcast=broadcast)
def density_reference(ksd_reference):
    ksd, fdm = ksd_reference
    dmat_rho_wg = get_density_fdm(ksd, fdm, 'dmat')
    ksd_rho_wg = get_density_fdm(ksd, fdm, 'ksd')
    return dict(dmat=dmat_rho_wg, ksd=ksd_rho_wg)


def test_ksd_vs_dmat_density(density_reference):
    ref_wg = density_reference['dmat']
    rho_wg = density_reference['ksd']
    err = calculate_error(rho_wg, ref_wg)
    atol = 2e-10
    assert err < atol


@pytest.fixture(scope='module')
def density(load_ksd):
    ksd, fdm = load_ksd
    if ksd.ksl.using_blacs:
        pytest.xfail('Scalapack is not supported')
    dmat_rho_wg = get_density_fdm(ksd, fdm, 'dmat')
    ksd_rho_wg = get_density_fdm(ksd, fdm, 'ksd')
    return dict(dmat=dmat_rho_wg, ksd=ksd_rho_wg)


@pytest.mark.parametrize('kind', ['ksd', 'dmat'])
def test_density(kind, density, load_ksd, density_reference):
    ksd, fdm = load_ksd
    ref_wg = density_reference[kind]
    rho_wg = fdm.dmat.density.finegd.collect(density[kind])
    err = calculate_error(rho_wg, ref_wg)
    atol = 1e-19
    assert err < atol


@pytest.mark.parametrize('kind', ['ksd', 'dmat'])
def test_dipole_moment_from_density(kind, density, load_ksd,
                                    dipole_moment_reference):
    ksd, fdm = load_ksd
    rho_wg = density[kind]
    dm_wv = np.empty((len(fdm.freq_w), 3), dtype=complex)
    dm_wv[:] = np.nan + 1j * np.nan
    for w in range(len(fdm.freq_w)):
        dm_v = ksd.density.finegd.calculate_dipole_moment(rho_wg[w])
        dm_wv[w, :] = dm_v

    ref_wv = dipole_moment_reference
    err = calculate_error(dm_wv, ref_wv)
    atol = 5e-7
    assert err < atol
