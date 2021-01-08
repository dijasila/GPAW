from functools import wraps
import numpy as np
import pytest

from ase.build import molecule

from gpaw import GPAW
from gpaw.mpi import world, serial_comm
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.tddft.folding import frequencies
from gpaw.tddft.units import au_to_eV, eV_to_au
from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw.utilities import compiled_with_sl


def only_on_master(comm):
    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if comm.rank == 0:
                func(*args, **kwargs)
            comm.barrier()
        return wrapped_func
    return wrap


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


# Options used in the fixtures and tests
kick_v = np.ones(3) * 1e-5
e_min = 0.0
e_max = 30.0
delta_e = 5.0
freq_w = np.arange(e_min, e_max + 0.5 * delta_e, delta_e)


@pytest.fixture
@only_on_master(world)
def calculate_system(in_tmp_dir):
    # Atoms
    atoms = molecule('NaCl')
    atoms.center(vacuum=4.0)

    # Ground-state calculation
    calc = GPAW(nbands=6,
                h=0.4,
                setups=dict(Na='1'),
                basis='dzp',
                mode='lcao',
                poissonsolver=PoissonSolver(eps=1e-16),
                convergence={'density': 1e-8},
                communicator=serial_comm,
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    width = 0.1
    td_calc = LCAOTDDFT('gs.gpw',
                        communicator=serial_comm,
                        txt='td.out')
    dmat = DensityMatrix(td_calc)
    ffreqs = frequencies(freq_w, 'Gauss', width)
    fdm = FrequencyDensityMatrix(td_calc, dmat, frequencies=ffreqs)
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(kick_v)
    td_calc.propagate(20, 3)
    fdm.write('fdm.ulm')

    # Calculate reference spectrum
    photoabsorption_spectrum('dm.dat', 'spec.dat', e_min=e_min, e_max=e_max,
                             delta_e=delta_e, width=width)

    # Calculate ground state with full unoccupied space
    unocc_calc = calc.fixed_density(nbands='nao',
                                    communicator=serial_comm,
                                    txt='unocc.out')
    unocc_calc.write('unocc.gpw', mode='all')


@pytest.fixture(params=parallel_i)
def build_ksdecomp(calculate_system, in_tmp_dir, request):
    # Construct KS decomposition with the parallelization options
    calc = GPAW('unocc.gpw', parallel=request.param, txt=None)
    ksd = KohnShamDecomposition(calc)
    ksd.initialize(calc)
    ksd.write('ksd.ulm')


@pytest.fixture(params=parallel_i)
def load_ksdecomp(build_ksdecomp, in_tmp_dir, request):
    # Load KS decomposition with the parallelization options
    calc = GPAW('unocc.gpw', parallel=request.param, txt=None)
    using_blacs = calc.wfs.ksl.using_blacs
    calc.initialize_positions()  # Initialize in order to calculate density
    ksd = KohnShamDecomposition(calc, 'ksd.ulm')
    dmat = DensityMatrix(calc)
    fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')
    return using_blacs, calc, ksd, dmat, fdm


def test_ksdecomp(calculate_system, load_ksdecomp, in_tmp_dir):
    atol = 1e-12
    rtol = 2e-3

    using_blacs, calc, ksd, dmat, fdm = load_ksdecomp

    # Read the reference values
    ref_wv = np.loadtxt('spec.dat')[:, 1:]

    for w in range(len(freq_w)):
        rho_uMM = fdm.FReDrho_wuMM[w]
        rho_uMM = [rho_uMM[0] * kick_v[0] / np.sum(kick_v**2)]
        freq = freq_w[w] * eV_to_au

        # KS transformation
        rho_up = ksd.transform(rho_uMM, broadcast=True)

        # Calculate dipole moment from matrix elements
        dmrho_vp = ksd.get_dipole_moment_contributions(rho_up)
        spec_vp = 2 * freq / np.pi * dmrho_vp.imag / au_to_eV
        spec_v = np.sum(spec_vp, axis=1)
        assert np.allclose(spec_v, ref_wv[w], atol=atol, rtol=rtol)

        # The remaining operations do not support scalapack
        if using_blacs:
            continue

        # Calculate dipole moment from density matrix
        rho_g = dmat.get_density([rho_uMM[0].imag])
        dm_v = dmat.density.finegd.calculate_dipole_moment(rho_g)
        spec_v = 2 * freq / np.pi * dm_v / au_to_eV
        assert np.allclose(spec_v, ref_wv[w], atol=atol, rtol=rtol)

        # Calculate dipole moment from induced density
        rho_g = ksd.get_density(calc.wfs, [rho_up[0].imag])
        dm_v = ksd.density.finegd.calculate_dipole_moment(rho_g)
        spec_v = 2 * freq / np.pi * dm_v / au_to_eV
        assert np.allclose(spec_v, ref_wv[w], atol=atol, rtol=rtol)
