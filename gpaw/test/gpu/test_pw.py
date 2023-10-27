import pytest
from ase import Atoms
from ase.units import Ha

from gpaw.new.calculation import DFTCalculation
from gpaw.mpi import size
from gpaw.poisson import FDPoissonSolver


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_gpu(dtype, gpu, mode):
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    if mode == 'fd':
        poisson = FDPoissonSolver()
    else:
        poisson = None
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': mode},
             dtype=dtype,
             poissonsolver=poisson,
             parallel={'gpu': gpu},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    energy = dft.results['energy'] * Ha
    if mode == 'pw':
        assert energy == pytest.approx(-16.032945, abs=1e-6)
    else:
        assert energy == pytest.approx(6.681945750355547, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('par', ['domain', 'kpt', 'band'])
@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('xc', ['LDA', 'PBE'])
def test_gpu_k(gpu, par, mode, xc):
    atoms = Atoms('H', pbc=True, cell=[1.0, 1.1, 1.1])
    if mode == 'fd':
        poisson = FDPoissonSolver()
    else:
        poisson = None
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': mode},
             spinpol=True,
             xc=xc,
             convergence={'density': 1e-8},
             kpts=(4, 1, 1),
             poissonsolver=poisson,
             parallel={'gpu': gpu,
                       par: size},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    dft.forces()
    if mode == 'pw':
        dft.stress()
    energy = dft.results['energy'] * Ha
    ref = {'LDAfd': -17.371536887378824,
           'PBEfd': -17.020743089737195,
           'PBEpw': -17.304186,
           'LDApw': -17.653433}[xc + mode]
    assert energy == pytest.approx(ref, abs=1e-6)
