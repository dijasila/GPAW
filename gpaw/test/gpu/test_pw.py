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
<<<<<<< HEAD
@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_gpu_k(gpu, par, mode):
=======
@pytest.mark.parametrize('xc', ['LDA','PBE'])
def test_gpu_pw_k(gpu, par, xc):
>>>>>>> gpu-pbe2
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
             kpts=(4, 1, 1),
             poissonsolver=poisson,
             parallel={'gpu': gpu,
                       par: size},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
<<<<<<< HEAD
    if mode == 'pw':
        dft.forces()
        dft.stress()
    energy = dft.results['energy'] * Ha
    if mode == 'pw':
        assert energy == pytest.approx(-17.653433, abs=1e-6)
    else:
        assert energy == pytest.approx(-17.371538, abs=1e-6)
=======
    dft.forces()
    #dft.stress()
    energy = dft.results['energy'] * Ha
    ref = {'PBE': -17.304186,
           'LDA': -17.653433 }
    assert energy == pytest.approx(ref[xc], abs=1e-6)
>>>>>>> gpu-pbe2
