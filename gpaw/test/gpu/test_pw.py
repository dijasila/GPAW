import pytest
from ase import Atoms
from ase.units import Ha

from gpaw.new.calculation import DFTCalculation
from gpaw.mpi import size


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('gpu', [False, True])
def test_gpu_pw(dtype, gpu):
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw'},
             dtype=dtype,
             parallel={'gpu': gpu},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    energy = dft.results['energy'] * Ha
    assert energy == pytest.approx(-16.032945, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('par', ['domain', 'kpt', 'band'])
@pytest.mark.parametrize('xc', ['LDA', 'PBE'])
def test_gpu_pw_k(gpu, par, xc):
    atoms = Atoms('H', pbc=True, cell=[1.0, 1.1, 1.1])
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw'},
             spinpol=True,
             xc=xc,
             kpts=(4, 1, 1),
             parallel={'gpu': gpu,
                       par: size},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    dft.forces()
    #dft.stress()
    energy = dft.results['energy'] * Ha
    ref = {'PBE': -17.304186,
           'LDA': -17.653433 }
    assert energy == pytest.approx(ref[xc], abs=1e-6)
