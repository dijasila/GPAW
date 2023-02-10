import pytest
from ase import Atoms
from ase.units import Ha

from gpaw import SCIPY_VERSION
from gpaw.new.calculation import DFTCalculation


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.skipif(SCIPY_VERSION < [1, 6], reason='Too old scipy')
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
