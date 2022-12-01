import pytest
from ase import Atoms
from gpaw.new.calculation import DFTCalculation


@pytest.mark.gpu
@pytest.mark.serial
def test_gpu_pw():
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw', 'force_complex_dtype': True},
             parallel={'gpu': True},
             setups='ae'),
        log='-')
    dft.converge()
