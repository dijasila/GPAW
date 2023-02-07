import pytest
from ase import Atoms
from gpaw.new.calculation import DFTCalculation
from gpaw import SCIPY_VERSION


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.skipif(SCIPY_VERSION < [1, 6], reason='Too old scipy')
def test_gpu_pw():
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    energies = []
    for gpu in [False, True]:
        dft = DFTCalculation.from_parameters(
            atoms,
            dict(mode={'name': 'pw', 'force_complex_dtype': True},
                 parallel={'gpu': gpu},
                 setups='paw'),
            log=None)
        dft.converge()
        dft.energies()
        energy = dft.results['energy']
        energies.append(energy)
    e0, e1 = energies
    assert e1 == pytest.approx(e0, abs=1e-14)
