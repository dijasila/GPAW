import pytest
from ase import Atoms
from gpaw import GPAW


# @pytest.mark.gpu
# @pytest.mark.skipif(size > 2, reason='Not implemented')
# @pytest.mark.parametrize('gpu', [False, True])
# @pytest.mark.parametrize('par', ['domain', 'kpt'])
def test_gpu_pw_k():
    gpu = 0
    atoms = Atoms('H', pbc=True, cell=[1, 1, 1])
    atoms.calc = GPAW(
        mode={'name': 'pw'},
        kpts=(4, 1, 1),
        parallel={'gpu': gpu})
    atoms.get_potential_energy()
    atoms.cell[2, 2] = 1.1
    atoms.get_potential_energy()
