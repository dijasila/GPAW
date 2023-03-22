import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW
from gpaw.mpi import size


@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.parametrize(
    'gpu',
    [False,
     pytest.param(True, marks=pytest.mark.gpu)])
def test_new_cell(gpu):
    atoms = Atoms('H', pbc=True, cell=[1, 1, 1])
    atoms.calc = GPAW(
        mode={'name': 'pw'},
        kpts=(4, 1, 1),
        parallel={'gpu': gpu})
    e0 = atoms.get_potential_energy()
    assert e0 == pytest.approx(-19.579937435888795)
    atoms.cell[2, 2] = 0.9
    e1 = atoms.get_potential_energy()
    assert e1 - e0 == pytest.approx(-1.0902883222695756)
