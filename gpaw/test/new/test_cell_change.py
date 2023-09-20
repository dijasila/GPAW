import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW
from gpaw.mpi import size, broadcast_string
from io import StringIO


@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.parametrize('gpu', [False, True])
def test_new_cell(gpu):
    a = 2.1
    atoms = Atoms('Li', pbc=True, cell=[a, a, a])
    output = StringIO()
    atoms.calc = GPAW(
        mode={'name': 'pw'},
        kpts=(4, 1, 1),
        parallel={'gpu': gpu},
        txt=output)
    e0 = atoms.get_potential_energy()
    s0 = atoms.get_stress()
    f0 = atoms.get_forces()
    print(e0, s0, f0)
    assert e0 == pytest.approx(-3.302561607168984)
    assert f0 == pytest.approx(0, abs=1e-15)
    assert s0 == pytest.approx([-0.28223875, -0.28223875, -0.28223875,
                                0.0, 0.0, 0.0])

    atoms.cell[2, 2] = 0.9 * a
    e1 = atoms.get_potential_energy()
    s1 = atoms.get_stress()
    f1 = atoms.get_forces()
    print(e1, s1, f1)
    assert e1 == pytest.approx(-2.985067947312407)
    assert f1 == pytest.approx(0, abs=1e-15)
    assert s1 == pytest.approx([-0.3837174, -0.3837174, -0.43328543,
                                0.0, 0.0, 0.0])
    out = broadcast_string(output.getvalue() or None)
    assert '# Interpolating wave fun' in out
