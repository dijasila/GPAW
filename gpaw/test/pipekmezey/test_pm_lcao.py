import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey


@pytest.mark.pipekmezey
def test_pipekmezey_lcao(in_tmp_dir, needs_ase_master):

    atoms = Atoms('CO',
                  positions=[[0, 0, 0],
                             [0, 0, 1.128]])
    atoms.center(vacuum=5)

    calc = GPAW(mode='lcao',
                h=0.24,
                convergence={'density': 1e-4})

    calc.atoms = atoms
    calc.calculate()

    PM = PipekMezey(calc=calc, seed=42)
    PM.localize()

    P = PM.get_function_value()

    assert P == pytest.approx(3.3756, abs=0.0001)
