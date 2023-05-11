import pytest
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey
import numpy
from gpaw.test import equal

numpy.random.seed(25)

@pytest.mark.pipekmezey
def test_pipekmezey_pw(in_tmp_dir):

    atoms = Atoms('CO', 
                  positions=[[0,0,0],
                             [0,0,1.128]])
    atoms.center(vacuum=5)

    calc = GPAW(mode=PW(200), 
                h=0.24, 
                convergence={'density' : 1e-4,
                             'eigenstates' : 1e-4})
    
    calc.atoms = atoms
    calc.calculate()

    PM = PipekMezey(calc=calc)
    PM.localize()

    P = PM.get_function_value()

    equal(P, 3.3138, 0.0001)
