import numpy as np
from ase import Atoms

from gpaw.gpu import cupy as cp
from gpaw.new.calculation import DFTCalculation


def test_gpu_pw():
    a = cp.empty(2)
    b = np.asarray(a)
    assert isinstance(b, np.ndarray)
    assert b.dtype == float

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
