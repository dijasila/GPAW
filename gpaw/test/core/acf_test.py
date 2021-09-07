import numpy as np
import pytest
from gpaw.core import UniformGrid, PlaneWaves, PlaneWaveAtomCenteredFunctions
from gpaw.mpi import world


@pytest.mark.ci
def test_acf():
    a = 2.5
    n = 20

    # comm = world.new_communicator([world.rank])
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world)
    pw = PlaneWaves(ecut=10, grid=grid)
    alpha = 4.0
    s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
    basis = PlaneWaveAtomCenteredFunctions([[s]],
                                           positions=[[0.5, 0.5, 0.5]],
                                           pw=pw)

    coefs = basis.layout.empty()
    if 0 in coefs:
        coefs[0] = [1.0]
    f1 = pw.zeros()
    basis.add_to(f1, coefs)
    