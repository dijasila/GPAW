import numpy as np
import pytest
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.mpi import world


@pytest.mark.ci
def test_acf():
    a = 2.5
    n = 20

    # comm = world.new_communicator([world.rank])
    grid = UniformGrid(cell=[a, a, a], size=(n, n, n), comm=world,
                       dtype=complex)

    alpha = 4.0
    s = (0, 3.0, lambda r: np.exp(-alpha * r**2))

    basis = grid.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]])
    coefs = basis.layout.empty()
    if 0 in coefs:
        print(coefs[0])
        coefs[0] = [1.0]
    f1 = grid.zeros()
    basis.add_to(f1, coefs)

    if 0:
        from sympy import integrate, exp, oo, var
        a, r = var('a, r')
        integrate(exp(-a * r**2) * r**2, (r, 0, oo))

    I = np.pi / 2 / alpha**1.5
    assert f1.integrate() == pytest.approx(I)

    f2 = f1.gather(broadcast=True)
    x, y = f2.xy(10, 10, ...)
    y0 = np.exp(-alpha * (x - a / 2)**2) / (4 * np.pi)**0.5
    assert abs(y - y0).max() == pytest.approx(0.0, abs=0.001)

    pw = PlaneWaves(ecut=50, cell=grid.cell, dtype=complex, comm=world)

    basis = pw.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]])

    coefs = basis.layout.empty(transposed=True)
    if 0 in coefs:
        print(coefs[0])
        coefs[0] = [1.0]

    f1 = pw.zeros()
    basis.add_to(f1, coefs)
    assert f1.integrate() == pytest.approx(I)
    f2 = f1.gather(broadcast=True)
    r2 = f2.ifft(grid=grid.new(comm=None))
    x, y = r2.xy(10, 10, ...)
    assert abs(y - y0).max() == pytest.approx(0.0, abs=0.002)
