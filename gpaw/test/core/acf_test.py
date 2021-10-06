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
    pw = PlaneWaves(ecut=50, grid=grid)
    alpha = 4.0
    s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
    basis = pw.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]])

    coefs = basis.layout.empty()
    if 0 in coefs:
        coefs[0] = [1.0]
    f1 = pw.zeros()
    basis.add_to(f1, coefs)
    r1 = f1.ifft()
    x, y = r1.xy(10, 10, ...)
    print(x)
    print(y)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    basis = grid.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]])
    f1 = grid.zeros()
    basis.add_to(f1, coefs)
    x, y = f1.xy(10, 10, ...)
    print(x)
    print(y)
    plt.plot(x, y)
    plt.plot(x, np.exp(-alpha * (x - a / 2)**2) / (4 * np.pi)**0.5)
    plt.show()
    return

    import gpaw.mpi as mpi
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.kpt_descriptor import KPointDescriptor
    from gpaw.lfc import LocalizedFunctionsCollection as LFC
    from gpaw.pw.descriptor import PWDescriptor
    from gpaw.pw.lfc import PWLFC
    from gpaw.spline import Spline
    from gpaw.test import equal

    rc = 3.0
    r = np.linspace(0, rc, 100)
    gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
    spos_ac = np.array([(0.5, 0.5, 0.5)])
    pd = PWDescriptor(45, gd, complex, kd)

    eikr = np.ascontiguousarray(
        np.exp(2j * np.pi * np.dot(np.indices(gd.N_c).T,
                                   (kpts / gd.N_c).T).T)[0])

    for l in range(3):
        print(l)
        s = Spline(l, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))

        lfc1 = LFC(gd, [[s]], kd, dtype=complex)
        lfc2 = PWLFC([[s]], pd)

        c_axi = {0: np.zeros((1, 2 * l + 1), complex)}
        c_axi[0][0, 0] = 1.9 - 4.5j
        c_axiv = {0: np.zeros((1, 2 * l + 1, 3), complex)}

        b1 = gd.zeros(1, dtype=complex)
        b2 = pd.zeros(1, dtype=complex)

        for lfc, b in [(lfc1, b1), (lfc2, b2)]:
            lfc.set_positions(spos_ac)
            lfc.add(b, c_axi, 0)

        b2 = pd.ifft(b2[0]) * eikr
        equal(abs(b2 - b1[0]).max(), 0, 0.001)
