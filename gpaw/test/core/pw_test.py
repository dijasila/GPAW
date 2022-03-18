import numpy as np
import pytest
from gpaw.core import UniformGrid, PlaneWaves
from gpaw.mpi import world
from gpaw.core.plane_waves import find_reciprocal_vectors
from math import pi


@pytest.mark.ci
def test_pw_redist():
    a = 2.5
    pw = PlaneWaves(ecut=10, cell=[a, a, a], comm=world)
    f1 = pw.empty()
    f1.data[:] = 1.0
    f2 = f1.gather()
    if f2 is not None:
        assert (f2.data == 1.0).all()
        assert f2.desc.comm.size == 1


def test_pw_integrate():
    a = 1.0
    decomp = {1: [[0, 4], [0, 4], [0, 4]],
              2: [[0, 2, 4], [0, 4], [0, 4]],
              4: [[0, 2, 4], [0, 2, 4], [0, 4]],
              8: [[0, 1, 2, 3, 4], [0, 2, 4], [0, 4]]}[world.size]
    grid = UniformGrid(cell=[a, a, a], size=(4, 4, 4), comm=world,
                       decomp=decomp)
    gridc = grid.new(dtype=complex)

    g1 = grid.empty()
    g1.data[:] = 1.0

    g2 = grid.empty()
    g2.data[:] = 1.0
    g2.data += [0, 1, 0, -1]

    g3 = gridc.empty()
    g3.data[:] = 1.0

    g4 = gridc.empty()
    g4.data[:] = 1.0
    g4.data += [0, 1, 0, -1]

    g5 = gridc.empty()
    g5.data[:] = 1.0 + 1.0j
    g5.data += [0, 1, 0, -1]

    ecut = 0.5 * (2 * np.pi / a)**2 * 1.01
    for g in [g1, g2, g3, g4, g5]:
        pw = PlaneWaves(cell=g.desc.cell, dtype=g.desc.dtype,
                        ecut=ecut, comm=world)
        f = g.fft(pw=pw)
        print(f.data)

        gg = g.new()
        gg.scatter_from(f.gather(broadcast=True)
                        .ifft(grid=g.desc.new(comm=None)))
        assert (g.data == gg.data).all()

        i1 = g.integrate()
        i2 = f.integrate()
        assert i1 == i2
        assert type(i1) == g.desc.dtype

        i1 = g.integrate(g)
        i2 = f.integrate(f)
        assert i1 == i2
        assert type(i1) == g.desc.dtype

        g1 = g.desc.empty(1)
        g1.data[:] = g.data
        m1 = g1.matrix_elements(g1)
        assert (i1 == m1.data).all()

        f1 = f.desc.empty(1)
        f1.data[:] = f.data
        m2 = f1.matrix_elements(f1)
        assert (i2 == m2.data).all()


def test_grr():
    from ase.units import Ha, Bohr
    grid = UniformGrid(cell=[2 / Bohr, 2 / Bohr, 2.737166 / Bohr],
                       size=(9, 9, 12),
                       comm=world)
    pw = PlaneWaves(ecut=340 / Ha, cell=grid.cell, comm=world)
    print(pw.G_plus_k_Gv.shape)
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.pw.descriptor import PWDescriptor
    g = GridDescriptor((9, 9, 12), [2 / Bohr, 2 / Bohr, 2.737166 / Bohr])
    p = PWDescriptor(340 / Ha, g)
    print(p.get_reciprocal_vectors().shape)
    assert (p.get_reciprocal_vectors() == pw.G_plus_k_Gv).all()


def test_find_g():
    G, e, i = find_reciprocal_vectors(0.501 * (2 * pi)**2,
                                      np.eye(3))
    assert i.T.tolist() == [[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, -1],
                            [0, 1, 0],
                            [0, -1, 0],
                            [1, 0, 0],
                            [-1, 0, 0]]
    G, e, i = find_reciprocal_vectors(0.501 * (2 * pi)**2,
                                      np.eye(3),
                                      dtype=float)
    assert i.T.tolist() == [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]
    G, e, i = find_reciprocal_vectors(0.5 * (2 * pi)**2,
                                      np.eye(3),
                                      kpt=np.array([0.1, 0, 0]))
    assert i.T.tolist() == [[0, 0, 0],
                            [-1, 0, 0]]
