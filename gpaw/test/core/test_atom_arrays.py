import numpy as np
from gpaw.core.atom_arrays import AtomArraysLayout, AtomDistribution
from gpaw.mpi import world
import _gpaw


def test_aa_to_full():
    d = np.array([[1, 2, 4],
                  [2, 3, 5],
                  [4, 5, 6]], dtype=float)
    a = AtomArraysLayout([(3, 3)]).empty()
    a[0][:] = d
    p = a.to_lower_triangle()
    assert (p[0] == [1, 2, 3, 4, 5, 6]).all()
    assert (p.to_full()[0] == d).all()


def test_scatter_from():
    N = 9
    atomdist1 = AtomDistribution([0] * N, world)
    b1 = AtomArraysLayout([(3, 3)] * N, atomdist=atomdist1).empty(2)
    for a, b_sii in b1.items():
        assert world.rank == 0
        b_sii[0] = a
        b_sii[1] = 2 * a
    b2 = b1.gather()
    if world.rank == 0:
        assert (b1.data == b2.data).all()
    atomdist3 = AtomDistribution.from_number_of_atoms(N, world)
    b3 = b1.layout.new(atomdist=atomdist3).empty(2)
    b3.scatter_from(b2.data if b2 is not None else None)
    for a, b_sii in b3.items():
        assert (b_sii[0] == a).all()
        assert (b_sii[1] == 2 * a).all()


def test_gather():
    """Two atoms on rank-1."""
    r = min(1, world.size - 1)
    ranks = [r, r]
    atomdist = AtomDistribution(ranks, world)
    D_asii = AtomArraysLayout([(1, 1)] * 2, atomdist=atomdist).empty(1)
    if world.rank == r:
        D_asii[0][:] = 1
        D_asii[1][:] = 2
    D2_asii = D_asii.gather(broadcast=True)
    assert D2_asii.data.shape == (1, 2)
    for a, D_sii in D2_asii.items():
        assert D_sii[0, 0, 0] == a + 1


def test_dh():
    ni_a = np.arange(2, 5, dtype=np.int32)
    dH_asii = AtomArraysLayout([(n, n) for n in ni_a]).empty(1)
    dH_asii.data[:] = 2.0
    P_ani = AtomArraysLayout(ni_a, dtype=complex).empty(20)
    P_ani.data[:] = 1.0 + 2.0j
    out_ani = P_ani.new()
    _gpaw.dH_aii_times_P_ani_gpu(
        dH_asii.data[0], ni_a, P_ani.data, out_ani.data)
    print(out_ani.data)
