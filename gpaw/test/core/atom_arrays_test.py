import numpy as np
from gpaw.core.atom_arrays import AtomArraysLayout, AtomDistribution
from gpaw.mpi import world


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