import numpy as np
import cupy as cp
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

def primes():
    count = 3
    
    while True:
        isprime = True
        
        for x in range(2, int(math.sqrt(count) + 1)):
            if count % x == 0: 
                isprime = False
                break
        
        if isprime:
            yield count
        
        count += 1

def test_dh(xp):
    ni_a = [2, 3, 4, 17] #np.arange(2, 5, dtype=xp.int32)
    dH_asii = AtomArraysLayout([(n, n) for n in ni_a], xp=xp).empty()
    primeiter = primes()
    dH_asii.data[:] = xp.arange(1, 2**2+3**2+4**2+17**2+1)
    P_ani = AtomArraysLayout(ni_a, dtype=complex, xp=xp).empty(300)
    P_ani.data[:] = 0.0
    for n in range(300):
        I = 0
        for a, ni in enumerate(ni_a):
            for i in range(ni):
                P_ani.data[n, I] = i + 1 + 14.4j*i + a + n * 2.2
                I += 1

    out_ani = P_ani.new()
    out_ani[0][:] = 100
    out_ani[1][:] = 200
    out_ani[2][:] = 300
    out_ani[2][:] = 400
    _gpaw.dH_aii_times_P_ani_gpu(
        dH_asii.data, xp.asarray(ni_a, dtype=xp.int32), P_ani.data, out_ani.data)
    out2_ani = out_ani.new()
    for a, dH_ii in dH_asii.items():
        out2_ani[a][:] = P_ani[a] @ dH_ii
    print(out_ani.data, 'gpu')
    print(out2_ani.data, 'ref')
    print(out2_ani.data - out_ani.data, 'diff')
    assert xp.allclose(out2_ani.data, out_ani.data)
test_dh(cp)
