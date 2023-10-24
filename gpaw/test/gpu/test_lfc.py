import pytest
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
import gpaw.mpi as mpi
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.gpu import cupy as cp
from gpaw.core.atom_arrays import AtomArraysLayout


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [float, complex])
def test_lfc(gpu, dtype):
    s = Spline(0, 5, np.exp(-0.3*np.arange(0, 5, 0.001)**2))
    n = 40
    a = 8
    gd = GridDescriptor((64, 64, 64), (a, a, a), comm=mpi.world)

    c = LFC(gd, [[s]], dtype=dtype, integral=100000.0)
    c_gpu = LFC(gd, [[s]], dtype=dtype, xp=cp, integral=100000.0)
    c.set_positions([(0, 0, 0 + 0.25 * i) for i in [0]])
    c_gpu.set_positions([(0, 0, 0 + 0.25 * i) for i in [0]])

    P_ani = {}
    for a in c.my_atom_indices:
        P_ani[a] = np.ones((1,), dtype=dtype)

    P_gpu_ani = AtomArraysLayout([1], dtype=dtype, xp=cp).empty()
    P_gpu_ani.data[:] = 1.0

    b = gd.zeros(dtype=dtype)
    b_gpu = cp.zeros_like(b)
    q = {float: 0, complex: 0}[dtype]
    c.add(b, P_ani, q=q)
    c_gpu.add(b_gpu, P_gpu_ani, q=q)
    b_ref = b_gpu.get()
    from matplotlib import pyplot as plt
    plt.plot(b_ref[0,0,:])
    plt.plot(b[0,0,:])
    plt.show()
    assert b == pytest.approx(b_ref, abs=1e-12)

    c.integrate(b, P_ani, q=q)
    c_gpu.integrate(b_gpu, P_gpu_ani, q=q)

    for a, P_ni in P_ani.items():
        assert P_ni == pytest.approx(P_gpu_ani[a].get(), abs=1e-12)
