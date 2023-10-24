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
    s = Spline(0, 1.0, [1.0, 0.5, 0.0])
    n = 40
    a = 8.0
    gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.world)

    c = LFC(gd, [[s], [s], [s]], dtype=dtype)
    c_gpu = LFC(gd, [[s], [s], [s]], dtype=dtype, xp=cp)
    c.set_positions([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])
    c_gpu.set_positions([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])

    P_ani = {}
    for a in c.my_atom_indices:
        P_ani[a] = np.ones((1,), dtype=dtype)

    P_gpu_ani = AtomArraysLayout([1, 1, 1], dtype=dtype, xp=cp).empty()
    P_gpu_ani.data[:] = 1.0

    b = gd.zeros(dtype=dtype)
    b_gpu = cp.zeros_like(b)
    q = {float: -1, complex: 0}[dtype]
    c.add(b, P_ani, q=q)
    c_gpu.add(b_gpu, P_gpu_ani, q=q)
    b_ref = b_gpu.get()

    assert b == pytest.approx(b_ref, abs=1e-12)

    c.integrate(b, P_ani, q=q)
    c_gpu.integrate(b_gpu, P_gpu_ani, q=q)

    for a, P_ni in P_ani.items():
        assert P_ni == pytest.approx(P_gpu_ani[a].get(), abs=1e-12)
