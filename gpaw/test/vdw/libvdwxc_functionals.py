from __future__ import print_function
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc.libvdwxc import VDWDF, VDWDF2, VDWDFCX

# This test verifies that the results returned by the van der Waals
# functionals implemented in libvdwxc do not change.

N_c = np.array([23, 10, 6])
gd = GridDescriptor(N_c, N_c * 0.2, pbc_c=(1, 0, 1))

n_sg = gd.zeros(1)
nG_sg = gd.collect(n_sg)
if gd.comm.rank == 0:
    gen = np.random.RandomState(0)
    nG_sg[:] = gen.rand(*nG_sg.shape)
gd.distribute(nG_sg, n_sg)


def test(vdwxcclass, Eref=None, nvref=None):
    xc = vdwxcclass()
    xc._initialize(gd)
    v_sg = gd.zeros(1)
    E = xc.calculate(gd, n_sg, v_sg)
    nv = gd.integrate(n_sg * v_sg, global_integral=True)
    nv = float(nv)  # Comes out as an array due to spin axis

    Eerr = None if Eref is None else abs(E - Eref)
    nverr = None if nvref is None else abs(nv - nvref)

    if gd.comm.rank == 0:
        print(vdwxcclass.__name__)
        print('=' * len(vdwxcclass.__name__))
        print('E  = %15s vs Eref  = %15s :: Eerr  = %15s' % (E, Eref, Eerr))
        print('nv = %15s vs nvref = %15s :: nverr = %15s' % (nv, nvref, nverr))
        print()
    gd.comm.barrier()

    if Eerr is not None:
        assert Eerr < 1e-11
    if nverr is not None:
        assert nverr < 1e-11

test(VDWDF, -3.73732366504, -4.77652923808)
test(VDWDF2, -3.75680663781, -4.79124091301)
test(VDWDFCX, -3.62973365771, -4.6752434767)
