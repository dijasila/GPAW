from gpaw.transformers import Transformer
import numpy as npy
import RandomArray as ra
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain

p = 0
domain = Domain((8.0, 8.0, 8.0), periodic=(p, p, p))
n = 20
gd1 = GridDescriptor(domain, (n, n, n))
a1 = gd1.zeros()
ra.seed(1, 2)
a1[:] = ra.random(a1.shape)
gd2 = gd1.refine()
a2 = gd2.zeros()
i = Transformer(gd1, gd2).apply
i(a1, a2)
assert abs(npy.sum(a1.flat) - npy.sum(a2.flat) / 8) < 1e-10
r = Transformer(gd2, gd1).apply
a2[0] = 0.0
a2[:, 0] = 0.0
a2[:, :, 0] = 0.0
a2[-1] = 0.0
a2[:, -1] = 0.0
a2[:, :, -1] = 0.0
r(a2, a1)
assert abs(npy.sum(a1.flat) - npy.sum(a2.flat) / 8) < 1e-10
