from gridpaw.transformers import Interpolator
import Numeric as num
import RandomArray as ra
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.domain import Domain


p = 0
domain = Domain((8.0, 8.0, 8.0), periodic=(p, p, p))
n = 20
gd1 = GridDescriptor(domain, (n, n, n))
a1 = gd1.array()
ra.seed(1, 2)
a1[:] = ra.random(a1.shape)
#gd1.Zero(a1)
a1[0] = 0.0
a1[:, 0] = 0.0
a1[:, :, 0] = 0.0
gd2 = GridDescriptor(domain, (2*n, 2*n, 2*n))
a2 = gd2.array()
i = Interpolator(gd1, 1).apply
i(a1, a2)
print num.sum(a1.flat)
print num.sum(a1.flat) / 8
##print a2
