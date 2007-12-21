from math import sqrt, exp, pi
from gpaw.vdw import VanDerWaals
import numpy as npy
n = 48
d = npy.ones((2 * n, n, n), npy.Float)
a = 4.0
c = a / 2
h = a / n
for x in range(2 * n):
    for z in range(n):
        for y in range(n):
            r = sqrt((x * h - c)**2 + (y * h - c)**2 + (z * h - c)**2)
            d[x, y, z] = exp(-2 * r) / pi

print npy.sum(d.flat) * h**3
uc = npy.array([(2 * a, 0, 0),
                (0,     a, 0),
                (0,     0, a)])
e1 = VanDerWaals(d, unitcell=uc,xcname='revPBE').GetEnergy(n=4)
d += d[::-1].copy()
e2 = VanDerWaals(d, unitcell=uc,xcname='revPBE').GetEnergy(n=4)
print  'revPBE',e1, e2, 2 * e1 - e2
#RPBE
e1 = VanDerWaals(d, unitcell=uc,xcname='RPBE').GetEnergy(n=4)
d += d[::-1].copy()
e2 = VanDerWaals(d, unitcell=uc,xcname='RPBE').GetEnergy(n=4)
print 'RPBE',e1, e2, 2 * e1 - e2

#pbc using mic
for x in range(2 * n):
    for z in range(n):
        for y in range(n):
            r = sqrt((x * h - c)**2 + (y * h - c)**2 + (z * h - c)**2)
            d[x, y, z] = exp(-2 * r) / pi
                                    
e1 = VanDerWaals(d, unitcell=uc,xcname='revPBE',pbc='mic').GetEnergy(n=4)
d += d[::-1].copy()
e2 = VanDerWaals(d, unitcell=uc,xcname='revPBE',pbc='mic').GetEnergy(n=4)
print  'revPBE mic',e1, e2, 2 * e1 - e2
