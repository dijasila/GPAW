import numpy as np
from gpaw.things import (PlaneWaves,
                         ReciprocalSpaceAtomCenteredFunctions, UniformGrid,
                         gaussian)

a = 2.5
n = 20

ug = UniformGrid(cell=[a, a, a], size=(n, n, n))
wfs = ug.empty(3)
wfs._data[:] = 1.0
kpts = [(0, 0, 0), (0.5, 0, 0)]

w2 = ug.redistributor(ug).redistribute(wfs)

ibz = []
for kpt in kpts:
    pws = PlaneWaves(ecut=300, ug=ug.new(kpt=kpt))
    wfs = pws.zeros(3, dist=...)
    ibz.append(wfs)

s = gaussian(l=0, alpha=4.0, rcut=3.0)
basis = ReciprocalSpaceAtomCenteredFunctions(
    [[s]],
    positions=[[0.5, 0.5, 0.5]],
    kpts=kpts)
for kpt, wfs in zip(kpts, ibz):
    coefs = {0: np.ones((3, 1))}
    basis.add(coefs, wfs)

# basis = basis.new(positions=[[0, 0, 0]])

for wfs in ibz:
    wfs[0].ifft().plot((10, 10))

