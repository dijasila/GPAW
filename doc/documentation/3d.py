import numpy as np
from gpaw.core import (PlaneWaves,
                       PlaneWaveAtomicOrbitals, UniformGrid)
from gpaw.mpi import world

a = 2.5
n = 20

comm = world.new_communicator([world.rank])
grid1 = UniformGrid(cell=[a, a, a], size=(n, n, n), dist=comm)
wfs = grid1.empty(3)
wfs.data[:] = 1.0
grid = grid1.new(dist=world)
kpts = [(0, 0, 0), (0.5, 0, 0)]

w2 = grid1.redistribute(wfs)


ibz = []
for kpt in kpts:
    pws = PlaneWaves(ecut=300, grid=grid.new(kpt=kpt))
    wfs = pws.zeros(3)
    ibz.append(wfs)

alpha = 4.0
s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
basis = PlaneWaveAtomicOrbitals([[s]],
                                positions=[[0.5, 0.5, 0.5]])
for kpt, wfs in zip(kpts, ibz):
    coefs = {0: np.ones((3, 1))}
    basis.add(coefs, wfs)

# basis = basis.new(positions=[[0, 0, 0]])

for wfs in ibz:
    wfs[0].ifft().plot((10, 10))

