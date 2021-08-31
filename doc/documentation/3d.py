from gpaw.things import (Matrix, PlaneWaves,
                         ReciprocalSpaceAtomCenteredFunctions, UniformGrid,
                         gaussian)

a = 2.5
n = 20
c = a / 2

ug = UniformGrid(cell=[a, a, a], size=(n, n, n), dist=...)
wfs = ug.empty(3)
wfs._data[:] = 1.0
kpts = [(0, 0, 0), (0.5, 0, 0)]

ibz = []
for kpt in kpts:
    pws = PlaneWaves(ecut=300, ug=ug.new(kpt=kpt))
    wfs = pws.zeros(3, comm=...)
    ibz.append(wfs)

s = gaussian(l=0, alpha=4.0, rcut=3.0)
basis = ReciprocalSpaceAtomCenteredFunctions(
    [[s]],
    positions=[[c, c, c]],
    kpts=kpts)
for kpt, wfs in zip(kpts, ibz):
    coefs = Matrix((3, 1))
    coefs._data[:] = 1.0
    basis.add(coefs, wfs)

basis = basis.new(positions=[[0, 0, 0]])
