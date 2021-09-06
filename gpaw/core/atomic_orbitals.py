import numpy as np
from gpaw.pw.lfc import PWLFC
from gpaw.pw.descriptor import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor


class AtomicOrbitals:
    def __init__(self, functions, positions, dist=None):
        self.functions = functions
        self.positions = np.array(positions)

        if dist is None:
            dist = serial_comm

        if not isinstance(dist, AtomicOrbitalDistribution):
            comm = dist
            ranks = np.zeros(len(positions), int)
            atomdist = AtomDistribution(ranks, comm)
            dist = AtomicOrbitalDistribution([len(f) for f in functions],
                                             atomdist)
        self.dist = dist
        self.size = dist.total_size

    def empty(self,
              shape: tuple[int] = None,
              dist: MPIComm | ShapeDistribution | None = None
              ) -> AtomicOrbitalCoefficients:
        dist = create_shape_distributuion(shape, dist)
        array = np.empty(dist.shape + self.dist.size, self.dtype)
        return AtomicOrbitalCoefficients(array, self, dist)

    def zeros(self, shape=(), dist=None) -> UniformGridFunctions:
        funcs = self.empty(shape, dist)
        funcs.data[:] = 0.0
        return funcs


def distribute_atoms(positions, grid, kind):
    if kind == 'master':
        return np.zeros(len(positions), int)
    1 / 0


class AtomicOrbitalDistribution:
    def __init__(self,
                 nfunctions: list[int],
                 atomdist: AtomDistribution):
        self.nfunctions = nfunctions
        self.atomdist = atomdist
        self.indices = []
        self.size = 0
        I1 = 0
        for a in atomdist.indices:
            I2 = I1 + nfunctions[a]
            self.indices.append((a, I1, I2))
            self.size += I2 - I1

    @property
    def total_size(self):
        return sum(self.nfunctions)


class AtomDistribution:
    def __init__(self, ranks, comm):
        self.comm = comm
        self.ranks = ranks
        self.indices = np.where(ranks == comm.rank)[0]


class AtomicOrbitalCoefficients:
    def __init__(self, aodist):

class UniformGridAtomicOrbitals:
    def __init__(self, functions, grid, positions):
        AtomicOrbitals.__init__(self, functions, positions, grid)
        self.grid = grid


class PlaneWaveAtomicOrbitals:
    def __init__(self, functions, pws, positions):
        AtomicOrbitals.__init__(self, functions, positions, pws.grid)
        self.pws = pws
        gd = pws.grid.gd
        kd = KPointDescriptor(np.array([pws.grid.kpt]))
        pd = PWDescriptor(pws.ecut, gd, kd=kd)
        self.lfc = PWLFC(functions, pd)
        self.lfc.set_positions(self.positions)

    def add(self, coefs, functions):
        self.lfc.add(functions.data, coefs, q=0)


class