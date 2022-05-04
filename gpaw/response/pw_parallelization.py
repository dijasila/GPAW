import numpy as np
from gpaw.blacs import BlacsDescriptor, BlacsGrid, Redistributor


class GaGb:
    def __init__(self, blockcomm, nG):
        self.blockcomm = blockcomm
        mynG = (nG + blockcomm.size - 1) // blockcomm.size
        self.Ga = min(blockcomm.rank * mynG, nG)
        self.Gb = min(self.Ga + mynG, nG)
        self.nGlocal = self.Gb - self.Ga
        self.nG = nG

        self.myslice = slice(self.Ga, self.Gb)


def block_partition(supercomm, nblocks):
    assert supercomm.size % nblocks == 0, supercomm.size
    rank1 = supercomm.rank // nblocks * nblocks
    rank2 = rank1 + nblocks
    blockcomm = supercomm.new_communicator(range(rank1, rank2))
    ranks = range(supercomm.rank % nblocks, supercomm.size, nblocks)
    if nblocks == 1:
        assert len(ranks) == supercomm.size
        transverse_comm = supercomm
    else:
        transverse_comm = supercomm.new_communicator(ranks)
    assert blockcomm.size * transverse_comm.size == supercomm.size
    return blockcomm, transverse_comm


class PlaneWaveBlockDistributor:
    """Functionality to shuffle block distribution of pair functions
    in the plane wave basis."""

    def __init__(self, world, blockcomm, intrablockcomm,
                 wd, GaGb):
        self.world = world
        self.blockcomm = blockcomm
        self.intrablockcomm = intrablockcomm
        self.wd = wd
        self.GaGb = GaGb

    def redistribute(self, in_wGG, out_x=None):
        """Redistribute array.

        Switch between two kinds of parallel distributions:

        1) parallel over G-vectors (second dimension of in_wGG)
        2) parallel over frequency (first dimension of in_wGG)

        Returns new array using the memory in the 1-d array out_x.
        """

        comm = self.blockcomm

        if comm.size == 1:
            return in_wGG

        nw = len(self.wd)
        nG = in_wGG.shape[2]
        mynw = (nw + comm.size - 1) // comm.size
        mynG = (nG + comm.size - 1) // comm.size

        bg1 = BlacsGrid(comm, comm.size, 1)
        bg2 = BlacsGrid(comm, 1, comm.size)
        md1 = BlacsDescriptor(bg1, nw, nG**2, mynw, nG**2)
        md2 = BlacsDescriptor(bg2, nw, nG**2, nw, mynG * nG)

        if len(in_wGG) == nw:
            mdin = md2
            mdout = md1
        else:
            mdin = md1
            mdout = md2

        r = Redistributor(comm, mdin, mdout)

        outshape = (mdout.shape[0], mdout.shape[1] // nG, nG)
        if out_x is None:
            out_wGG = np.empty(outshape, complex)
        else:
            out_wGG = out_x[:np.product(outshape)].reshape(outshape)

        r.redistribute(in_wGG.reshape(mdin.shape),
                       out_wGG.reshape(mdout.shape))

        return out_wGG

    def distribute_frequencies(self, in_wGG):
        """Distribute frequencies to all cores."""

        world = self.world
        comm = self.blockcomm

        if world.size == 1:
            return in_wGG

        nw = len(self.wd)
        nG = in_wGG.shape[2]
        mynw = (nw + world.size - 1) // world.size
        mynG = (nG + comm.size - 1) // comm.size

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.blockcomm.size == 1:
            return in_wGG[wa:wb].copy()

        if self.intrablockcomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = in_wGG.reshape((nw, -1))
        else:
            bg1 = BlacsGrid(None, 1, 1)
            # bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)

        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)

        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))

        return out_wGG
