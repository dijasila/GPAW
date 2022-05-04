import numpy as np
from gpaw.blacs import BlacsDescriptor, BlacsGrid, Redistributor


class GaGb:
    def __init__(self, blockcomm, nG):
        self.blockcomm = blockcomm
        self.nG = nG

        self.mynG = (nG + blockcomm.size - 1) // blockcomm.size
        self.Ga = min(blockcomm.rank * self.mynG, nG)
        self.Gb = min(self.Ga + self.mynG, nG)
        self.nGlocal = self.Gb - self.Ga

        self.myslice = slice(self.Ga, self.Gb)


def block_partition(comm, nblocks):
    assert comm.size % nblocks == 0, comm.size
    rank1 = comm.rank // nblocks * nblocks
    rank2 = rank1 + nblocks
    blockcomm = comm.new_communicator(range(rank1, rank2))
    ranks = range(comm.rank % nblocks, comm.size, nblocks)
    if nblocks == 1:
        assert len(ranks) == comm.size
        intrablockcomm = comm
    else:
        intrablockcomm = comm.new_communicator(ranks)
    assert blockcomm.size * intrablockcomm.size == comm.size
    return blockcomm, intrablockcomm


class PlaneWaveBlockDistributor:
    """Functionality to shuffle block distribution of pair functions
    in the plane wave basis."""

    def __init__(self, world, blockcomm, intrablockcomm,
                 wd, GaGb):
        self.world = world
        self.blockcomm = blockcomm
        self.intrablockcomm = intrablockcomm

        # NB: It seems that the only thing we need from the wd is
        # nw, why we should consider simply passing nw instead.
        # Also, it would seem beneficial to make a wawb object
        # similar to GaGb, in order to reduce code duplication.
        self.nw = len(wd)

        # CAUTION: Currently the redistribute functionality is used
        # also for arrays with different distribution than GaGb.
        # For this reason, GaGb is actually not used below, but we
        # keep for now, to make the vision more clear.
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

        nw = self.nw
        mynw = (nw + comm.size - 1) // comm.size
        # CAUTION: It is not always the case that nG == self.GaGb.nG
        # This happens, because chi0 wants to reuse the function
        # for the heads and wings part as well
        nG = in_wGG.shape[2]
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

        nw = self.nw
        mynw = (nw + world.size - 1) // world.size
        nG = in_wGG.shape[2]
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
