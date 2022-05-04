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
