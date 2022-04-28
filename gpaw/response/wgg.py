"""Parallelization scheme for frequency–planewave–planewave arrays."""
from gpaw.mpi import world
from gpaw.response.hacks import GaGb
import numpy as np
from gpaw.matrix import suggest_blocking


def get_blocksize(length, commsize):
    return -(-length // commsize)


def get_strides(cpugrid):
    return np.array([cpugrid[1] * cpugrid[2], cpugrid[2], 1], int)


class Grid:
    def __init__(self, comm, shape, cpugrid):
        self.comm = comm
        self.shape = shape
        self.blocksize = tuple([get_blocksize(size, commsize)
                                for size, commsize in zip(shape, cpugrid)])
        self.cpugrid = cpugrid

        self.myparpos = self.rank2parpos(self.comm.rank)

        n_cp = self.get_n_cp()

        shape = []
        for i in range(3):
            n_p = n_cp[i]
            parpos = self.myparpos[i]
            size = n_p[parpos + 1] - n_p[parpos]
            shape.append(size)
        self.myshape = tuple(shape)

    def get_n_cp(self):
        domains_cp = []

        for i in range(3):
            n_p = np.empty(self.cpugrid[i] + 1, int)
            n_p[0] = 0
            n_p[1:] = self.blocksize[i]
            n_p[:] = n_p.cumsum().clip(0, self.shape[i] - 1)
            domains_cp.append(n_p)

        return domains_cp

    def rank2parpos(self, rank):
        # XXX Borrowing from gd -- we should eliminate this duplication.

        strides = get_strides(self.cpugrid)
        cpugrid_coord = np.array(
            [rank // strides[0],
             (rank % strides[0]) // strides[1],
             rank % strides[1]])

        return cpugrid_coord

    def redistribute(self, dstgrid, srcarray, dstarray):
        from gpaw.utilities.grid_redistribute import general_redistribute
        domains1 = self.get_n_cp()
        domains2 = dstgrid.get_n_cp()
        general_redistribute(self.comm, domains1, domains2,
                             self.rank2parpos, dstgrid.rank2parpos,
                             srcarray, dstarray, behavior='overwrite')


def find_wgg_process_grid(size):
    # Use sqrt(cores) for w parallelization and the remaining sqrt(cores)
    # for G parallelization.

    infinity = 10000000
    ggsize, wsize, _ = suggest_blocking(infinity, size)
    gsize1, gsize2, _ = suggest_blocking(infinity, ggsize)

    return wsize, gsize1, gsize2


def main():
    nW = 7
    nG = 30

    rng = np.random.RandomState(world.rank)

    comm = world

    gagb = GaGb(comm, nG)

    WGG = (nW, nG, nG)

    cpugrid = find_wgg_process_grid(comm.size)

    grid1 = Grid(comm, WGG, (1, comm.size, 1))
    grid2 = Grid(comm, WGG, cpugrid)

    x_WgG = np.zeros(grid1.myshape, complex)
    x1_WgG = np.zeros(grid1.myshape, complex)

    print(comm.rank, grid1.myshape, grid2.myshape)

    x_wgg = np.zeros(grid2.myshape, complex)
    grid1.redistribute(grid2, x_WgG, x_wgg)
    grid2.redistribute(grid1, x_wgg, x1_WgG)

    assert np.allclose(x_WgG, x1_WgG)


if __name__ == '__main__':
    main()
