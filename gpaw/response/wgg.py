"""Parallelization scheme for frequency–planewave–planewave arrays."""
from gpaw.mpi import world
from gpaw.response.hacks import block_partition
import numpy as np
from gpaw.matrix import suggest_blocking


def get_blocksize(length, commsize):
    return -(-length // commsize)


def get_strides(cpugrid):
    return np.array([cpugrid[1] * cpugrid[2], cpugrid[2], 1], int)


class Grid:
    def __init__(self, comm, shape, cpugrid, blocksize=None):
        self.comm = comm
        self.shape = shape
        if blocksize is None:
            blocksize = [get_blocksize(size, commsize)
                         for size, commsize in zip(shape, cpugrid)]
            # XXX scalapack blocksize hack
            blocksize[1] = blocksize[2] = max(blocksize[1:])

        # XXX Our scalapack interface does NOT like it when blocksizes
        # are not the same.  There must be a bug.
        assert blocksize[1] == blocksize[2]

        self.blocksize = tuple(blocksize)
        self.cpugrid = cpugrid

        self.myparpos = self.rank2parpos(self.comm.rank)

        n_cp = self.get_domains()

        shape = []
        for i in range(3):
            n_p = n_cp[i]
            parpos = self.myparpos[i]
            size = n_p[parpos + 1] - n_p[parpos]
            shape.append(size)
        self.myshape = tuple(shape)

    # TODO inherit these from array descriptor
    def zeros(self, dtype=float):
        return np.zeros(self.myshape, dtype=dtype)

    def get_domains(self):
        """Get definition of domains.

        Returns domains_cp where domains_cp[c][r + 1] - domains_cp[c][r]
        is the number of points in domain r along direction c.

        The second axis contains the "fencepost" locations
        of the grid: [0, blocksize, 2 * blocksize, ...]
        """
        domains_cp = []

        for i in range(3):
            n_p = np.empty(self.cpugrid[i] + 1, int)
            n_p[0] = 0
            n_p[1:] = self.blocksize[i]
            n_p[:] = n_p.cumsum().clip(0, self.shape[i])
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
        domains1 = self.get_domains()
        domains2 = dstgrid.get_domains()
        general_redistribute(self.comm, domains1, domains2,
                             self.rank2parpos, dstgrid.rank2parpos,
                             srcarray, dstarray, behavior='overwrite')


def find_wgg_process_grid(commsize, nG):
    # Use sqrt(cores) for w parallelization and the remaining sqrt(cores)
    # for G parallelization.

    # We first assume we have an infinite matrix and just want a "good"
    # balance between frequency grid and ScaLAPACK:
    # infinity = 10000 * nG
    #ggsize, wsize, _ = suggest_blocking(infinity, commsize)
    wsize = 2
    ggsize = commsize

    # Next get ScaLAPACK row and columns based on real matrix size:
    gsize1, gsize2, _ = suggest_blocking(nG, ggsize)

    return wsize, gsize1, gsize2

def get_x_WGG(WGG_grid):
    x_WGG = WGG_grid.zeros(dtype=complex)
    rng = np.random.RandomState(42)

    x_WGG.flat[:] = rng.random(x_WGG.size)
    # XXX write also to imaginary parts

    nG = x_WGG.shape[1]

    xinv_WGG = np.zeros_like(x_WGG)
    if WGG_grid.comm.rank == 0:
        assert x_WGG.shape == WGG_grid.myshape
        for iw, x_GG in enumerate(x_WGG):
            x_GG += x_GG.T.copy()
            x_GG += np.identity(nG) * 5
            eigs = np.linalg.eigvals(x_GG)
            assert all(eigs.real) > 0
            xinv_WGG[iw] = np.linalg.inv(x_GG)
    else:
        assert np.prod(x_WGG.shape) == 0
    return x_WGG, xinv_WGG


def main(comm=world):
    nW = 7
    nG = 31

    #cpugrid = find_wgg_process_grid(comm.size, nG)
    cpugrid = (2, 3, 2)
    WGG = (nW, nG, nG)

    dtype = complex

    # Build serial grid (data only on rank 0)
    # and establish matrix and its inverse
    WGG_grid = Grid(comm, WGG, cpugrid, blocksize=WGG)
    x_WGG, xinv_WGG = get_x_WGG(WGG_grid)

    # Distribute to WgG grid:
    WgG_grid = Grid(comm, WGG, (1, comm.size, 1))
    x_WgG = np.zeros(WgG_grid.myshape, dtype=dtype)
    WGG_grid.redistribute(WgG_grid, x_WGG, x_WgG)

    # Build wgg grid choosing scalapack
    nscalapack_cores = np.prod(cpugrid[1:])
    blacs_comm, wcomm = block_partition(comm, nscalapack_cores)
    assert wcomm.size == cpugrid[0]
    assert blacs_comm.size * wcomm.size == comm.size
    wgg_grid = Grid(comm, WGG, cpugrid)
    print(f'cpugrid={cpugrid} blocksize={wgg_grid.blocksize} '
          f'shape={wgg_grid.shape} myshape={wgg_grid.myshape}')

    x_wgg = wgg_grid.zeros(dtype=dtype)
    WgG_grid.redistribute(wgg_grid, x_WgG, x_wgg)

    # By now let's distribute wgg back to WgG to check that numbers
    # are the same:
    x1_WgG = WgG_grid.zeros(dtype=dtype)
    wgg_grid.redistribute(WgG_grid, x_wgg, x1_WgG)
    assert np.allclose(x_WgG, x1_WgG)

    from gpaw.blacs import BlacsGrid
    from gpaw.utilities.scalapack import scalapack_inverse
    for iw, x_gg in enumerate(x_wgg):
        bg = BlacsGrid(blacs_comm, *cpugrid[1:][::-1])
        desc = bg.new_descriptor(*wgg_grid.shape[1:], *wgg_grid.blocksize[1:])

        xtmp_gg = desc.empty(dtype=dtype)
        xtmp_gg[:] = x_gg.T
        scalapack_inverse(desc, xtmp_gg, 'U')
        x_gg[:] = xtmp_gg.T

    # Distribute the inverse wgg back to WGG:
    inv_x_WGG = WGG_grid.zeros(dtype=dtype)
    wgg_grid.redistribute(WGG_grid, x_wgg, inv_x_WGG)

    from gpaw.utilities.tools import tri2full
    if comm.rank == 0:
        for inv_x_GG in inv_x_WGG:
            tri2full(inv_x_GG, 'L')

        for x_GG, inv_x_GG in zip(x_WGG, inv_x_WGG):
            assert np.allclose(x_GG @ inv_x_GG, np.identity(nG))


if __name__ == '__main__':
    main()
