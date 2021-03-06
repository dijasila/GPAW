import numpy as np
from scipy.linalg import eigh
from gpaw.blacs import BlacsGrid, Redistributor


class ScipyDiagonalizer:
    def __init__(self, communicator_list):
        self.scalapack_communicator, self.grid_communicator, self.band_communicator = communicator_list

    def diagonalize(self, A, B, v, debug=False):
        """[summary]

        Parameters
        ----------
        A : [type]
            [description]
        B : [type]
            [description]
        v : [type]
            [description]
        """

        if self.scalapack_communicator.rank == 0 and self.grid_communicator.rank == 0 and self.band_communicator.rank == 0:
            if debug:
                H_NN[np.triu_indices(2 * B, 1)] = 42.0
                S_NN[np.triu_indices(2 * B, 1)] = 42.0

            v[:], A[:] = eigh(
                    A, B,
                    lower=True,
                    check_finite=debug)

class ScalapackDiagonalizer:
    def __init__(self, arraysize, grid_nrows, grid_ncols, *, communicator_list, dtype, blocksize=64, eigvecs_as_columns=True):

        assert ((arraysize) % 2 == 0)
        self.arraysize = arraysize
        self.dtype = dtype
        self.grid_nrows = grid_nrows
        self.grid_ncols = grid_ncols
        self.blocksize = blocksize
        self.eigvecs_as_columns = eigvecs_as_columns
        self.scalapack_communicator, self.grid_communicator, self.band_communicator = communicator_list

        self.blacsgrid = BlacsGrid(self.scalapack_communicator, grid_nrows, grid_ncols)
        self.distributed_descriptor = self.blacsgrid.new_descriptor(arraysize, arraysize, blocksize, blocksize)
        self.head_rank_descriptor = self.blacsgrid.new_descriptor(arraysize, arraysize, arraysize, arraysize)

        self.head_to_all_redistributor = Redistributor(self.scalapack_communicator, self.head_rank_descriptor, self.distributed_descriptor)

        self.all_to_head_redistributor = Redistributor(self.scalapack_communicator,self.distributed_descriptor, self.head_rank_descriptor)

    def diagonalize(self, A, B, eps):
        """[summary]

        Parameters
        ----------
        A : [type]
            [description]
        B : [type]
            [description]
        eps : [type]
            [description]
        """

        Asc_MM = self.head_rank_descriptor.zeros(dtype=self.dtype)
        Bsc_MM = self.head_rank_descriptor.zeros(dtype=self.dtype)
        vec_MM = self.head_rank_descriptor.zeros(dtype=self.dtype)

        Asc_mm = self.distributed_descriptor.zeros(dtype=self.dtype)
        Bsc_mm = self.distributed_descriptor.zeros(dtype=self.dtype)
        vec_mm = self.distributed_descriptor.zeros(dtype=self.dtype)

        temporary_eps = np.zeros([self.arraysize])
        if self.scalapack_communicator.rank == 0:
            Asc_MM[:, :] = A.copy()
            Bsc_MM[:, :] = B.copy()

        self.head_to_all_redistributor.redistribute(Asc_MM, Asc_mm)
        self.head_to_all_redistributor.redistribute(Bsc_MM, Bsc_mm)

        self.distributed_descriptor.general_diagonalize_dc(
                        Asc_mm, Bsc_mm.copy(), vec_mm, temporary_eps)

        self.all_to_head_redistributor.redistribute(vec_mm, vec_MM, uplo='G')

        if self.scalapack_communicator.rank == 0 and self.grid_communicator.rank == 0 and self.band_communicator.rank == 0:
            A[:, :] = vec_MM.conj().T.copy()
            eps[:] = temporary_eps.copy()
        # self.head_to_all_redistributor.redistribute(vec_MM, vec_mm)







