import numpy as np
from scipy.linalg import eigh
from gpaw.blacs import BlacsGrid, Redistributor


class ScipyDiagonalizer:
    def __init__(self):
        pass

    def diagonalize(self, A, B, eps, is_gridband_master, debug):
        """[summary]

        Parameters
        ----------
        A : [type]
            [description]
        B : [type]
            [description]
        eps : [type]
            [description]
        is_gridband_master : bool
            [description]
        debug : [type]
            [description]
        """

        if is_gridband_master:
            eps[:], A[:] = eigh(A, B, lower=True, check_finite=debug)


class ScalapackDiagonalizer:
    def __init__(
        self,
        arraysize,
        grid_nrows,
        grid_ncols,
        *,
        dtype,
        scalapack_communicator,
        blocksize=64,
    ):

        assert (arraysize) % 2 == 0
        self.arraysize = arraysize
        self.blocksize = blocksize
        self.dtype = dtype
        self.scalapack_communicator = scalapack_communicator

        self.blacsgrid = BlacsGrid(
            self.scalapack_communicator, grid_nrows, grid_ncols
        )
        self.distributed_descriptor = self.blacsgrid.new_descriptor(
            arraysize, arraysize, blocksize, blocksize
        )
        self.head_rank_descriptor = self.blacsgrid.new_descriptor(
            arraysize, arraysize, arraysize, arraysize
        )

        self.head_to_all_redistributor = Redistributor(
            self.scalapack_communicator,
            self.head_rank_descriptor,
            self.distributed_descriptor,
        )

        self.all_to_head_redistributor = Redistributor(
            self.scalapack_communicator,
            self.distributed_descriptor,
            self.head_rank_descriptor,
        )

    def diagonalize(self, A, B, eps, is_gridband_master, debug):
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
            Asc_MM[:, :] = A
            Bsc_MM[:, :] = B

        self.head_to_all_redistributor.redistribute(Asc_MM, Asc_mm)
        self.head_to_all_redistributor.redistribute(Bsc_MM, Bsc_mm)

        self.distributed_descriptor.general_diagonalize_dc(
            Asc_mm, Bsc_mm, vec_mm, temporary_eps
        )

        # vec_MM contains the eigenvectors in 'Fortran form'. They need to be
        # transpose-conjugated before they are consistent with Scipy behaviour
        self.all_to_head_redistributor.redistribute(vec_mm, vec_MM, uplo="G")

        if is_gridband_master:
            assert self.scalapack_communicator.rank == 0
            A[:, :] = vec_MM.conj().T
            eps[:] = temporary_eps
