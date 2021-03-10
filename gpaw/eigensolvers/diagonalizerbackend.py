"""Module for Numpy array diagonalization with Scipy/Scalapack."""
import numpy as np
from scipy.linalg import eigh

from gpaw.blacs import BlacsGrid, Redistributor


class ScipyDiagonalizer:
    """Diagonalizer class that uses scipy.linalg.eigh.

    The ScipyDiagonalizer wraps scipy.linalg.eigh to solve a
    (generalized) eigenproblem on one core.
    """

    def __init__(self):
        pass

    def diagonalize(self, A, B, eps, is_master, debug):
        """Solves the eigenproblem A @ x = eps [B] @ x.

        The problem is solved inplace, so when done, A has the eigenvectors as
        columns and eps has the eigenvalues.
        B is overwritten for potential increase in performance.

        Parameters
        ----------
        A : Numpy array
            Left-hand matrix of the eigenproblem. After running, the
            eigenvectors are the column vectors of this array.
        B : Numpy array
            Right-hand overlap matrix of the eigenproblem.
        eps : Numpy array
            1D vector containing the eigenvalues of the solved eigenproblem.
        is_master : bool
            A boolean to mark which rank to perform the diagonalization on.
            Since Scipy's diagonalizer is not MPI-parallelized, the
            diagonalization needs (necessarily) to be done on the rank which
            has the arrays.
        debug : bool
            Flag to check for finiteness when running in debug mode.
        """
        if is_master:
            eps[:], A[:] = eigh(
                A, B, lower=True, check_finite=debug, overwrite_b=True
            )


class ScalapackDiagonalizer:
    """Diagonalizer class that uses general_diagonalize_dc.

    The ScalapackDiagonalizer wraps general_diagonalize_dc to solve a
    (generalized) eigenproblem on one core.
    """

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
        """Initialize grids, communicators, redistributors.

        Parameters
        ----------
        arraysize : int
            The side length of the square matrix to diagonalize.
        grid_nrows : int
            Number of rows in the BLACS grid.
        grid_ncols : int
            Number of columns in the BLACS grid.
        dtype : type
            `float` or `complex`, the datatype of the eigenproblem.
        scalapack_communicator : MPICommunicator
            The communicator object over which Scalapack diagonalizes.
        blocksize : int, optional
            The block size in the 2D block cyclic data distribution.
            The default value of 64 is universally good.
        """
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

    def diagonalize(self, A, B, eps, is_master, debug):
        """Solves the eigenproblem A @ x = eps [B] @ x.
        
        The problem is solved inplace, so when done, A has the eigenvectors
        as columns and eps has the eigenvalues.

        Parameters
        ----------
        A : Numpy array
            Left-hand matrix of the eigenproblem. After running, the
            eigenvectors are the column vectors of this array.
        B : Numpy array
            Right-hand overlap matrix of the eigenproblem.
        eps : Numpy array
            1D vector containing the eigenvalues of the solved eigenproblem.
        is_master : bool
            A boolean to mark which rank to perform the diagonalization on.
            Used to know which ranks to redistribute the Numpy arrays to/from.
        debug : bool
            Flag to check for finiteness when running in debug mode.
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

        if is_master:
            assert self.scalapack_communicator.rank == 0
            # Conjugate-transpose here since general_diagonalize_dc gives us
            # Fortran-convention eigenvectors.
            A[:, :] = vec_MM.conj().T
            eps[:] = temporary_eps
