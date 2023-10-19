"""Module for Numpy array diagonalization with Scipy/Scalapack."""
import numpy as np
from scipy.linalg import eigh
from ase.utils import lazyproperty

from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.utilities.tools import tri2full


class ScipyDiagonalizer:
    """Diagonalizer class that uses scipy.linalg.eigh.

    The ScipyDiagonalizer wraps scipy.linalg.eigh to solve a
    (generalized) eigenproblem on one core.
    """

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
                A, B, lower=True, check_finite=debug, overwrite_b=True,
                overwrite_a=True)


class ParallelDiagonalizer:
    """Diagonalizer class that uses ScaLAPACK/ELPA.

    The class wraps general_diagonalize_dc or similar Elpa function to solve a
    (generalized) eigenproblem.
    """

    def __init__(
            self,
            arraysize: int,
            grid_nrows: int,
            grid_ncols: int,
            *,
            dtype,
            scalapack_communicator,
            blocksize: int = 64,
            use_elpa: bool = False):
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

        self.blacsgrid = BlacsGrid(
            scalapack_communicator, grid_nrows, grid_ncols)
        self.distributed_descriptor = self.blacsgrid.new_descriptor(
            arraysize, arraysize, blocksize, blocksize)
        self.head_rank_descriptor = self.blacsgrid.new_descriptor(
            arraysize, arraysize, arraysize, arraysize)

        self.head_to_all_redistributor = Redistributor(
            self.blacsgrid.comm,
            self.head_rank_descriptor,
            self.distributed_descriptor)

        self.all_to_head_redistributor = Redistributor(
            self.blacsgrid.comm,
            self.distributed_descriptor,
            self.head_rank_descriptor)

        self.use_elpa = use_elpa

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
        if is_master:
            assert self.blacsgrid.comm.rank == 0
            Asc_MM[:, :] = A
            Bsc_MM[:, :] = B

            # tri2full is only necessary if we are using Elpa.
            # We should perhaps investigate if there's a way to tell
            # Elpa only to use lower, so we don't have to tri2full.
            if self.use_elpa:
                tri2full(Bsc_MM)
                tri2full(Asc_MM)

        self.head_to_all_redistributor.redistribute(Asc_MM, Asc_mm)
        self.head_to_all_redistributor.redistribute(Bsc_MM, Bsc_mm)

        if self.use_elpa:
            self._elpa.general_diagonalize(
                Asc_mm, Bsc_mm, vec_mm, temporary_eps)
        else:
            self.distributed_descriptor.general_diagonalize_dc(
                Asc_mm, Bsc_mm, vec_mm, temporary_eps)

        # vec_MM contains the eigenvectors in 'Fortran form'. They need to be
        # transpose-conjugated before they are consistent with Scipy behaviour
        self.all_to_head_redistributor.redistribute(vec_mm, vec_MM, uplo="G")

        if is_master:
            # Conjugate-transpose here since general_diagonalize_dc gives us
            # Fortran-convention eigenvectors.
            A[:, :] = vec_MM.conj().T
            eps[:] = temporary_eps

    @lazyproperty
    def _elpa(self):
        from gpaw.utilities.elpa import LibElpa
        elpa = LibElpa(self.distributed_descriptor, solver='2stage')
        gpu_kwargs = {}
        if 1:
            gpu_kwargs = {
                'nvidia-gpu': 1,
                'use_gpu_id': 0,
                'gpu_cholesky': 1,
                'gpu_hermitian_multiply': 1
            }
        elpa.elpa_set1(gpu_kwargs)
        return elpa
