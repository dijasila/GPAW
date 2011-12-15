import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.mpi import rank

class LinearSolver:
    """Base class for solving sets of linear equations A.x = b iteratively."""

    def __init__(self, sort_bands=True, tolerance=1e-8,
                 max_iterations=1000, eps=1e-15):
        """Tolerance should not be smaller than attainable accuracy, which is
        order of kappa(A) * eps, where kappa(A) is the (spectral) condition
        number of the matrix. The maximum number of iterations should be
        significantly less than matrix size, approximately
        .5 sqrt(kappa) ln(2/tolerance). A small number is treated as zero
        if it's magnitude is smaller than argument eps.
        
        Parameters
        ----------
        sort_bands: bool
            determines whether to allow sorting of band by convergence
        tolerance: float
            tolerance for the norm of the residual ||b - A.x||^2
        max_iterations: integer
            maximum number of iterations
        eps: float
            if abs(rho) or omega < eps, it's regarded as zero
            and the method breaks down

        """

        self.gd = None
        self.bd = None
        self.timer = None
        self.dtype = None
        self.tol = tolerance
        self.maxiter = max_iterations
        self.niter = -1
        self.sort_bands = sort_bands
        self.initialized = False

        self.conv_n = None
        self.perm_n = None
        self.internals = ('conv_n', 'perm_n')

        if eps <= tolerance:
            self.eps = eps
        else:
            raise ValueError('%s: Invalid tolerance (tol = %le < eps = %le).'
                             % (self.__class__.__name__, tolerance, eps))

    def initialize(self, wfs):
        if self.initialized:
            raise RuntimeError('%s: Solver has already been initialized.'
                               % self.__class__.__name__)

        self.gd = wfs.gd
        self.bd = wfs.bd
        self.timer = wfs.timer
        self.dtype = wfs.dtype

        nvec = self.bd.mynbands
        self.conv_n = np.empty(nvec, dtype=bool)
        self.perm_n = np.empty(nvec, dtype=int)

        # Return without setting self.initialized to True

    def swap(self, x1, x2, *args):
        if x1 == x2:
            return

        # Swap internal state variables e.g. convergence flags and permutation
        for name in self.internals:
            b_x = getattr(self, name)
            tmp = b_x[x1]
            b_x[x1] = b_x[x2]
            b_x[x2] = tmp

        # Perform swap of real-space data using buffer
        tmp_G = self.gd.empty(dtype=self.dtype)
        for a_xG in args:
            tmp_G[:] = a_xG[x1]
            a_xG[x1] = a_xG[x2]
            a_xG[x2] = tmp_G

    def sort(self, *args):
        """Sort so that converged bands are placed contiguously at the end."""

        self.timer.start('Sort')

        # Number of converged bands and how many that can stay where they are
        nconv = np.sum(self.conv_n)
        nstay = np.sum(self.conv_n[-nconv:])

        while nstay < nconv:
            # Swap the lowest converged band with the highest unconverged
            x1 = np.min(np.arange(self.bd.mynbands)[self.conv_n])
            x2 = np.max(np.arange(self.bd.mynbands)[~self.conv_n])
            self.swap(x1, x2, *args)
            nconv = np.sum(self.conv_n)
            nstay = np.sum(self.conv_n[-nconv:])

        self.timer.stop('Sort')

    def restore(self, *args):
        """Restore original band order by undoing all permutations."""

        self.timer.start('Restore')

        for x1,n1 in enumerate(self.perm_n):
            n2 = x1
            x2 = np.argwhere(self.perm_n == n2).item()
            self.swap(x1, x2, *args)

        self.timer.stop('Restore')

    def solve(self, A, x_nG, b_nG, slow_convergence_iters=None):
        raise RuntimeError("LinearSolver member function 'solve' is virtual.")

    def __repr__(self):
        classname = self.__class__.__name__
        return '%s(sort_bands=%s, tolerance=%le, max_iterations=%d, eps=%le)' \
            % (classname, self.sort_bands, self.tol, self.maxiter, self.eps)

