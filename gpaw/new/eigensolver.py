"""Module defining an eigensolver cfg-class."""

import numpy as np
from ase.dft.bandgap import _bandgap
from ase.units import Ha
from ase.utils.timing import timer

from gpaw.utilities import unpack
from gpaw.utilities.blas import axpy
from gpaw.xc.hybrid import HybridXC


def reshape(a_x, shape):
    """Get an ndarray of size shape from a_x buffer."""
    return a_x.ravel()[:np.prod(shape)].reshape(shape)


class Eigensolver:
    def __init__(self, keep_htpsit=True, blocksize=1):
        self.keep_htpsit = keep_htpsit
        self.initialized = False
        self.Htpsit_nG = None
        self.error = np.inf
        self.blocksize = blocksize
        self.orthonormalization_required = True

    def initialize(self, wfs):
        self.timer = wfs.timer
        self.world = wfs.world
        self.kpt_comm = wfs.kd.comm
        self.band_comm = wfs.bd.comm
        self.dtype = wfs.dtype
        self.bd = wfs.bd
        self.nbands = wfs.bd.nbands
        self.mynbands = wfs.bd.mynbands

        if wfs.bd.comm.size > 1:
            self.keep_htpsit = False

        if self.keep_htpsit:
            self.Htpsit_nG = np.empty_like(wfs.work_array)

        # Preconditioner for the electronic gradients:
        self.preconditioner = wfs.make_preconditioner(self.blocksize)

        for kpt in wfs.kpt_u:
            if kpt.eps_n is None:
                kpt.eps_n = np.empty(self.mynbands)

        self.initialized = True

    def reset(self):
        self.initialized = False

    def weights(self, wfs):
        """Calculate convergence weights for all eigenstates."""
        weight_un = np.zeros((len(wfs.kpt_u), self.bd.mynbands))

        if isinstance(self.nbands_converge, int):
            # Converge fixed number of bands:
            n = self.nbands_converge - self.bd.beg
            if n > 0:
                for weight_n, kpt in zip(weight_un, wfs.kpt_u):
                    weight_n[:n] = kpt.weight
        elif self.nbands_converge == 'occupied':
            # Conveged occupied bands:
            for weight_n, kpt in zip(weight_un, wfs.kpt_u):
                if kpt.f_n is None:  # no eigenvalues yet
                    weight_n[:] = np.inf
                else:
                    # Methfessel-Paxton distribution can give negative
                    # occupation numbers - so we take the absolute value:
                    weight_n[:] = np.abs(kpt.f_n)
        else:
            # Converge state with energy up to CBM + delta:
            assert self.nbands_converge.startswith('CBM+')
            delta = float(self.nbands_converge[4:]) / Ha

            if wfs.kpt_u[0].f_n is None:
                weight_un[:] = np.inf  # no eigenvalues yet
            else:
                # Collect all eigenvalues and calculate band gap:
                efermi = np.mean(wfs.fermi_levels)
                eps_skn = np.array(
                    [[wfs.collect_eigenvalues(k, spin) - efermi
                      for k in range(wfs.kd.nibzkpts)]
                     for spin in range(wfs.nspins)])
                if wfs.world.rank > 0:
                    eps_skn = np.empty((wfs.nspins,
                                        wfs.kd.nibzkpts,
                                        wfs.bd.nbands))
                wfs.world.broadcast(eps_skn, 0)
                try:
                    # Find bandgap + positions of CBM:
                    gap, _, (s, k, n) = _bandgap(eps_skn,
                                                 spin=None, direct=False)
                except ValueError:
                    gap = 0.0

                if gap == 0.0:
                    cbm = efermi
                else:
                    cbm = efermi + eps_skn[s, k, n]

                ecut = cbm + delta

                for weight_n, kpt in zip(weight_un, wfs.kpt_u):
                    weight_n[kpt.eps_n < ecut] = kpt.weight

                if (eps_skn[:, :, -1] < ecut - efermi).any():
                    # We don't have enough bands!
                    weight_un[:] = np.inf

        return weight_un

    def iterate(self, ham, wfs):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement *iterate_one_k_point* method for a single iteration of
        a single kpoint.
        """

        if not self.initialized:
            if isinstance(ham.xc, HybridXC):
                self.blocksize = wfs.bd.mynbands
            self.initialize(wfs)

        weight_un = self.weights(wfs)

        error = 0.0
        for kpt, weights in zip(wfs.kpt_u, weight_un):
            if not wfs.orthonormalized:
                wfs.orthonormalize(kpt)
            e = self.iterate_one_k_point(ham, wfs, kpt, weights)
            error += e
            if self.orthonormalization_required:
                wfs.orthonormalize(kpt)

        wfs.orthonormalized = True
        self.error = self.band_comm.sum(self.kpt_comm.sum(error))

    def iterate_one_k_point(self, ham, kpt):
        """Implemented in subclasses."""
        raise NotImplementedError

    def calculate_residuals(self, kpt, wfs, ham, dH, dS, psit, projections,
                            eps_n,
                            R, tmp1, tmp2,
                            n_x=None, calculate_change=False):
        """Calculate residual.

        From R=Ht*psit calculate R=H*psit-eps*S*psit."""

        for R_G, eps, psit_G in zip(R.data, eps_n, psit.data):
            axpy(-eps, psit_G, R_G)

        dH(projections, out=tmp1)
        tmp2.data[:] = projections.data * eps_n
        dS(tmp2, out=tmp2)
        tmp1.data -= tmp2.data

        ham.xc.add_correction(kpt, psit.data, R.data,
                              projections,
                              tmp1,
                              n_x,
                              calculate_change)
        kpt.projectors.add_to(R, tmp1)

    def estimate_memory(self, mem, wfs):
        gridmem = wfs.bytes_per_wave_function()

        keep_htpsit = self.keep_htpsit and (wfs.bd.mynbands == wfs.bd.nbands)

        if keep_htpsit:
            mem.subnode('Htpsit', wfs.bd.nbands * gridmem)
        else:
            mem.subnode('No Htpsit', 0)

        mem.subnode('eps_n', wfs.bd.mynbands * mem.floatsize)
        mem.subnode('eps_N', wfs.bd.nbands * mem.floatsize)
        mem.subnode('Preconditioner', 4 * gridmem)
        mem.subnode('Work', gridmem)
