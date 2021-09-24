from __future__ import annotations
import numpy as np
from functools import partial
from gpaw.core.arrays import DistributedArrays as DA
from gpaw.setup import Setups
from gpaw.typing import Array1D, Array2D, ArrayLike1D
from gpaw.new.brillouin import IBZ
from gpaw.mpi import MPIComm


class IBZWaveFunctions:
    def __init__(self,
                 ibz: IBZ,
                 ranks: ArrayLike1D,
                 kpt_comm: MPIComm,
                 mykpts: list[WaveFunctions],
                 nelectrons: int):
        self.ibz = ibz
        self.ranks = ranks
        self.kpt_comm = kpt_comm
        self.mykpts = mykpts
        self.nelectrons = nelectrons

    def __iter__(self):
        for wfs in self.mykpts:
            yield wfs

    @staticmethod
    def from_random_numbers(cfg,
                            nbands: int) -> IBZWaveFunctions:
        ibz = cfg.ibz
        assert len(ibz) == 1
        ranks = [0]
        band_comm = cfg.communicators['b']
        kpt_comm = cfg.communicators['k']

        mykpts = []
        for kpt, weight, rank in zip(ibz.points, ibz.weights, ranks):
            if rank != kpt_comm.rank:
                continue
            basis = cfg.grid.new(kpt=kpt)
            wfs = WaveFunctions.from_random_numbers(basis, weight,
                                                    nbands, band_comm,
                                                    cfg.setups,
                                                    cfg.positions)
            mykpts.append(wfs)

        return IBZWaveFunctions(ibz, ranks, kpt_comm, mykpts, cfg.nelectrons)

    def orthonormalize(self, work_array=None):
        for wfs in self:
            wfs.orthonormalize(work_array)

    def calculate_occs(self, occs_calc):
        for wfs in self:
            wfs._occs = np.ones(2)


class WaveFunctions:
    def __init__(self,
                 wave_functions: DA,
                 spin: int | None,
                 setups: Setups,
                 positions: Array2D,
                 weight: float = 1.0):
        self.wave_functions = wave_functions
        self.spin = spin
        self.setups = setups
        self.weight = weight

        self._projections = None
        self.projectors = setups.create_projectors(wave_functions.layout,
                                                   positions)
        self.orthonormalized = False

        self._eigs: Array1D | None = None
        self._occs: Array1D | None = None

    @property
    def eigs(self) -> Array1D:
        if self._eigs is None:
            raise ValueError
        return self._eigs

    @property
    def occs(self) -> Array1D:
        if self._occs is None:
            raise ValueError
        return self._occs

    @property
    def projections(self):
        if self._projections is None:
            self._projections = self.projectors.integrate(self.wave_functions)
        return self._projections

    @property
    def myeigs(self):
        assert self.wave_functions.comm.size == 1
        return self.eigs

    @staticmethod
    def from_random_numbers(basis, weight, nbands, band_comm, setups,
                            positions):
        wfs = basis.random(nbands, band_comm)
        return WaveFunctions(wfs, 0, setups, positions)

    def orthonormalize(self, work_array=None):
        if self.orthonormalized:
            return
        wfs = self.wave_functions
        domain_comm = wfs.layout.comm

        projections = self.projections

        projections2 = projections.new()
        wfs2 = wfs.new(data=work_array)

        dS = partial(self.setups.overlap_correction, out=projections2)

        S = wfs.matrix_elements(wfs, domain_sum=False)
        projections.matrix_elements(projections, function=dS,
                                    domain_sum=False, out=S, add_to_out=True)
        domain_comm.sum(S.data, 0)
        if domain_comm.rank == 0:
            S.invcholesky()
        # S now contains the inverse of the Cholesky factorization
        domain_comm.broadcast(S.data, 0)
        # cc ??????

        S.multiply(wfs, out=wfs2)
        projections.matrix.multiply(S, opb='T', out=projections2)
        wfs.data[:] = wfs2.data
        projections.data[:] = projections2.data

        self.orthonormalized = True

    def subspace_diagonalize(self,
                             Ht,
                             dH,
                             work_array=None,
                             Htpsit=None,
                             scalapack_parameters=(None, 1, 1, -1)):
        """

        Ht(in, out)::

           ~   ^   ~
           H = T + v

        dH::

            ~  ~    a    ~  ~
          <psi|p> dH    <p|psi>
              m i   ij    j   n
        """
        self.orthonormalize(work_array)
        psit = self.wave_functions
        projections = self.projections
        psit2 = psit.new(data=work_array)
        projections2 = projections.new()
        domain_comm = psit.layout.comm

        Ht = partial(Ht, out=psit2, spin=0)
        dH = partial(dH, out=projections2, spin=0)
        H = psit.matrix_elements(psit, function=Ht, domain_sum=False)
        projections.matrix_elements(projections, function=dH,
                                    domain_sum=False, out=H, add_to_out=True)
        domain_comm.sum(H.data, 0)

        if domain_comm.rank == 0:
            slcomm, r, c, b = scalapack_parameters
            if r == c == 1:
                slcomm = None
            self._eigs = H.eigh(scalapack=(slcomm, r, c, b))
            # H.data[n, :] now contains the n'th eigenvector and eps_n[n]
            # the n'th eigenvalue

        domain_comm.broadcast(H.data, 0)
        domain_comm.broadcast(self.eigs, 0)

        if Htpsit is not None:
            H.multiply(psit2, out=Htpsit)

        H.multiply(psit, out=psit2)
        psit.data[:] = psit2.data
        projections.matrix.multiply(H, opb='T', out=projections2)
        projections.data[:] = projections2.data
