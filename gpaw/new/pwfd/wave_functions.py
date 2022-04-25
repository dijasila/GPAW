from __future__ import annotations

from functools import partial
from math import pi

import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.core.uniform_grid import UniformGrid, UniformGridFunctions
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.typing import Array2D, Array3D, ArrayND, Vector
from gpaw.fftw import get_efficient_fft_size


class PWFDWaveFunctions(WaveFunctions):
    def __init__(self,
                 psit_nX: DistributedArrays,
                 spin: int,
                 q: int,
                 k: int,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 atomdist: AtomDistribution,
                 weight: float = 1.0,
                 ncomponents: int = 1):
        assert isinstance(atomdist, AtomDistribution)
        self.psit_nX = psit_nX
        super().__init__(setups=setups,
                         nbands=psit_nX.dims[0],
                         spin=spin,
                         q=q,
                         k=k,
                         kpt_c=psit_nX.desc.kpt_c,
                         fracpos_ac=fracpos_ac,
                         atomdist=atomdist,
                         weight=weight,
                         ncomponents=ncomponents,
                         dtype=psit_nX.desc.dtype,
                         domain_comm=psit_nX.desc.comm,
                         band_comm=psit_nX.comm)
        self.pt_aiX = None
        self.orthonormalized = False

    def __del__(self):
        data = self.psit_nX.data
        if hasattr(data, 'fd'):
            data.fd.close()

    def array_shape(self, global_shape=False):
        if global_shape:
            return self.psit_nX.desc.global_shape()
        return self.psit_nX.desc.myshape

    def __len__(self):
        return self.psit_nX.dims[0]

    @classmethod
    def from_wfs(cls,
                 wfs: WaveFunctions,
                 psit_nX,
                 fracpos_ac,
                 atomdist):
        return cls(psit_nX,
                   wfs.spin,
                   wfs.setups,
                   fracpos_ac,
                   atomdist,
                   wfs.weight,
                   wfs.spin_degeneracy)

    @property
    def P_ani(self):
        if self._P_ani is None:
            if self.pt_aiX is None:
                self.pt_aiX = self.psit_nX.desc.atom_centered_functions(
                    [setup.pt_j for setup in self.setups],
                    self.fracpos_ac,
                    atomdist=self.atomdist)
            self._P_ani = self.pt_aiX.empty(self.psit_nX.dims,
                                            self.psit_nX.comm)
            self.pt_aiX.integrate(self.psit_nX, self._P_ani)
        return self._P_ani

    def move(self, fracpos_ac, atomdist):
        self._P_ani = None
        self.orthonormalized = False
        self.pt_aiX.move(fracpos_ac, atomdist)
        self._eig_n = None
        self._occ_n = None

    def add_to_density(self,
                       nt_sR,
                       D_asii: AtomArrays) -> None:
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        self.psit_nX.abs_square(weights=occ_n, out=nt_sR[self.spin])
        self.add_to_atomic_density_matrices(occ_n, D_asii)

    def orthonormalize(self, work_array_nX: ArrayND = None):
        r"""Orthonormalize wave functions.

        Computes the overlap matrix:::

               /~ _ *~ _   _   ---  a  * a   a
          S  = |ψ(r) ψ(r) dr + >  (P  ) P  ΔS
           mn  / m    n        ---  im   jn  ij
                               aij

        With `LSL^\dagger=1`, we update the wave functions and projections
        inplace like this:::

                  -- *      a    -- *  a
            Ψ  <- > L  Ψ,  P  <- > L  P
             m    -- mn n   in   -- mn in
                  n

        """
        if self.orthonormalized:
            return
        psit_nX = self.psit_nX
        domain_comm = psit_nX.desc.comm

        P_ani = self.P_ani

        P2_ani = P_ani.new()
        psit2_nX = psit_nX.new(data=work_array_nX)

        dS = self.setups.overlap_correction

        # We are actually calculating S^*:
        S = psit_nX.matrix_elements(psit_nX, domain_sum=False, cc=True)
        dS(P_ani, out=P2_ani)
        P_ani.matrix.multiply(P2_ani, opb='C', symmetric=True, out=S, beta=1.0)
        domain_comm.sum(S.data, 0)

        if domain_comm.rank == 0:
            S.invcholesky()

        domain_comm.broadcast(S.data, 0)
        # S now contains L^*

        S.multiply(psit_nX, out=psit2_nX)
        S.multiply(P_ani, out=P2_ani)
        psit_nX.data[:] = psit2_nX.data
        P_ani.data[:] = P2_ani.data

        self.orthonormalized = True

    def subspace_diagonalize(self,
                             Ht,
                             dH,
                             work_array: ArrayND = None,
                             Htpsit_nX=None,
                             scalapack_parameters=(None, 1, 1, -1)):
        """

        Ht(in, out):::

           ~   ^   ~
           H = T + v

        dH:::

           ~ ~    a  ~  ~
          <𝜓|p> ΔH  <p |𝜓>
            m i   ij  j  n
        """
        self.orthonormalize(work_array)
        psit_nX = self.psit_nX
        P_ani = self.P_ani
        psit2_nX = psit_nX.new(data=work_array)
        P2_ani = P_ani.new()
        domain_comm = psit_nX.desc.comm

        Ht = partial(Ht, out=psit2_nX, spin=self.spin)
        H = psit_nX.matrix_elements(psit_nX,
                                    function=Ht,
                                    domain_sum=False,
                                    cc=True)
        dH(P_ani, out=P2_ani, spin=self.spin)
        P_ani.matrix.multiply(P2_ani, opb='C', symmetric=True,
                              out=H, beta=1.0)
        domain_comm.sum(H.data, 0)

        if domain_comm.rank == 0:
            slcomm, r, c, b = scalapack_parameters
            if r == c == 1:
                slcomm = None
            self._eig_n = H.eigh(scalapack=(slcomm, r, c, b))
            H.complex_conjugate()
            # H.data[n, :] now contains the n'th eigenvector and eps_n[n]
            # the n'th eigenvalue
        else:
            self._eig_n = np.empty(psit_nX.dims)

        domain_comm.broadcast(H.data, 0)
        domain_comm.broadcast(self._eig_n, 0)
        if Htpsit_nX is not None:
            H.multiply(psit2_nX, out=Htpsit_nX)

        H.multiply(psit_nX, out=psit2_nX)
        psit_nX.data[:] = psit2_nX.data
        H.multiply(P_ani, out=P2_ani)
        P_ani.data[:] = P2_ani.data

    def force_contribution(self, dH_asii: AtomArrays, F_av: Array2D):
        F_avni = self.pt_aiX.derivative(self.psit_nX)
        myocc_n = self.weight * self.spin_degeneracy * self.myocc_n
        for a, F_vni in F_avni.items():
            F_vni = F_vni.conj()
            F_vni *= myocc_n[:, np.newaxis]
            dH_ii = dH_asii[a][self.spin]
            P_ni = self.P_ani[a]
            F_vii = np.einsum('vni, nj, jk -> vik', F_vni, P_ni, dH_ii)
            F_vni *= self.myeig_n[:, np.newaxis]
            dO_ii = self.setups[a].dO_ii
            F_vii -= np.einsum('vni, nj, jk -> vik', F_vni, P_ni, dO_ii)
            F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

    def collect(self,
                n1: int = 0,
                n2: int = 0) -> PWFDWaveFunctions:
        """Collect range of bands to master of band and domain
        communicators."""
        # Also collect projections instead of recomputing XXX
        n2 = n2 or len(self) + n2
        band_comm = self.psit_nX.comm
        domain_comm = self.psit_nX.desc.comm
        nbands = len(self)
        mynbands = (nbands + band_comm.size - 1) // band_comm.size
        rank1, b1 = divmod(n1, mynbands)
        rank2, b2 = divmod(n2, mynbands)
        if band_comm.rank == 0:
            if domain_comm.rank == 0:
                psit_nX = self.psit_nX.desc.new(comm=None).empty(n2 - n1)
            rank = rank1
            ba = b1
            na = n1
            while (rank, ba) < (rank2, b2):
                bb = min((rank + 1) * mynbands, nbands) - rank * mynbands
                if rank == rank2 and bb > b2:
                    bb = b2
                nb = na + bb - ba
                if bb > ba:
                    if rank == 0:
                        psit_bX = self.psit_nX[ba:bb].gather()
                        if domain_comm.rank == 0:
                            psit_nX.data[:bb - ba] = psit_bX.data
                    else:
                        if domain_comm.rank == 0:
                            band_comm.receive(psit_nX.data[na - n1:nb - n1],
                                              rank)
                rank += 1
                ba = 0
                na = nb
            if domain_comm.rank == 0:
                return PWFDWaveFunctions(psit_nX,
                                         self.spin,
                                         self.q,
                                         self.k,
                                         self.setups,
                                         self.fracpos_ac,
                                         self.atomdist.gather(),
                                         self.weight,
                                         self.ncomponents)
        else:
            rank = band_comm.rank
            ranka, ba = max((rank1, b1), (rank, 0))
            rankb, bb = min((rank2, b2), (rank, self.psit_nX.mydims[0]))
            if (ranka, ba) < (rankb, bb):
                assert ranka == rankb == rank
                band_comm.send(self.psit_nX.data[ba:bb])

    def dipole_matrix_elements(self,
                               center_v: Vector = None) -> Array3D:
        """Calculate dipole matrix-elements.

        :::

           _    /  _ ~ ~ _   ---  a  a  _a
           μ  = | dr 𝜓 𝜓 r + >   P  P  Δμ
            mn  /     m n    ---  im jn  ij
                             aij

        Parameters
        ----------
        center_v:
            Center of molecule.  Defaults to center of cell.

        Returns
        -------
        Array3D:
            matrix elements in atomic units.
        """
        cell_cv = self.psit_nX.desc.cell_cv

        if center_v is None:
            center_v = cell_cv.sum(0) * 0.5

        dipole_nnv = np.zeros((len(self), len(self), 3))

        scenter_c = np.linalg.solve(cell_cv.T, center_v)
        spos_ac = self.fracpos_ac.copy()
        spos_ac -= scenter_c - 0.5
        spos_ac %= 1.0
        spos_ac += scenter_c - 0.5
        position_av = spos_ac.dot(cell_cv)

        R_aiiv = []
        for setup, position_v in zip(self.setups, position_av):
            Delta_iiL = setup.Delta_iiL
            R_iiv = Delta_iiL[:, :, [3, 1, 2]] * (4 * pi / 3)**0.5
            R_iiv += position_v * setup.Delta_iiL[:, :, :1] * (4 * pi)**0.5
            R_aiiv.append(R_iiv)

        for a, P_ni in self.P_ani.items():
            dipole_nnv += np.einsum('mi, ijv, nj -> mnv',
                                    P_ni, R_aiiv[a], P_ni)

        self.psit_nX.desc.comm.sum(dipole_nnv)

        if isinstance(self.psit_nX, UniformGridFunctions):
            psit_nR = self.psit_nX
        else:
            # Find size of fft grid large enough to store square of wfs.
            pw = self.psit_nX.desc
            s1, s2, s3 = pw.indices_cG.ptp(axis=1)
            assert pw.dtype == float
            # Last dimension is special because dtype=float:
            size_c = [2 * s1 + 2,
                      2 * s2 + 2,
                      4 * s3 + 2]
            size_c = [get_efficient_fft_size(N, 2) for N in size_c]
            grid = UniformGrid(cell=pw.cell_cv, size=size_c)
            psit_nR = self.psit_nX.ifft(grid=grid)

        for na, psita_R in enumerate(psit_nR):
            for nb, psitb_R in enumerate(psit_nR[:na + 1]):
                d_v = (psita_R * psitb_R).moment(center_v)
                dipole_nnv[na, nb] += d_v
                if na != nb:
                    dipole_nnv[nb, na] += d_v

        return dipole_nnv

    def gather_wave_function_coefficients(self) -> np.ndarray:
        psit_nX = self.psit_nX.gather()  # gather X
        if psit_nX is not None:
            data_nX = psit_nX.matrix.gather()  # gather n
            if data_nX.dist.comm.rank == 0:
                # XXX PW-gamma-point mode: float or complex matrix.dtype?
                return data_nX.data.view(
                    psit_nX.data.dtype).reshape(psit_nX.data.shape)
        return None

    def receive(self, kpt_comm, rank):
        """PWFDWaveFunctions(
                 psit_nX: DistributedArrays,
                 spin: int,
                 q: int,
                 k: int,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 weight: float = 1.0,
                 ncomponents: int = 1):
        """
        return 42
