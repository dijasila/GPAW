from __future__ import annotations

from functools import partial
from math import pi

import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays
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
                 weight: float = 1.0,
                 ncomponents: int = 1):
        self.psit_nX = psit_nX
        super().__init__(setups=setups,
                         nbands=psit_nX.dims[0],
                         spin=spin,
                         q=q,
                         k=k,
                         kpt_c=psit_nX.desc.kpt_c,
                         weight=weight,
                         ncomponents=ncomponents,
                         dtype=psit_nX.desc.dtype,
                         domain_comm=psit_nX.desc.comm,
                         band_comm=psit_nX.comm)
        self.fracpos_ac = fracpos_ac
        self.pt_aiX = None
        self.orthonormalized = False

    def array_shape(self, global_shape=False):
        if global_shape:
            return self.psit_nX.desc.myshape
        return self.psit_nX.desc.global_shape()

    def __len__(self):
        return self.psit_nX.dims[0]

    @classmethod
    def from_wfs(cls, wfs: WaveFunctions, psit_nX, fracpos_ac):
        return cls(psit_nX,
                   wfs.spin,
                   wfs.setups,
                   fracpos_ac,
                   wfs.weight,
                   wfs.spin_degeneracy)

    @property
    def P_ain(self):
        if self._P_ain is None:
            if self.pt_aiX is None:
                self.pt_aiX = self.setups.create_projectors(self.psit_nX.desc,
                                                            self.fracpos_ac)
            self._P_ain = self.pt_aiX.empty(self.psit_nX.dims,
                                            self.psit_nX.comm,
                                            transposed=True)
            self.pt_aiX.integrate(self.psit_nX, self._P_ain)
        return self._P_ain

    def move(self, fracpos_ac):
        self._P_ain = None
        self.orthonormalized = False
        self.pt_aiX.move(fracpos_ac)
        self._eig_n = None
        self._occ_n = None

    def add_to_density(self,
                       nt_sR,
                       D_asii: AtomArrays) -> None:
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        self.psit_nX.abs_square(weights=occ_n, out=nt_sR[self.spin])
        self.add_to_atomic_density_matrices(occ_n, D_asii)

    def orthonormalize(self, work_array_nX: ArrayND = None):
        if self.orthonormalized:
            return
        psit_nX = self.psit_nX
        domain_comm = psit_nX.desc.comm

        P_ain = self.P_ain

        P2_ain = P_ain.new()
        psit2_nX = psit_nX.new(data=work_array_nX)

        dS = self.setups.overlap_correction

        S = psit_nX.matrix_elements(psit_nX, domain_sum=False)
        dS(P_ain, out=P2_ain)
        P_ain.matrix.multiply(P2_ain, opa='C', symmetric=True, out=S, beta=1.0)
        domain_comm.sum(S.data, 0)

        if domain_comm.rank == 0:
            S.invcholesky()
        # S now contains the inverse of the Cholesky factorization
        domain_comm.broadcast(S.data, 0)
        # cc ??????

        S.multiply(psit_nX, out=psit2_nX)
        P_ain.matrix.multiply(S, opb='T', out=P2_ain)
        psit_nX.data[:] = psit2_nX.data
        P_ain.data[:] = P2_ain.data

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

           ~ ~    a    ~  ~
          <𝜓|p> dH    <p |𝜓>
            m i   ij    j  n
        """
        self.orthonormalize(work_array)
        psit_nX = self.psit_nX
        P_ain = self.P_ain
        psit2_nX = psit_nX.new(data=work_array)
        P2_ain = P_ain.new()
        domain_comm = psit_nX.desc.comm

        Ht = partial(Ht, out=psit2_nX, spin=self.spin)
        H = psit_nX.matrix_elements(psit_nX, function=Ht, domain_sum=False)
        dH(P_ain, out=P2_ain, spin=self.spin)
        P_ain.matrix.multiply(P2_ain, opa='C', symmetric=True,
                              out=H, beta=1.0)
        domain_comm.sum(H.data, 0)

        if domain_comm.rank == 0:
            slcomm, r, c, b = scalapack_parameters
            if r == c == 1:
                slcomm = None
            self._eig_n = H.eigh(scalapack=(slcomm, r, c, b))
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
        P_ain.matrix.multiply(H, opb='T', out=P2_ain)
        P_ain.data[:] = P2_ain.data

    def force_contribution(self, dH_asii: AtomArrays, F_av: Array2D):
        F_ainv = self.pt_aiX.derivative(self.psit_nX)
        myocc_n = self.weight * self.spin_degeneracy * self.myocc_n
        for a, F_inv in F_ainv.items():
            F_inv = F_inv.conj()
            F_inv *= myocc_n[:, np.newaxis]
            dH_ii = dH_asii[a][self.spin]
            P_in = self.P_ain[a]
            F_vii = np.einsum('inv, jn, jk -> vik', F_inv, P_in, dH_ii)
            F_inv *= self.myeig_n[:, np.newaxis]
            dO_ii = self.setups[a].dO_ii
            F_vii -= np.einsum('inv, jn, jk -> vik', F_inv, P_in, dO_ii)
            F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

    def collect(self, n1=0, n2=0):
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
                               center_v: Vector | None) -> Array3D:
        """Calculate dipole matrix-elements.

        :::

             /  _ ~ ~ _   ---  a  a _a
             | dr 𝜓 𝜓 r + >   P  P  D
             /     m n    ---  i  j  ij
                          aij

        Parameters
        ----------
        center_v:
            Center of molecule.  Defaults to center of cell.

        Returns matrix elements in atomic units.
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

        for a, P_in in self.P_ain.items():
            dipole_nnv += np.einsum('im, ijv, jn -> mnv',
                                    P_in, R_aiiv[a], P_in)

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
        psit_nX = self.psit_nX.gather()
        if psit_nX is not None:
            data_nX = psit_nX.matrix.gather()
            if data_nX.dist.comm.rank == 0:
                # XXX PW-gamma-point mode: float or complex matrix.dtype?
                return data_nX.data.view(psit_nX.data.dtype)
        return None
