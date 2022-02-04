from __future__ import annotations

from functools import partial

import numpy as np
from gpaw.core.arrays import DistributedArrays as DA
from gpaw.core.atom_arrays import AtomArrays
from gpaw.setup import Setups
from gpaw.typing import Array1D, Array2D, ArrayND
from gpaw.mpi import MPIComm


class WaveFunctions:
    domain_comm: MPIComm
    band_comm: MPIComm
    nbands: int
    P_ain: AtomArrays

    def __init__(self,
                 spin: int | None,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2,
                 dtype=float):
        self.spin = spin
        self.setups = setups
        self.weight = weight
        self.spin_degeneracy = spin_degeneracy
        self.dtype = dtype

        self._P_ain = None

        self._eig_n: Array1D | None = None
        self._occ_n: Array1D | None = None

    @property
    def eig_n(self) -> Array1D:
        if self._eig_n is None:
            raise ValueError
        return self._eig_n

    @property
    def occ_n(self) -> Array1D:
        if self._occ_n is None:
            raise ValueError
        return self._occ_n

    @property
    def myeig_n(self):
        assert self.band_comm.size == 1
        return self.eig_n

    @property
    def myocc_n(self):
        assert self.band_comm.size == 1
        return self.occ_n

    def add_to_atomic_density_matrices(self,
                                       occ_n,
                                       D_asii: AtomArrays) -> None:
        for D_sii, P_in in zip(D_asii.values(), self.P_ain.values()):
            D_sii[self.spin] += np.einsum('in, n, jn -> ij',
                                          P_in.conj(), occ_n, P_in).real


class PWFDWaveFunctions(WaveFunctions):
    def __init__(self,
                 psit_nX: DA,
                 spin: int | None,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2):
        self.psit_nX = psit_nX
        super().__init__(spin, setups, fracpos_ac, weight, spin_degeneracy,
                         dtype=psit_nX.desc.dtype)
        self.pt_aiX = setups.create_projectors(self.psit_nX.desc,
                                               fracpos_ac)
        self.orthonormalized = False
        self.domain_comm = psit_nX.desc.comm
        self.band_comm = psit_nX.comm
        self.nbands = psit_nX.dims[0]

    @property
    def P_ain(self):
        if self._P_ain is None:
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

        Ht(in, out)::

           ~   ^   ~
           H = T + v

        dH::

            ~  ~    a    ~  ~
          <psi|p> dH    <p|psi>
              m i   ij    j   n
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
