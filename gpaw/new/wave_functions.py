from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
from ase.units import Ha
from gpaw.core.arrays import DistributedArrays as DA
from gpaw.core.atom_arrays import AtomArrays
from gpaw.mpi import MPIComm
from gpaw.new.brillouin import IBZ
from gpaw.setup import Setups
from gpaw.typing import Array1D, Array2D, ArrayND
from gpaw.utilities.debug import frozen


@frozen
class IBZWaveFunctions:
    def __init__(self,
                 ibz: IBZ,
                 rank_k: Sequence[int],
                 kpt_comm: MPIComm,
                 wfs_qs: list[list[WaveFunctions]],
                 nelectrons: float,
                 spin_degeneracy: int = 2):
        """Collection of wave function objects for k-points in the IBZ."""
        self.ibz = ibz
        self.rank_k = rank_k
        self.kpt_comm = kpt_comm
        self.wfs_qs = wfs_qs
        self.nelectrons = nelectrons
        self.fermi_levels = None
        self.collinear = False
        self.spin_degeneracy = spin_degeneracy

        self.band_comm = wfs_qs[0][0].psit_nX.comm
        self.domain_comm = wfs_qs[0][0].psit_nX.desc.comm

        self.nbands = wfs_qs[0][0].psit_nX.dims[0]

        self.q_k = {}  # ibz index to local index
        q = 0
        for k, rank in enumerate(rank_k):
            if rank == kpt_comm.rank:
                self.q_k[k] = q
                q += 1
        self.energies: dict[str, float] = {}

    def __str__(self):
        return str(self.ibz)

    def __iter__(self):
        for wfs_s in self.wfs_qs:
            yield from wfs_s

    @classmethod
    def from_random_numbers(cls,
                            ibz,
                            band_comm,
                            kpt_comm,
                            desc,
                            setups,
                            fracpos_ac,
                            nbands: int,
                            nelectrons: float,
                            dtype=None) -> IBZWaveFunctions:
        rank_k = ibz.ranks(kpt_comm)

        wfs_q = []
        for kpt_c, weight, rank in zip(ibz.kpt_kc, ibz.weight_k, rank_k):
            if rank != kpt_comm.rank:
                continue
            desck = desc.new(kpt=kpt_c, dtype=dtype)
            wfs = WaveFunctions.from_random_numbers(desck, weight,
                                                    nbands, band_comm,
                                                    setups,
                                                    fracpos_ac)
            wfs_q.append(wfs)

        return cls(ibz, rank_k, kpt_comm, wfs_q, nelectrons)

    def move(self, fracpos_ac):
        self.ibz.symmetries.check_positions(fracpos_ac)
        self.energies.clear()
        for wfs in self:
            wfs._P_ain = None
            wfs.orthonormalized = False
            wfs.pt_aiX.move(fracpos_ac)
            wfs._eig_n = None
            wfs._occ_n = None

    def orthonormalize(self, work_array_nX: ArrayND = None):
        for wfs in self:
            wfs.orthonormalize(work_array_nX)

    def calculate_occs(self, occ_calc, fixed_fermi_level=False):
        degeneracy = self.spin_degeneracy

        # u index is q and s combined
        occ_un, fermi_levels, e_entropy = occ_calc.calculate(
            nelectrons=self.nelectrons / degeneracy,
            eigenvalues=[wfs.eig_n * Ha for wfs in self],
            weights=[wfs.weight for wfs in self],
            fermi_levels_guess=(None
                                if self.fermi_levels is None else
                                self.fermi_levels * Ha))

        if not fixed_fermi_level or self.fermi_levels is None:
            self.fermi_levels = np.array(fermi_levels) / Ha

        for occ_n, wfs in zip(occ_un, self):
            wfs._occ_n = occ_n

        e_entropy *= degeneracy / Ha
        e_band = 0.0
        for wfs in self:
            e_band += wfs.occ_n @ wfs.eig_n * wfs.weight * degeneracy
        e_band = self.kpt_comm.sum(e_band)
        self.energies = {
            'band': e_band,
            'entropy': e_entropy,
            'extrapolation': e_entropy * occ_calc.extrapolate_factor}

    def add_to_density(self, nt_sR, D_asii) -> None:
        for wfs in self:
            wfs.add_to_density(nt_sR, D_asii)
        self.kpt_comm.sum(nt_sR.data)
        self.kpt_comm.sum(D_asii.data)

    def get_eigs_and_occs(self, k=0, s=0):
        if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
            rank = self.rank_k[k]
            if rank == self.kpt_comm.rank:
                wfs = self.wfs_qs[self.q_k[k]][s]
                if rank == 0:
                    return wfs._eig_n, wfs._occ_n
                self.kpt_comm.send(wfs._eig_n, 0)
                self.kpt_comm.send(wfs._occ_n, 0)
            elif self.kpt_comm.rank == 0:
                eig_n = np.empty(self.nbands)
                occ_n = np.empty(self.nbands)
                self.kpt_comm.receive(eig_n, rank)
                self.kpt_comm.receive(occ_n, rank)
                return eig_n, occ_n
        return np.zeros(0), np.zeros(0)

    def forces(self, dH_asii: AtomArrays):
        F_av = np.zeros((dH_asii.natoms, 3))
        for wfs in self:
            wfs.force_contribution(dH_asii, F_av)
        self.kpt_comm.sum(F_av)
        return F_av

    def write(self, writer, skip_wfs):
        writer.write(fermi_levels=self.fermi_levels)

    def write_summary(self, log):
        fl = self.fermi_levels * Ha
        assert len(fl) == 1
        log(f'\nFermi level: {fl[0]:.3f}')

        ibz = self.ibz

        for k, (x, y, z) in enumerate(ibz.kpt_kc):
            log(f'\nkpt = [{x:.3f}, {y:.3f}, {z:.3f}], '
                f'weight = {ibz.weight_k[k]:.3f}:')

            if self.spin_degeneracy == 2:
                log('  Band      eig [eV]   occ [0-2]')
                eigs, occs = self.get_eigs_and_occs(k)
                for n, (e, f) in enumerate(zip(eigs * Ha, occs)):
                    log(f'  {n:4} {e:13.3f}   {2 * f:9.3f}')
            else:
                log('  Band      eig [eV]   occ [0-1]'
                    '      eig [eV]   occ [0-1]')
                eigs1, occs1 = self.get_eigs_and_occs(k, 0)
                eigs2, occs2 = self.get_eigs_and_occs(k, 1)
                for n, (e1, f1, e2, f2) in enumerate(zip(eigs1 * Ha,
                                                         occs1,
                                                         eigs2 * Ha,
                                                         occs2)):
                    log(f'  {n:4} {e1:13.3f}   {f1:9.3f}'
                        f'    {e2:10.3f}   {f2:9.3f}')
            if k == 3:
                break


@frozen
class WaveFunctions:
    def __init__(self,
                 psit_nX: DA,
                 spin: int | None,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2):
        self.psit_nX = psit_nX
        self.spin = spin
        self.setups = setups
        self.weight = weight
        self.spin_degeneracy = spin_degeneracy

        self._P_ain = None
        self.pt_aiX = setups.create_projectors(self.psit_nX.desc,
                                               fracpos_ac)
        self.orthonormalized = False

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
        assert self.psit_nX.comm.size == 1
        return self.eig_n

    @property
    def myocc_n(self):
        assert self.psit_nX.comm.size == 1
        return self.occ_n

    @property
    def P_ain(self):
        if self._P_ain is None:
            self._P_ain = self.pt_aiX.empty(self.psit_nX.dims,
                                            self.psit_nX.comm,
                                            transposed=True)
            self.pt_aiX.integrate(self.psit_nX, self._P_ain)
        return self._P_ain

    @classmethod
    def from_random_numbers(cls, desc, weight, nbands, band_comm, setups,
                            fracpos_ac):
        wfs = desc.random(nbands, band_comm)
        return cls(wfs, 0, setups, fracpos_ac, weight)

    def add_to_density(self,
                       nt_sR,
                       D_asii: AtomArrays) -> None:
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        self.psit_nX.abs_square(weights=occ_n, out=nt_sR[self.spin])

        for D_sii, P_in in zip(D_asii.values(), self.P_ain.values()):
            D_sii[self.spin] += np.einsum('in, n, jn -> ij',
                                          P_in.conj(), occ_n, P_in).real

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
