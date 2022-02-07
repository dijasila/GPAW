from __future__ import annotations

import numpy as np
from ase.dft.bandgap import bandgap
from ase.units import Ha
from gpaw.core.atom_arrays import AtomArrays
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.brillouin import IBZ
from gpaw.new.wave_functions import WaveFunctions


class IBZWaveFunctions:
    def __init__(self,
                 ibz: IBZ,
                 nelectrons: float,
                 ncomponents: int,
                 create_wfs_func,
                 kpt_comm: MPIComm = serial_comm):
        """Collection of wave function objects for k-points in the IBZ."""
        self.ibz = ibz
        self.kpt_comm = kpt_comm
        self.nelectrons = nelectrons

        self.ncomponents = ncomponents
        self.collinear = (ncomponents == 4)
        self.spin_degeneracy = ncomponents % 2 + 1
        self.nspins = ncomponents % 3

        self.rank_k = ibz.ranks(kpt_comm)
        mask_k = (self.rank_k == kpt_comm.rank)
        kpt_qc = ibz.kpt_kc[mask_k]
        weight_q = ibz.weight_k[mask_k]

        self.wfs_qs: list[list[WaveFunctions]] = []
        for q, (kpt_c, weight) in enumerate(zip(kpt_qc, weight_q)):
            wfs_s = []
            for spin in range(self.nspins):
                wfs = create_wfs_func(spin, q, kpt_c, weight)
                wfs_s.append(wfs)
            self.wfs_qs.append(wfs_s)

        self.band_comm = wfs.band_comm
        self.domain_comm = wfs.domain_comm
        self.dtype = wfs.dtype
        self.nbands = wfs.nbands

        self.fermi_levels = None

        # IBZ-index to local index:
        self.q_k = {k: q for q, k in enumerate(np.arange(len(ibz))[mask_k])}

        self.energies: dict[str, float] = {}

    def __str__(self):
        return (f'{self.ibz}\n'
                f'Valence electrons: {self.nelectrons}\n'
                f'Spin-degeneracy: {self.spin_degeneracy}')

    def __iter__(self):
        for wfs_s in self.wfs_qs:
            yield from wfs_s

    def move(self, fracpos_ac):
        self.ibz.symmetries.check_positions(fracpos_ac)
        self.energies.clear()
        for wfs in self:
            wfs.move(fracpos_ac)

    def orthonormalize(self, work_array_nX: np.ndarray = None):
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

        nspins = 2 // self.spin_degeneracy
        nkpts = len(ibz)
        eig_skn = np.empty((nspins, nkpts, self.nbands))
        occ_skn = np.empty((nspins, nkpts, self.nbands))
        for k in range(nkpts):
            for s in range(nspins):
                (eig_skn[s, k, :],
                 occ_skn[s, k, :]) = self.get_eigs_and_occs(k, s)

        for k, (x, y, z) in enumerate(ibz.kpt_kc):
            log(f'\nkpt = [{x:.3f}, {y:.3f}, {z:.3f}], '
                f'weight = {ibz.weight_k[k]:.3f}:')

            if self.spin_degeneracy == 2:
                log('  Band      eig [eV]   occ [0-2]')
                for n, (e, f) in enumerate(zip(eig_skn[0, k] * Ha,
                                               occ_skn[0, k])):
                    log(f'  {n:4} {e:13.3f}   {2 * f:9.3f}')
            else:
                log('  Band      eig [eV]   occ [0-1]'
                    '      eig [eV]   occ [0-1]')
                for n, (e1, f1, e2, f2) in enumerate(zip(eig_skn[0, k] * Ha,
                                                         occ_skn[0, k],
                                                         eig_skn[1, k] * Ha,
                                                         occ_skn[1, k])):
                    log(f'  {n:4} {e1:13.3f}   {f1:9.3f}'
                        f'    {e2:10.3f}   {f2:9.3f}')
            if k == 3:
                break

        try:
            bandgap(eigenvalues=eig_skn * Ha,
                    efermi=fl[0],
                    output=log.fd,
                    kpts=ibz.kpt_kc)
        except ValueError:
            # Maybe we only have the occupied bands and no empty bands
            pass

    def get_homo_lumo(self, spin=None):
        """Return HOMO and LUMO eigenvalues."""
        if spin is None:
            if self.spin_degeneracy == 2:
                return self.get_homo_lumo(0)
            h0, l0 = self.get_homo_lumo(0)
            h1, l1 = self.get_homo_lumo(1)
            return np.array([max(h0, h1), min(l0, l1)])

        n = int(round(self.nelectrons)) // 2
        assert 2 * n == self.nelectrons
        homo = self.kpt_comm.max(max(wfs._eig_n[n - 1] for wfs in self))
        lumo = self.kpt_comm.min(min(wfs._eig_n[n] for wfs in self))

        return np.array([homo, lumo])
