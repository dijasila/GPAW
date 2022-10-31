import numpy as np
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.typing import Array2D
from gpaw.new import zip


def forces(ibzwfs: LCAOWaveFunctions) -> Array2D:
    domain_comm = ibzwfs.domain_comm

    dThetadR_qvMM, dTdR_qvMM = ibzwfs.manytci.O_qMM_T_qMM(
        domain_comm,
        0, ibzwfs.setups.nao,
        False, derivative=True)
    dPdR_aqvMi = ibzwfs.manytci.P_aqMi(
        ibzwfs.atomdist.indices,
        derivative=True)

    domain_comm.sum(dThetadR_qvMM)
    domain_comm.sum(dTdR_qvMM)

    F_av = np.zeros((len(ibzwfs.setups), 3))
    indices = []
    M1 = 0
    for a, P_Mi in ibzwfs.P_aMi.items():
        M2 = M1 + len(P_Mi)
        indices.append((a, M1, M2))
        M1 = M2

    for wfs, dTdR_vMM in zip(ibzwfs, dTdR_qvMM):
        #Transpose?
        rho_MM = wfs.calculate_density_matrix()
        erho_MM = wfs.calculate_density_matrix(eigs=True)

        add_kinetic_term(rho_MM, dTdR_vMM, F_av, indices)
        Fpot_av = get_pot_term()
        Ftheta_av = get_den_mat_term()
        Frho_av = get_den_mat_paw_term()
        Fatom_av = get_atomic_density_term()

    return Fkin_av + Fpot_av + Ftheta_av + Frho_av + Fatom_av


def add_kinetic_term(rho_MM, dTdR_vMM, F_av, indices):
    """Calculate Kinetic energy term in LCAO

    :::

                      dT
     _a        --- --   μν
     F += 2 Re >   >  ---- ρ
               --- --  _    νμ
               μ=a ν  dR
                        νμ
            """

    for a, M1, M2 in indices:
        F_av[a, :] += 2 * np.einsum('vmM, Mm -> v',
                                    dTdR_vMM[:, M1:M2],
                                    rho_MM[:, M1:M2])


def get_den_mat_term(self):
    """Calculate density matrix term in LCAO"""
    Ftheta_av = np.zeros_like(self.Fref_av)
    # Density matrix contribution due to basis overlap
    #
    #            ----- d Theta
    #  a          \           mu nu
    # F  += -2 Re  )   ------------  E
    #             /        d R        nu mu
    #            -----        mu nu
    #         mu in a; nu
    #
    Ftheta_av = np.zeros_like(Ftheta_av)
    for u, kpt in enumerate(self.kpt_u):
        dThetadRE_vMM = (self.dThetadR_qvMM[kpt.q] *
                         self.ET_uMM[u][np.newaxis]).real
        for a, M1, M2 in self.my_slices():
            Ftheta_av[a, :] += \
                -2.0 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
    return Ftheta_av


def get_pot_term(self):
    """Calculate potential term"""
    Fpot_av = np.zeros_like(self.Fref_av)
    # Potential contribution
    #
    #           -----      /  d Phi  (r)
    #  a         \        |        mu    ~
    # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
    #            /        |     d R                nu          nu mu
    #           -----    /         a
    #        mu in a; nu
    #
    self.timer.start('Potential')
    vt_sG = self.hamiltonian.vt_sG
    Fpot_av = np.zeros_like(Fpot_av)
    for u, kpt in enumerate(self.kpt_u):
        vt_G = vt_sG[kpt.s]
        Fpot_av += self.bfs.calculate_force_contribution(vt_G,
                                                         self.rhoT_uMM[u],
                                                         kpt.q)
    self.timer.stop('Potential')

    return Fpot_av


def get_den_mat_paw_term(self):
    """Calcualte PAW correction"""
    # TO DO: split this function into
    # _get_den_mat_paw_term (which calculate Frho_av) and
    # get_paw_correction (which calculate ZE_MM)
    # Density matrix contribution from PAW correction
    #
    #           -----                        -----
    #  a         \      a                     \     b
    # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
    #            /      mu nu  nu mu          /     mu nu  nu mu
    #           -----                        -----
    #           mu nu                    b; mu in a; nu
    #
    # with
    #                  b*
    #         -----  dP
    #   b      \       i mu    b   b
    #  Z     =  )   -------- dS   P
    #   mu nu  /     dR        ij  j nu
    #         -----    b mu
    #           ij
    #
    self.timer.start('Paw correction')
    Frho_av = np.zeros_like(self.Fref_av)
    for u, kpt in enumerate(self.kpt_u):
        work_MM = np.zeros((self.mynao, self.nao), self.dtype)
        ZE_MM = None
        for b in self.my_atom_indices:
            setup = self.setups[b]
            dO_ii = np.asarray(setup.dO_ii, self.dtype)
            dOP_iM = np.zeros((setup.ni, self.nao), self.dtype)
            mmm(1.0, dO_ii, 'N', self.P_aqMi[b][kpt.q], 'C', 0.0, dOP_iM)
            for v in range(3):
                mmm(1.0,
                    self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop],
                    'N',
                    dOP_iM, 'N',
                    0.0, work_MM)
                ZE_MM = (work_MM * self.ET_uMM[u]).real
                for a, M1, M2 in self.slices():
                    dE = 2 * ZE_MM[M1:M2].sum()
                    Frho_av[a, v] -= dE  # the "b; mu in a; nu" term
                    Frho_av[b, v] += dE  # the "mu nu" term
    self.timer.stop('Paw correction')
    return Frho_av

def _get_den_mat_paw_term(self):
    # THIS doesn't work in parallel
    # Density matrix contribution from PAW correction
    #
    #           -----                        -----
    #  a         \      a                     \     b
    # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
    #            /      mu nu  nu mu          /     mu nu  nu mu
    #           -----                        -----
    #           mu nu                    b; mu in a; nu
    #
    # with
    #                  b*
    #         -----  dP
    #   b      \       i mu    b   b
    #  Z     =  )   -------- dS   P
    #   mu nu  /     dR        ij  j nu
    #         -----    b mu
    #           ij
    #
    Frho_av = np.zeros_like(self.Fref_av)
    self.timer.start('add paw correction')
    ZE_MM = self.get_paw_correction()
    for u, kpt in enumerate(self.kpt_u):
        for b in self.my_atom_indices:
            for v in range(3):
                for a, M1, M2 in self.slices():
                    dE = 2 * ZE_MM[u, b, v, M1:M2].sum()
                    Frho_av[a, v] -= dE.real  # the "b; mu in a; nu" term
                    Frho_av[b, v] += dE.real  # the "mu nu" term
    self.timer.stop('add paw correction')
    return Frho_av

def get_paw_correction(self):
    # THIS doesn't work in parallel
    #  <Phi_nu|pt_i>O_ii<dPt_i/dR|Phi_mu>
    self.timer.start('get paw correction')
    ZE_MM = np.zeros((len(self.kpt_u), len(self.my_atom_indices), 3,
                      self.mynao, self.nao), self.dtype)
    for u, kpt in enumerate(self.kpt_u):
        work_MM = np.zeros((self.mynao, self.nao), self.dtype)
        for b in self.my_atom_indices:
            setup = self.setups[b]
            dO_ii = np.asarray(setup.dO_ii, self.dtype)
            dOP_iM = np.zeros((setup.ni, self.nao), self.dtype)
            mmm(1.0, dO_ii, 'N', self.P_aqMi[b][kpt.q], 'C', 0.0, dOP_iM)
            for v in range(3):
                mmm(1.0,
                    self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop],
                    'N',
                    dOP_iM, 'N',
                    0.0, work_MM)
                ZE_MM[u, b, v, :, :] = (work_MM * self.ET_uMM[u]).real
    self.timer.stop('get paw correction')
    return ZE_MM

def get_atomic_density_term(self):
    Fatom_av = np.zeros_like(self.Fref_av)
    # Atomic density contribution
    #            -----                         -----
    #  a          \     a                       \     b
    # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
    #             /     mu nu    nu mu          /     mu nu    nu mu
    #            -----                         -----
    #            mu nu                     b; mu in a; nu
    #
    #                  b*
    #         ----- d P
    #  b       \       i mu   b   b
    # A     =   )   ------- dH   P
    #  mu nu   /    d R       ij  j nu
    #         -----    b mu
    #           ij
    #
    self.timer.start('Atomic Hamiltonian force')
    Fatom_av = np.zeros_like(Fatom_av)
    for u, kpt in enumerate(self.kpt_u):
        for b in self.my_atom_indices:
            H_ii = np.asarray(unpack(self.dH_asp[b][kpt.s]), self.dtype)
            HP_iM = gemmdot(H_ii, np.ascontiguousarray(
                            self.P_aqMi[b][kpt.q].T.conj()))
            for v in range(3):
                dPdR_Mi = \
                    self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop]
                ArhoT_MM = \
                    (gemmdot(dPdR_Mi, HP_iM) * self.rhoT_uMM[u]).real
                for a, M1, M2 in self.slices():
                    dE = 2 * ArhoT_MM[M1:M2].sum()
                    Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                    Fatom_av[b, v] -= dE  # the "mu nu" term
    self.timer.stop('Atomic Hamiltonian force')

    return Fatom_av

def get_den_mat_block_blacs(self, f_n, C_nM, redistributor):
    rho1_mm = self.ksl.calculate_blocked_density_matrix(f_n,
                                                        C_nM).conj()
    rho_mm = redistributor.redistribute(rho1_mm)
    return rho_mm

def get_pot_term_blacs(self):
    Fpot_av = np.zeros_like(self.Fref_av)
    from gpaw.blacs import BlacsGrid, Redistributor
    self.grid = BlacsGrid(self.ksl.block_comm, self.gd.comm.size,
                          self.bd.comm.size)
    self.blocksize1 = -(-self.nao // self.grid.nprow)
    self.blocksize2 = -(-self.nao // self.grid.npcol)
    desc = self.grid.new_descriptor(self.nao, self.nao,
                                    self.blocksize1, self.blocksize2)
    vt_sG = self.hamiltonian.vt_sG
    self.rhoT_umm = []
    self.ET_umm = []
    self.redistributor = Redistributor(self.grid.comm,
                                       self.ksl.mmdescriptor, desc)
    Fpot_av = np.zeros_like(self.Fref_av)
    for u, kpt in enumerate(self.kpt_u):
        self.timer.start('Get density matrix')
        rhoT_mm = self.get_den_mat_block_blacs(kpt.f_n, kpt.C_nM,
                                               self.redistributor)
        self.rhoT_umm.append(rhoT_mm)
        self.timer.stop('Get density matrix')
        self.timer.start('Potential')
        rhoT_mM = self.ksl.distribute_to_columns(rhoT_mm, desc)
        vt_G = vt_sG[kpt.s]
        Fpot_av += self.bfs.calculate_force_contribution(vt_G, rhoT_mM,
                                                         kpt.q)
        del rhoT_mM
        self.timer.stop('Potential')

    return Fpot_av

def get_kin_and_den_term_blacs(self):
    Fkin_av_sum = np.zeros_like(self.Fref_av)
    Ftheta_av_sum = np.zeros_like(self.Fref_av)
    # pcutoff_a = [max([pt.get_cutoff() for pt in setup.pt_j])
    #              for setup in self.setups]
    # phicutoff_a = [max([phit.get_cutoff() for phit in setup.phit_j])
    #                for setup in self.setups]
    # XXX should probably use bdsize x gdsize instead
    # That would be consistent with some existing grids
    # I'm not sure if this is correct
    # XXX what are rows and columns actually?
    dH_asp = self.hamiltonian.dH_asp
    self.timer.start('Get density matrix')
    for kpt in self.kpt_u:
        ET_mm = self.get_den_mat_block_blacs(kpt.f_n * kpt.eps_n, kpt.C_nM,
                                             self.redistributor)
        self.ET_umm.append(ET_mm)
    self.timer.stop('Get density matrix')
    self.M1start = self.blocksize1 * self.grid.myrow
    self.M2start = self.blocksize2 * self.grid.mycol
    self.M1stop = min(self.M1start + self.blocksize1, self.nao)
    self.M2stop = min(self.M2start + self.blocksize2, self.nao)
    self.m1max = self.M1stop - self.M1start
    self.m2max = self.M2stop - self.M2start
    # from gpaw.lcao.overlap import TwoCenterIntegralCalculator
    self.timer.start('Prepare TCI loop')
    self.M_a = self.bfs.M_a
    Fkin2_av = np.zeros_like(self.Fref_av)
    Ftheta2_av = np.zeros_like(self.Fref_av)
    self.atompairs = self.newtci.a1a2.get_atompairs()
    self.timer.start('broadcast dH')
    self.alldH_asp = {}
    for a in range(len(self.setups)):
        gdrank = self.bfs.sphere_a[a].rank
        if gdrank == self.gd.rank:
            dH_sp = dH_asp[a]
        else:
            ni = self.setups[a].ni
            dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
        self.gd.comm.broadcast(dH_sp, gdrank)
        # okay, now everyone gets copies of dH_sp
        self.alldH_asp[a] = dH_sp
    self.timer.stop('broadcast dH')
    # This will get sort of hairy.  We need to account for some
    # three-center overlaps, such as:
    #
    #         a1
    #      Phi   ~a3    a3  ~a3     a2     a2,a1
    #   < ----  |p  > dH   <p   |Phi  > rho
    #      dR
    #
    # To this end we will loop over all pairs of atoms (a1, a3),
    # and then a sub-loop over (a3, a2).
    self.timer.stop('Prepare TCI loop')
    self.timer.start('Not so complicated loop')
    for (a1, a2) in self.atompairs:
        if a1 >= a2:
            # Actually this leads to bad load balance.
            # We should take a1 > a2 or a1 < a2 equally many times.
            # Maybe decide which of these choices
            # depending on whether a2 % 1 == 0
            continue
        m1start = self.M_a[a1] - self.M1start
        m2start = self.M_a[a2] - self.M2start
        if m1start >= self.blocksize1 or m2start >= self.blocksize2:
            continue  # (we have only one block per CPU)
        nm1 = self.setups[a1].nao
        nm2 = self.setups[a2].nao
        m1stop = min(m1start + nm1, self.m1max)
        m2stop = min(m2start + nm2, self.m2max)
        if m1stop <= 0 or m2stop <= 0:
            continue
        m1start = max(m1start, 0)
        m2start = max(m2start, 0)
        J1start = max(0, self.M1start - self.M_a[a1])
        J2start = max(0, self.M2start - self.M_a[a2])
        M1stop = J1start + m1stop - m1start
        J2stop = J2start + m2stop - m2start
        dThetadR_qvmm, dTdR_qvmm = self.newtci.dOdR_dTdR(a1, a2)
        for u, kpt in enumerate(self.kpt_u):
            rhoT_mm = self.rhoT_umm[u][m1start:m1stop, m2start:m2stop]
            ET_mm = self.ET_umm[u][m1start:m1stop, m2start:m2stop]
            Fkin_v = 2.0 * (dTdR_qvmm[kpt.q][:, J1start:M1stop,
                                             J2start:J2stop] *
                            rhoT_mm[np.newaxis]).real.sum(-1).sum(-1)
            Ftheta_v = 2.0 * (dThetadR_qvmm[kpt.q][:, J1start:M1stop,
                                                   J2start:J2stop] *
                              ET_mm[np.newaxis]).real.sum(-1).sum(-1)
            Fkin2_av[a1] += Fkin_v
            Fkin2_av[a2] -= Fkin_v
            Ftheta2_av[a1] -= Ftheta_v
            Ftheta2_av[a2] += Ftheta_v
    Fkin_av = Fkin2_av
    Ftheta_av = Ftheta2_av
    self.timer.stop('Not so complicated loop')

    Fkin_av_sum += Fkin_av
    Ftheta_av_sum += Ftheta_av

    return Fkin_av_sum, Ftheta_av_sum

def get_at_den_and_den_paw_blacs(self):
    Fatom_av = np.zeros_like(self.Fref_av)
    Frho_av = np.zeros_like(self.Fref_av)
    Fatom_av_sum = np.zeros_like(self.Fref_av)
    Frho_av_sum = np.zeros_like(self.Fref_av)
    self.dHP_and_dSP_aauim = {}
    self.a2values = {}
    for (a2, a3) in self.atompairs:
        if a3 not in self.a2values:
            self.a2values[a3] = []
        self.a2values[a3].append(a2)

    self.timer.start('Complicated loop')
    for a1, a3 in self.atompairs:
        if a1 == a3:
            # Functions reside on same atom, so their overlap
            # does not change when atom is displaced
            continue
        m1start = self.M_a[a1] - self.M1start
        if m1start >= self.blocksize1:
            continue
        nm1 = self.setups[a1].nao
        m1stop = min(m1start + nm1, self.m1max)
        if m1stop <= 0:
            continue
        dPdR_qvim = self.newtci.dPdR(a3, a1)
        if dPdR_qvim is None:
            continue
        dPdR_qvmi = -dPdR_qvim.transpose(0, 1, 3, 2).conj()
        m1start = max(m1start, 0)
        J1start = max(0, self.M1start - self.M_a[a1])
        J1stop = J1start + m1stop - m1start
        dPdR_qvmi = dPdR_qvmi[:, :, J1start:J1stop, :].copy()
        for a2 in self.a2values[a3]:
            m2start = self.M_a[a2] - self.M2start
            if m2start >= self.blocksize2:
                continue
            nm2 = self.setups[a2].nao
            m2stop = min(m2start + nm2, self.m2max)
            if m2stop <= 0:
                continue
            m2start = max(m2start, 0)
            J2start = max(0, self.M2start - self.M_a[a2])
            J2stop = J2start + m2stop - m2start
            if (a2, a3) in self.dHP_and_dSP_aauim:
                dHP_uim, dSP_uim = self.dHP_and_dSP_aauim[(a2, a3)]
            else:
                P_qim = self.newtci.P(a3, a2)
                if P_qim is None:
                    continue
                P_qmi = P_qim.transpose(0, 2, 1).conj()
                P_qmi = P_qmi[:, J2start:J2stop].copy()
                dH_sp = self.alldH_asp[a3]
                dS_ii = self.setups[a3].dO_ii
                dHP_uim = []
                dSP_uim = []
                for u, kpt in enumerate(self.kpt_u):
                    dH_ii = unpack(dH_sp[kpt.s])
                    dHP_im = np.dot(P_qmi[kpt.q], dH_ii).T.conj()
                    # XXX only need nq of these,
                    # but the looping is over all u
                    dSP_im = np.dot(P_qmi[kpt.q], dS_ii).T.conj()
                    dHP_uim.append(dHP_im)
                    dSP_uim.append(dSP_im)
                    self.dHP_and_dSP_aauim[(a2, a3)] = dHP_uim, dSP_uim
            for u, kpt in enumerate(self.kpt_u):
                rhoT_mm = self.rhoT_umm[u][m1start:m1stop, m2start:m2stop]
                ET_mm = self.ET_umm[u][m1start:m1stop, m2start:m2stop]
                dPdRdHP_vmm = np.dot(dPdR_qvmi[kpt.q], dHP_uim[u])
                dPdRdSP_vmm = np.dot(dPdR_qvmi[kpt.q], dSP_uim[u])
                Fatom_c = 2.0 * (dPdRdHP_vmm *
                                 rhoT_mm).real.sum(-1).sum(-1)
                Frho_c = 2.0 * (dPdRdSP_vmm *
                                ET_mm).real.sum(-1).sum(-1)
                Fatom_av[a1] += Fatom_c
                Fatom_av[a3] -= Fatom_c
                Frho_av[a1] -= Frho_c
                Frho_av[a3] += Frho_c
    self.timer.stop('Complicated loop')

    Fatom_av_sum += Fatom_av
    Frho_av_sum += Frho_av

    return Fatom_av_sum, Frho_av_sum
