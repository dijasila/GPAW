class LCAOHamiltonian:
    """Hamiltonian class for LCAO-basis calculations."""

    def __init__(self, domain, setups):
        self.domain = domain
        self.setups = setups
        self.ng = 2**12

        # Derivative overlaps should be evaluated lazily rather than
        # during initialization  to save memory/time. This is not implemented
        # yet, so presently we disable this.  Change behaviour by setting
        # this boolean.
        self.lcao_forces = False # XXX

        self.initialize_splines(self):
    def initialize(self, paw):
        self.ibzk_kc = paw.ibzk_kc
        self.gamma = paw.gamma
        self.dtype = paw.dtype

    def initialize_lcao(self):
        """Setting up S_kmm, T_kmm and P_kmi for LCAO calculations.

        ======    ==============================================
        S_kmm     Overlap between pairs of basis-functions
        T_kmm     Kinetic-Energy operator
        P_kmi     Overlap between basis-functions and projectors
        ======    ==============================================
        """

        nkpts = len(self.ibzk_kc)

        self.nao = 0
        for nucleus in self.nuclei:
            nucleus.m = self.nao
            self.nao += nucleus.get_number_of_atomic_orbitals()

        for nucleus in self.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nucleus.P_kmi = npy.zeros((nkpts, self.nao, ni), self.dtype)

        if self.lcao_forces:
            for nucleus in self.nuclei:
                ni = nucleus.get_number_of_partial_waves()
                nucleus.dPdR_kcmi = npy.zeros((nkpts, 3, self.nao, ni),
                                              self.dtype)
                # XXX Create "masks" on the nuclei which specify signs
                # and zeros for overlap derivatives.
                # This is inefficient and only a temporary hack!
                m1 = nucleus.m
                m2 = m1 + nucleus.get_number_of_atomic_orbitals()
                mask_mm = npy.zeros((self.nao, self.nao))
                mask_mm[:, m1:m2] = 1.
                mask_mm[m1:m2, :] = -1.
                mask_mm[m1:m2, m1:m2] = 0.
                nucleus.mask_mm = mask_mm

        if self.tci is None:
            self.tci = TwoCenterIntegrals(self.setups, self.ng)

        self.S_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        self.T_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        if self.lcao_forces:
            self.dSdR_kcmm = npy.zeros((nkpts, 3, self.nao, self.nao),
                                       self.dtype)
            self.dTdR_kcmm = npy.zeros((nkpts, 3, self.nao, self.nao),
                                       self.dtype)

        cell_cv = self.gd.domain.cell_cv

        atoms = Atoms(positions=[npy.dot(n.spos_c, cell_cv)
                                 for n in self.nuclei],
                      cell=cell_cv,
                      pbc=self.gd.domain.pbc_c)

        nl = NeighborList([max([phit.get_cutoff()
                                for phit in n.setup.phit_j])
                           for n in self.nuclei], skin=0, sorted=True)
        nl.update(atoms)

        for a, nucleusa in enumerate(self.nuclei):
            sposa = nucleusa.spos_c
            i, offsets = nl.get_neighbors(a)
            for b, offset in zip(i, offsets):
                assert b >= a
                selfinteraction = (a == b and offset.any())
                ma = nucleusa.m
                nucleusb = self.nuclei[b]
                sposb = nucleusb.spos_c + offset

                d = -npy.dot(sposb - sposa, cell_cv)
                r = sqrt(npy.dot(d, d))
                rlY_lm = []
                drlYdR_lmc = []
                for l in range(5):
                    rlY_m = npy.empty(2 * l + 1)
                    Yl(l, d, rlY_m)
                    rlY_lm.append(rlY_m)

                    if self.lcao_forces:
                        drlYdR_mc = npy.empty((2 * l + 1, 3))
                        for m in range(2 * l + 1):
                            L = l**2 + m
                            drlYdR_mc[m, :] = nablaYL(L, d)
                        drlYdR_lmc.append(drlYdR_mc)

                phase_k = npy.exp(-2j * pi * npy.dot(self.ibzk_kc, offset))
                phase_k.shape = (-1, 1, 1)

                # Calculate basis-basis overlaps:
                self.st(a, b, r, d, rlY_lm, drlYdR_lmc, phase_k,
                        selfinteraction)

                # Calculate basis-projector function overlaps:
                # So what's the reason for (-1)**l ?
                # Better do the same thing with drlYdR
                self.p(a, b, r, d,
                       [rlY_m * (-1)**l
                        for l, rlY_m in enumerate(rlY_lm)],
                       [drlYdR_mc * (-1)**l
                        for l, drlYdR_mc in enumerate(drlYdR_lmc)],
                       phase_k)
                if a != b or offset.any():
                    self.p(b, a, r, d,
                           rlY_lm, drlYdR_lmc,
                           phase_k.conj())

        # Only lower triangle matrix elements of S and T have been calculated
        # so far.  Better fill out the rest
        if self.lcao_forces:
            tri1 = npy.tri(self.nao)
            tri2 = npy.tri(self.nao, None, -1)
            def tri2full(matrix, op=1):
                return tri1 * matrix + (op * tri2 * matrix).transpose().conj()

            for S_mm, T_mm, dSdR_cmm, dTdR_cmm in zip(self.S_kmm,
                                                      self.T_kmm,
                                                      self.dSdR_kcmm,
                                                      self.dTdR_kcmm):
                S_mm[:] = tri2full(S_mm)
                T_mm[:] = tri2full(T_mm)
                for c in range(3):
                    dSdR_cmm[c, :, :] = tri2full(dSdR_cmm[c], -1) # XXX
                    dTdR_cmm[c, :, :] = tri2full(dTdR_cmm[c], -1) # XXX

            # These will be the *unmodified* basis function overlaps
            # XXX We may be able to avoid remembering these.
            # The derivative dSdR which depends on the projectors is
            # presently found during force calculations, which means it is
            # not necessary here
            #self.Theta_kmm = self.S_kmm.copy()
            self.dThetadR_kcmm = self.dSdR_kcmm.copy()

        # Add adjustment from O_ii, having already calculated <phi_m1|phi_m2>:
        #
        #                         -----
        #                          \            ~a   a   ~a
        # S    = <phi  | phi  > +   )   <phi  | p > O   <p | phi  >
        #  m1m2      m1     m2     /        m1   i   ij   j     m2
        #                         -----
        #                          aij
        #

        if self.gd.comm.size > 1:
            self.S_kmm /= self.gd.comm.size

        for nucleus in self.my_nuclei:
            dO_ii = nucleus.setup.O_ii
            for S_mm, P_mi in zip(self.S_kmm, nucleus.P_kmi):
                S_mm += npy.dot(P_mi, npy.inner(dO_ii, P_mi).conj())

        if self.gd.comm.size > 1:
            self.gd.comm.sum(self.S_kmm)

    def st(self, a, b, r, R, rlY_lm, drlYdR_lmc, phase_k, selfinteraction):
        """Calculate overlaps and kinetic energy matrix elements for the
        (a,b) pair of atoms."""

        setupa = self.nuclei[a].setup
        ma = self.nuclei[a].m
        nucleusb = self.nuclei[b]
        setupb = nucleusb.setup
        for ja, phita in enumerate(setupa.phit_j):
            ida = (setupa.symbol, ja)
            la = phita.get_angular_momentum_number()
            ma2 = ma + 2 * la + 1
            mb = nucleusb.m
            for jb, phitb in enumerate(setupb.phit_j):
                idb = (setupb.symbol, jb)
                lb = phitb.get_angular_momentum_number()
                mb2 = mb + 2 * lb + 1
                (s_mm, t_mm, dSdR_cmm, dTdR_cmm) = \
                    self.tci.st_overlap3(ida, idb, la, lb, r, R, rlY_lm,
                                         drlYdR_lmc)

                if self.gamma:
                    if selfinteraction:
                        self.S_kmm[0, ma:ma2, mb:mb2] += s_mm.T
                        self.T_kmm[0, ma:ma2, mb:mb2] += t_mm.T
                    self.S_kmm[0, mb:mb2, ma:ma2] += s_mm
                    self.T_kmm[0, mb:mb2, ma:ma2] += t_mm
                else:
                    s_kmm = s_mm[None, :, :] * phase_k.conj()
                    t_kmm = t_mm[None, :, :] * phase_k.conj()
                    if selfinteraction:
                        s1_kmm = s_kmm.transpose(0, 2, 1).conj()
                        t1_kmm = t_kmm.transpose(0, 2, 1).conj()
                        self.S_kmm[:, ma:ma2, mb:mb2] += s1_kmm
                        self.T_kmm[:, ma:ma2, mb:mb2] += t1_kmm
                    self.S_kmm[:, mb:mb2, ma:ma2] += s_kmm
                    self.T_kmm[:, mb:mb2, ma:ma2] += t_kmm

                if self.lcao_forces:
                    # the below is more or less copy-paste of the above
                    # XXX do this in a less silly way
                    if self.gamma:
                        if selfinteraction:
                            dSdRT_cmm = npy.transpose(dSdR_cmm, (0, 2, 1))
                            dTdRT_cmm = npy.transpose(dTdR_cmm, (0, 2, 1))
                            self.dSdR_kcmm[0, :, ma:ma2, mb:mb2] += dSdRT_cmm
                            self.dTdR_kcmm[0, :, ma:ma2, mb:mb2] += dTdRT_cmm
                        self.dSdR_kcmm[0, :, mb:mb2, ma:ma2] += dSdR_cmm
                        self.dTdR_kcmm[0, :, mb:mb2, ma:ma2] += dTdR_cmm
                    else:
                        # XXX cumbersome
                        phase_kc = phase_k[:, None, :, :].repeat(3, axis=1)
                        dSdR_kcmm = dSdR_cmm[None, :, :, :] * phase_kc.conj()
                        dTdR_kcmm = dTdR_cmm[None, :, :, :] * phase_kc.conj()

                        if selfinteraction:
                            dSdR1_kcmm = dSdR_kcmm.transpose(0, 1, 3, 2).conj()
                            dTdR1_kcmm = dTdR_kcmm.transpose(0, 1, 3, 2).conj()
                            self.dSdR_kcmm[:, :, ma:ma2, mb:mb2] += dSdR1_kcmm
                            self.dTdR_kcmm[:, :, ma:ma2, mb:mb2] += dTdR1_kcmm
                        self.dSdR_kcmm[:, :, mb:mb2, ma:ma2] += dSdR_kcmm
                        self.dTdR_kcmm[:, :, mb:mb2, ma:ma2] += dTdR_kcmm

                mb = mb2
            ma = ma2


    def p(self, a, b, r, R, rlY_lm, drlYdR_lm, phase_k):
        """Calculate basis-projector functions overlaps for the (a,b) pair
        of atoms."""

        nucleusb = self.nuclei[b]

        if not (self.lcao_forces or nucleusb.in_this_domain):
            return

        setupa = self.nuclei[a].setup
        ma = self.nuclei[a].m
        setupb = nucleusb.setup
        for ja, phita in enumerate(setupa.phit_j):
            ida = (setupa.symbol, ja)
            la = phita.get_angular_momentum_number()
            ma2 = ma + 2 * la + 1
            ib = 0
            for jb, ptb in enumerate(setupb.pt_j):
                idb = (setupb.symbol, jb)
                lb = ptb.get_angular_momentum_number()
                ib2 = ib + 2 * lb + 1
                p_mi, dPdR_cmi = self.tci.p(ida, idb, la, lb, r, R,
                                            rlY_lm, drlYdR_lm)
                if self.gamma and nucleusb.in_this_domain:
                    nucleusb.P_kmi[0, ma:ma2, ib:ib2] += p_mi
                elif nucleusb.in_this_domain:
                    nucleusb.P_kmi[:, ma:ma2, ib:ib2] += (p_mi[None, :, :] *
                                                          phase_k)

                if self.lcao_forces:
                    if self.gamma:
                        nucleusb.dPdR_kcmi[0, :, ma:ma2, ib:ib2] += dPdR_cmi
                    else: # XXX phase_kc
                        phase_kc = phase_k[:, None, :, :].repeat(3, axis=1)
                        nucleusb.dPdR_kcmi[:, :, ma:ma2, ib:ib2] += \
                            dPdR_cmi[None, :, :, :] * phase_kc

                ib = ib2
            ma = ma2


