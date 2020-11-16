import warnings

import numpy as np

from ase.units import Hartree

from gpaw.occupations import ZeroKelvin


class OccupationsMOM(ZeroKelvin):
    def __init__(self, occupations,
                 constraints=None,
                 space='full',
                 width=0.0,
                 width_increment=0.01,
                 niter_smearing=None):
        ZeroKelvin.__init__(self, True)
        self.occupations = np.array(occupations)
        self.constraints = constraints
        self.space = space
        self.width = width / Hartree
        self.width_increment = width_increment / Hartree
        self.niter_smearing = niter_smearing

        self.name = 'mom'
        self.iters = 0
        self.ne = None

        if self.space == 'reduced':
            assert self.constraints is not None, \
                'Provide constraints as MOMConstraint objects'
        if self.space == 'full' and self.width != 0:
            warnings.warn("Smearing is not used when space='full'")

    def todict(self):
        dct = {'name': self.name}
        if self.width != 0.0:
            dct['width'] = self.width * Hartree
        return dct

    def __str__(self):
        s = 'Occupation numbers:\n'
        s += '  Delta SCF with Maximum Overlap Method\n'
        s += '  Smearing of constraints: '
        if self.width == 0.0:
            s += 'off\n'
        else:
            s += '{0:.4f} eV\n'.format(self.width * Hartree)
        return s

    def calculate(self, wfs):
        occ = self.occupations.copy()

        if self.iters == 0 and self.space == 'full':
            self.occupation = occ
            ZeroKelvin.calculate(self, wfs)
            self.initialize_reference_orbitals(wfs)

        for kpt in wfs.kpt_u:
            if self.space == 'full':
                if self.iters == 0:
                    continue
                else:
                    occ[kpt.s].fill(0)

                    # Compute projections within equally occupied subspaces
                    f_n_unique = np.unique(kpt.f_n)
                    for f in f_n_unique:
                        if f >= 1.0e-10:
                            occupied = kpt.f_n == f
                            n_occ = len(kpt.f_n[occupied])
                            P = self.calculate_mom_projections(wfs, kpt, f)
                            P_max = np.argpartition(P, -n_occ)[-n_occ:]
                            P_max.sort() # Do we need this?

                            occ[kpt.s][P_max] = f

            elif self.space == 'reduced':
                for c in self.constraints:
                    if c[2] != kpt.s:
                        continue

                    orb = c[1]
                    max_overlap = orb.get_maximum_overlap(wfs, kpt,
                                                          c[0], self.iters)

                    occ_new = occ[kpt.s][max_overlap] + c[0]
                    if (occ_new < 0.0) or (occ_new > kpt.weight):
                        continue

                    if self.width != 0.0:
                        # Gaussian smearing of constraints
                        mask, gauss = self.smear_gaussian(kpt, occ,
                                                          c[0], max_overlap)
                        occ[kpt.s][mask] += (c[0] * gauss)
                    else:
                        occ[kpt.s][max_overlap] += c[0]

        if self.ne is None:
            self.ne = occ.sum(1)
        else:
            # TODO: Works only for spin polarized calculations
            for kpt in wfs.kpt_u:
                 occ[kpt.s] = self.check_number_of_electrons(kpt, occ[kpt.s])

        self.occupation = occ
        ZeroKelvin.calculate(self, wfs)

        self.iters += 1

    def spin_paired(self, wfs):
        return self.fixed_moment(wfs)

    def fixed_moment(self, wfs):
        egs_name = getattr(wfs.eigensolver, "name", None)
        magmom = 0.0

        for kpt in wfs.kpt_u:
            wfs.bd.distribute(self.occupation[kpt.s], kpt.f_n)

            # Compute the magnetic moment
            if wfs.nspins == 2:
                if kpt.s == 0:
                    magmom += self.occupation[kpt.s].sum()
                else:
                    magmom -= self.occupation[kpt.s].sum()

        self.magmom = wfs.kptband_comm.sum(magmom)

    def initialize_reference_orbitals(self, wfs):
        if wfs.mode == 'lcao':
            self.c_ref = {}

            for kpt in wfs.kpt_u:
                self.c_ref[kpt.s] = {}

                # Initialize ref orbitals for each equally occupied subspace
                f_n_unique = np.unique(kpt.f_n)
                for f in f_n_unique:
                    if f >= 1.0e-10:
                        occupied = kpt.f_n == f
                        self.c_ref[kpt.s][f] = kpt.C_nM[occupied].copy()
        else:
            self.wf = {}
            self.p_an = {}
            for kpt in wfs.kpt_u:
                self.wf[kpt.s] = {}
                self.p_an[kpt.s] = {}

                # Initialize ref orbitals for each equally occupied subspace
                f_n_unique = np.unique(kpt.f_n)
                for f in f_n_unique:
                    if f >= 1.0e-10:
                        occupied = kpt.f_n == f
                        # Pseudo wave functions
                        self.wf[kpt.s][f] = kpt.psit_nG[occupied].copy()
                        # PAW projection
                        self.p_an[kpt.s][f] = \
                            dict([(a, np.dot(wfs.setups[a].dO_ii,
                                             P_ni[occupied].T))
                                  for a, P_ni in kpt.P_ani.items()])

    def calculate_mom_projections(self, wfs, kpt, f):
        if wfs.mode == 'lcao':
            P = np.dot(self.c_ref[kpt.s][f].conj(),
                       np.dot(kpt.S_MM, kpt.C_nM.T))
            P = np.sum(np.absolute(P)**2, axis=0)
            P = P ** 0.5
        else:
            # Pseudo wave functions overlaps
            P = wfs.integrate(self.wf[kpt.s][f][:], kpt.psit_nG[:], True)

            # PAW corrections
            P_corr = np.zeros_like(P)
            for a, p_a in self.p_an[kpt.s][f].items():
                P_corr += np.dot(kpt.P_ani[a].conj(), p_a).T
            P_corr = np.ascontiguousarray(P_corr)
            wfs.gd.comm.sum(P_corr)

            # Sum pseudo wave and PAW contributions
            P += P_corr
            P = np.sum(np.absolute(P)**2, axis=0)
            P = P ** 0.5

        return P

    def sort_wavefunctions(self, wfs, kpt):
        # Works only for LCAO calculations
        occupied = kpt.f_n > 1.0e-10
        n_occ = len(kpt.f_n[occupied])

        if n_occ == 0.0:
            return

        if np.min(kpt.f_n[:n_occ]) == 0:
            ind_occ = np.argwhere(occupied)
            ind_unocc = np.argwhere(~occupied)
            ind = np.vstack((ind_occ, ind_unocc))

            # Sort coefficients, occupation numbers, eigenvalues
            if wfs.mode == 'lcao':
                kpt.C_nM = np.squeeze(kpt.C_nM[ind])
            else:
                kpt.psit_nG[:] = np.squeeze(kpt.psit_nG[ind])
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            kpt.f_n = np.squeeze(kpt.f_n[ind])
            kpt.eps_n = np.squeeze(kpt.eps_n[ind])

    def smear_gaussian(self, kpt, occ, c, n):
        if c < 0:
            mask = (occ[kpt.s] != 0)
        else:
            mask = (occ[kpt.s] == 0)

        e = kpt.eps_n[mask]
        de2 = -(e - kpt.eps_n[n]) ** 2
        gauss = (1 / (self.width * np.sqrt(2 * np.pi)) *
                 np.exp(de2 / (2 * self.width ** 2)))
        gauss /= sum(gauss)

        return mask, gauss

    def check_number_of_electrons(self, kpt, occ):
        ne_diff = occ.sum() - self.ne[kpt.s]
        lumo = int(self.ne[kpt.s])
        homo = int(lumo - 1)

        # Check that total number of electrons is conserved
        while ne_diff != 0:
            if ne_diff < 0:
                occ[lumo] += 1
                lumo += 1
                ne_diff += 1
            else:
                occ[homo] -= 1
                homo -= 1
                ne_diff -= 1

        return occ

    def reset(self, wfs):
        if self.iters > 1:
            self.iters = 0
            if self.space == 'full':
                for u, kpt in enumerate(wfs.kpt_u):
                    self.occupations[u] = kpt.f_n

    def init_ref_orb2(self,wfs):
        self.iters = 0
        self.initialize_reference_orbitals(wfs)
        self.iters += 1

class MOMConstraint:
    def __init__(self, n, nstart=0, nend=None):
        self.n = n
        self.nstart = nstart
        self.nend = nend

    def initialize(self, wfs, c):
        nocc = wfs.nvalence // 2
        if self.nend is None:
            if c < 0:
                self.nend = nocc
            else:
                self.nend = self.nbands - nocc
        if c < 0:
            self.ini = self.nstart
            self.fin = self.nend
        else:
            self.ini = nocc + self.nstart
            self.fin = nocc + self.nend

    def update_target_orbital(self, wfs, kpt):
        if wfs.mode == 'lcao':
            self.c_n = kpt.C_nM[self.n].copy()
        else:
            self.wf_n = kpt.psit_nG[self.n].copy()
            self.p_an = dict([(a, np.dot(wfs.setups[a].dO_ii, P_ni[self.n]))
                               for a, P_ni in kpt.P_ani.items()])

    def get_maximum_overlap(self, wfs, kpt, c, iters):
        self.nbands = wfs.bd.nbands

        if iters == 0:
            self.initialize(wfs, c)
            self.update_target_orbital(wfs, kpt)
        ini = self.ini
        fin = self.fin

        P_n = np.zeros(self.nbands)
        if wfs.mode == 'lcao':
            if kpt.S_MM is None:
                return self.n
            else:
                P_n[ini:fin] = np.dot(self.c_n.conj(),
                                  np.dot(kpt.S_MM, kpt.C_nM[ini:fin].T))
        else:
            # Pseudo wave functions overlaps
            P_n[ini:fin] = wfs.integrate(self.wf_n, kpt.psit_nG[ini:fin], False)

            if iters > 1:
                # Add PAW corrections
                for a, p_a in self.p_an.items():
                    P_n[ini:fin] += np.dot(kpt.P_ani[a][ini:fin].conj(), p_a)
            wfs.gd.comm.sum(P_n)

        P_n = P_n ** 2

        # Update index of target orbital
        self.n = np.argmax(P_n)

        if iters == 1:
            # If positions have changed than overlap operators change
            # So reinitialize reference orbitals
            self.update_target_orbital(wfs, kpt)

        return self.n

